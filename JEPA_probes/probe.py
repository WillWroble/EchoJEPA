"""Study-level JEPA probes with EchoFocus aggregation.

Caches video-level embeddings in output_dir on first run.
Re-runs with same output_dir skip extraction and go straight to probes.

Usage:
    python -u probe.py \
        --checkpoint /path/to/latest.pt \
        --output_dir results/epoch240 \
        --train_manifest manifests/jepa_probe_platon_train.txt \
        --val_manifest manifests/jepa_probe_platon_val.txt
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from pathlib import Path

from encoder import load_encoder, extract_embeddings


# ---------------------------------------------------------------------------
# Model (mean-pool EchoFocus)
# ---------------------------------------------------------------------------

class EchoFocus(nn.Module):
    def __init__(self, input_dim=768, n_heads=8, ff_dim=768, dropout=0.1, n_targets=1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(input_dim)
        self.head = nn.Linear(input_dim, n_targets)

    def forward(self, x):
        h = self.encoder(x)
        h = h.mean(dim=1)
        h = self.norm(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StudyDataset(Dataset):
    def __init__(self, study_ids, emb_by_study, labels, n_videos=48):
        self.ids = study_ids
        self.emb = emb_by_study
        self.labels = labels
        self.n = n_videos

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        emb = self.emb[sid]
        n = emb.shape[0]
        sel = np.random.choice(n, self.n, replace=(n < self.n))
        return torch.from_numpy(emb[sel]), torch.tensor(self.labels[sid], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

def get_embeddings(checkpoint, manifest, data_dir, output_dir, split,
                   videos_per_study, device):
    cache = Path(output_dir) / f'{split}_embeddings.npz'
    if cache.exists():
        print(f"Cache hit: {cache}", flush=True)
        return np.load(cache, allow_pickle=True)
    print(f"Cache miss — extracting {split}...", flush=True)
    sids = [l.strip() for l in open(manifest)]
    model = load_encoder(checkpoint, device)
    #data = extract_embeddings(model, sids, data_dir, videos_per_study, device)
    
    data = extract_embeddings(model, sids, data_dir, videos_per_study, device,
                              partial_path=str(Path(output_dir) / f'_partial_{split}.npz'))
    
    np.savez(cache, **data)
    print(f"Cached → {cache} ({data['embeddings'].shape})", flush=True)
    del model; torch.cuda.empty_cache()
    return data


def group_by_study(data):
    out = {}
    for emb, sid in zip(data['embeddings'], data['study_ids'].astype(str)):
        out.setdefault(sid, []).append(emb)
    return {k: np.stack(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_probe(model, train_loader, val_loader, epochs, lr, device, task):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss() if task == 'cls' else nn.MSELoss()
    best_loss, best_sd = float('inf'), None

    for ep in range(epochs):
        model.train()
        t_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            t_loss += loss.item()

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                v_loss += loss_fn(model(x), y).item()
        v_loss /= len(val_loader)

        if v_loss < best_loss:
            best_loss = v_loss
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 5 == 0:
            print(f"  ep {ep+1}: train={t_loss/len(train_loader):.4f} val={v_loss:.4f}",
                  flush=True)

    model.load_state_dict(best_sd)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_classification(model, loader, names, device):
    preds, labels = [], []
    for x, y in loader:
        preds.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
        labels.append(y.numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)

    rows = []
    for i, name in enumerate(names):
        y = labels[:, i]
        if y.sum() < 5 or (1 - y).sum() < 5:
            continue
        rows.append({'code': name, 'auroc': roc_auc_score(y, preds[:, i]),
                     'n_pos': int(y.sum())})
    return pd.DataFrame(rows).sort_values('auroc', ascending=False)


@torch.no_grad()
def eval_regression(model, loader, names, scaler, device):
    preds, labels = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).cpu().numpy())
        labels.append(y.numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)

    # inverse standardize
    preds = preds * scaler['std'] + scaler['mean']
    labels = labels * scaler['std'] + scaler['mean']

    rows = []
    for i, name in enumerate(names):
        mask = ~np.isnan(labels[:, i])
        if mask.sum() < 10:
            continue
        y, p = labels[mask, i], preds[mask, i]
        ss_res = ((y - p) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        rows.append({'measurement': name, 'mae': np.abs(y - p).mean(),
                     'r2': 1 - ss_res / ss_tot if ss_tot > 0 else 0,
                     'n': int(mask.sum())})
    return pd.DataFrame(rows).sort_values('r2', ascending=False)


# ---------------------------------------------------------------------------
# UMAPs
# ---------------------------------------------------------------------------

def make_umaps(data, fyler_df, meas_df, output_dir, n_sub=50000):
    import umap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    embs = data['embeddings']
    sids = data['study_ids'].astype(str)
    if len(embs) > n_sub:
        idx = np.random.choice(len(embs), n_sub, replace=False)
        embs, sids = embs[idx], sids[idx]

    print(f"UMAP on {len(embs)} points...", flush=True)
    coords = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine',
                        random_state=42).fit_transform(embs)

    # CHD: color by total positive Fyler codes
    if fyler_df is not None:
        fcols = [c for c in fyler_df.columns if c.startswith('fyler_')]
        lookup = fyler_df.set_index('sid')[fcols]
        counts = []
        for s in sids:
            counts.append(lookup.loc[s].sum() if s in lookup.index else np.nan)
        counts = np.array(counts, dtype=float)
        mask = ~np.isnan(counts)
        if mask.sum() > 100:
            fig, ax = plt.subplots(figsize=(10, 8))
            sc = ax.scatter(coords[mask, 0], coords[mask, 1],
                            c=counts[mask], cmap='viridis', s=1, alpha=0.4)
            plt.colorbar(sc, ax=ax, label='# Fyler codes')
            ax.set_title('JEPA embeddings — Fyler burden')
            plt.tight_layout()
            plt.savefig(Path(output_dir) / 'umap_chd.png', dpi=200)
            plt.close()
            print("Saved umap_chd.png", flush=True)

    # EF: color by ejection fraction
    if meas_df is not None and 'EF05' in meas_df.columns:
        lookup = meas_df.drop_duplicates('eid').set_index('eid')['EF05']
        vals = np.array([lookup.get(s, np.nan) for s in sids], dtype=float)
        mask = ~np.isnan(vals)
        if mask.sum() > 100:
            fig, ax = plt.subplots(figsize=(10, 8))
            sc = ax.scatter(coords[mask, 0], coords[mask, 1],
                            c=vals[mask], cmap='RdYlBu_r', s=1, alpha=0.4)
            plt.colorbar(sc, ax=ax, label='EF')
            ax.set_title('JEPA embeddings — Ejection Fraction')
            plt.tight_layout()
            plt.savefig(Path(output_dir) / 'umap_ef.png', dpi=200)
            plt.close()
            print("Saved umap_ef.png", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--train_manifest', required=True)
    p.add_argument('--val_manifest', required=True)
    p.add_argument('--data_dir', default='/lab-share/Cardio-Mayourian-e2/Public/Echo_Internal_30k')
    p.add_argument('--fyler_labels', default='/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/fyler_labels_v2.csv')
    p.add_argument('--fyler_lines', default='/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/fyler_lines.csv')
    p.add_argument('--measurements', default='/lab-share/Cardio-Mayourian-e2/Public/Echo_Labels/echo_measurements_090425.csv')
    p.add_argument('--videos_per_study', type=int, default=48)
    p.add_argument('--min_pos', type=int, default=20)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- extract / load cache ----
    train_data = get_embeddings(args.checkpoint, args.train_manifest, args.data_dir,
                                args.output_dir, 'train', args.videos_per_study, args.device)
    val_data = get_embeddings(args.checkpoint, args.val_manifest, args.data_dir,
                              args.output_dir, 'val', args.videos_per_study, args.device)

    train_by = group_by_study(train_data)
    val_by = group_by_study(val_data)

    # ---- Fyler probes ----
    print("Loading Fyler labels...", flush=True)
    fyler_df = pd.read_csv(args.fyler_labels)
    fyler_df['sid'] = fyler_df['sid'].astype(str)
    fcols = [c for c in fyler_df.columns if c.startswith('fyler_')]

    lines_df = pd.read_csv(args.fyler_lines)
    code_map = dict(zip(lines_df['fyler_code'].astype(str).str.zfill(4), lines_df['line']))

    # filter codes with enough positives in train
    train_sids = set(train_by)
    ft = fyler_df[fyler_df['sid'].isin(train_sids)]
    valid_codes = [c for c in fcols if ft[c].sum() >= args.min_pos]
    print(f"  {len(valid_codes)} codes with >={args.min_pos} pos in train", flush=True)

    fyler_idx = fyler_df.set_index('sid')

    def fyler_labels(sids):
        return {s: fyler_idx.loc[s, valid_codes].values.astype(np.float32)
                for s in sids if s in fyler_idx.index}

    tr_fy = fyler_labels(train_by)
    va_fy = fyler_labels(val_by)
    tr_ids = sorted(set(train_by) & set(tr_fy))
    va_ids = sorted(set(val_by) & set(va_fy))
    print(f"  Fyler: {len(tr_ids)} train, {len(va_ids)} val", flush=True)

    if tr_ids:
        print("Training Fyler probe...", flush=True)
        tr_dl = DataLoader(StudyDataset(tr_ids, train_by, tr_fy, args.videos_per_study),
                           batch_size=args.batch_size, shuffle=True, num_workers=4)
        va_dl = DataLoader(StudyDataset(va_ids, val_by, va_fy, args.videos_per_study),
                           batch_size=args.batch_size, num_workers=4)
        model = EchoFocus(n_targets=len(valid_codes)).to(args.device)
        model = train_probe(model, tr_dl, va_dl, args.epochs, args.lr, args.device, 'cls')

        names = [code_map.get(c.replace('fyler_', ''), c) for c in valid_codes]
        res = eval_classification(model, va_dl, names, args.device)
        res.to_csv(out / 'fyler_aurocs.csv', index=False)
        print(f"  mean={res['auroc'].mean():.4f}  median={res['auroc'].median():.4f}  "
              f"({len(res)} codes) → fyler_aurocs.csv", flush=True)

    # ---- Measurement probes ----
    print("Loading measurements...", flush=True)
    meas_df = pd.read_csv(args.measurements)
    meas_df['eid'] = meas_df['eid'].astype(str)
    meas_cols = ['AA01', 'AR01', 'EF05', 'LD05', 'LE05', 'LE07', 'LM12', 'LS04',
                 'MA02', 'MP01', 'PA02', 'TA01', 'LA34', 'RA06', 'RV19', 'ST39',
                 'ST49', 'RV32']
    meas_cols = [c for c in meas_cols if c in meas_df.columns]
    meas_idx = meas_df.drop_duplicates('eid').set_index('eid')

    # filter columns with enough data in train
    train_meas_sids = set(train_by) & set(meas_idx.index.astype(str))
    valid_meas = [c for c in meas_cols
                  if meas_idx.loc[meas_idx.index.astype(str).isin(train_meas_sids), c]
                  .notna().sum() >= 100]
    print(f"  {len(valid_meas)} measurements with >=100 values", flush=True)

    if valid_meas:
        def meas_labels(sids):
            out = {}
            for s in sids:
                if s in meas_idx.index.astype(str).values:
                    row = meas_idx.loc[meas_idx.index.astype(str) == s].iloc[0]
                    out[s] = np.array([row[c] if pd.notna(row[c]) else np.nan
                                       for c in valid_meas], dtype=np.float32)
            return out

        tr_ms = meas_labels(train_by)
        va_ms = meas_labels(val_by)
        tr_m_ids = sorted(set(train_by) & set(tr_ms))
        va_m_ids = sorted(set(val_by) & set(va_ms))
        print(f"  Measurements: {len(tr_m_ids)} train, {len(va_m_ids)} val", flush=True)

        # standardize on train (NaN-safe)
        train_vals = np.stack([tr_ms[s] for s in tr_m_ids])
        scaler = {
            'mean': np.nanmean(train_vals, axis=0),
            'std': np.nanstd(train_vals, axis=0) + 1e-8,
        }
        for d in [tr_ms, va_ms]:
            for s in d:
                raw = d[s].copy()
                d[s] = (raw - scaler['mean']) / scaler['std']
                d[s][np.isnan(raw)] = 0.0  # zero after standardize for masked positions

        print("Training measurement probe...", flush=True)
        tr_dl = DataLoader(StudyDataset(tr_m_ids, train_by, tr_ms, args.videos_per_study),
                           batch_size=args.batch_size, shuffle=True, num_workers=4)
        va_dl = DataLoader(StudyDataset(va_m_ids, val_by, va_ms, args.videos_per_study),
                           batch_size=args.batch_size, num_workers=4)
        model = EchoFocus(n_targets=len(valid_meas)).to(args.device)
        model = train_probe(model, tr_dl, va_dl, args.epochs, args.lr, args.device, 'reg')

        res = eval_regression(model, va_dl, valid_meas, scaler, args.device)
        res.to_csv(out / 'measurement_results.csv', index=False)
        print(f"  → measurement_results.csv ({len(res)} measurements)", flush=True)
        for _, r in res.iterrows():
            print(f"    {r['measurement']}: MAE={r['mae']:.4f} R2={r['r2']:.4f} (n={r['n']})",
                  flush=True)

    # ---- UMAPs ----
    print("Generating UMAPs...", flush=True)
    make_umaps(val_data, fyler_df, meas_df, args.output_dir)

    print("Done.", flush=True)


if __name__ == '__main__':
    main()
