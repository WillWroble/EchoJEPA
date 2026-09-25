"""UMAP of v3 clip encoder embeddings colored by study.

Loads trained ViT-L + AttentivePooler from video_pretraining_v3,
extracts per-clip 1024d CLS embeddings, produces UMAP +
intra/inter similarity histogram (from analyze.py).

Usage:
    python -u umap_clip_encoder.py \
        --encoder_checkpoint .../echojepa_vitl_mimic_bch_cooldown/latest.pt \
        --v3_checkpoint .../video_pretraining_v3/results/v1/latest.pt \
        --data_dir .../Echo_Internal_30k \
        --manifest .../platon_val.txt \
        --output_dir results/v3_umap \
        --n_studies 50
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import umap

import src.models.vision_transformer as vit
from src.models.attentive_pooler import AttentivePooler
from encoder import load_and_sample_clips, _clean_backbone_key


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(encoder_ckpt, v3_ckpt, device, model_name='vit_large',
                embed_dim=1024, num_heads=16, depth=4):
    encoder_fn = getattr(vit, model_name)
    encoder = encoder_fn(
        img_size=(224, 224), patch_size=16, num_frames=16,
        tubelet_size=2, use_rope=True, use_sdpa=True, uniform_power=True,
    )
    #ckpt = torch.load(encoder_ckpt, map_location='cpu', weights_only=False)
    #state = ckpt.get('target_encoder') or ckpt.get('encoder')
    #state = _clean_backbone_key(state)
    
    v7_ckpt = torch.load(encoder_ckpt, map_location='cpu', weights_only=False)
    state = {k.replace('module.', ''): v for k, v in v7_ckpt['encoder_vitl'].items()}

    msg = encoder.load_state_dict(state, strict=False)
    print(f"  encoder: missing={len(msg.missing_keys)} "
          f"unexpected={len(msg.unexpected_keys)}", flush=True)
    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    pooler = AttentivePooler(
        num_queries=1, embed_dim=embed_dim,
        num_heads=num_heads, depth=depth,
    )
    v3 = torch.load(v3_ckpt, map_location='cpu', weights_only=False)
    pooler_state = {k.replace('module.', ''): v for k, v in v3['pooler'].items()}
    pooler.load_state_dict(pooler_state)
    pooler.to(device).eval()
    for p in pooler.parameters():
        p.requires_grad_(False)

    print(f"  pooler: {sum(p.numel() for p in pooler.parameters()) / 1e6:.1f}M params",
          flush=True)
    return encoder, pooler


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_clip_embeddings(encoder, pooler, study_ids, data_dir,
                            num_clips=1, device='cuda'):
    data_dir = Path(data_dir)
    all_embs, all_sids = [], []

    for i, sid in enumerate(study_ids):
        study_dir = data_dir / f"{sid}_trim"
        avis = sorted(study_dir.glob("*.avi")) if study_dir.exists() else []
        if not avis:
            continue
        for avi in avis:
            clips = load_and_sample_clips(avi, num_clips=num_clips)
            if clips is None:
                continue
            clips = clips.to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                patches = encoder(clips)
                #clip_embs = pooler(patches).squeeze(1)
                clip_embs = patches.mean(dim=1)
            for emb in clip_embs.float().cpu().numpy():
                all_embs.append(emb)
                all_sids.append(sid)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(study_ids)} studies "
                  f"({len(all_embs)} clips)", flush=True)

    return np.stack(all_embs).astype(np.float32), np.array(all_sids)


# ---------------------------------------------------------------------------
# Analysis (from video_pretraining/analyze.py)
# ---------------------------------------------------------------------------

def sample_by_study(embs, study_ids, n_studies=200):
    unique = np.unique(study_ids)
    chosen = np.random.choice(unique, min(n_studies, len(unique)), replace=False)
    mask = np.isin(study_ids, chosen)
    return embs[mask], study_ids[mask]


def similarity_ratios(embs, study_ids):
    sim = cosine_similarity(embs)
    ids = np.array(study_ids)
    same = (ids[:, None] == ids[None, :])
    np.fill_diagonal(same, False)
    diff = ~(ids[:, None] == ids[None, :])
    intra = sim[same].mean()
    inter = sim[diff].mean()
    return intra, inter, intra / inter if inter != 0 else float("inf")


def knn_enrichment(embs, study_ids, k=10):
    sim = cosine_similarity(embs)
    np.fill_diagonal(sim, -1)
    ids = np.array(study_ids)
    enrichments = []
    for i in range(len(embs)):
        neighbors = np.argsort(sim[i])[-k:]
        enrichments.append((ids[neighbors] == ids[i]).sum() / k)
    return np.mean(enrichments)


def mean_first_neighbor_rank(embs, study_ids):
    sim = cosine_similarity(embs)
    np.fill_diagonal(sim, -1)
    ids = np.array(study_ids)
    ranks = []
    for i in range(len(embs)):
        sorted_idx = np.argsort(sim[i])[::-1]
        sorted_ids = ids[sorted_idx]
        same = np.where(sorted_ids == ids[i])[0]
        if len(same) > 0:
            ranks.append(same[0] + 1)
    return np.mean(ranks), np.median(ranks)


def within_study_variance(embs, study_ids):
    ids = np.array(study_ids)
    variances = []
    for sid in np.unique(ids):
        mask = ids == sid
        if mask.sum() < 2:
            continue
        variances.append(np.var(embs[mask], axis=0).mean())
    return np.mean(variances)


def plot_umap(embs, study_ids, title, ax):
    coords = umap.UMAP(n_neighbors=15, min_dist=0.1,
                        random_state=42).fit_transform(embs)
    uid, int_ids = np.unique(study_ids, return_inverse=True)
    ax.scatter(coords[:, 0], coords[:, 1],
               c=int_ids % 20, cmap="tab20", s=1, alpha=0.5)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_similarity_histograms(embs, study_ids, title, ax):
    sim = cosine_similarity(embs)
    ids = np.array(study_ids)
    same = (ids[:, None] == ids[None, :])
    np.fill_diagonal(same, False)
    diff = ~same & ~np.eye(len(ids), dtype=bool)
    intra_sims = sim[same]
    inter_sims = np.random.choice(sim[diff],
                                  min(len(intra_sims) * 5, diff.sum()),
                                  replace=False)
    ax.hist(inter_sims, bins=80, alpha=0.5, label="inter-study", density=True)
    ax.hist(intra_sims, bins=80, alpha=0.5, label="intra-study", density=True)
    ax.set_title(title)
    ax.legend()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--encoder_checkpoint', required=True)
    p.add_argument('--v3_checkpoint', required=False)
    p.add_argument('--data_dir', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--n_studies', type=int, default=50)
    p.add_argument('--num_clips', type=int, default=1)
    p.add_argument('--model_name', default='vit_large')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / 'clip_embeddings.npz'

    np.random.seed(42)

    # Sample studies from manifest
    study_ids = [str(int(float(x)))
                 for x in Path(args.manifest).read_text().split()]
    np.random.shuffle(study_ids)
    study_ids = study_ids[:args.n_studies]
    print(f"Selected {len(study_ids)} studies", flush=True)

    # Extract or load cache
    if cache.exists():
        print(f"Cache hit: {cache}", flush=True)
        data = np.load(cache)
        embs = data['embeddings']
        sids = data['study_ids']
    else:
        print("Loading models...", flush=True)
        encoder, pooler = load_models(
            args.encoder_checkpoint, args.v3_checkpoint,
            args.device, args.model_name)

        print("Extracting clip embeddings...", flush=True)
        embs, sids = extract_clip_embeddings(
            encoder, pooler, study_ids, args.data_dir,
            num_clips=args.num_clips, device=args.device)

        np.savez(cache, embeddings=embs, study_ids=sids)
        print(f"Cached → {cache} ({embs.shape})", flush=True)
        del encoder, pooler
        torch.cuda.empty_cache()

    print(f"\nEmbeddings: {embs.shape} from {len(np.unique(sids))} studies")
    print(f"Avg clips/study: {len(embs) / len(np.unique(sids)):.0f}")

    # Metrics
    intra, inter, ratio = similarity_ratios(embs, sids)
    knn = knn_enrichment(embs, sids, k=10)
    var = within_study_variance(embs, sids)
    rank_mean, rank_med = mean_first_neighbor_rank(embs, sids)

    print(f"\n{'Metric':<35} {'Value':>10}")
    print("-" * 47)
    print(f"{'Intra-study similarity':<35} {intra:>10.4f}")
    print(f"{'Inter-study similarity':<35} {inter:>10.4f}")
    print(f"{'Intra/Inter ratio':<35} {ratio:>10.2f}x")
    print(f"{'KNN-10 study enrichment':<35} {knn:>10.4f}")
    print(f"{'Within-study variance':<35} {var:>10.6f}")
    print(f"{'First same-study neighbor (mean)':<35} {rank_mean:>10.1f}")
    print(f"{'First same-study neighbor (median)':<35} {rank_med:>10.1f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_similarity_histograms(embs, sids, "Cosine similarity", axes[0])
    plot_umap(embs, sids, "UMAP", axes[1])
    plt.tight_layout()
    fig_path = out / 'umap_clip_encoder.png'
    plt.savefig(fig_path, dpi=150)
    print(f"\nSaved figure to {fig_path}")


if __name__ == '__main__':
    main()
