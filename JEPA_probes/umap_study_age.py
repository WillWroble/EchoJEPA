"""UMAP of JEPA clip embeddings colored by study date.

Samples 50K videos from val manifest studies, encodes, plots.
Checks for equipment/era clustering in the embedding space.

Usage:
    python -u umap_study_age.py \
        --checkpoint /path/to/latest.pt \
        --output_dir results/epoch240
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime

from encoder import load_encoder, encode_video


def parse_study_date(s):
    try:
        return datetime.strptime(str(s).strip(), '%m%d%y%H%M%S')
    except Exception:
        return None


def to_fractional_year(dt):
    if dt is None:
        return np.nan
    return dt.year + (dt.timetuple().tm_yday - 1) / 365.25


def plot_colored(coords, values, title, path, cmap='viridis'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    valid = ~np.isnan(values)
    if (~valid).any():
        ax.scatter(coords[~valid, 0], coords[~valid, 1], s=0.3, alpha=0.1, c='lightgray')
    vmin = np.nanpercentile(values, 2)
    vmax = np.nanpercentile(values, 98)
    sc = ax.scatter(coords[valid, 0], coords[valid, 1], s=0.5, alpha=0.3,
                    c=values[valid], cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(sc, ax=ax, shrink=0.8)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--val_manifest', default='/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/jepa_probe_platon_val.txt')
    p.add_argument('--data_dir', default='/lab-share/Cardio-Mayourian-e2/Public/Echo_Pulled/Echo_Internal_30k')
    p.add_argument('--reports', default='/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/echo_reports_v2.csv')
    p.add_argument('--n_samples', type=int, default=50000)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / 'study_date_embeddings.npz'

    # load study dates
    reports = pd.read_csv(args.reports, dtype=str, usecols=['study_id', 'study_date'])
    reports = reports.drop_duplicates('study_id').set_index('study_id')
    date_lookup = {}
    for sid in reports.index:
        dt = parse_study_date(reports.loc[sid, 'study_date'])
        date_lookup[sid] = to_fractional_year(dt)
    print(f"Loaded {len(date_lookup)} study dates", flush=True)

    # glob all AVIs from val studies
    val_sids = [l.strip() for l in open(args.val_manifest)]
    data_dir = Path(args.data_dir)
    all_avis = []
    for sid in val_sids:
        study_dir = data_dir / f"{sid}_trim"
        if study_dir.exists():
            for avi in study_dir.glob("*.avi"):
                all_avis.append((sid, avi))
    print(f"Found {len(all_avis)} videos across {len(val_sids)} studies", flush=True)

    # subsample
    rng = np.random.RandomState(42)
    if len(all_avis) > args.n_samples:
        idx = rng.choice(len(all_avis), args.n_samples, replace=False)
        all_avis = [all_avis[i] for i in idx]
    print(f"Sampled {len(all_avis)} videos", flush=True)

    # extract or load cache
    if cache.exists():
        print(f"Cache hit: {cache}", flush=True)
        data = np.load(cache, allow_pickle=True)
        embs = data['embeddings']
        years = data['years']
    else:
        print("Extracting embeddings...", flush=True)
        model = load_encoder(args.checkpoint, args.device)

        embs, years = [], []
        for i, (sid, avi) in enumerate(all_avis):
            emb = encode_video(model, avi, args.device)
            if emb is not None:
                embs.append(emb)
                years.append(date_lookup.get(sid, np.nan))
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(all_avis)} ({len(embs)} ok)", flush=True)

        embs = np.stack(embs).astype(np.float32)
        years = np.array(years, dtype=np.float32)
        np.savez(cache, embeddings=embs, years=years)
        print(f"Cached → {cache} ({embs.shape})", flush=True)
        del model; torch.cuda.empty_cache()

    # UMAP
    from umap import UMAP
    print(f"Running UMAP on {len(embs)} points...", flush=True)
    coords = UMAP(n_neighbors=30, min_dist=0.3, metric='cosine',
                   random_state=42).fit_transform(embs)

    plot_colored(coords, years, 'Study Date', out / 'umap_study_date.png')


if __name__ == '__main__':
    main()
