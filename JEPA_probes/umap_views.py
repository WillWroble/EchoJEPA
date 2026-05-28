"""UMAP of JEPA clip embeddings colored by echocardiographic view.

Usage:
    python -u umap_views.py \
        --checkpoint /path/to/latest.pt \
        --output_dir results/epoch240
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from encoder import load_encoder, encode_video


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--view_labels', default='/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/view_labels.csv')
    p.add_argument('--data_dir', default='/lab-share/Cardio-Mayourian-e2/Public/Echo_Internal_30k')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / 'view_embeddings.npz'

    view_cols = ['parasternal_long', 'parasternal_short', 'subxiphoid_long',
                 'apical_4_chamber', 'other']

    df = pd.read_csv(args.view_labels)
    print(f"View labels: {len(df)} videos", flush=True)

    # extract or load cache
    if cache.exists():
        print(f"Cache hit: {cache}", flush=True)
        data = np.load(cache, allow_pickle=True)
        embs = data['embeddings']
        labels = data['labels']
    else:
        print("Extracting view embeddings...", flush=True)
        model = load_encoder(args.checkpoint, args.device)
        data_dir = Path(args.data_dir)

        embs, labels, kept = [], [], []
        for i, row in df.iterrows():
            avi = data_dir / f"{row['echo_id']}_trim" / row['video_filename']
            emb = encode_video(model, avi, args.device)
            if emb is not None:
                embs.append(emb)
                labels.append([row[c] for c in view_cols])
                kept.append(i)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(df)} ({len(embs)} ok)", flush=True)

        embs = np.stack(embs).astype(np.float32)
        labels = np.array(labels, dtype=np.float32)
        np.savez(cache, embeddings=embs, labels=labels)
        print(f"Cached → {cache} ({embs.shape})", flush=True)
        del model; torch.cuda.empty_cache()

    # UMAP
    import umap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # assign each video its primary view (argmax of binary columns)
    view_idx = labels.argmax(axis=1)
    view_names = ['PLAX', 'PSAX', 'Subxiphoid', 'A4C', 'Other']

    print(f"Running UMAP on {len(embs)} points...", flush=True)
    coords = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine',
                        random_state=42).fit_transform(embs)

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#999999']
    fig, ax = plt.subplots(figsize=(10, 8))
    for vi in range(len(view_names)):
        mask = view_idx == vi
        if mask.sum() == 0:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=colors[vi], s=2, alpha=0.5, label=f'{view_names[vi]} ({mask.sum()})')
    ax.legend(markerscale=4)
    ax.set_title('JEPA clip embeddings by echo view')
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    plt.tight_layout()
    plt.savefig(out / 'umap_views.png', dpi=200)
    plt.close()
    print(f"Saved → {out / 'umap_views.png'}", flush=True)


if __name__ == '__main__':
    main()
