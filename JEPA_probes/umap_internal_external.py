"""UMAP of JEPA clip embeddings colored by internal vs external.

Usage:
    python -u umap_internal_external.py \
        --checkpoint /path/to/latest.pt \
        --output_dir results/internal_external
"""

import argparse
import numpy as np
import torch
from pathlib import Path

from encoder import load_encoder, encode_video


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--internal_manifest', default='/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/platon_train.txt')
    p.add_argument('--external_manifest', default='/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/study_outside.txt')
    p.add_argument('--internal_dir', default='/lab-share/Cardio-Mayourian-e2/Public/Echo_Pulled/Echo_Internal_30k')
    p.add_argument('--external_dir', default='/lab-share/Cardio-Mayourian-e2/Public/Echo_Pulled/Echo_Outside')
    p.add_argument('--n_samples', type=int, default=50000)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / 'internal_external_embeddings.npz'
    rng = np.random.RandomState(42)

    if cache.exists():
        print(f"Cache hit: {cache}", flush=True)
        data = np.load(cache)
        embs = data['embeddings']
        is_external = data['is_external']
    else:
        def collect_avis(manifest, data_dir):
            sids = [l.strip() for l in open(manifest)]
            dd = Path(data_dir)
            avis = []
            for sid in sids:
                sd = dd / f"{sid}_trim"
                if sd.exists():
                    for avi in sd.glob("*.avi"):
                        avis.append(avi)
            return avis

        print("Collecting internal AVIs...", flush=True)
        internal_avis = collect_avis(args.internal_manifest, args.internal_dir)
        print(f"  {len(internal_avis)} internal videos", flush=True)

        print("Collecting external AVIs...", flush=True)
        external_avis = collect_avis(args.external_manifest, args.external_dir)
        print(f"  {len(external_avis)} external videos", flush=True)

        all_avis = [(avi, 0) for avi in internal_avis] + [(avi, 1) for avi in external_avis]
        if len(all_avis) > args.n_samples:
            idx = rng.choice(len(all_avis), args.n_samples, replace=False)
            all_avis = [all_avis[i] for i in idx]
        print(f"Sampled {sum(1 for _,e in all_avis if e==0)} internal + "
              f"{sum(1 for _,e in all_avis if e==1)} external", flush=True)

        print("Extracting embeddings...", flush=True)
        model = load_encoder(args.checkpoint, args.device)

        embs, labels = [], []
        for i, (avi, ext) in enumerate(all_avis):
            emb = encode_video(model, avi, args.device)
            if emb is not None:
                embs.append(emb)
                labels.append(ext)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(all_avis)} ({len(embs)} ok)", flush=True)

        embs = np.stack(embs).astype(np.float32)
        is_external = np.array(labels, dtype=np.int32)
        np.savez(cache, embeddings=embs, is_external=is_external)
        print(f"Cached → {cache} ({embs.shape})", flush=True)
        del model; torch.cuda.empty_cache()

    # UMAP
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from umap import UMAP

    print(f"Running UMAP on {len(embs)} points...", flush=True)
    coords = UMAP(n_neighbors=30, min_dist=0.3, metric='cosine',
                   random_state=42).fit_transform(embs)

    fig, ax = plt.subplots(figsize=(8, 7))
    for label, name, color in [(0, 'Internal', '#1f77b4'), (1, 'External', '#ff7f0e')]:
        mask = is_external == label
        ax.scatter(coords[mask, 0], coords[mask, 1], s=0.5, alpha=0.3,
                   c=color, label=f"{name} ({mask.sum():,})")
    ax.legend(markerscale=10)
    ax.set_title('Internal vs External')
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out / 'umap_internal_external.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {out / 'umap_internal_external.png'}", flush=True)


if __name__ == '__main__':
    main()
