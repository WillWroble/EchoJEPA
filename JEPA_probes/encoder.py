"""JEPA clip extraction. Logic from video_pretraining_v2/train.py and
src/datasets/video_dataset.py. Sequential reads + DataLoader workers."""

import sys
sys.path.insert(0, '/lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA')

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

import src.models.vision_transformer as vit

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)


def _clean_backbone_key(state_dict):
    for key, val in state_dict.copy().items():
        _ = state_dict.pop(key)
        key = key.replace("module.", "").replace("backbone.", "")
        state_dict[key] = val
    return state_dict


def load_encoder(checkpoint_path, device='cuda'):
    model = vit.vit_base(
        img_size=(224, 224), patch_size=16, num_frames=16,
        tubelet_size=2, use_rope=True, use_sdpa=True, uniform_power=True,
    )
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ckpt.get('target_encoder') or ckpt.get('encoder')
    state = _clean_backbone_key(state)
    msg = model.load_state_dict(state, strict=False)
    print(f"  encoder: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}", flush=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def preprocess_clip(buffer, resolution=224):
    x = torch.from_numpy(buffer).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(resolution, resolution), mode='bilinear', align_corners=False)
    x = x.permute(1, 0, 2, 3)
    return (x - MEAN) / STD


def load_and_sample_clips(avi_path, num_clips=4, fpc=16, fps=8):
    cap = cv2.VideoCapture(str(avi_path))
    if not cap.isOpened():
        return None
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < 1 or video_fps < 1:
        cap.release()
        return None

    fstp = max(1, int(video_fps // max(1, fps)))
    clip_len = fpc * fstp
    V = frame_count
    partition_len = V // num_clips

    all_indices = []
    for i in range(num_clips):
        if partition_len > clip_len:
            end_indx = clip_len
            indices = np.linspace(0, end_indx, num=fpc)
            indices = np.clip(indices, 0, end_indx - 1).astype(np.int64)
            indices = indices + i * partition_len
        else:
            sample_len = min(clip_len, V) - 1
            base = max(1, sample_len // fstp)
            indices = np.linspace(0, sample_len, num=base)
            if base < fpc:
                indices = np.concatenate((indices, np.ones(fpc - base) * sample_len))
            indices = np.clip(indices, 0, max(0, V - 1)).astype(np.int64)
            clip_step = 0
            if V > clip_len and num_clips > 1:
                clip_step = (V - clip_len) // (num_clips - 1)
            indices = indices + i * clip_step
        all_indices.extend(indices.tolist())

    all_indices = np.clip(all_indices, 0, V - 1).astype(np.int64)

    # sequential read (10-100x faster than seeking on AVIs)
    max_idx = int(max(all_indices))
    all_frames = []
    for fidx in range(max_idx + 1):
        ret, frame = cap.read()
        if ret:
            all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        elif all_frames:
            all_frames.append(all_frames[-1])
        else:
            all_frames.append(np.zeros(
                (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                 int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), dtype=np.uint8))
    cap.release()

    frames = [all_frames[idx] for idx in all_indices]
    buffer = np.stack(frames)
    clips = [preprocess_clip(buffer[i * fpc:(i + 1) * fpc]) for i in range(num_clips)]
    return torch.stack(clips)


@torch.no_grad()
def encode_video(model, avi_path, device='cuda'):
    clips = load_and_sample_clips(avi_path)
    if clips is None:
        return None
    clips = clips.to(device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        patches = model(clips)
        emb = patches.mean(dim=1).mean(dim=0)
    return emb.float().cpu().numpy()


class _VideoDataset(Dataset):
    def __init__(self, avi_list):
        self.items = avi_list

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sid, avi_path = self.items[idx]
        clips = load_and_sample_clips(avi_path)
        if clips is None:
            return torch.empty(0), sid, Path(avi_path).name
        return clips, sid, Path(avi_path).name


def _collate(batch):
    valid = [(c, s, f) for c, s, f in batch if c.numel() > 0]
    if not valid:
        return None, [], [], []
    clips = torch.cat([c for c, _, _ in valid])
    sids = [s for _, s, _ in valid]
    fnames = [f for _, _, f in valid]
    n_clips = [c.shape[0] for c, _, _ in valid]
    return clips, sids, fnames, n_clips


@torch.no_grad()
def extract_embeddings(encoder, study_ids, data_dir, videos_per_study=48, device='cuda',
                       partial_path=None, batch_size=16, num_workers=4):
    data_dir = Path(data_dir)

    # build flat (sid, avi_path) list
    print("  building video list...", flush=True)
    avi_list = []
    for sid in study_ids:
        study_dir = data_dir / f"{sid}_trim"
        avis = sorted(study_dir.glob("*.avi")) if study_dir.exists() else []
        if not avis:
            continue
        if len(avis) > videos_per_study:
            idx = np.random.RandomState(int(sid) % 2**31).choice(
                len(avis), videos_per_study, replace=False)
            avis = [avis[j] for j in sorted(idx)]
        for avi in avis:
            avi_list.append((sid, str(avi)))
    print(f"  {len(avi_list)} videos from {len(set(s for s,_ in avi_list))} studies", flush=True)

    # resume from partial
    all_embs, all_sids, all_fnames = [], [], []
    start_video = 0
    if partial_path and Path(partial_path).exists():
        prev = np.load(partial_path, allow_pickle=True)
        all_embs = list(prev['embeddings'])
        all_sids = list(prev['study_ids'])
        all_fnames = list(prev['filenames'])
        start_video = int(prev['next_video'])
        print(f"  resuming from video {start_video} ({len(all_embs)} cached)", flush=True)

    remaining = avi_list[start_video:]
    dataset = _VideoDataset(remaining)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True,
                        collate_fn=_collate, drop_last=False, prefetch_factor=2)

    for step, (clips, sids, fnames, n_clips) in enumerate(loader):
        if clips is None:
            continue
        clips = clips.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            patches = encoder(clips)
            clip_embs = patches.mean(dim=1)
        clip_embs = clip_embs.float().cpu().numpy()
        ptr = 0
        for sid, fname, nc in zip(sids, fnames, n_clips):
            vid_emb = clip_embs[ptr:ptr + nc].mean(axis=0)
            all_embs.append(vid_emb)
            all_sids.append(sid)
            all_fnames.append(fname)
            ptr += nc

        if (step + 1) % 50 == 0:
            done = start_video + min((step + 1) * batch_size, len(remaining))
            print(f"  {done}/{len(avi_list)} videos ({len(all_embs)} embedded)", flush=True)

        if partial_path and (step + 1) % 500 == 0:
            next_v = start_video + min((step + 1) * batch_size, len(remaining))
            np.savez(partial_path,
                     embeddings=np.stack(all_embs).astype(np.float32),
                     study_ids=np.array(all_sids),
                     filenames=np.array(all_fnames),
                     next_video=np.array(next_v))
            print(f"  checkpoint saved ({len(all_embs)} videos)", flush=True)

    print(f"  done: {len(all_embs)} videos from {len(set(all_sids))} studies", flush=True)
    if partial_path and Path(partial_path).exists():
        Path(partial_path).unlink()
    return {
        'embeddings': np.stack(all_embs).astype(np.float32),
        'study_ids': np.array(all_sids),
        'filenames': np.array(all_fnames),
    }
