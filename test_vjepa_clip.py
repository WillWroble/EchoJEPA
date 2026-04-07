"""Test: one AVI → one 768d clip embedding via V-JEPA 2.1 ViT-B."""

import torch
import cv2
import numpy as np

# ── Load encoder ──
print("Loading V-JEPA 2.1 ViT-B encoder...", flush=True)
encoder, predictor = torch.hub.load(
    'facebookresearch/vjepa2', 'vjepa2_1_vit_base_384', trust_repo=True
)
encoder = encoder.cuda().eval()
print(f"Encoder: {encoder.embed_dim}d, {sum(p.numel() for p in encoder.parameters())/1e6:.1f}M params", flush=True)

# ── Read one clip from a real AVI ──
avi_path = "/lab-share/Cardio-Mayourian-e2/Public/Echo_Pulled/Echo_Internal_30k/2689491_trim/20150701RP611710572816ECHOB31611710572816_1_1.avi"
cap = cv2.VideoCapture(avi_path)
native_fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {frame_count} frames @ {native_fps:.0f}fps", flush=True)

# Sample 16 frames at ~8fps (every ~4th frame at 30fps)
target_fps = 8
step = max(1, round(native_fps / target_fps))
max_start = max(0, frame_count - step * 16)
start = np.random.randint(0, max_start + 1) if max_start > 0 else 0

frames = []
for i in range(16):
    idx = start + i * step
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        frame = frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)
cap.release()

# (16, 224, 224, 3) → (1, 3, 16, 224, 224) float32 normalized
clip = np.stack(frames)  # T H W C
clip = torch.from_numpy(clip).permute(3, 0, 1, 2).float() / 255.0  # C T H W
# ImageNet normalization
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
clip = (clip - mean) / std
clip = clip.unsqueeze(0).cuda()  # 1 x 3 x 16 x 224 x 224

print(f"Input shape: {clip.shape}", flush=True)

# ── Forward pass ──
with torch.no_grad():
    patch_tokens = encoder(clip)  # (1, N_patches, 768)

print(f"Patch tokens: {patch_tokens.shape}", flush=True)

# Mean pool → clip embedding
clip_embedding = patch_tokens.mean(dim=1)  # (1, 768)
print(f"Clip embedding: {clip_embedding.shape}", flush=True)
print(f"Norm: {clip_embedding.norm().item():.2f}", flush=True)
print("Done.", flush=True)
