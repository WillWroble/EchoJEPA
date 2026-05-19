
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import pathlib
import warnings
from logging import getLogger

import cv2

import numpy as np
import pandas as pd
import torch
import torchvision
import pydicom

from src.datasets.utils.dataloader import ConcatIndices, MonitoredDataset, NondeterministicDataLoader
from src.datasets.utils.weighted_sampler import DistributedWeightedSampler

_GLOBAL_SEED = 0
logger = getLogger()
  


def _worker_init_fn(_):
    # keep each worker to 1 CPU thread to avoid oversubscription
    try:
        import torch as _torch, cv2, os as _os
        _torch.set_num_threads(1)
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass
        _os.environ["OMP_NUM_THREADS"] = "1"
        _os.environ["MKL_NUM_THREADS"] = "1"
    except Exception:
        pass


def make_videodataset(
    data_paths,
    batch_size,
    frames_per_clip=8,
    dataset_fpcs=None,
    frame_step=4,
    duration=None,
    fps=None,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(10**9),
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    deterministic=True,
    log_dir=None,
):
    dataset = VideoDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        dataset_fpcs=dataset_fpcs,
        duration=duration,
        fps=fps,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        shared_transform=shared_transform,
        transform=transform,
    )

    log_dir = pathlib.Path(log_dir) if log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        # Worker ID will replace '%w'
        resource_log_filename = log_dir / f"resource_file_{rank}_%w.csv"
        dataset = MonitoredDataset(
            dataset=dataset,
            log_filename=str(resource_log_filename),
            log_interval=10.0,
            monitor_interval=5.0,
        )

    logger.info("VideoDataset dataset created")
    if datasets_weights is not None:
        dist_sampler = DistributedWeightedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )

    dl_kwargs = dict(
        dataset=dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
        worker_init_fn=_worker_init_fn,
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = 1  # safe default; change as needed

    if deterministic:
        data_loader = torch.utils.data.DataLoader(**dl_kwargs)
    else:
        # custom loader variant with relaxed determinism
        data_loader = NondeterministicDataLoader(**dl_kwargs)

    logger.info("VideoDataset unsupervised data loader created")
    return dataset, data_loader, dist_sampler


class VideoDataset(torch.utils.data.Dataset):
    """
    Video classification dataset that supports both local filesystem and S3 paths.
    """

    def __init__(
        self,
        data_paths,
        datasets_weights=None,
        frames_per_clip=16,
        fps=None,
        dataset_fpcs=None,
        frame_step=4,
        num_clips=1,
        transform=None,
        shared_transform=None,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        duration=None,  # duration in seconds
    ):
        self.data_paths = data_paths
        self.datasets_weights = datasets_weights
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.duration = duration
        self.fps = fps

        # Initialize S3 client lazily per worker (avoid pickling/FD sharing)
        self.s3_client = None

        if sum([v is not None for v in (fps, duration, frame_step)]) != 1:
            raise ValueError(f"Must specify exactly one of either {fps=}, {duration=}, or {frame_step=}.")

        if isinstance(data_paths, str):
            data_paths = [data_paths]

        if dataset_fpcs is None:
            self.dataset_fpcs = [frames_per_clip for _ in data_paths]
        else:
            if len(dataset_fpcs) != len(data_paths):
                raise ValueError("Frames per clip not properly specified for data paths")
            self.dataset_fpcs = dataset_fpcs


        # Load video paths and labels from the annotation file(s)
        samples, labels = [], []
        self.num_samples_per_dataset = []
        for data_path in self.data_paths:
            if data_path.endswith(".csv"):
                try:
                    data = pd.read_csv(data_path, header=None, delimiter=" ")
                except pd.errors.ParserError:
                    data = pd.read_csv(data_path, header=None, delimiter="::")
                samples.extend(list(data.values[:, 0]))
                labels.extend(list(data.values[:, 1]))
                self.num_samples_per_dataset.append(len(data))
            elif data_path.endswith(".npy"):
                data = np.load(data_path, allow_pickle=True)
                data = [repr(x)[1:-1] for x in data]
                samples.extend(data)
                labels.extend([0] * len(data))
                self.num_samples_per_dataset.append(len(data))

        self.per_dataset_indices = ConcatIndices(self.num_samples_per_dataset)

        self.sample_weights = None
        if self.datasets_weights is not None:
            self.sample_weights = []
            for dw, ns in zip(self.datasets_weights, self.num_samples_per_dataset):
                self.sample_weights.extend([dw / ns] * ns)

        self.samples = samples
        self.labels = labels

        logger.info(f"Loaded {len(self.samples)} samples")
        if len(self.samples) > 0:
            logger.info(f"First 5 samples: {self.samples[:5]}")
            logger.info(f"Sample types: {[type(s) for s in self.samples[:5]]}")

    # ---------- S3 helper ----------
    def _ensure_s3_client(self):
        if self.s3_client is None:
            self.s3_client = boto3.client(
                "s3",
                config=Config(
                    max_pool_connections=32,
                    retries={"max_attempts": 5, "mode": "standard"},
                ),
            )

    # ---------- Dataset API ----------
    def __getitem__(self, index):
        # Keep trying new indices until a valid sample is loaded (matches default behavior)
        while True:
            sample_path = self.samples[index]
            if isinstance(sample_path, str):
                is_image = sample_path.split(".")[-1].lower() in ("jpg", "jpeg", "png")
                loaded = self.get_item_image(index) if is_image else self.get_item_video(index)
                if loaded:
                    return loaded
            warnings.warn(f"Retrying with new sample, failed to load: {self.samples[index]}")
            index = np.random.randint(len(self))

    def get_item_video(self, index):
        sample_uri = self.samples[index]
        dataset_idx, _ = self.per_dataset_indices[index]
        frames_per_clip = self.dataset_fpcs[dataset_idx]

        #buffer, clip_indices = self.loadvideo_cv2(sample_uri, frames_per_clip)
        
        if sample_uri.endswith('.dcm'):
            buffer, clip_indices = self.loadvideo_dcm(sample_uri, frames_per_clip)
        else:
            buffer, clip_indices = self.loadvideo_cv2(sample_uri, frames_per_clip)
        if buffer is None or len(buffer) == 0:
            return None

        label = self.labels[index]

        def split_into_clips(video):
            fpc = frames_per_clip
            nc = self.num_clips
            return [video[i * fpc : (i + 1) * fpc] for i in range(nc)]

        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        buffer = split_into_clips(buffer)
        if self.transform is not None:
            buffer = [self.transform(clip) for clip in buffer]

        return buffer, label, clip_indices

    def get_item_image(self, index):
        sample_uri = self.samples[index]
        dataset_idx, _ = self.per_dataset_indices[index]
        fpc = self.dataset_fpcs[dataset_idx]

        try:
            if isinstance(sample_uri, str) and sample_uri.startswith("s3://"):
                # S3 image
                self._ensure_s3_client()
                bucket_name, key = sample_uri.replace("s3://", "").split("/", 1)
                response = self.s3_client.get_object(Bucket=bucket_name, Key=key)
                image_bytes = response["Body"].read()
                image_tensor = torchvision.io.decode_image(
                    torch.from_numpy(np.frombuffer(image_bytes, np.uint8)),
                    mode=torchvision.io.ImageReadMode.RGB,
                )
            else:
                # Local image
                image_tensor = torchvision.io.read_image(path=sample_uri, mode=torchvision.io.ImageReadMode.RGB)
        except Exception as e:
            logger.warning(f"Failed to load image {sample_uri}: {e}")
            return None

        label = self.labels[index]
        clip_indices = [np.arange(start=0, stop=fpc, dtype=np.int32)]

        # Expand to [T, H, W, 3]
        buffer = image_tensor.unsqueeze(dim=0).repeat((fpc, 1, 1, 1))
        buffer = buffer.permute((0, 2, 3, 1))

        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        if self.transform is not None:
            buffer = [self.transform(buffer)]

        return buffer, label, clip_indices

    def debug_sample_loading(self, index):
        sample_uri = self.samples[index]
        print(f"Attempting to load sample {index}: {sample_uri}")
        print(f"Sample type: {type(sample_uri)}")
        print(f"Is string: {isinstance(sample_uri, str)}")
        print(f"Starts with s3://: {sample_uri.startswith('s3://') if isinstance(sample_uri, str) else False}")

        if self.s3_client is None:
            print("S3 client not initialized")
            return

        try:
            bucket_name, key = sample_uri.replace("s3://", "").split("/", 1)
            print(f"Bucket: {bucket_name}, Key: {key}")
            response = self.s3_client.head_object(Bucket=bucket_name, Key=key)
            print(f"Object exists, size: {response['ContentLength']}")
        except Exception as e:
            print(f"S3 error: {e}")

    # ---------- Core video loader adapted for BCH ----------
    def loadvideo_cv2(self, sample_uri, fpc):
        """Load video frames using cv2. Drop-in replacement for loadvideo_decord.

        Returns (buffer[T,H,W,3] uint8 RGB, clip_indices) or ([], None).
        """
        fname = sample_uri
        if not os.path.exists(fname):
            warnings.warn(f"video path not found fname='{fname}'")
            return [], None

        _fsize = os.path.getsize(fname)
        if self.filter_long_videos and _fsize > self.filter_long_videos:
            warnings.warn(f"skipping long video of size _fsize={_fsize} (bytes)")
            return [], None

        cap = cv2.VideoCapture(fname)
        if not cap.isOpened():
            return [], None

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 1 or video_fps < 1:
            cap.release()
            return [], None

        return self._sample_from_cap(cap, fpc, video_fps, frame_count)

    
    
    def loadvideo_dcm(self, sample_uri, fpc):
        """Load video frames from a multi-frame DICOM."""
        if not os.path.exists(sample_uri):
            warnings.warn(f"DICOM path not found: '{sample_uri}'")
            return [], None

        try:
            ds = pydicom.dcmread(sample_uri)
            frames = ds.pixel_array  # (N, H, W, 3) YBR_FULL_422
        except Exception as e:
            warnings.warn(f"Failed to read DICOM {sample_uri}: {e}")
            return [], None

        nf = getattr(ds, 'NumberOfFrames', 1)
        if int(nf) < 2:
            return [], None

        # YBR -> RGB
        frames = np.stack([cv2.cvtColor(f, cv2.COLOR_YCrCb2RGB) for f in frames])

        frame_count = len(frames)
        video_fps = getattr(ds, 'CineRate', 30)  # fallback 30fps

        # Reuse the same sampling logic via a fake cv2 cap interface
        # Easier: just index directly into the array
        return self._sample_from_frames(frames, fpc, float(video_fps), frame_count)
    def _sample_from_cap(self, cap, fpc, video_fps, frame_count):
        """Sample frames from cv2 capture. Replaces _sample_from_vr."""
        fstp = self.frame_step
        if self.duration is not None or self.fps is not None:
            if self.duration is not None:
                fstp = max(1, int(self.duration * video_fps / fpc))
            else:
                fstp = max(1, int(video_fps // max(1, self.fps)))

        clip_len = int(fpc * fstp)
        V = frame_count

        if self.filter_short_videos and V < clip_len:
            warnings.warn(f"skipping video of length {V}")
            cap.release()
            return [], None

        partition_len = V // self.num_clips
        all_indices, clip_indices = [], []

        for i in range(self.num_clips):
            if partition_len > clip_len:
                if self.random_clip_sampling:
                    end_indx = np.random.randint(clip_len, partition_len)
                else:
                    end_indx = clip_len
                start_indx = end_indx - clip_len
                indices = np.linspace(start_indx, end_indx, num=fpc)
                indices = np.clip(indices, start_indx, end_indx - 1).astype(np.int64)
                indices = indices + i * partition_len
            else:
                sample_len = min(clip_len, V) - 1
                base = max(1, sample_len // fstp)
                indices = np.linspace(0, sample_len, num=base)
                if base < fpc:
                    indices = np.concatenate((indices, np.ones(fpc - base) * sample_len))
                indices = np.clip(indices, 0, max(0, V - 1)).astype(np.int64)
                clip_step = 0
                if V > clip_len and self.num_clips > 1:
                    clip_step = (V - clip_len) // (self.num_clips - 1)
                indices = indices + i * clip_step

            clip_indices.append(indices)
            all_indices.extend(list(indices))

        all_indices = np.clip(all_indices, 0, V - 1).astype(np.int64)

        frames = []
        last_frame = None
        for idx in all_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                last_frame = frame
            else:
                frame = last_frame if last_frame is not None else np.zeros(
                    (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), dtype=np.uint8
                )
            frames.append(frame)
        cap.release()

        buffer = np.stack(frames)  # (T, H, W, 3) uint8 RGB
        return buffer, clip_indices
    
    
    
    
    def _sample_from_frames(self, frames, fpc, video_fps, frame_count):
        fstp = self.frame_step
        if self.duration is not None or self.fps is not None:
            if self.duration is not None:
                fstp = max(1, int(self.duration * video_fps / fpc))
            else:
                fstp = max(1, int(video_fps // max(1, self.fps)))

        clip_len = int(fpc * fstp)
        V = frame_count

        if self.filter_short_videos and V < clip_len:
            return [], None

        partition_len = V // self.num_clips
        all_indices, clip_indices = [], []

        for i in range(self.num_clips):
            if partition_len > clip_len:
                if self.random_clip_sampling:
                    end_indx = np.random.randint(clip_len, partition_len)
                else:
                    end_indx = clip_len
                start_indx = end_indx - clip_len
                indices = np.linspace(start_indx, end_indx, num=fpc)
                indices = np.clip(indices, start_indx, end_indx - 1).astype(np.int64)
                indices = indices + i * partition_len
            else:
                sample_len = min(clip_len, V) - 1
                base = max(1, sample_len // fstp)
                indices = np.linspace(0, sample_len, num=base)
                if base < fpc:
                    indices = np.concatenate((indices, np.ones(fpc - base) * sample_len))
                indices = np.clip(indices, 0, max(0, V - 1)).astype(np.int64)
                clip_step = 0
                if V > clip_len and self.num_clips > 1:
                    clip_step = (V - clip_len) // (self.num_clips - 1)
                indices = indices + i * clip_step

            clip_indices.append(indices)
            all_indices.extend(list(indices))

        all_indices = np.clip(all_indices, 0, V - 1).astype(np.int64)
        buffer = frames[all_indices]
        return buffer, clip_indices
    def __len__(self):
        return len(self.samples)



# ---------------------------------------------------------------------------
# Intra-study batch sampler (for intra-study contrastive learning)
# ---------------------------------------------------------------------------

from collections import defaultdict
from torch.utils.data import Sampler


class IntraStudyBatchSampler(Sampler):
    """Yields batches of M studies x K clips, grouped by integer label.

    Each yielded batch is a flat list of M*K dataset indices, where every K
    consecutive indices share the same label (study_id_int). DDP-aware: each
    rank consumes a disjoint stride of studies per epoch.
    """

    def __init__(self, labels, m_studies, k_clips, num_replicas=1, rank=0, seed=0):
        self.m = m_studies
        self.k = k_clips
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        self.study_to_idxs = defaultdict(list)
        for i, lab in enumerate(labels):
            self.study_to_idxs[int(lab)].append(i)
        for lab in self.study_to_idxs:
            self.study_to_idxs[lab] = np.array(self.study_to_idxs[lab], dtype=np.int64)
        self.study_ids = np.array(sorted(self.study_to_idxs.keys()), dtype=np.int64)

        per_rank = len(self.study_ids) // num_replicas
        self._num_batches = per_rank // m_studies
        self._studies_per_rank = self._num_batches * m_studies

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self._num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.study_ids))
        my = self.study_ids[order][self.rank::self.num_replicas][:self._studies_per_rank]

        for start in range(0, len(my), self.m):
            batch_studies = my[start:start + self.m]
            indices = []
            for sid in batch_studies:
                pool = self.study_to_idxs[int(sid)]
                replace = len(pool) < self.k
                chosen = rng.choice(len(pool), size=self.k, replace=replace)
                indices.extend(int(pool[c]) for c in chosen)
            yield indices


class IntraMRNBatchSampler:
    def __init__(self, labels, sid_to_mrn, m_mrns, num_replicas=1, rank=0, seed=0):
        # labels: clip-level study_id list (5.7M entries, ints)
        # sid_to_mrn: dict[int study_id -> int mrn_id]
        # m_mrns: number of MRNs per local batch
        from collections import defaultdict
        self.m_mrns = m_mrns
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed

        # study_id -> [clip_idxs]
        study_to_idxs = defaultdict(list)
        for i, sid in enumerate(labels):
            study_to_idxs[int(sid)].append(i)

        # mrn -> [study_ids that have at least 1 clip]
        mrn_to_studies = defaultdict(list)
        for sid, idxs in study_to_idxs.items():
            mrn = sid_to_mrn.get(sid)
            if mrn is not None and len(idxs) > 0:
                mrn_to_studies[mrn].append(sid)

        # Keep only multi-visit MRNs
        self.mrn_to_studies = {m: s for m, s in mrn_to_studies.items() if len(s) >= 2}
        self.study_to_idxs = dict(study_to_idxs)
        self.mrns = sorted(self.mrn_to_studies.keys())

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        import random
        rng = random.Random(self.seed + self.epoch)
        mrns = self.mrns[:]
        rng.shuffle(mrns)
        # split across ranks
        mrns = mrns[self.rank::self.num_replicas]
        # batch them
        for i in range(0, len(mrns) - self.m_mrns + 1, self.m_mrns):
            batch_mrns = mrns[i:i + self.m_mrns]
            batch = []
            for mrn in batch_mrns:
                s1, s2 = rng.sample(self.mrn_to_studies[mrn], 2)
                c1 = rng.choice(self.study_to_idxs[s1])
                c2 = rng.choice(self.study_to_idxs[s2])
                batch.extend([c1, c2])
            yield batch

    def __len__(self):
        return len(self.mrns) // (self.num_replicas * self.m_mrns)
