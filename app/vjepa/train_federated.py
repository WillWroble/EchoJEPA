"""
Federated EchoJEPA — Flower Client

Same config-driven design as the rest of the codebase. Each site
gets its own YAML with folder, manifest, and a `federated` section:

    federated:
      server: 100.64.175.34:8080
      mode: 1          # 1=all, 2=pooled EMA, 3=encoder only
      local_steps: 300
      sync_momentum: 0.998

Usage:
    python -m app.vjepa.train_federated \
        --fname configs/train/vitb16/fl-internal.yaml \
        --device cuda:0
"""

import argparse
import copy
import gc
import os
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import flwr as fl

from app.vjepa.transforms import make_transforms
from app.vjepa.utils import init_opt, init_video_model
from src.datasets.data_manager import init_data
from src.masks.multiseq_multiblock3d import MaskCollator
from src.masks.utils import apply_masks
from src.utils.logging import AverageMeter, get_logger
import random
logger = get_logger(__name__, force=True)

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True
# -- Serialization --

def state_dict_to_numpy(state_dict):
    return [v.cpu().numpy() for v in state_dict.values()]


def load_numpy_into_model(model, params):
    state_dict = OrderedDict(
        {k: torch.from_numpy(v) for k, v in zip(model.state_dict().keys(), params)}
    )
    model.load_state_dict(state_dict)


# -- Trainer --

class JEPATrainer:
    """Holds JEPA models, data, optimizer. Runs local training rounds."""

    def __init__(self, args, device):
        cfgs_data = args.get("data")
        cfgs_data_aug = args.get("data_aug")
        cfgs_mask = args.get("mask")
        cfgs_model = args.get("model")
        cfgs_opt = args.get("optimization")
        cfgs_meta = args.get("meta")
        cfgs_loss = args.get("loss")

        # checkpoint folder from config
        self.folder = args.get("folder")
        Path(self.folder).mkdir(parents=True, exist_ok=True)

        self.device = device
        self.loss_exp = cfgs_loss.get("loss_exp", 1.0)

        # dtype
        self.dtype = getattr(torch, cfgs_meta.get("dtype", "float16"))
        self.mixed_precision = self.dtype != torch.float32

        # data
        dataset_type = cfgs_data.get("dataset_type", "VideoDataset")
        dataset_paths = cfgs_data.get("datasets")
        datasets_weights = cfgs_data.get("datasets_weights", [1.0])
        batch_size = cfgs_data.get("batch_size", 64)
        dataset_fpcs = cfgs_data.get("dataset_fpcs", [16])
        fps = cfgs_data.get("fps", 8)
        num_workers = cfgs_data.get("num_workers", 8)
        pin_mem = cfgs_data.get("pin_mem", True)
        persistent_workers = cfgs_data.get("persistent_workers", True)
        crop_size = cfgs_data.get("crop_size", 224)
        patch_size = cfgs_data.get("patch_size", 16)
        tubelet_size = cfgs_data.get("tubelet_size", 2)

        # model
        model_name = cfgs_model.get("model_name", "vit_base")
        pred_depth = cfgs_model.get("pred_depth", 6)
        pred_embed_dim = cfgs_model.get("pred_embed_dim", 384)
        pred_num_heads = cfgs_model.get("pred_num_heads", None)
        uniform_power = cfgs_model.get("uniform_power", True)
        use_mask_tokens = cfgs_model.get("use_mask_tokens", True)
        num_mask_tokens = cfgs_model.get("num_mask_tokens", 2)
        zero_init_mask_tokens = cfgs_model.get("zero_init_mask_tokens", True)
        use_sdpa = cfgs_meta.get("use_sdpa", True)
        use_rope = cfgs_model.get("use_rope", False)
        use_silu = cfgs_model.get("use_silu", False)
        use_pred_silu = cfgs_model.get("use_pred_silu", False)
        wide_silu = cfgs_model.get("wide_silu", False)
        use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)

        # optimization
        ema = cfgs_opt.get("ema", [0.998, 0.998])
        ipe = cfgs_opt.get("ipe", 300)
        ipe_scale = cfgs_opt.get("ipe_scale", 1.25)
        num_epochs = cfgs_opt.get("epochs", 240)
        warmup = cfgs_opt.get("warmup", 40)
        wd = cfgs_opt.get("weight_decay", 0.04)
        final_wd = cfgs_opt.get("final_weight_decay", 0.04)
        start_lr = cfgs_opt.get("start_lr", 1e-5)
        lr = cfgs_opt.get("lr", 1e-4)
        final_lr = cfgs_opt.get("final_lr", 1e-4)
        is_anneal = cfgs_opt.get("is_anneal", False)
        betas = cfgs_opt.get("betas", (0.9, 0.999))
        eps = cfgs_opt.get("eps", 1e-8)

        # augmentation
        ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [0.9, 1.1])
        rr_scale = cfgs_data_aug.get("random_resize_scale", [0.5, 1.0])
        reprob = cfgs_data_aug.get("reprob", 0.0)
        use_aa = cfgs_data_aug.get("auto_augment", False)
        motion_shift = cfgs_data_aug.get("motion_shift", False)

        # --- models (no DDP) ---
        logger.info("Initializing models...")
        self.encoder, self.predictor = init_video_model(
            device=device,
            patch_size=patch_size,
            max_num_frames=max(dataset_fpcs),
            tubelet_size=tubelet_size,
            model_name=model_name,
            crop_size=crop_size,
            pred_depth=pred_depth,
            pred_num_heads=pred_num_heads,
            pred_embed_dim=pred_embed_dim,
            uniform_power=uniform_power,
            use_mask_tokens=use_mask_tokens,
            num_mask_tokens=num_mask_tokens,
            zero_init_mask_tokens=zero_init_mask_tokens,
            use_sdpa=use_sdpa,
            use_rope=use_rope,
            use_silu=use_silu,
            use_pred_silu=use_pred_silu,
            wide_silu=wide_silu,
            use_activation_checkpointing=use_activation_checkpointing,
        )
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # --- optimizer ---
        self.optimizer, self.scaler, self.scheduler, self.wd_scheduler = init_opt(
            is_anneal=is_anneal,
            encoder=self.encoder,
            predictor=self.predictor,
            wd=wd,
            final_wd=final_wd,
            start_lr=start_lr,
            ref_lr=lr,
            final_lr=final_lr,
            iterations_per_epoch=ipe,
            warmup=warmup,
            num_epochs=num_epochs,
            ipe_scale=ipe_scale,
            mixed_precision=self.mixed_precision,
            betas=betas,
            eps=eps,
        )

        # --- EMA schedule ---
        total_steps = int(ipe * num_epochs * ipe_scale)
        self.momentum_scheduler = iter(
            ema[0] + i * (ema[1] - ema[0]) / total_steps
            for i in range(total_steps + 1)
        )

        # --- data ---
        logger.info("Initializing data...")
        mask_collator = MaskCollator(
            cfgs_mask=cfgs_mask,
            dataset_fpcs=dataset_fpcs,
            crop_size=crop_size,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
        )
        transform = make_transforms(
            random_horizontal_flip=False,
            random_resize_aspect_ratio=ar_range,
            random_resize_scale=rr_scale,
            reprob=reprob,
            auto_augment=use_aa,
            motion_shift=motion_shift,
            crop_size=crop_size,
        )
        self.loader, self.sampler = init_data(
            data=dataset_type,
            root_path=dataset_paths,
            batch_size=batch_size,
            training=True,
            dataset_fpcs=dataset_fpcs,
            fps=fps,
            transform=transform,
            rank=0,
            world_size=1,
            datasets_weights=datasets_weights,
            persistent_workers=persistent_workers,
            collator=mask_collator,
            num_workers=num_workers,
            pin_mem=pin_mem,
            log_dir=None,
        )
        self.batch_size = batch_size
        self._loader_iter = iter(self.loader)
        self._epoch = 0

        # --- pretrained checkpoint ---
        ckpt_path = cfgs_opt.get("anneal_ckpt")
        if cfgs_opt.get("force_load_pretrain") and ckpt_path and os.path.exists(ckpt_path):
            logger.info(f"Loading pretrained weights from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            for name, model in [("encoder", self.encoder), ("target_encoder", self.target_encoder),
                                ("predictor", self.predictor)]:
                if name in ckpt:
                    sd = {k.replace("module.", ""): v for k, v in ckpt[name].items()}
                    current = model.state_dict()
                    filtered = {k: v for k, v in sd.items() if k in current and v.shape == current[k].shape}
                    msg = model.load_state_dict(filtered, strict=False)
                    logger.info(f"Loaded {name}: {msg}")
            del ckpt
            gc.collect()

        # parameter counts for mode 1 splitting
        self.n_encoder = len(list(self.encoder.state_dict()))
        self.n_predictor = len(list(self.predictor.state_dict()))

        # save config alongside checkpoints
        params_path = os.path.join(self.folder, "params-federated.yaml")
        with open(params_path, "w") as f:
            yaml.dump(args, f)

        self.step_count = 0
        logger.info("Trainer initialized.")

    def _next_batch(self):
        try:
            return next(self._loader_iter)
        except (StopIteration, Exception):
            self._epoch += 1
            self.sampler.set_epoch(self._epoch)
            self._loader_iter = iter(self.loader)
            return next(self._loader_iter)

    def train_round(self, local_steps, do_ema=True):
        """Run local_steps of JEPA training. Returns (avg_loss, num_samples)."""
        self.encoder.train()
        self.predictor.train()
        loss_meter = AverageMeter()
        num_samples = 0

        for _ in range(local_steps):
            sample = self._next_batch()

            clips, masks_enc, masks_pred = [], [], []
            for fpc_sample in sample:
                udata, m_enc, m_pred = fpc_sample
                clips.append(udata[0][0].to(self.device, non_blocking=True))
                masks_enc.append([m.to(self.device, non_blocking=True) for m in m_enc])
                masks_pred.append([m.to(self.device, non_blocking=True) for m in m_pred])

            num_samples += clips[0].shape[0]

            self.scheduler.step()
            self.wd_scheduler.step()

            with torch.amp.autocast("cuda", dtype=self.dtype, enabled=self.mixed_precision):
                with torch.no_grad():
                    h = self.target_encoder(clips)
                    h = [F.layer_norm(hi, (hi.size(-1),)) for hi in h]

                z = self.encoder(clips, masks_enc)
                z = self.predictor(z, masks_enc, masks_pred)

                h_masked = [apply_masks(hi, mi, concat=False) for hi, mi in zip(h, masks_pred)]
                loss, n = 0, 0
                for zi, hi in zip(z, h_masked):
                    for zij, hij in zip(zi, hi):
                        loss += torch.mean(torch.abs(zij - hij) ** self.loss_exp) / self.loss_exp
                        n += 1
                loss /= n

            if self.mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            self.optimizer.zero_grad()

            if do_ema:
                m = next(self.momentum_scheduler)
                with torch.no_grad():
                    for p_enc, p_tgt in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                        p_tgt.data.mul_(m).add_(p_enc.data, alpha=1 - m)
            else:
                _ = next(self.momentum_scheduler)

            loss_meter.update(float(loss))
            self.step_count += 1

            if self.step_count % 50 == 0:
                logger.info(f"[step {self.step_count}] loss: {loss_meter.avg:.4f}")
                gc.collect()

        logger.info(f"Round complete — {local_steps} steps, avg loss: {loss_meter.avg:.4f}")
        return loss_meter.avg, num_samples

    def ema_update_target_from_pool(self, pooled_params, momentum):
        with torch.no_grad():
            for key, new_val in zip(self.target_encoder.state_dict().keys(), pooled_params):
                param = self.target_encoder.state_dict()[key]
                param.mul_(momentum).add_(
                    torch.from_numpy(new_val).to(param.device), alpha=1 - momentum,
                )

    def save_checkpoint(self, round_num):
        torch.save({
            "encoder": self.encoder.state_dict(),
            "predictor": self.predictor.state_dict(),
            "target_encoder": self.target_encoder.state_dict(),
            "opt": self.optimizer.state_dict(),
            "scaler": None if self.scaler is None else self.scaler.state_dict(),
            "round": round_num,
            "step": self.step_count,
        }, os.path.join(self.folder, "latest.pt"))

# -- Flower Client --

class EchoJEPAClient(fl.client.NumPyClient):

    def __init__(self, trainer, mode, local_steps, sync_momentum=0.998):
        self.trainer = trainer
        self.mode = mode
        self.local_steps = local_steps
        self.sync_momentum = sync_momentum
        self.round_num = 0

    def get_parameters(self, config):
        if self.mode == 1:
            return (
                state_dict_to_numpy(self.trainer.encoder.state_dict())
                + state_dict_to_numpy(self.trainer.predictor.state_dict())
                + state_dict_to_numpy(self.trainer.target_encoder.state_dict())
            )
        return state_dict_to_numpy(self.trainer.encoder.state_dict())

    def fit(self, parameters, config):
        n_enc = self.trainer.n_encoder
        n_pred = self.trainer.n_predictor

        if self.mode == 1:
            load_numpy_into_model(self.trainer.encoder, parameters[:n_enc])
            load_numpy_into_model(self.trainer.predictor, parameters[n_enc:n_enc + n_pred])
            load_numpy_into_model(self.trainer.target_encoder, parameters[n_enc + n_pred:])
        elif self.mode == 2:
            self.trainer.ema_update_target_from_pool(parameters, self.sync_momentum)
        elif self.mode == 3:
            load_numpy_into_model(self.trainer.encoder, parameters)

        do_ema = (self.mode != 2)
        avg_loss, num_samples = self.trainer.train_round(self.local_steps, do_ema)
        self.round_num += 1

        if self.round_num % 300 == 0:
            self.trainer.save_checkpoint(self.round_num)
        logger.info(f"=== Round {self.round_num} complete (mode {self.mode}) ===")

        return self.get_parameters(config), num_samples, {"loss": float(avg_loss)}

    def evaluate(self, parameters, config):
        return 0.0, 0, {}


# -- Main --

def main():
    parser = argparse.ArgumentParser(description="Federated EchoJEPA Client")
    parser.add_argument("--fname", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    with open(args.fname) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    cfgs_fl = config.get("federated")
    server = cfgs_fl.get("server")
    mode = cfgs_fl.get("mode")
    local_steps = cfgs_fl.get("local_steps")
    sync_momentum = cfgs_fl.get("sync_momentum", 0.998)

    trainer = JEPATrainer(config, torch.device(args.device))
    client = EchoJEPAClient(trainer, mode, local_steps, sync_momentum)

    logger.info(f"Starting FL client — mode {mode}, server {server}")
    fl.client.start_client(
        server_address=server,
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()
