from __future__ import annotations

import argparse
import functools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import Dataset
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import get_cosine_schedule_with_warmup, set_seed

from src.config.settings import load_settings
from src.vae import VAEWithAdapter, load_diffusers_vae


VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _parse_image_size(image_size: str | None) -> tuple[int, int] | None:
    if image_size is None:
        return None
    text = str(image_size).strip().lower()
    if "x" in text:
        w_text, h_text = text.split("x", maxsplit=1)
        return int(w_text), int(h_text)
    value = int(text)
    return value, value


def _load_hyperparameters(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Hyperparameter file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Hyperparameter root must be a JSON object.")
    return data


def _build_round_index(dataset_path: Path) -> dict[str, dict[int, Path]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    index: dict[str, dict[int, Path]] = {}
    files = sorted(
        p for p in dataset_path.rglob("*") if p.is_file() and p.suffix.lower() in VALID_SUFFIXES
    )
    for path in files:
        stem = path.stem
        if "_" not in stem:
            continue
        sample_id, round_text = stem.rsplit("_", maxsplit=1)
        if not round_text.isdigit():
            continue
        round_idx = int(round_text)
        if sample_id not in index:
            index[sample_id] = {}
        index[sample_id][round_idx] = path
    return index


def _build_pair_records(
    *,
    round_index: dict[str, dict[int, Path]],
    pair_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample_id in sorted(round_index.keys()):
        rounds = round_index[sample_id]
        for spec in pair_specs:
            source_round = int(spec["source_round"])
            target_round = int(spec["target_round"])
            if source_round not in rounds or target_round not in rounds:
                continue
            weight = float(spec["probability"])
            records.append(
                {
                    "sample_id": sample_id,
                    "source_path": str(rounds[source_round]),
                    "target_path": str(rounds[target_round]),
                    "source_round": source_round,
                    "target_round": target_round,
                    "pair_name": f"{source_round}->{target_round}",
                    "weight": weight,
                }
            )
    if not records:
        raise RuntimeError("No valid training pairs found from dataset and sampling config.")
    return records


def _load_image_tensor(path: str, image_size: tuple[int, int] | None) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if image_size is not None:
            rgb = rgb.resize(image_size, Image.Resampling.LANCZOS)
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor * 2.0 - 1.0


def _collate_batch(
    batch: list[dict[str, Any]],
    image_size: tuple[int, int] | None,
) -> dict[str, Any]:
    sources = [_load_image_tensor(item["source_path"], image_size) for item in batch]
    targets = [_load_image_tensor(item["target_path"], image_size) for item in batch]
    return {
        "source": torch.stack(sources, dim=0),
        "target": torch.stack(targets, dim=0),
        "pair_name": [item["pair_name"] for item in batch],
    }


def _build_loss_fn(loss_cfg: dict[str, Any]):
    loss_type = str(loss_cfg.get("type", "l1")).strip().lower()
    
    # 对于 latent 损失，不需要加载额外的模型
    if loss_type in ["l1", "latent"]:
        return loss_type, None
    elif loss_type == "lpips" or loss_type == "mixed":
        try:
            import lpips
        except ImportError as exc:
            raise RuntimeError("Missing dependency `lpips`.") from exc
        network = str(loss_cfg.get("lpips_net", "alex"))
        model = lpips.LPIPS(net=network)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return loss_type, model
    raise ValueError("`loss.type` must be `l1`, `latent`, `lpips` or `mixed`.")


def _save_checkpoint(
    *,
    accelerator: Accelerator,
    model: VAEWithAdapter,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    output_dir: Path,
    global_step: int,
    epoch: int,
    avg_loss: float,
) -> None:
    if not accelerator.is_main_process:
        return

    ckpt_dir = output_dir / "checkpoints" / f"step_{global_step:08d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)

    torch.save(
        {
            "adapter_state_dict": unwrapped.adapter.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "avg_loss": avg_loss,
        },
        ckpt_dir / "adapter_checkpoint.pt",
    )


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.train.train",
        description="Train VAE latent adapter on round-pair regression targets.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/config/config.json"),
        help="Path to base config JSON.",
    )
    parser.add_argument(
        "--hyperparams",
        type=Path,
        default=Path("src/config/hyperparameters.json"),
        help="Path to hyperparameter JSON.",
    )
    return parser


def main() -> None:
    args = _arg_parser().parse_args()
    app_settings = load_settings(args.config)
    hyper = _load_hyperparameters(args.hyperparams)

    dataset_cfg = dict(hyper.get("dataset", {}))
    sampling_cfg = dict(hyper.get("sampling", {}))
    train_cfg = dict(hyper.get("train", {}))
    adapter_cfg = dict(hyper.get("adapter", {}))
    loss_cfg = dict(hyper.get("loss", {}))

    dataset_path = Path(str(dataset_cfg.get("path", "data/train_dataset")))
    image_size = _parse_image_size(dataset_cfg.get("image_size"))
    output_dir = Path(str(train_cfg.get("output_dir", "results/train_adapter")))
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_specs = list(sampling_cfg.get("pairs", []))
    if not pair_specs:
        raise ValueError("`sampling.pairs` must not be empty.")

    batch_size = int(train_cfg.get("batch_size", 8))
    num_workers = int(train_cfg.get("num_workers", 4))
    num_epochs = int(train_cfg.get("num_epochs", 5))
    learning_rate = float(train_cfg.get("learning_rate", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    adam_beta1 = float(train_cfg.get("adam_beta1", 0.9))
    adam_beta2 = float(train_cfg.get("adam_beta2", 0.999))
    adam_epsilon = float(train_cfg.get("adam_epsilon", 1e-8))
    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    gradient_accumulation_steps = int(train_cfg.get("gradient_accumulation_steps", 1))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    mixed_precision = str(train_cfg.get("mixed_precision", "fp16"))
    seed = int(train_cfg.get("seed", 42))
    save_every_steps = int(train_cfg.get("save_every_steps", 500))
    log_every_steps = int(train_cfg.get("log_every_steps", 20))

    set_seed(seed)
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    round_index = _build_round_index(dataset_path)
    pair_records = _build_pair_records(round_index=round_index, pair_specs=pair_specs)
    hf_dataset = Dataset.from_list(pair_records)

    pair_weights = torch.tensor([float(item["weight"]) for item in pair_records], dtype=torch.double)
    steps_per_epoch_raw = train_cfg.get("steps_per_epoch")
    if steps_per_epoch_raw is None:
        steps_per_epoch = max(1, math.ceil(len(pair_records) / max(batch_size, 1)))
    else:
        steps_per_epoch = int(steps_per_epoch_raw)
    sampler = WeightedRandomSampler(
        weights=pair_weights,
        num_samples=steps_per_epoch * batch_size,
        replacement=True,
    )
    collate_fn = functools.partial(_collate_batch, image_size=image_size)
    train_loader = DataLoader(
        hf_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    vae = load_diffusers_vae(
        model_type=app_settings.vae.model_type,
        model_path=app_settings.vae.model_path,
        dtype_name=app_settings.vae.dtype,
    )
    latent_channels = int(getattr(vae.config, "latent_channels", 4))
    from src.vae.reconstructor import AdapterSettings, InterventionSettings
    
    adapter_settings = AdapterSettings(
        enabled=True,
        hidden_multiplier=int(adapter_cfg.get("hidden_multiplier", 4)),
        num_blocks=int(adapter_cfg.get("num_blocks", 4)),
        dropout=float(adapter_cfg.get("dropout", 0.0)),
    )
    
    intervention_dict = dict(hyper.get("intervention", {}))
    intervention_settings = InterventionSettings(
        enabled=bool(intervention_dict.get("enabled", True)),
        strength=float(intervention_dict.get("strength", 0.22)),
        direction_suppress=float(intervention_dict.get("direction_suppress", 0.45)),
        radial_jitter=float(intervention_dict.get("radial_jitter", 0.16)),
        max_freq_suppress=float(intervention_dict.get("max_freq_suppress", 0.35)),
        noise_std=float(intervention_dict.get("noise_std", 0.015)),
        channel_dropout_prob=float(intervention_dict.get("channel_dropout_prob", 0.02)),
        channel_shuffle_prob=float(intervention_dict.get("channel_shuffle_prob", 0.03)),
        preserve_mean=bool(intervention_dict.get("preserve_mean", True)),
        preserve_std=bool(intervention_dict.get("preserve_std", True)),
    )

    model = VAEWithAdapter(
        vae=vae,
        latent_channels=latent_channels,
        adapter_cfg=adapter_settings,
        intervention_cfg=intervention_settings,
    )

    optimizer = torch.optim.AdamW(
        model.adapter.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        eps=adam_epsilon,
        weight_decay=weight_decay,
    )
    total_train_steps = steps_per_epoch * num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )

    loss_type, lpips_model = _build_loss_fn(loss_cfg)

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    if lpips_model is not None:
        lpips_model = lpips_model.to(accelerator.device)

    if accelerator.is_main_process:
        (output_dir / "train_config_snapshot.json").write_text(
            json.dumps(
                {
                    "config_path": str(args.config),
                    "hyperparams_path": str(args.hyperparams),
                    "dataset_path": str(dataset_path),
                    "num_pair_records": len(pair_records),
                    "steps_per_epoch": steps_per_epoch,
                    "num_epochs": num_epochs,
                    "loss_type": loss_type,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    global_step = 0
    vae_dtype = next(model.vae.parameters()).dtype 
    for epoch in range(num_epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        
        epoch_rgb_bias_sum = torch.zeros(3, device=accelerator.device, dtype=torch.float32)
        epoch_rgb_bias_count = 0

        for batch in train_loader:
            with accelerator.accumulate(model):
                source = batch["source"].to(device=accelerator.device, dtype=vae_dtype)
                target = batch["target"].to(device=accelerator.device, dtype=vae_dtype)

                # ---------------------------------------------------------
                # Loss Calculation Logic
                # ---------------------------------------------------------
                if loss_type == "latent":
                    # 1. Get Prediction Latent (Forward pass with return_latent=True)
                    pred_latent = model(source, return_latent=True)
                    
                    # 2. Get Target Latent (Detached, no grad)
                    # Since VAE is frozen, we can call this directly
                    with torch.no_grad():
                        # Use the same deterministic encoding logic as forward
                        # Note: model.encode_latent uses the fixed generator if configured
                        target_latent = model.encode_latent(target)

                    # 3. Calculate Latent Loss
                    # L1 Loss (Pixel/Value accuracy)
                    loss_l1 = F.l1_loss(pred_latent, target_latent)
                    
                    # Cosine Similarity Loss (Direction/Semantic consistency)
                    # Flatten: [B, C, H, W] -> [B, C*H*W]
                    pred_flat = pred_latent.view(pred_latent.shape[0], -1)
                    target_flat = target_latent.view(target_latent.shape[0], -1)
                    
                    # Cosine Similarity: 1 - cos(theta)
                    # We want to minimize this, pushing theta -> 0
                    loss_cosine = 1.0 - F.cosine_similarity(pred_flat, target_flat, dim=1).mean()
                    
                    # Combined Loss (Weights can be tuned, 0.5 is a safe start)
                    loss = loss_l1 + 0.5 *loss_cosine
                    
                elif loss_type == "l1":
                    prediction = model(source)
                    loss = F.l1_loss(prediction, target)
                    
                elif loss_type == "lpips":
                    if lpips_model is None:
                        raise RuntimeError("LPIPS model is not initialized.")
                    prediction = model(source)
                    loss = lpips_model(prediction.float(), target.float()).mean()
                    
                elif loss_type == "mixed":
                    if lpips_model is None:
                        raise RuntimeError("LPIPS model is not initialized.")
                    prediction = model(source)
                    loss = F.l1_loss(prediction, target) + 0.2 * lpips_model(prediction.float(), target.float()).mean()
                else:
                    raise ValueError(f"Unknown loss type: {loss_type}")

                accelerator.backward(loss)

                # ---- grad norm (adapter only) ----
                grad_norm_value = None
                if accelerator.sync_gradients:
                    grad_norm_tensor = accelerator.clip_grad_norm_(
                        model.adapter.parameters(), max_grad_norm
                    )
                    if isinstance(grad_norm_tensor, torch.Tensor):
                        grad_norm_value = float(grad_norm_tensor.detach().item())
                    else:
                        grad_norm_value = float(grad_norm_tensor)
                # ----------------------------------

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            epoch_loss_sum += float(loss.detach().item())
            epoch_loss_count += 1

            # Log RGB bias only for pixel-space losses or if decoded
            if loss_type != "latent":
                # Need to decode for bias check if we did latent training, or just use existing prediction
                if loss_type == "latent":
                     # Optional: Can decode pred_latent for monitoring, but costs VRAM
                     pass
                else:
                    rgb_bias = (prediction.detach() - target.detach()).mean(dim=(0, 2, 3)).float()
                    epoch_rgb_bias_sum += rgb_bias
                    epoch_rgb_bias_count += 1

            if accelerator.is_main_process and global_step % log_every_steps == 0:
                avg_loss = epoch_loss_sum / max(epoch_loss_count, 1)
                
                log_str = f"[train] epoch={epoch + 1}/{num_epochs} step={global_step} loss={avg_loss:.6f}"
                
                if loss_type == "latent":
                    log_str += f" (l1={loss_l1.item():.4f} cos={loss_cosine.item():.4f})"
                else:
                    avg_rgb_bias = (epoch_rgb_bias_sum / max(epoch_rgb_bias_count, 1)).detach().cpu().numpy()
                    log_str += f" rgb_bias=({avg_rgb_bias[0]:+.4f},{avg_rgb_bias[1]:+.4f},{avg_rgb_bias[2]:+.4f})"

                grad_text = "n/a" if grad_norm_value is None else f"{grad_norm_value:.4f}"
                log_str += f" grad={grad_text}"
                print(log_str)

            if global_step % save_every_steps == 0:
                avg_loss = epoch_loss_sum / max(epoch_loss_count, 1)
                _save_checkpoint(
                    accelerator=accelerator,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    output_dir=output_dir,
                    global_step=global_step,
                    epoch=epoch,
                    avg_loss=avg_loss,
                )


        if accelerator.is_main_process:
            epoch_loss = epoch_loss_sum / max(epoch_loss_count, 1)
            print(f"[epoch] {epoch + 1}/{num_epochs} avg_loss={epoch_loss:.6f}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        torch.save(
            {
                "adapter_state_dict": unwrapped.adapter.state_dict(),
                "scale": unwrapped.scale,
                "latent_channels": latent_channels,
            },
            output_dir / "adapter_final.pt",
        )
        print(f"[done] adapter={output_dir / 'adapter_final.pt'}")


if __name__ == "__main__":
    main()