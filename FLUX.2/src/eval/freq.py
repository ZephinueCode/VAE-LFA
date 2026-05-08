from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import AppConfig, load_config
from src.utils.data import iter_samples, resolve_eval_prompts
from src.eval.metrics import _build_lpips, _lpips_vs_base, _ssim_global, _to_gray, _to_lpips_tensor


def resolve_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    value = mapping.get(str(name).strip().lower(), torch.float32)
    if value != torch.float32 and not torch.cuda.is_available():
        return torch.float32
    return value


def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
    return transform(image).unsqueeze(0)


def postprocess_tensor(tensor: torch.Tensor) -> Image.Image:
    tensor = (tensor / 2 + 0.5).clamp(0, 1)
    tensor = tensor.detach().cpu().permute(0, 2, 3, 1).float().numpy()[0]
    return Image.fromarray((tensor * 255).astype("uint8"))


def pil_to_unit_tensor(image: Image.Image) -> torch.Tensor:
    tensor = T.ToTensor()(image).unsqueeze(0)
    return tensor


def unit_tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.clamp(0, 1).detach().cpu().permute(0, 2, 3, 1).float().numpy()[0]
    return Image.fromarray((tensor * 255).astype("uint8"))


def lowpass_filter(x: torch.Tensor, kernel: int = 9) -> torch.Tensor:
    if kernel <= 1:
        return x
    pad = kernel // 2
    x_pad = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="replicate")
    return torch.nn.functional.avg_pool2d(x_pad, kernel_size=kernel, stride=1)


def round_to_multiple(value: int, multiple: int) -> int:
    value = max(int(value), multiple)
    return max(multiple, (value // multiple) * multiple)


def center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def resolve_target_size(config: AppConfig, image: Image.Image, model_type: str) -> tuple[int, int]:
    cfg_height = config.inference.height
    cfg_width = config.inference.width

    crop_side = min(image.size)
    if cfg_width is not None and cfg_height is not None:
        target = min(int(cfg_width), int(cfg_height))
    elif cfg_width is not None:
        target = int(cfg_width)
    elif cfg_height is not None:
        target = int(cfg_height)
    else:
        target = min(int(crop_side), 1024)

    if model_type == "kandinsky5":
        # Kandinsky's square bucket is 1024x1024.
        return 1024, 1024

    resolved = round_to_multiple(int(round(target)), multiple=16)
    return resolved, resolved


def normalize_model_type(value: str) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "flux": "flux2",
        "flux2": "flux2",
        "kandinsky": "kandinsky5",
        "kandinsky5": "kandinsky5",
        "k5": "kandinsky5",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported model type: {value}")
    return aliases[normalized]


def load_pipeline(config: AppConfig, model_type: str, dtype: torch.dtype) -> Any:
    common_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "local_files_only": config.model.local_files_only,
        "low_cpu_mem_usage": config.model.low_cpu_mem_usage,
    }
    if config.model.revision:
        common_kwargs["revision"] = config.model.revision
    if config.model.variant:
        common_kwargs["variant"] = config.model.variant

    if model_type == "flux2":
        from diffusers import Flux2KleinPipeline

        pipe_cls = Flux2KleinPipeline
    else:
        from diffusers import Kandinsky5I2IPipeline

        pipe_cls = Kandinsky5I2IPipeline
    try:
        pipe = pipe_cls.from_pretrained(config.model.model_path, **common_kwargs)
    except TypeError:
        pipe = pipe_cls.from_pretrained(
            config.model.model_path,
            torch_dtype=dtype,
        )
    if config.model.enable_model_cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(config.model.device)
    return pipe


def resolve_module_device_dtype(module: Any) -> tuple[torch.device, torch.dtype]:
    for tensor in list(module.parameters()) + list(module.buffers()):
        if tensor is not None and tensor.device.type != "meta":
            return tensor.device, tensor.dtype
    return torch.device("cpu"), torch.float32


def resolve_vae_io_device_dtype(pipe: Any) -> tuple[torch.device, torch.dtype]:
    vae_device, vae_dtype = resolve_module_device_dtype(pipe.vae)
    execution_device = getattr(pipe, "_execution_device", None)
    if execution_device is not None:
        execution_device = torch.device(execution_device)
        if execution_device.type != "cpu":
            return execution_device, vae_dtype
    return vae_device, vae_dtype


def encode_scaled_latents(pipe: Any, image: Image.Image) -> torch.Tensor:
    vae_device, vae_dtype = resolve_vae_io_device_dtype(pipe)
    vae_input = preprocess_image(image).to(device=vae_device, dtype=vae_dtype)
    with torch.no_grad():
        raw_latents = pipe.vae.encode(vae_input).latent_dist.mode()
    shift = float(getattr(pipe.vae.config, "shift_factor", 0.0))
    scale = float(getattr(pipe.vae.config, "scaling_factor", 1.0))
    return (raw_latents - shift) * scale


def decode_scaled_latents(pipe: Any, latents: torch.Tensor) -> Image.Image:
    shift = float(getattr(pipe.vae.config, "shift_factor", 0.0))
    scale = float(getattr(pipe.vae.config, "scaling_factor", 1.0))
    vae_device, vae_dtype = resolve_vae_io_device_dtype(pipe)
    with torch.no_grad():
        decode_input = ((latents / scale) + shift).to(device=vae_device, dtype=vae_dtype)
        dec_tensor = pipe.vae.decode(decode_input).sample
    return postprocess_tensor(dec_tensor)


def save_grid(images: list[Image.Image], out_path: Path) -> None:
    if not images:
        raise ValueError("No images provided for grid.")
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    cell_w = max(widths)
    cell_h = max(heights)
    canvas = Image.new("RGB", (len(images) * cell_w, cell_h), color=(255, 255, 255))
    for idx, image in enumerate(images):
        canvas.paste(image, (idx * cell_w, 0))
    canvas.save(out_path)


def save_rows_grid(rows: list[list[Image.Image]], out_path: Path) -> None:
    if not rows or not rows[0]:
        raise ValueError("No image rows provided for grid.")
    flat = [img for row in rows for img in row]
    cell_w = max(img.width for img in flat)
    cell_h = max(img.height for img in flat)
    cols = max(len(row) for row in rows)
    canvas = Image.new("RGB", (cols * cell_w, len(rows) * cell_h), color=(255, 255, 255))
    for row_idx, row in enumerate(rows):
        for col_idx, image in enumerate(row):
            canvas.paste(image, (col_idx * cell_w, row_idx * cell_h))
    canvas.save(out_path)


def build_breakdown(
    pipe: Any,
    image: Image.Image,
    *,
    kernel: int,
    square_size: tuple[int, int],
    output_dir: Path | None = None,
    prefix: str = "",
) -> dict[str, Any]:
    prepared = center_crop_square(image).resize((square_size[1], square_size[0]), resample=Image.Resampling.LANCZOS)
    full_latents = encode_scaled_latents(pipe, prepared)
    low_latents = lowpass_filter(full_latents.float(), kernel=kernel).to(dtype=full_latents.dtype)
    high_latents = full_latents - low_latents
    zero_latents = torch.zeros_like(full_latents)

    low_img = decode_scaled_latents(pipe, low_latents)
    high_img = decode_scaled_latents(pipe, high_latents)
    zero_img = decode_scaled_latents(pipe, zero_latents)

    low_tensor = pil_to_unit_tensor(low_img)
    high_tensor = pil_to_unit_tensor(high_img)
    zero_tensor = pil_to_unit_tensor(zero_img)
    pixel_add_tensor = (low_tensor + (high_tensor - zero_tensor)).clamp(0, 1)
    pixel_add_img = unit_tensor_to_pil(pixel_add_tensor)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        prepared.save(output_dir / f"{prefix}input_square.png")
        low_img.save(output_dir / f"{prefix}decode_low_only.png")
        high_img.save(output_dir / f"{prefix}decode_high_only.png")
        zero_img.save(output_dir / f"{prefix}decode_zero.png")
        pixel_add_img.save(output_dir / f"{prefix}decode_pixel_add.png")

    low_energy = float((low_latents.float() ** 2).sum().item())
    high_energy = float((high_latents.float() ** 2).sum().item())
    total_energy = max(low_energy + high_energy, 1e-12)

    return {
        "input_image": prepared,
        "low_image": low_img,
        "high_image": high_img,
        "pixel_add_image": pixel_add_img,
        "latent_shape": list(full_latents.shape),
        "low_energy": low_energy,
        "high_energy": high_energy,
        "low_ratio": low_energy / total_energy,
        "high_ratio": high_energy / total_energy,
    }


def load_single_sample(config: AppConfig, image_path: Path):
    original_path = config.data.input_path
    original_max_samples = config.data.max_samples
    try:
        config.data.input_path = image_path
        config.data.max_samples = 1
        sample = next(iter(iter_samples(config.data)))
    finally:
        config.data.input_path = original_path
        config.data.max_samples = original_max_samples
    return sample


def load_input_samples(config: AppConfig, input_path: Path):
    original_path = config.data.input_path
    original_max_samples = config.data.max_samples
    try:
        config.data.input_path = input_path
        samples = list(iter_samples(config.data))
    finally:
        config.data.input_path = original_path
        config.data.max_samples = original_max_samples
    return samples


def unique_sample_name(sample, seen: dict[str, int]) -> str:
    group_name = sample.image_path.parent.name or "default"
    base_id = sample.sample_id.strip() or "sample"
    composite = f"{group_name}_{base_id}"
    count = seen.get(composite, 0)
    seen[composite] = count + 1
    return composite if count == 0 else f"{composite}_{count:03d}"


def build_runner(
    config: AppConfig,
    *,
    model_type: str,
    use_lpt: bool,
    augmentation: str,
    lowpass_filter: str,
    lowpass_sigma: float,
):
    if model_type == "flux2":
        from src.model.model import FluxKleinRunner

        return FluxKleinRunner(
            config=config,
            use_lpt=use_lpt,
            augmentation=augmentation,
            lowpass_filter=lowpass_filter,
            lowpass_sigma=lowpass_sigma,
        )

    from src.model.model_kandinsky5 import Kandinsky5Runner

    return Kandinsky5Runner(
        config=config,
        use_lpt=use_lpt,
        augmentation=augmentation,
        lowpass_filter=lowpass_filter,
        lowpass_sigma=lowpass_sigma,
    )


def run_vae_rollout(
    pipe: Any,
    *,
    image: Image.Image,
    num_rounds: int,
    run_dir: Path,
    square_size: tuple[int, int],
) -> list[str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    prepared = center_crop_square(image).resize((square_size[1], square_size[0]), resample=Image.Resampling.LANCZOS)
    input_path = run_dir / "round_000_input.png"
    prepared.save(input_path)

    round_images = [str(input_path)]
    current_image = prepared
    for round_idx in range(1, num_rounds + 1):
        latents = encode_scaled_latents(pipe, current_image)
        current_image = decode_scaled_latents(pipe, latents)
        round_path = run_dir / f"round_{round_idx:03d}.png"
        current_image.save(round_path)
        round_images.append(str(round_path))

    return round_images


def pil_to_np(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def compute_component_metrics(
    image_a: Image.Image,
    image_b: Image.Image,
    *,
    lpips_model: Any,
    torch_module: Any,
    lpips_device: str,
) -> dict[str, float]:
    arr_a = pil_to_np(image_a)
    arr_b = pil_to_np(image_b)
    gray_a = _to_gray(arr_a)
    gray_b = _to_gray(arr_b)
    base_lpips_tensor = _to_lpips_tensor(arr_a, torch_module=torch_module, device=lpips_device)
    return {
        "l1_drift": float(np.mean(np.abs(arr_b - arr_a))),
        "ssim": float(_ssim_global(gray_a, gray_b)),
        "lpips": float(
            _lpips_vs_base(
                base_tensor=base_lpips_tensor,
                current=arr_b,
                lpips_model=lpips_model,
                torch_module=torch_module,
                device=lpips_device,
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect low/high-frequency contributions inside VAE latent space.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML/JSON config.")
    parser.add_argument("--image", type=Path, required=True, help="Input image path, JSON path, or directory.")
    parser.add_argument("--model-type", type=str, default="flux2", help="flux2 or kandinsky5")
    parser.add_argument("--kernel", type=int, default=9, help="AvgPool kernel size for low-pass split.")
    parser.add_argument("--outdir", type=Path, default=None, help="Optional output directory.")
    parser.add_argument("--eval", type=str, choices=["noop", "cycle", "long_chain"], default=None, help="If set, first run an iterative editing rollout and visualize every round.")
    parser.add_argument("--vae-only", action="store_true", help="In rollout mode, use a pure multi-round VAE encode/decode baseline instead of model editing.")
    parser.add_argument("--cpu-offload", action="store_true", help="Enable model CPU offload for this freq run to reduce GPU memory usage.")
    parser.add_argument("--latent", type=str, default="false", help="Enable latent pass-through during iterative rollout (true/false).")
    parser.add_argument("--augmentation", type=str, choices=["none", "momentum", "soft"], default="momentum", help="Latent alignment strategy for iterative rollout. Use `none` for pure VAE bypass without extra interference.")
    parser.add_argument("--lowpass-filter", type=str, choices=["avg_pool", "fft_gaussian"], default="avg_pool", help="Low/high frequency split used by iterative momentum and soft augmentation.")
    parser.add_argument("--lowpass-sigma", type=float, default=0.25, help="Gaussian sigma for iterative rollout when using fft_gaussian.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.cpu_offload:
        config.model.enable_model_cpu_offload = True
    model_type = normalize_model_type(args.model_type)
    dtype = resolve_torch_dtype(config.model.torch_dtype)

    if not args.image.exists():
        raise FileNotFoundError(f"Input path not found: {args.image}")

    output_name = f"freq_{args.image.stem}_{model_type}" if args.image.is_file() else f"freq_{args.image.name}_{model_type}"
    output_dir = args.outdir or (config.output.output_dir / output_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_input_samples(config, args.image)
    if not samples:
        raise ValueError(f"No samples found under: {args.image}")

    seen_names: dict[str, int] = {}
    sample_summaries: list[dict[str, Any]] = []

    if args.eval is None:
        if args.vae_only:
            raise ValueError("--vae-only requires --eval.")
        print("Loading pipeline...")
        pipe = load_pipeline(config, model_type=model_type, dtype=dtype)
        for idx, sample in enumerate(samples, start=1):
            sample_name = unique_sample_name(sample, seen_names)
            sample_dir = output_dir if len(samples) == 1 else (output_dir / sample_name)
            square_image = center_crop_square(sample.image)
            height, width = resolve_target_size(config, square_image, model_type=model_type)
            square_size = (height, width)
            print(f"[{idx}/{len(samples)}] Encoding VAE latents for {sample_name}...")
            result = build_breakdown(
                pipe,
                sample.image,
                kernel=args.kernel,
                square_size=square_size,
                output_dir=sample_dir,
            )
            save_grid(
                [
                    result["input_image"],
                    result["low_image"],
                    result["high_image"],
                    result["pixel_add_image"],
                ],
                sample_dir / "freq_grid.png",
            )
            sample_summaries.append(
                {
                    "sample_id": sample_name,
                    "image_path": str(sample.image_path),
                    "model_type": model_type,
                    "kernel": int(args.kernel),
                    "resolved_height": int(height),
                    "resolved_width": int(width),
                    "latent_shape": result["latent_shape"],
                    "low_energy": result["low_energy"],
                    "high_energy": result["high_energy"],
                    "low_ratio": result["low_ratio"],
                    "high_ratio": result["high_ratio"],
                    "outputs": {
                        "sample_dir": str(sample_dir),
                        "grid": str(sample_dir / "freq_grid.png"),
                    },
                }
            )

        summary_path = output_dir / "summary.json"
        summary = {
            "input_path": str(args.image),
            "model_type": model_type,
            "model_path": config.model.model_path,
            "kernel": int(args.kernel),
            "num_samples": len(sample_summaries),
            "samples": sample_summaries,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved frequency breakdown to {output_dir}")
        return

    use_lpt = args.latent.strip().lower() in {"true", "1", "yes", "t"}
    augmentation = args.augmentation.strip().lower()
    lowpass_filter = args.lowpass_filter.strip().lower()
    lpips_model, torch_module, lpips_device = _build_lpips()
    pipe = None
    runner = None
    if args.vae_only:
        print("Loading pipeline for VAE-only rollout...")
        pipe = load_pipeline(config, model_type=model_type, dtype=dtype)
    else:
        runner = build_runner(
            config,
            model_type=model_type,
            use_lpt=use_lpt,
            augmentation=augmentation,
            lowpass_filter=lowpass_filter,
            lowpass_sigma=args.lowpass_sigma,
        )

    try:
        for idx, sample in enumerate(samples, start=1):
            sample_name = unique_sample_name(sample, seen_names)
            sample_dir = output_dir if len(samples) == 1 else (output_dir / sample_name)
            rollout_dir = sample_dir / "round_images"
            per_round_dir = sample_dir / "per_round"
            prompts = resolve_eval_prompts(sample, args.eval)
            square_image = center_crop_square(sample.image)
            height, width = resolve_target_size(config, square_image, model_type=model_type)
            square_size = (height, width)
            if args.vae_only:
                print(f"[{idx}/{len(samples)}] Running VAE-only rollout for {sample_name}...")
                round_images = run_vae_rollout(
                    pipe,
                    image=sample.image,
                    num_rounds=len(prompts),
                    run_dir=rollout_dir,
                    square_size=square_size,
                )
            else:
                print(f"[{idx}/{len(samples)}] Running rollout for {sample_name}...")
                trace = runner.run_sample(
                    sample_id=sample_name,
                    image=sample.image,
                    prompts=prompts,
                    run_dir=rollout_dir,
                )
                round_images = trace.round_images

            round_rows: list[list[Image.Image]] = []
            rounds_summary: list[dict[str, Any]] = []
            selected_round_images: dict[int, dict[str, Image.Image]] = {}
            analysis_pipe = pipe if args.vae_only else runner.pipe
            for round_idx, image_path_str in enumerate(round_images[1:], start=1):
                with Image.open(image_path_str) as round_image:
                    result = build_breakdown(
                        analysis_pipe,
                        round_image.convert("RGB"),
                        kernel=args.kernel,
                        square_size=square_size,
                        output_dir=per_round_dir,
                        prefix=f"round_{round_idx:03d}_",
                    )
                round_rows.append(
                    [
                        result["input_image"],
                        result["low_image"],
                        result["high_image"],
                        result["pixel_add_image"],
                    ]
                )
                rounds_summary.append(
                    {
                        "round": round_idx,
                        "prompt": prompts[round_idx - 1],
                        "image_path": image_path_str,
                        "latent_shape": result["latent_shape"],
                        "low_energy": result["low_energy"],
                        "high_energy": result["high_energy"],
                        "low_ratio": result["low_ratio"],
                        "high_ratio": result["high_ratio"],
                    }
                )
                if round_idx in {1, 10}:
                    selected_round_images[round_idx] = {
                        "low": result["low_image"],
                        "high": result["high_image"],
                    }

            grid_name = f"freq_grid_{args.eval}.png"
            save_rows_grid(round_rows, sample_dir / grid_name)
            round_a = 1 if 1 in selected_round_images else None
            round_b = 10 if 10 in selected_round_images else (len(rounds_summary) if len(rounds_summary) in selected_round_images else None)
            component_metrics = None
            if round_a is not None and round_b is not None and round_a != round_b:
                component_metrics = {
                    "round_a": round_a,
                    "round_b": round_b,
                    "low": compute_component_metrics(
                        selected_round_images[round_a]["low"],
                        selected_round_images[round_b]["low"],
                        lpips_model=lpips_model,
                        torch_module=torch_module,
                        lpips_device=lpips_device,
                    ),
                    "high": compute_component_metrics(
                        selected_round_images[round_a]["high"],
                        selected_round_images[round_b]["high"],
                        lpips_model=lpips_model,
                        torch_module=torch_module,
                        lpips_device=lpips_device,
                    ),
                }

            sample_summary = {
                "sample_id": sample_name,
                "image_path": str(sample.image_path),
                "meta_path": None if sample.meta_path is None else str(sample.meta_path),
                "model_type": model_type,
                "eval_mode": args.eval,
                "kernel": int(args.kernel),
                "resolved_height": int(height),
                "resolved_width": int(width),
                "rollout_backend": "vae_only" if args.vae_only else "model",
                "latent_pass_through": use_lpt,
                "augmentation": augmentation,
                "lowpass_filter": lowpass_filter,
                "lowpass_sigma": args.lowpass_sigma,
                "num_rounds": len(rounds_summary),
                "outputs": {
                    "sample_dir": str(sample_dir),
                    "rollout_dir": str(rollout_dir),
                    "per_round_dir": str(per_round_dir),
                    "grid": str(sample_dir / grid_name),
                },
                "round1_vs_round10_component_metrics": component_metrics,
                "rounds": rounds_summary,
            }
            sample_summaries.append(sample_summary)
            (sample_dir / f"summary_{args.eval}.json").write_text(json.dumps(sample_summary, indent=2), encoding="utf-8")
    finally:
        if runner is not None:
            runner.close()

    component_metric_entries = [
        {
            "sample_id": sample["sample_id"],
            "image_path": sample["image_path"],
            "metrics": sample["round1_vs_round10_component_metrics"],
        }
        for sample in sample_summaries
        if sample.get("round1_vs_round10_component_metrics") is not None
    ]
    component_aggregate = None
    if component_metric_entries:
        low_l1 = [entry["metrics"]["low"]["l1_drift"] for entry in component_metric_entries]
        low_ssim = [entry["metrics"]["low"]["ssim"] for entry in component_metric_entries]
        low_lpips = [entry["metrics"]["low"]["lpips"] for entry in component_metric_entries]
        high_l1 = [entry["metrics"]["high"]["l1_drift"] for entry in component_metric_entries]
        high_ssim = [entry["metrics"]["high"]["ssim"] for entry in component_metric_entries]
        high_lpips = [entry["metrics"]["high"]["lpips"] for entry in component_metric_entries]
        component_aggregate = {
            "num_samples": len(component_metric_entries),
            "low": {
                "l1_drift": float(np.mean(low_l1)),
                "ssim": float(np.mean(low_ssim)),
                "lpips": float(np.mean(low_lpips)),
            },
            "high": {
                "l1_drift": float(np.mean(high_l1)),
                "ssim": float(np.mean(high_ssim)),
                "lpips": float(np.mean(high_lpips)),
            },
        }

    summary = {
        "input_path": str(args.image),
        "model_type": model_type,
        "model_path": config.model.model_path,
        "eval_mode": args.eval,
        "kernel": int(args.kernel),
        "rollout_backend": "vae_only" if args.vae_only else "model",
        "latent_pass_through": use_lpt,
        "augmentation": augmentation,
        "lowpass_filter": lowpass_filter,
        "lowpass_sigma": args.lowpass_sigma,
        "num_samples": len(sample_summaries),
        "round1_vs_round10_component_metrics": {
            "aggregate": component_aggregate,
            "per_sample": component_metric_entries,
        },
        "samples": sample_summaries,
    }
    (output_dir / f"summary_{args.eval}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / f"round1_vs_round10_component_metrics_{args.eval}.json").write_text(
        json.dumps(
            {
                "input_path": str(args.image),
                "model_type": model_type,
                "eval_mode": args.eval,
                "rollout_backend": "vae_only" if args.vae_only else "model",
                "latent_pass_through": use_lpt,
                "augmentation": augmentation,
                "aggregate": component_aggregate,
                "per_sample": component_metric_entries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved iterative frequency breakdown to {output_dir}")


if __name__ == "__main__":
    main()
