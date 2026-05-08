from __future__ import annotations

import argparse
import contextlib
import csv
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import yaml
from PIL import Image


DEFAULT_IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
DEFAULT_NOOP_PROMPT = "Keep the image unchanged."


@dataclass
class ModelConfig:
    model_path: str = "./sd"
    torch_dtype: str = "float16"
    device: str = "cuda"
    local_files_only: bool = False
    enable_model_cpu_offload: bool = False
    low_cpu_mem_usage: bool = True
    variant: str | None = None
    revision: str | None = None


@dataclass
class DataConfig:
    input_path: Path
    max_samples: int | None = None
    recursive: bool = True
    image_suffixes: list[str] = field(default_factory=lambda: list(DEFAULT_IMAGE_SUFFIXES))


@dataclass
class InferenceConfig:
    seed: int = 0
    seed_stride: int = 1
    height: int | None = None
    width: int | None = None
    guidance_scale: float = 7.5
    image_guidance_scale: float = 1.5
    num_inference_steps: int = 28
    strength: float | None = None
    negative_prompt: str = ""
    free_form_mask: bool = True
    max_sequence_length: int | None = None


@dataclass
class OutputConfig:
    output_dir: Path = Path("results")
    run_name: str | None = None
    save_round_images: bool = True


@dataclass
class AppConfig:
    model: ModelConfig
    data: DataConfig
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


@dataclass
class ImageSample:
    sample_id: str
    image: Image.Image
    image_path: Path
    meta_path: Path | None
    prompts_by_eval: dict[str, list[str]]


@dataclass
class RunTrace:
    sample_id: str
    round_images: list[str]


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python run_ultraedit_metrics.py",
        description="UltraEdit metrics evaluation with optional latent momentum.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON/YAML config.")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["sd3.5", "sd3", "sd35", "ultraedit"],
        default="sd3.5",
        help="Compatibility flag. This script always uses SD3 UltraEdit.",
    )
    parser.add_argument(
        "--eval",
        type=str,
        choices=["noop", "cycle", "long_chain"],
        required=True,
        help="Evaluation protocol to run for every sample.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Override data.max_samples.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Override data.input_path.")
    parser.add_argument("--run-name", type=str, default=None, help="Override output.run_name.")
    parser.add_argument(
        "--latent",
        type=str,
        default="false",
        help="Enable latent pass-through (true/false).",
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        choices=["momentum", "soft"],
        default="momentum",
        help="Compatibility flag. Only momentum is supported when --latent true.",
    )
    parser.add_argument(
        "--lowpass-filter",
        type=str,
        choices=["avg_pool", "fft_gaussian"],
        default="avg_pool",
        help="Low/high frequency split used by momentum.",
    )
    parser.add_argument(
        "--lowpass-sigma",
        type=float,
        default=0.25,
        help="Gaussian sigma for --lowpass-filter fft_gaussian.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=None,
        help="Compatibility flag. Ignored by UltraEdit.",
    )
    return parser


def _normalize_model_type(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"sd3.5", "sd3", "sd35", "ultraedit"}:
        return "sd3.5"
    raise ValueError(f"Unsupported model type: {value}")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {k: _to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def _load_raw(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    data = yaml.safe_load(text) if suffix in {".yaml", ".yml"} else json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a JSON/YAML object.")
    return data


def _as_dict(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"`{name}` must be an object.")
    return value


def load_config(path: Path) -> AppConfig:
    raw = _load_raw(path)
    model_raw = _as_dict(raw.get("model"), "model")
    data_raw = _as_dict(raw.get("data"), "data")
    infer_raw = _as_dict(raw.get("inference"), "inference")
    output_raw = _as_dict(raw.get("output"), "output")

    if "input_path" not in data_raw:
        raise ValueError("`data.input_path` is required.")

    model = ModelConfig(
        model_path=str(model_raw.get("model_path", ModelConfig.model_path)),
        torch_dtype=str(model_raw.get("torch_dtype", ModelConfig.torch_dtype)),
        device=str(model_raw.get("device", ModelConfig.device)),
        local_files_only=bool(model_raw.get("local_files_only", ModelConfig.local_files_only)),
        enable_model_cpu_offload=bool(
            model_raw.get("enable_model_cpu_offload", ModelConfig.enable_model_cpu_offload)
        ),
        low_cpu_mem_usage=bool(model_raw.get("low_cpu_mem_usage", ModelConfig.low_cpu_mem_usage)),
        variant=model_raw.get("variant"),
        revision=model_raw.get("revision"),
    )
    data = DataConfig(
        input_path=Path(str(data_raw["input_path"])),
        max_samples=None if data_raw.get("max_samples") is None else int(data_raw["max_samples"]),
        recursive=bool(data_raw.get("recursive", True)),
        image_suffixes=[str(item).lower() for item in data_raw.get("image_suffixes", DEFAULT_IMAGE_SUFFIXES)],
    )
    inference = InferenceConfig(
        seed=int(infer_raw.get("seed", InferenceConfig.seed)),
        seed_stride=int(infer_raw.get("seed_stride", InferenceConfig.seed_stride)),
        height=None if infer_raw.get("height") is None else int(infer_raw["height"]),
        width=None if infer_raw.get("width") is None else int(infer_raw["width"]),
        guidance_scale=float(infer_raw.get("guidance_scale", InferenceConfig.guidance_scale)),
        image_guidance_scale=float(
            infer_raw.get("image_guidance_scale", InferenceConfig.image_guidance_scale)
        ),
        num_inference_steps=int(
            infer_raw.get("num_inference_steps", InferenceConfig.num_inference_steps)
        ),
        strength=None if infer_raw.get("strength") is None else float(infer_raw["strength"]),
        negative_prompt=str(infer_raw.get("negative_prompt", InferenceConfig.negative_prompt)),
        free_form_mask=bool(infer_raw.get("free_form_mask", InferenceConfig.free_form_mask)),
        max_sequence_length=(
            None
            if infer_raw.get("max_sequence_length") is None
            else int(infer_raw["max_sequence_length"])
        ),
    )
    output = OutputConfig(
        output_dir=Path(str(output_raw.get("output_dir", OutputConfig.output_dir))),
        run_name=output_raw.get("run_name"),
        save_round_images=bool(output_raw.get("save_round_images", OutputConfig.save_round_images)),
    )
    return AppConfig(model=model, data=data, inference=inference, output=output)


def create_run_dir(config: AppConfig) -> Path:
    run_name = config.output.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = config.output.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _is_supported(path: Path, suffixes: set[str]) -> bool:
    return path.suffix.lower() in suffixes


def _load_prompts_from_json(meta_path: Path) -> dict[str, list[str]]:
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Metadata file must contain a JSON object: {meta_path}")

    prompts_by_eval: dict[str, list[str]] = {}
    for key in ("cycle", "long_chain"):
        value = raw.get(key, [])
        if value is None:
            value = []
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"`{key}` must be a list of strings in {meta_path}")
        prompts_by_eval[key] = [item.strip() for item in value]
    return prompts_by_eval


def _load_image_sample(image_path: Path, meta_path: Path | None) -> ImageSample:
    prompts_by_eval: dict[str, list[str]] = {}
    if meta_path is not None:
        prompts_by_eval.update(_load_prompts_from_json(meta_path))
    noop_rounds = max(
        len(prompts_by_eval.get("cycle", [])),
        len(prompts_by_eval.get("long_chain", [])),
        1,
    )
    prompts_by_eval["noop"] = [DEFAULT_NOOP_PROMPT] * noop_rounds

    with Image.open(image_path) as image:
        rgb = image.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
        return ImageSample(
            sample_id=image_path.stem,
            image=rgb,
            image_path=image_path,
            meta_path=meta_path,
            prompts_by_eval=prompts_by_eval,
        )


def _iter_from_image_file(path: Path) -> Iterator[ImageSample]:
    meta_path = path.with_suffix(".json")
    yield _load_image_sample(path, meta_path if meta_path.exists() else None)


def _iter_from_json_file(path: Path) -> Iterator[ImageSample]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Metadata file must contain a JSON object: {path}")

    image_name = raw.get("image")
    if not isinstance(image_name, str) or not image_name.strip():
        raise ValueError(f"`image` must be a non-empty string in {path}")

    image_path = (path.parent / image_name).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image referenced by metadata not found: {image_path}")

    yield _load_image_sample(image_path, path)


def _iter_from_directory(data_cfg: DataConfig) -> Iterator[ImageSample]:
    suffixes = {value.lower() for value in data_cfg.image_suffixes}
    globber = data_cfg.input_path.rglob if data_cfg.recursive else data_cfg.input_path.glob
    files = sorted(p for p in globber("*") if p.is_file() and _is_supported(p, suffixes))
    for image_path in files:
        meta_path = image_path.with_suffix(".json")
        yield _load_image_sample(image_path, meta_path if meta_path.exists() else None)


def iter_samples(data_cfg: DataConfig) -> Iterator[ImageSample]:
    path = data_cfg.input_path
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")

    generated = 0
    if path.is_file():
        if path.suffix.lower() == ".json":
            iterator = _iter_from_json_file(path)
        else:
            suffixes = {value.lower() for value in data_cfg.image_suffixes}
            if not _is_supported(path, suffixes):
                raise ValueError(f"Unsupported input file suffix: {path.suffix}")
            iterator = _iter_from_image_file(path)
    else:
        iterator = _iter_from_directory(data_cfg)

    for sample in iterator:
        if data_cfg.max_samples is not None and generated >= data_cfg.max_samples:
            break
        generated += 1
        yield sample


def resolve_eval_prompts(sample: ImageSample, eval_mode: str) -> list[str]:
    prompts = sample.prompts_by_eval.get(eval_mode, [])
    if not prompts:
        detail = f"Metadata file: {sample.meta_path}" if sample.meta_path is not None else "No metadata file found."
        raise ValueError(
            f"Sample `{sample.sample_id}` does not provide prompts for eval mode `{eval_mode}`. {detail}"
        )
    return list(prompts)


def _load_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return np.asarray(rgb, dtype=np.float32) / 255.0


def _align_to_shape(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    if image.shape[:2] == (height, width):
        return image
    pil = Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8))
    resized = pil.resize((width, height), Image.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _to_gray(image: np.ndarray) -> np.ndarray:
    return 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]


def _ssim_global(a_gray: np.ndarray, b_gray: np.ndarray) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    mu_a = float(np.mean(a_gray))
    mu_b = float(np.mean(b_gray))
    var_a = float(np.var(a_gray))
    var_b = float(np.var(b_gray))
    cov_ab = float(np.mean((a_gray - mu_a) * (b_gray - mu_b)))
    numerator = (2 * mu_a * mu_b + c1) * (2 * cov_ab + c2)
    denominator = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
    if abs(denominator) <= 1e-12:
        return 1.0
    return float(numerator / denominator)


def _build_lpips():
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("Missing dependency for LPIPS. Install with `pip install lpips`.") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = lpips.LPIPS(net="alex")
    model = model.to(device)
    model.eval()
    return model, device


def _to_lpips_tensor(image: np.ndarray, *, device: str):
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=torch.float32)
    return tensor * 2.0 - 1.0


def _lpips_vs_base(*, base_tensor: Any, current: np.ndarray, lpips_model: Any, device: str) -> float:
    current_tensor = _to_lpips_tensor(current, device=device)
    with torch.no_grad():
        score = lpips_model(base_tensor, current_tensor)
    return float(score.detach().mean().cpu().item())


def _group_name(row: dict[str, Any]) -> str:
    group = row.get("group")
    if isinstance(group, str) and group.strip():
        return group.strip()
    image_path = row.get("image_path")
    if image_path:
        return Path(str(image_path)).parent.name or "default"
    return "default"


def _append_metric(store: dict[int, dict[str, list[float]]], round_idx: int, *, l1: float, ssim: float, lpips: float) -> None:
    if round_idx not in store:
        store[round_idx] = {"l1": [], "ssim": [], "lpips": []}
    store[round_idx]["l1"].append(l1)
    store[round_idx]["ssim"].append(ssim)
    store[round_idx]["lpips"].append(lpips)


def _mean_results_from_stats(store: dict[int, dict[str, list[float]]]) -> list[dict[str, float]]:
    mean_results: list[dict[str, float]] = []
    for round_idx in sorted(store.keys()):
        stats = store[round_idx]
        mean_results.append(
            {
                "round": round_idx,
                "l1": float(np.mean(stats["l1"])),
                "ssim": float(np.mean(stats["ssim"])),
                "lpips": float(np.mean(stats["lpips"])),
            }
        )
    return mean_results


def analyze_metrics(
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    eval_mode: str | None = None,
    model_type: str | None = None,
) -> dict[str, Any]:
    if not records:
        return {"error": "No metric records found."}

    output_dir.mkdir(parents=True, exist_ok=True)
    lpips_model, lpips_device = _build_lpips()

    per_sample_results = {}
    aggregated_stats = {}
    grouped_per_sample: dict[str, dict[str, list[dict[str, float]]]] = {}
    grouped_stats: dict[str, dict[int, dict[str, list[float]]]] = {}

    for row in records:
        sample_id = row["sample_id"]
        group = _group_name(row)
        image_paths = [Path(p) for p in row["round_images"]]
        if len(image_paths) < 2:
            continue

        images = [_load_image(path) for path in image_paths]
        base_h, base_w = images[0].shape[:2]
        aligned = [_align_to_shape(img, (base_h, base_w)) for img in images]

        base = aligned[0]
        base_gray = _to_gray(base)
        base_lpips_tensor = _to_lpips_tensor(base, device=lpips_device)

        sample_metrics = []
        for idx in range(1, len(aligned)):
            current = aligned[idx]
            current_gray = _to_gray(current)
            l1 = float(np.mean(np.abs(current - base)))
            ssim = _ssim_global(base_gray, current_gray)
            lpips_val = _lpips_vs_base(
                base_tensor=base_lpips_tensor,
                current=current,
                lpips_model=lpips_model,
                device=lpips_device,
            )
            sample_metrics.append({"round": idx, "l1": l1, "ssim": ssim, "lpips": lpips_val})
            _append_metric(aggregated_stats, idx, l1=l1, ssim=ssim, lpips=lpips_val)
            _append_metric(grouped_stats.setdefault(group, {}), idx, l1=l1, ssim=ssim, lpips=lpips_val)

        per_sample_results[sample_id] = sample_metrics
        grouped_per_sample.setdefault(group, {})[sample_id] = sample_metrics

    mean_results = _mean_results_from_stats(aggregated_stats)
    group_summaries = {}
    for group_name in sorted(grouped_stats.keys()):
        group_mean_results = _mean_results_from_stats(grouped_stats[group_name])
        group_summaries[group_name] = {
            "num_samples": len(grouped_per_sample.get(group_name, {})),
            "num_rounds": len(group_mean_results),
            "mean_metrics": group_mean_results,
            "per_sample": grouped_per_sample.get(group_name, {}),
            "final_mean_metrics": group_mean_results[-1] if group_mean_results else None,
        }

    csv_path = output_dir / "metrics_by_group_and_round.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "round", "l1", "ssim", "lpips"])
        for item in mean_results:
            writer.writerow(["overall", item["round"], item["l1"], item["ssim"], item["lpips"]])
        for group_name, payload in group_summaries.items():
            for item in payload["mean_metrics"]:
                writer.writerow([group_name, item["round"], item["l1"], item["ssim"], item["lpips"]])

    summary = {
        "evaluation": "metrics",
        "eval_mode": eval_mode,
        "model_type": model_type,
        "num_samples": len(per_sample_results),
        "num_rounds": len(mean_results),
        "mean_metrics": mean_results,
        "overall": {
            "num_samples": len(per_sample_results),
            "num_rounds": len(mean_results),
            "mean_metrics": mean_results,
            "per_sample": per_sample_results,
            "final_mean_metrics": mean_results[-1] if mean_results else None,
        },
        "by_group": group_summaries,
        "per_sample": per_sample_results,
    }
    summary_path = output_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_path = output_dir / "metrics_mean_l1_ssim_lpips.png"
    try:
        import matplotlib.pyplot as plt

        rounds = [r["round"] for r in mean_results]
        l1_vals = [r["l1"] for r in mean_results]
        ssim_vals = [r["ssim"] for r in mean_results]
        lpips_vals = [r["lpips"] for r in mean_results]

        fig = plt.figure(figsize=(8.2, 4.8))
        plt.plot(rounds, l1_vals, marker="o", linewidth=1.6, label="Mean L1 Drift")
        plt.plot(rounds, ssim_vals, marker="o", linewidth=1.6, label="Mean SSIM")
        plt.plot(rounds, lpips_vals, marker="o", linewidth=1.6, label="Mean LPIPS")
        plt.xlabel("Round")
        plt.ylabel("Metric Value")
        y_max = max(1.0, max(l1_vals + ssim_vals + lpips_vals) * 1.05) if mean_results else 1.0
        plt.ylim(0.0, y_max)
        plt.title("Mean L1 Drift, SSIM & LPIPS Across All Samples")
        plt.grid(alpha=0.3)
        plt.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
    except ImportError:
        pass

    return {
        "summary_path": str(summary_path),
        "group_round_csv_path": str(csv_path),
        "plot_path": str(plot_path),
        "final_mean_metrics": mean_results[-1] if mean_results else None,
        "final_mean_metrics_by_group": {
            group_name: payload["final_mean_metrics"] for group_name, payload in group_summaries.items()
        },
    }


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


class UltraEditRunner:
    def __init__(
        self,
        config: AppConfig,
        use_lpt: bool = False,
        augmentation: str = "momentum",
        lowpass_filter: str = "avg_pool",
        lowpass_sigma: float = 0.25,
    ):
        self.config = config
        self.use_lpt = use_lpt
        self.augmentation = augmentation
        self.lowpass_filter = str(lowpass_filter).strip().lower()
        self.lowpass_sigma = max(1e-6, float(lowpass_sigma))
        self.momentum_lowpass_kernel = 9
        self.momentum_mean_decay = 0.95
        self.momentum_log_std_decay = 0.85
        self._momentum_mean: torch.Tensor | None = None
        self._momentum_log_std: torch.Tensor | None = None

        if self.use_lpt and self.augmentation != "momentum":
            raise ValueError("UltraEdit latent mode currently supports only --augmentation momentum.")
        if not self.config.inference.free_form_mask:
            raise ValueError("UltraEdit script currently supports only free-form editing with free_form_mask: true.")

        self.dtype = resolve_torch_dtype(config.model.torch_dtype)
        self.device = config.model.device
        self.pipe = self._load_pipeline(config)

    def _load_pipeline(self, config: AppConfig) -> Any:
        try:
            from diffusers import StableDiffusion3InstructPix2PixPipeline
        except ImportError as exc:
            raise ImportError(
                "StableDiffusion3InstructPix2PixPipeline is unavailable. "
                "Please install the UltraEdit forked diffusers in this separate environment."
            ) from exc

        kwargs: dict[str, Any] = {
            "torch_dtype": self.dtype,
            "local_files_only": config.model.local_files_only,
            "low_cpu_mem_usage": config.model.low_cpu_mem_usage,
        }
        if config.model.revision:
            kwargs["revision"] = config.model.revision
        if config.model.variant:
            kwargs["variant"] = config.model.variant

        pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained(config.model.model_path, **kwargs)
        if config.model.enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(config.model.device)
        return pipe

    def _build_generator(self, seed: int) -> torch.Generator:
        gen_device = self.device
        if gen_device.startswith("cuda") and not torch.cuda.is_available():
            gen_device = "cpu"
        generator = torch.Generator(device=gen_device)
        generator.manual_seed(seed)
        return generator

    def _reset_momentum_stats(self) -> None:
        self._momentum_mean = None
        self._momentum_log_std = None

    @staticmethod
    def _round_to_multiple(value: int, multiple: int = 16) -> int:
        value = max(int(value), multiple)
        return max(multiple, (value // multiple) * multiple)

    def _resolve_target_size(self, image: Image.Image) -> tuple[int, int]:
        cfg_height = self.config.inference.height
        cfg_width = self.config.inference.width
        if cfg_height is not None and cfg_width is not None:
            return int(cfg_height), int(cfg_width)

        width, height = map(float, image.size)
        if cfg_width is not None:
            scale = float(cfg_width) / max(width, 1.0)
            width = float(cfg_width)
            height = height * scale
        elif cfg_height is not None:
            scale = float(cfg_height) / max(height, 1.0)
            height = float(cfg_height)
            width = width * scale
        else:
            max_side = 1280.0
            long_edge = max(width, height)
            if long_edge > max_side:
                scale = max_side / long_edge
                width = width * scale
                height = height * scale

        resolved_height = self._round_to_multiple(int(round(height)))
        resolved_width = self._round_to_multiple(int(round(width)))
        return resolved_height, resolved_width

    @staticmethod
    def _lowpass_latent_with_kernel(x: torch.Tensor, kernel: int) -> torch.Tensor:
        k = int(kernel)
        if k <= 1:
            return x
        if k % 2 == 0:
            k += 1
        pad = k // 2
        x_pad = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="replicate")
        return torch.nn.functional.avg_pool2d(x_pad, kernel_size=k, stride=1)

    @staticmethod
    def _fft_gaussian_lowpass_latent(x: torch.Tensor, sigma: float) -> torch.Tensor:
        _, _, h, w = x.shape
        if h < 2 or w < 2:
            return x
        sigma = max(1e-6, float(sigma))
        x_f = x.float()
        fft = torch.fft.fftshift(torch.fft.fft2(x_f, dim=(-2, -1)), dim=(-2, -1))
        y, x_grid = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=x.device, dtype=torch.float32),
            torch.linspace(-1.0, 1.0, w, device=x.device, dtype=torch.float32),
            indexing="ij",
        )
        radius = torch.sqrt(x_grid.square() + y.square())
        mask = torch.exp(-0.5 * (radius / sigma).square()).view(1, 1, h, w)
        low_fft = fft * mask
        low = torch.fft.ifft2(torch.fft.ifftshift(low_fft, dim=(-2, -1)), dim=(-2, -1)).real
        return low.to(dtype=x.dtype)

    def _lowpass_latent(self, x: torch.Tensor) -> torch.Tensor:
        if self.lowpass_filter == "fft_gaussian":
            return self._fft_gaussian_lowpass_latent(x, self.lowpass_sigma)
        return self._lowpass_latent_with_kernel(x, self.momentum_lowpass_kernel)

    @staticmethod
    def _channel_moments(latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        work = latents.detach().float()
        mean = work.mean(dim=(2, 3), keepdim=True)
        std = work.std(dim=(2, 3), unbiased=False, keepdim=True).clamp_min(1e-6)
        return mean, std

    def _low_frequency_moments(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        low_latents = self._lowpass_latent(latents.float())
        mean, std = self._channel_moments(low_latents)
        return low_latents, mean, std

    def _update_momentum_moments(self, mean: torch.Tensor, log_std: torch.Tensor) -> None:
        mean = mean.detach().float()
        log_std = log_std.detach().float()
        if self._momentum_mean is None or self._momentum_log_std is None:
            self._momentum_mean = mean
            self._momentum_log_std = log_std
            return
        if self._momentum_mean.shape != mean.shape:
            self._momentum_mean = mean
            self._momentum_log_std = log_std
            return

        mean_decay = max(0.0, min(0.999, float(self.momentum_mean_decay)))
        std_decay = max(0.0, min(0.999, float(self.momentum_log_std_decay)))
        self._momentum_mean = mean_decay * self._momentum_mean + (1.0 - mean_decay) * mean
        self._momentum_log_std = std_decay * self._momentum_log_std + (1.0 - std_decay) * log_std

    def _align_low_frequency_to_moments(
        self,
        after: torch.Tensor,
        *,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        strength: float,
    ) -> torch.Tensor:
        gate = max(0.0, min(1.0, float(strength)))
        if gate <= 0.0:
            return after
        after_f = after.float()
        low_after, mean_a, std_a = self._low_frequency_moments(after_f)
        high_after = after_f - low_after
        low_aligned = (low_after - mean_a) / (std_a + 1e-6)
        low_aligned = low_aligned * (target_std.to(after.device, torch.float32) + 1e-6)
        low_aligned = low_aligned + target_mean.to(after.device, torch.float32)
        low_out = torch.lerp(low_after, low_aligned, gate)
        out = low_out + high_after
        return torch.nan_to_num(out, nan=0.0, posinf=1e3, neginf=-1e3).to(after.dtype)

    def _restore_moments_momentum(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        _, mean_a, std_a = self._low_frequency_moments(after)
        log_std_a = std_a.log()
        if (
            self._momentum_mean is None
            or self._momentum_log_std is None
            or self._momentum_mean.shape != mean_a.shape
        ):
            _, mean_b, std_b = self._low_frequency_moments(before)
            self._momentum_mean = mean_b.detach().float()
            self._momentum_log_std = std_b.log().detach().float()

        mean_t = self._momentum_mean.to(device=after.device, dtype=torch.float32)
        std_t = self._momentum_log_std.to(device=after.device, dtype=torch.float32).exp()
        out = self._align_low_frequency_to_moments(after, target_mean=mean_t, target_std=std_t, strength=1.0)
        self._update_momentum_moments(mean_a, log_std_a)
        return out

    def _decode_latents(self, latents: torch.Tensor) -> Image.Image:
        with torch.no_grad():
            scale = getattr(self.pipe.vae.config, "scaling_factor", 1.0)
            decode_input = latents / scale
            dec_tensor = self.pipe.vae.decode(decode_input, return_dict=False)[0]
        tensor = dec_tensor.clamp(0, 1)
        tensor = tensor.detach().cpu().permute(0, 2, 3, 1).float().numpy()[0]
        return Image.fromarray((tensor * 255).astype(np.uint8))

    def _blank_mask(self, size: tuple[int, int]) -> Image.Image:
        return Image.new("RGB", size, (255, 255, 255))

    @contextlib.contextmanager
    def _encode_context(self, current_latents: torch.Tensor | None):
        orig_encode = self.pipe.vae.encode

        class MockVAEOutput:
            def __init__(self, latents: torch.Tensor):
                self.latents = latents

            @property
            def latent_dist(self):
                latents = self.latents

                class Dist:
                    def sample(self_dist, *args, **kwargs):
                        return latents

                    def mode(self_dist, *args, **kwargs):
                        return latents

                return Dist()

        def patched_encode(*args, **kwargs):
            if current_latents is not None:
                return MockVAEOutput(current_latents)
            return orig_encode(*args, **kwargs)

        self.pipe.vae.encode = patched_encode
        try:
            yield
        finally:
            self.pipe.vae.encode = orig_encode

    def _run_round(
        self,
        *,
        prompt: str,
        current_image_pil: Image.Image,
        current_latents: torch.Tensor | None,
        seed: int,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, Image.Image]:
        resized_image = current_image_pil.resize((width, height), resample=Image.Resampling.LANCZOS)
        mask_image = self._blank_mask((width, height))

        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "image": resized_image,
            "mask_img": mask_image,
            "negative_prompt": self.config.inference.negative_prompt,
            "height": height,
            "width": width,
            "num_inference_steps": self.config.inference.num_inference_steps,
            "guidance_scale": self.config.inference.guidance_scale,
            "image_guidance_scale": self.config.inference.image_guidance_scale,
            "generator": self._build_generator(seed),
            "output_type": "latent",
        }

        with torch.inference_mode():
            with self._encode_context(current_latents if (self.use_lpt and current_latents is not None) else None):
                output = self.pipe(**kwargs)

        latents_out = getattr(output, "images", None)
        if latents_out is None:
            latents_out = output[0] if isinstance(output, tuple) else output
        if isinstance(latents_out, (list, tuple)):
            if len(latents_out) == 0:
                raise RuntimeError("UltraEdit pipeline output is empty.")
            denoised_latents = latents_out[0]
        else:
            denoised_latents = latents_out
        denoised_latents = denoised_latents.unsqueeze(0) if denoised_latents.ndim == 3 else denoised_latents
        accepted_latents = denoised_latents

        if self.use_lpt and current_latents is not None:
            accepted_latents = self._restore_moments_momentum(before=current_latents, after=accepted_latents)

        vis_image = self._decode_latents(accepted_latents)
        return accepted_latents, vis_image

    def run_sample(
        self,
        *,
        sample_id: str,
        image: Image.Image,
        prompts: list[str],
        run_dir: Path,
    ) -> RunTrace:
        sample_dir = run_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        self._reset_momentum_stats()

        round_images: list[str] = []
        current_image_pil = image.convert("RGB")
        current_latents = None

        input_path = sample_dir / "round_000_input.png"
        current_image_pil.save(input_path)
        round_images.append(str(input_path))

        height, width = self._resolve_target_size(image)
        for round_idx, prompt in enumerate(prompts, start=1):
            seed = self.config.inference.seed + (round_idx - 1) * self.config.inference.seed_stride
            accepted_latents, vis_image = self._run_round(
                prompt=prompt,
                current_image_pil=current_image_pil,
                current_latents=current_latents,
                seed=seed,
                height=height,
                width=width,
            )
            if self.config.output.save_round_images:
                out_path = sample_dir / f"round_{round_idx:03d}.png"
                vis_image.save(out_path)
                round_images.append(str(out_path))
            current_image_pil = vis_image
            current_latents = accepted_latents if self.use_lpt else None

        return RunTrace(sample_id=sample_id, round_images=round_images)

    def close(self) -> None:
        pass


def main() -> None:
    args = _arg_parser().parse_args()
    model_type = _normalize_model_type(args.model_type)
    config = load_config(args.config)

    if args.max_samples is not None:
        config.data.max_samples = int(args.max_samples)
    if args.input_dir is not None:
        config.data.input_path = args.input_dir
    if args.run_name:
        config.output.run_name = args.run_name
    if args.strength is not None:
        config.inference.strength = float(args.strength)

    config.output.save_round_images = True
    run_dir = create_run_dir(config)
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    use_lpt = args.latent.strip().lower() in {"true", "1", "yes", "t"}
    augmentation = args.augmentation.strip().lower()
    lowpass_filter = args.lowpass_filter.strip().lower()

    runner = UltraEditRunner(
        config=config,
        use_lpt=use_lpt,
        augmentation=augmentation,
        lowpass_filter=lowpass_filter,
        lowpass_sigma=args.lowpass_sigma,
    )

    config_snapshot = {
        "resolved_config": _to_jsonable(config),
        "model_type": model_type,
        "model_backend": "sd3_ultraedit",
        "evaluation": "metrics",
        "eval_mode": args.eval,
        "ablation_latent_pass_through": use_lpt,
        "ablation_augmentation": augmentation,
        "ablation_lowpass_filter": lowpass_filter,
        "ablation_lowpass_sigma": args.lowpass_sigma,
        "note": "inference.strength is accepted for CLI compatibility but ignored by UltraEdit.",
    }
    (run_dir / "config_snapshot.json").write_text(
        json.dumps(config_snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    records: list[dict[str, Any]] = []
    sample_count = 0
    sample_name_counts: dict[str, int] = {}

    try:
        for sample in iter_samples(config.data):
            prompts = resolve_eval_prompts(sample, args.eval)
            group_name = sample.image_path.parent.name or "default"
            base_id = sample.sample_id.strip() or "sample"
            composite_base_id = f"{group_name}_{base_id}"
            seen = sample_name_counts.get(composite_base_id, 0)
            sample_name_counts[composite_base_id] = seen + 1
            sample_id = composite_base_id if seen == 0 else f"{composite_base_id}_{seen:03d}"

            trace = runner.run_sample(
                sample_id=sample_id,
                image=sample.image,
                prompts=prompts,
                run_dir=samples_dir,
            )
            sample_count += 1

            row = {
                "sample_id": trace.sample_id,
                "group": group_name,
                "eval_mode": args.eval,
                "image_path": str(sample.image_path),
                "meta_path": None if sample.meta_path is None else str(sample.meta_path),
                "prompts": prompts,
                "round_images": trace.round_images,
            }
            records.append(row)

            sample_trace_path = samples_dir / trace.sample_id / "metrics_trace.json"
            sample_trace_path.write_text(
                json.dumps(row, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    finally:
        runner.close()

    metrics_output = analyze_metrics(
        records,
        run_dir / "metrics",
        eval_mode=args.eval,
        model_type=model_type,
    )

    manifest = {
        "run_dir": str(run_dir),
        "sample_count": sample_count,
        "evaluation": "metrics",
        "eval_mode": args.eval,
        "model_type": model_type,
        "model_backend": "sd3_ultraedit",
        "outputs": {"metrics": metrics_output},
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[run] run_dir={run_dir}")
    print(f"[run] sample_count={sample_count}")
    print(f"[metrics] {metrics_output}")
    print(f"[manifest] {manifest_path}")


if __name__ == "__main__":
    main()
