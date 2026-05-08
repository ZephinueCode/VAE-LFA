from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]


@dataclass
class ModelConfig:
    model_path: str = "black-forest-labs/FLUX.2-klein-9B"
    torch_dtype: str = "bfloat16"
    device: str = "cuda"
    local_files_only: bool = False
    enable_model_cpu_offload: bool = True
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
    guidance_scale: float = 1.0
    num_inference_steps: int = 4
    strength: float | None = None
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


def _load_raw(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
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
        num_inference_steps=int(
            infer_raw.get("num_inference_steps", InferenceConfig.num_inference_steps)
        ),
        strength=None if infer_raw.get("strength") is None else float(infer_raw["strength"]),
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

    return AppConfig(
        model=model,
        data=data,
        inference=inference,
        output=output,
    )


def create_run_dir(config: AppConfig) -> Path:
    from datetime import datetime

    run_name = config.output.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = config.output.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
