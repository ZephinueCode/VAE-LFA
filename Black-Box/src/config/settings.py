from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class APISettings:
    base_url: str
    api_key_env: str
    model: str
    timeout_seconds: float
    size: str | None
    quality: str | None
    dashscope_endpoint: str
    dashscope_parameters: dict[str, Any]
    strength: float = 0.15

    @property
    def api_key(self) -> str:
        value = self.api_key_env.strip()
        if not value:
            raise ValueError("`api.api_key_env` is empty.")
        return value


@dataclass
class RunSettings:
    rounds: int
    input_image: Path | None
    input_dir: Path | None
    results_dir: Path
    run_name: str | None


@dataclass
class PromptSettings:
    per_round: list[str]


@dataclass
class InterventionSettings:
    enabled: bool = False
    kernel_size: int = 9
    mean_decay: float = 0.85
    std_decay: float = 0.85


@dataclass
class VAESettings:
    enabled: bool
    model_type: str
    model_path: Path
    device: str
    dtype: str


@dataclass
class AppSettings:
    api: APISettings
    run: RunSettings
    prompts: PromptSettings
    vae: VAESettings
    intervention: InterventionSettings


def _require_dict(value: Any, section_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"`{section_name}` must be an object.")
    return value


def load_settings(path: Path) -> AppSettings:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a JSON object.")

    api_raw = _require_dict(raw.get("api"), "api")
    run_raw = _require_dict(raw.get("run"), "run")
    prompts_raw = _require_dict(raw.get("prompts"), "prompts")
    vae_raw = _require_dict(raw.get("vae"), "vae")
    
    # 提取新的顶层 intervention 配置
    intervention_raw = raw.get("intervention", {})
    if not isinstance(intervention_raw, dict):
        raise ValueError("`intervention` must be an object.")

    dashscope_parameters_raw = api_raw.get("dashscope_parameters", {})
    if not isinstance(dashscope_parameters_raw, dict):
        raise ValueError("`api.dashscope_parameters` must be an object.")

    api = APISettings(
        base_url=str(api_raw["base_url"]),
        api_key_env=str(api_raw["api_key_env"]),
        model=str(api_raw["model"]),
        timeout_seconds=float(api_raw.get("timeout_seconds", 180.0)),
        size=api_raw.get("size"),
        quality=api_raw.get("quality"),
        dashscope_endpoint=str(
            api_raw.get("dashscope_endpoint", "/services/aigc/multimodal-generation/generation")
        ),
        dashscope_parameters=dict(dashscope_parameters_raw),
    )
    run = RunSettings(
        rounds=int(run_raw["rounds"]),
        input_image=Path(str(run_raw.get("input_image"))) if run_raw.get("input_image") else None,
        input_dir=Path(str(run_raw.get("input_dir"))) if run_raw.get("input_dir") else None,
        results_dir=Path(str(run_raw.get("results_dir", "results"))),
        run_name=run_raw.get("run_name"),
    )
    prompts = PromptSettings(
        per_round=[str(item) for item in prompts_raw["per_round"]],
    )

    # 极简化的 Intervention 加载
    intervention = InterventionSettings(
        enabled=bool(intervention_raw.get("enabled", False)),
        kernel_size=int(intervention_raw.get("kernel_size", 9)),
        mean_decay=float(intervention_raw.get("mean_decay", 0.85)),
        std_decay=float(intervention_raw.get("std_decay", 0.85))
    )

    # 极简化的 VAE 加载
    vae = VAESettings(
        enabled=bool(vae_raw.get("enabled", True)),
        model_type=str(vae_raw.get("model_type", "autoencoder_KL")),
        model_path=Path(str(vae_raw.get("model_path", "models/sd_vae"))),
        device=str(vae_raw.get("device", "cuda")),
        dtype=str(vae_raw.get("dtype", "float16"))
    )

    if run.rounds <= 0:
        raise ValueError("`run.rounds` must be > 0.")
    if len(prompts.per_round) != run.rounds:
        raise ValueError("`prompts.per_round` length must be exactly equal to `run.rounds`.")

    return AppSettings(api=api, run=run, prompts=prompts, vae=vae, intervention=intervention)