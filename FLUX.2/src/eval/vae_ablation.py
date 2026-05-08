from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.metrics import analyze_metrics
from src.model.model import FluxKleinRunner, RunTrace
from src.utils.config import AppConfig, load_config
from src.utils.data import iter_samples, resolve_eval_prompts


def unique_sample_name(sample, seen: dict[str, int]) -> str:
    group_name = sample.image_path.parent.name or "default"
    base_id = sample.sample_id.strip() or "sample"
    composite = f"{group_name}_{base_id}"
    count = seen.get(composite, 0)
    seen[composite] = count + 1
    return composite if count == 0 else f"{composite}_{count:03d}"


def create_ablation_dir(base_dir: Path, outdir: Path | None) -> Path:
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir
    run_name = f"vae_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    path = base_dir / run_name
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class VariantSpec:
    family: str
    key: str
    label: str
    align_scope: str
    anchor_strategy: str


FREQ_VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec("freq", "base", "FLUX.2_dit_only_base", "none", "prev"),
    VariantSpec("freq", "both", "both", "both", "ema"),
    VariantSpec("freq", "high_only", "high_only", "high", "ema"),
    VariantSpec("freq", "low_only", "low_only", "low", "ema"),
)

EMA_VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec("ema", "base", "FLUX.2_dit_only_base", "none", "prev"),
    VariantSpec("ema", "fixed", "fixed", "low", "fixed"),
    VariantSpec("ema", "prev", "prev", "low", "prev"),
    VariantSpec("ema", "ema", "ema", "low", "ema"),
)


class AblationFluxRunner(FluxKleinRunner):
    def __init__(
        self,
        config: AppConfig,
        *,
        align_scope: str,
        anchor_strategy: str,
        lowpass_filter: str,
        lowpass_sigma: float,
    ):
        super().__init__(
            config=config,
            use_lpt=True,
            augmentation="none",
            lowpass_filter=lowpass_filter,
            lowpass_sigma=lowpass_sigma,
        )
        self.align_scope = align_scope
        self.anchor_strategy = anchor_strategy

    def _select_anchor_latents(self, current_latents: torch.Tensor) -> torch.Tensor:
        if self.anchor_strategy == "prev":
            return current_latents

        state = self._anchor_state
        if self.anchor_strategy == "fixed" and state.init_latents is not None:
            return state.init_latents.to(device=current_latents.device, dtype=current_latents.dtype)
        if self.anchor_strategy == "ema" and state.ema_latents is not None:
            return state.ema_latents.to(device=current_latents.device, dtype=current_latents.dtype)
        return current_latents

    def _align_low_component(self, anchor: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        low_after = self._lowpass_latent(after.float())
        low_anchor = self._lowpass_latent(anchor.float())
        mean_t = low_anchor.mean(dim=(2, 3), keepdim=True)
        std_t = low_anchor.std(dim=(2, 3), unbiased=False, keepdim=True).clamp_min(1e-6)
        mean_a = low_after.mean(dim=(2, 3), keepdim=True)
        std_a = low_after.std(dim=(2, 3), unbiased=False, keepdim=True).clamp_min(1e-6)
        low_aligned = (low_after - mean_a) / (std_a + 1e-6)
        low_aligned = low_aligned * (std_t + 1e-6) + mean_t
        return torch.nan_to_num(low_aligned, nan=0.0, posinf=1e3, neginf=-1e3)

    def _align_high_component(self, anchor: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        low_after = self._lowpass_latent(after.float())
        high_after = after.float() - low_after
        low_anchor = self._lowpass_latent(anchor.float())
        high_anchor = anchor.float() - low_anchor
        return self._align_high_frequency_energy(
            anchor_high=high_anchor,
            after_high=high_after,
            strength=1.0,
        ).float()

    def _apply_ablation_alignment(self, anchor: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        if self.align_scope == "none":
            return after

        after_f = after.float()
        low_after = self._lowpass_latent(after_f)
        high_after = after_f - low_after

        out_low = low_after
        out_high = high_after
        if self.align_scope in {"low", "both"}:
            out_low = self._align_low_component(anchor, after)
        if self.align_scope in {"high", "both"}:
            out_high = self._align_high_component(anchor, after)

        out = out_low + out_high
        return torch.nan_to_num(out, nan=0.0, posinf=1e3, neginf=-1e3).to(dtype=after.dtype)

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
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "guidance_scale": self.config.inference.guidance_scale,
            "num_inference_steps": self.config.inference.num_inference_steps,
            "generator": self._build_generator(seed),
        }
        if self.config.inference.max_sequence_length is not None and "max_sequence_length" in self._call_signature.parameters:
            kwargs["max_sequence_length"] = self.config.inference.max_sequence_length
        if self.config.inference.strength is not None and "strength" in self._call_signature.parameters:
            kwargs["strength"] = self.config.inference.strength

        if self._supports_image_key is None:
            raise RuntimeError("Current Flux2KleinPipeline call signature does not support image-to-image input.")

        kwargs[self._supports_image_key] = current_image_pil
        kwargs["output_type"] = "latent"

        with torch.inference_mode():
            with self._encode_context(current_latents):
                output = self.pipe(**kwargs)

        latents_out = getattr(output, "images", None)
        if latents_out is None:
            latents_out = output[0] if isinstance(output, tuple) else output
        if isinstance(latents_out, (list, tuple)):
            if not latents_out:
                raise RuntimeError("Pipeline output is empty.")
            denoised_latents = latents_out[0]
        else:
            denoised_latents = latents_out

        denoised_latents = denoised_latents.unsqueeze(0) if denoised_latents.ndim == 3 else denoised_latents
        pure_latents = denoised_latents

        if current_latents is not None and self.align_scope != "none":
            anchor = self._select_anchor_latents(current_latents)
            pure_latents = self._apply_ablation_alignment(anchor=anchor, after=pure_latents)

        with torch.no_grad():
            shift = getattr(self.pipe.vae.config, "shift_factor", 0.0)
            scale = getattr(self.pipe.vae.config, "scaling_factor", 1.0)
            unscaled = (pure_latents / scale) + shift
            dec_tensor = self.pipe.vae.decode(unscaled).sample

        tensor = (dec_tensor / 2 + 0.5).clamp(0, 1)
        tensor = tensor.detach().cpu().permute(0, 2, 3, 1).float().numpy()[0]
        vis_image = Image.fromarray((tensor * 255).astype("uint8"))

        return pure_latents, vis_image


def collect_samples(config: AppConfig, input_dir: Path, max_samples: int | None) -> list[Any]:
    original_input = config.data.input_path
    original_max = config.data.max_samples
    try:
        config.data.input_path = input_dir
        config.data.max_samples = max_samples
        return list(iter_samples(config.data))
    finally:
        config.data.input_path = original_input
        config.data.max_samples = original_max


def run_variant(
    config: AppConfig,
    *,
    samples: list[Any],
    eval_mode: str,
    variant: VariantSpec,
    output_dir: Path,
    lowpass_filter: str,
    lowpass_sigma: float,
) -> dict[str, Any]:
    variant_dir = output_dir / eval_mode / variant.family / variant.key
    samples_dir = variant_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    runner = AblationFluxRunner(
        config=config,
        align_scope=variant.align_scope,
        anchor_strategy=variant.anchor_strategy,
        lowpass_filter=lowpass_filter,
        lowpass_sigma=lowpass_sigma,
    )

    seen_names: dict[str, int] = {}
    metric_records: list[dict[str, Any]] = []
    try:
        for idx, sample in enumerate(samples, start=1):
            sample_name = unique_sample_name(sample, seen_names)
            prompts = resolve_eval_prompts(sample, eval_mode)
            print(f"[{eval_mode}][{variant.key}] {idx}/{len(samples)} -> {sample_name}")
            trace: RunTrace = runner.run_sample(
                sample_id=sample_name,
                image=sample.image,
                prompts=prompts,
                run_dir=samples_dir,
            )
            metric_records.append(
                {
                    "sample_id": sample_name,
                    "group": sample.image_path.parent.name or "default",
                    "image_path": str(sample.image_path),
                    "round_images": trace.round_images,
                }
            )
    finally:
        runner.close()

    metrics_dir = variant_dir / "metrics"
    metrics_info = analyze_metrics(
        metric_records,
        metrics_dir,
        eval_mode=eval_mode,
        model_type="flux2_ablation",
    )

    return {
        "variant": variant.key,
        "label": variant.label,
        "family": variant.family,
        "align_scope": variant.align_scope,
        "anchor_strategy": variant.anchor_strategy,
        "metrics": metrics_info,
        "metrics_summary_path": metrics_info["summary_path"],
    }


def load_metrics_summary(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_round_metrics(summary: dict[str, Any], round_idx: int) -> dict[str, dict[str, float | None]]:
    by_group = summary.get("by_group", {})
    output: dict[str, dict[str, float | None]] = {}
    for group_name, payload in by_group.items():
        found = None
        for row in payload.get("mean_metrics", []):
            if int(row.get("round", -1)) == int(round_idx):
                found = row
                break
        output[group_name] = {
            "lpips": None if found is None else float(found["lpips"]),
            "l1": None if found is None else float(found["l1"]),
            "ssim": None if found is None else float(found["ssim"]),
        }
    return output


def build_family_summary(
    *,
    family: str,
    eval_modes: list[str],
    variants: list[VariantSpec],
    variant_runs: dict[tuple[str, str, str], dict[str, Any]],
    final_round: int,
    output_dir: Path,
) -> None:
    family_summary: dict[str, Any] = {
        "family": family,
        "final_round": int(final_round),
        "eval_modes": {},
    }
    csv_lines = ["eval_mode,variant,group,lpips,l1,ssim"]

    for eval_mode in eval_modes:
        family_summary["eval_modes"][eval_mode] = {}
        for variant in variants:
            run_payload = variant_runs[(family, eval_mode, variant.key)]
            metrics_summary = load_metrics_summary(run_payload["metrics_summary_path"])
            round_metrics = extract_round_metrics(metrics_summary, final_round)
            family_summary["eval_modes"][eval_mode][variant.key] = {
                "label": variant.label,
                "align_scope": variant.align_scope,
                "anchor_strategy": variant.anchor_strategy,
                "round_metrics_by_group": round_metrics,
                "metrics_summary_path": run_payload["metrics_summary_path"],
            }
            for group_name, metric_row in round_metrics.items():
                csv_lines.append(
                    ",".join(
                        [
                            eval_mode,
                            variant.key,
                            group_name,
                            "" if metric_row["lpips"] is None else f"{metric_row['lpips']:.6f}",
                            "" if metric_row["l1"] is None else f"{metric_row['l1']:.6f}",
                            "" if metric_row["ssim"] is None else f"{metric_row['ssim']:.6f}",
                        ]
                    )
                )

    (output_dir / f"{family}_summary_round{final_round}.json").write_text(
        json.dumps(family_summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / f"{family}_summary_round{final_round}.csv").write_text(
        "\n".join(csv_lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset-level LPIPS/SSIM/L1 ablations for FLUX.2 DiT-only editing.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the main YAML/JSON config.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Optional dataset root override.")
    parser.add_argument("--outdir", type=Path, default=None, help="Optional output directory.")
    parser.add_argument(
        "--eval-modes",
        nargs="+",
        choices=["noop", "cycle", "long_chain"],
        default=["noop", "cycle"],
        help="Editing protocols to run. Default: noop cycle",
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        choices=["freq", "ema"],
        default=["freq", "ema"],
        help="Which ablation families to run. Default: freq ema",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional dataset cap for smoke tests.")
    parser.add_argument("--final-round", type=int, default=10, help="Round index to export into summary tables.")
    parser.add_argument("--cpu-offload", action="store_true", help="Enable model CPU offload for this ablation run.")
    parser.add_argument("--lowpass-filter", choices=["avg_pool", "fft_gaussian"], default="avg_pool")
    parser.add_argument("--lowpass-sigma", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.input_dir is not None:
        config.data.input_path = args.input_dir
    if args.max_samples is not None:
        config.data.max_samples = args.max_samples
    if args.cpu_offload:
        config.model.enable_model_cpu_offload = True
    config.output.save_round_images = True

    output_dir = create_ablation_dir(config.output.output_dir, args.outdir)
    samples = collect_samples(config, config.data.input_path, config.data.max_samples)
    if not samples:
        raise ValueError(f"No samples found under: {config.data.input_path}")

    family_to_variants: dict[str, list[VariantSpec]] = {}
    if "freq" in args.ablations:
        family_to_variants["freq"] = list(FREQ_VARIANTS)
    if "ema" in args.ablations:
        family_to_variants["ema"] = list(EMA_VARIANTS)

    manifest: dict[str, Any] = {
        "config_path": str(args.config),
        "input_dir": str(config.data.input_path),
        "output_dir": str(output_dir),
        "num_samples": len(samples),
        "eval_modes": list(args.eval_modes),
        "ablations": list(args.ablations),
        "final_round": int(args.final_round),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    variant_runs: dict[tuple[str, str, str], dict[str, Any]] = {}
    for family, variants in family_to_variants.items():
        print(f"\n=== Running ablation family: {family} ===")
        for eval_mode in args.eval_modes:
            for variant in variants:
                variant_runs[(family, eval_mode, variant.key)] = run_variant(
                    config,
                    samples=samples,
                    eval_mode=eval_mode,
                    variant=variant,
                    output_dir=output_dir,
                    lowpass_filter=args.lowpass_filter,
                    lowpass_sigma=args.lowpass_sigma,
                )

        build_family_summary(
            family=family,
            eval_modes=list(args.eval_modes),
            variants=variants,
            variant_runs=variant_runs,
            final_round=args.final_round,
            output_dir=output_dir,
        )

    print(f"\nSaved ablation outputs to {output_dir}")


if __name__ == "__main__":
    main()
