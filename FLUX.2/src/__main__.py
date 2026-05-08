from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from src.eval.metrics import analyze_metrics
from src.utils.config import AppConfig, create_run_dir, load_config
from src.utils.data import iter_samples, resolve_eval_prompts


def _normalize_model_type(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized == "flux2":
        return "flux2"
    if normalized in {"kandinsky5", "kandinsky", "k5"}:
        return "kandinsky5"
    raise ValueError(f"Unsupported model type: {value}")


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src", description="ReGen metrics evaluation.")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON/YAML config.")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["flux2", "kandinsky5", "kandinsky", "k5"],
        default="flux2",
        help="Model backend to use.",
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
        help="Enable Latent Pass-Through (true/false) to bypass VAE loops.",
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        choices=["momentum", "soft"],
        default="momentum",
        help="Latent-space alignment strategy.",
    )
    parser.add_argument(
        "--lowpass-filter",
        type=str,
        choices=["avg_pool", "fft_gaussian"],
        default="avg_pool",
        help="Low/high frequency split used by momentum and soft augmentation.",
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
        help="Override img2img strength (default from config).",
    )
    return parser


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

    config_snapshot = {
        "resolved_config": _to_jsonable(config),
        "model_type": model_type,
        "evaluation": "metrics",
        "eval_mode": args.eval,
        "ablation_latent_pass_through": use_lpt,
        "ablation_augmentation": augmentation,
        "ablation_lowpass_filter": lowpass_filter,
        "ablation_lowpass_sigma": args.lowpass_sigma,
    }
    (run_dir / "config_snapshot.json").write_text(
        json.dumps(config_snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if model_type == "flux2":
        from src.model.model import FluxKleinRunner

        runner = FluxKleinRunner(
            config=config,
            use_lpt=use_lpt,
            augmentation=augmentation,
            lowpass_filter=lowpass_filter,
            lowpass_sigma=args.lowpass_sigma,
        )
    else:
        from src.model.model_kandinsky5 import Kandinsky5Runner

        runner = Kandinsky5Runner(
            config=config,
            use_lpt=use_lpt,
            augmentation=augmentation,
            lowpass_filter=lowpass_filter,
            lowpass_sigma=args.lowpass_sigma,
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
