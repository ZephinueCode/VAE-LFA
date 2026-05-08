from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from src.api import ImageEditClient
from src.config.settings import AppSettings
from src.inference.metrics import evaluate_sequence
from src.vae.reconstructor import VAEReconstructor


def _compute_overall_metrics(all_metrics: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    from collections import defaultdict
    import statistics

    # 按 round 分组
    rounds_data = defaultdict(list)
    for row in all_metrics:
        round_num = row["round"]
        rounds_data[round_num].append({
            "drift_l1_vs_base": row["drift_l1_vs_base"],
            "ssim_vs_base": row["ssim_vs_base"],
            "lpips_vs_base": row["lpips_vs_base"],
            "image": row["image"]
        })

    overall_rows = []
    for round_num in sorted(rounds_data.keys()):
        metrics_list = rounds_data[round_num]
        l1_values = [m["drift_l1_vs_base"] for m in metrics_list]
        ssim_values = [m["ssim_vs_base"] for m in metrics_list]
        lpips_values = [m["lpips_vs_base"] for m in metrics_list]

        overall_rows.append({
            "round": round_num,
            "mean_drift_l1_vs_base": statistics.mean(l1_values),
            "std_drift_l1_vs_base": statistics.stdev(l1_values) if len(l1_values) > 1 else 0,
            "mean_ssim_vs_base": statistics.mean(ssim_values),
            "std_ssim_vs_base": statistics.stdev(ssim_values) if len(ssim_values) > 1 else 0,
            "mean_lpips_vs_base": statistics.mean(lpips_values),
            "std_lpips_vs_base": statistics.stdev(lpips_values) if len(lpips_values) > 1 else 0,
            "num_images": len(metrics_list)
        })

    overall_csv_path = output_dir / "overall_metrics.csv"
    overall_plot_path = output_dir / "overall_metrics.png"
    overall_summary_path = output_dir / "overall_metrics_summary.json"

    # 保存 CSV
    import csv
    keys = ["round", "mean_drift_l1_vs_base", "std_drift_l1_vs_base", "mean_ssim_vs_base", "std_ssim_vs_base", "mean_lpips_vs_base", "std_lpips_vs_base", "num_images"]
    with overall_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in overall_rows:
            writer.writerow(row)

    # 保存图表
    import matplotlib.pyplot as plt
    rounds = [int(row["round"]) for row in overall_rows]
    l1_means = [float(row["mean_drift_l1_vs_base"]) for row in overall_rows]
    ssim_means = [float(row["mean_ssim_vs_base"]) for row in overall_rows]
    lpips_means = [float(row["mean_lpips_vs_base"]) for row in overall_rows]

    fig = plt.figure(figsize=(8.2, 4.8))
    plt.plot(rounds, l1_means, marker="o", linewidth=1.6, label="Mean L1 Drift vs Base")
    plt.plot(rounds, ssim_means, marker="o", linewidth=1.6, label="Mean SSIM vs Base")
    plt.plot(rounds, lpips_means, marker="o", linewidth=1.6, label="Mean LPIPS vs Base")
    plt.xlabel("Round")
    plt.ylabel("Mean Metric Value")
    y_max = max(1.0, max(l1_means + ssim_means + lpips_means) * 1.05)
    plt.ylim(0.0, y_max)
    plt.title("Overall Mean Metrics Across All Images")
    plt.grid(alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(overall_plot_path, dpi=150)
    plt.close(fig)

    summary = {
        "overall_metrics_csv": str(overall_csv_path),
        "overall_metrics_plot": str(overall_plot_path),
        "rows": overall_rows,
        "final": overall_rows[-1] if overall_rows else None,
    }
    overall_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "overall_metrics_csv": str(overall_csv_path),
        "overall_metrics_plot": str(overall_plot_path),
        "overall_summary_path": str(overall_summary_path),
    }


def _build_run_dir(results_dir: Path, run_name: str | None) -> Path:
    run_id = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_inference(config: AppSettings) -> dict[str, Any]:
    if config.run.input_dir and config.run.input_dir.exists():
        input_images = list(config.run.input_dir.glob("*.png")) + list(config.run.input_dir.glob("*.jpg")) + list(config.run.input_dir.glob("*.jpeg"))
        if not input_images:
            raise FileNotFoundError(f"No image files found in input directory: {config.run.input_dir}")
    elif config.run.input_image and config.run.input_image.exists():
        input_images = [config.run.input_image]
    else:
        raise FileNotFoundError("Neither input_image nor input_dir is specified or exists.")

    run_dir = _build_run_dir(config.run.results_dir, config.run.run_name)
    api_client = ImageEditClient(config.api)
    
    # 动态解析并初始化代理 VAE
    vae = VAEReconstructor(config.vae, config.intervention) if getattr(config.vae, "enabled", False) else None
    vae_mode = vae.inference_backend if vae is not None else "disabled"

    all_results = []
    all_metrics = []

    for input_image in input_images:
        image_name = input_image.stem
        image_run_dir = run_dir / image_name
        image_run_dir.mkdir(parents=True, exist_ok=True)

        seed_path = image_run_dir / f"round_000_input{input_image.suffix.lower()}"
        shutil.copyfile(input_image, seed_path)
        current_input = seed_path
        sequence_next_inputs: list[Path] = [seed_path]
        steps: list[dict[str, Any]] = []

        # =========================================================
        # 预热动量：用初始纯净图片作为最初的 Anchor (后续由 EMA 自动推演)
        # =========================================================
        if vae is not None and getattr(config.intervention, "enabled", False):
            vae.reset_momentum()
            with Image.open(seed_path) as seed_image:
                vae.init_momentum(seed_image)

        started_at = time.time()
        for round_idx in range(1, config.run.rounds + 1):
            prompt = config.prompts.per_round[round_idx - 1]
            raw_path = image_run_dir / f"round_{round_idx:03d}_raw.png"

            # 1. 调用闭源 DiT API
            image_bytes = api_client.edit_image(current_input, prompt)
            raw_path.write_bytes(image_bytes)

            # 2. 代理 VAE 介入：执行低频 EMA 动量对齐
            if vae is not None:
                with Image.open(raw_path) as raw_image:
                    vae_image = vae.reconstruct(raw_image)
                vae_path = image_run_dir / f"round_{round_idx:03d}_vae.png"
                vae_image.save(vae_path)
                next_input = vae_path
            else:
                next_input = raw_path

            sequence_next_inputs.append(next_input)
            steps.append(
                {
                    "round": round_idx,
                    "prompt": prompt,
                    "model": config.api.model,
                    "input_image": str(current_input),
                    "raw_output_image": str(raw_path),
                    "next_input_image": str(next_input),
                }
            )
            current_input = next_input

        metrics = evaluate_sequence(sequence_next_inputs, image_run_dir)
        all_metrics.extend(metrics["rows"])

        intervention_dict = {
            "enabled": getattr(config.intervention, "enabled", False),
            "kernel_size": getattr(config.intervention, "kernel_size", 9),
            "mean_decay": getattr(config.intervention, "mean_decay", 0.85),
            "std_decay": getattr(config.intervention, "std_decay", 0.85),
        }

        manifest = {
            "run_id": image_run_dir.name,
            "created_at": datetime.now().isoformat(),
            "duration_seconds": round(time.time() - started_at, 3),
            "config": {
                "api": {
                    "model": config.api.model,
                },
                "run": {
                    "rounds": config.run.rounds,
                    "input_image": str(input_image),
                },
                "vae": {
                    "enabled": getattr(config.vae, "enabled", False),
                    "inference_backend": vae_mode,
                    "model_type": getattr(config.vae, "model_type", "autoencoder_kl"),
                },
                "intervention": intervention_dict
            },
            "sequence_next_inputs": [str(path) for path in sequence_next_inputs],
            "steps": steps,
            "metrics": {
                "metrics_csv": metrics["metrics_csv"],
                "metrics_plot": metrics["metrics_plot"],
                "summary_path": metrics["summary_path"],
            },
        }
        manifest_path = image_run_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        all_results.append({
            "image_name": image_name,
            "run_dir": str(image_run_dir),
            "manifest_path": str(manifest_path),
            "metrics_csv": metrics["metrics_csv"],
            "metrics_plot": metrics["metrics_plot"],
        })

    # 计算总的均值
    if all_metrics:
        total_summary = _compute_overall_metrics(all_metrics, run_dir)
    else:
        total_summary = {}

    return {
        "run_dir": str(run_dir),
        "results": all_results,
        "overall_metrics": total_summary,
    }