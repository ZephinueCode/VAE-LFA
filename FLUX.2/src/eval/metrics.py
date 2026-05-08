from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


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
        import torch
    except ImportError as exc:
        raise RuntimeError("Missing dependency for LPIPS. Install with `pip install lpips`.") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = lpips.LPIPS(net="alex")
    model = model.to(device)
    model.eval()
    return model, torch, device


def _to_lpips_tensor(image: np.ndarray, *, torch_module: Any, device: str):
    tensor = torch_module.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=torch_module.float32)
    return tensor * 2.0 - 1.0


def _lpips_vs_base(*, base_tensor: Any, current: np.ndarray, lpips_model: Any, torch_module: Any, device: str) -> float:
    current_tensor = _to_lpips_tensor(current, torch_module=torch_module, device=device)
    with torch_module.no_grad():
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
    """
    对所有测试样本进行 LPIPS, SSIM, L1 Drift 评测，并求出每轮的均值。
    """
    if not records:
        return {"error": "No metric records found."}

    output_dir.mkdir(parents=True, exist_ok=True)
    lpips_model, torch_module, lpips_device = _build_lpips()

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
        base_lpips_tensor = _to_lpips_tensor(base, torch_module=torch_module, device=lpips_device)

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
                torch_module=torch_module,
                device=lpips_device,
            )
            
            sample_metrics.append({
                "round": idx,
                "l1": l1,
                "ssim": ssim,
                "lpips": lpips_val
            })

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

    # 画全局均值趋势图
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
