from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


def _load_image(path: Path) -> np.ndarray:
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


def _lpips_vs_base(
    *,
    base_tensor: Any,
    current: np.ndarray,
    lpips_model: Any,
    torch_module: Any,
    device: str,
) -> float:
    current_tensor = _to_lpips_tensor(current, torch_module=torch_module, device=device)
    with torch_module.no_grad():
        score = lpips_model(base_tensor, current_tensor)
    return float(score.detach().mean().cpu().item())


def _save_metrics_csv(rows: list[dict[str, Any]], path: Path) -> None:
    keys = ["round", "image", "drift_l1_vs_base", "ssim_vs_base", "lpips_vs_base"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_plot(rows: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    rounds = [int(row["round"]) for row in rows]
    l1_values = [float(row["drift_l1_vs_base"]) for row in rows]
    ssim_values = [float(row["ssim_vs_base"]) for row in rows]
    lpips_values = [float(row["lpips_vs_base"]) for row in rows]

    fig = plt.figure(figsize=(8.2, 4.8))
    plt.plot(rounds, l1_values, marker="o", linewidth=1.6, label="L1 Drift vs Base")
    plt.plot(rounds, ssim_values, marker="o", linewidth=1.6, label="SSIM vs Base")
    plt.plot(rounds, lpips_values, marker="o", linewidth=1.6, label="LPIPS vs Base")
    plt.xlabel("Round")
    plt.ylabel("Metric Value")
    y_max = max(1.0, max(l1_values + ssim_values + lpips_values) * 1.05)
    plt.ylim(0.0, y_max)
    plt.title("L1 Drift + SSIM + LPIPS")
    plt.grid(alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_sequence(image_paths: Sequence[Path], output_dir: Path) -> dict[str, Any]:
    if len(image_paths) < 2:
        raise ValueError("Need at least two images to compute metrics.")

    output_dir.mkdir(parents=True, exist_ok=True)
    images = [_load_image(path) for path in image_paths]
    base_h, base_w = images[0].shape[:2]
    aligned = [_align_to_shape(img, (base_h, base_w)) for img in images]

    base = aligned[0]
    base_gray = _to_gray(base)
    lpips_model, torch_module, lpips_device = _build_lpips()
    base_lpips_tensor = _to_lpips_tensor(base, torch_module=torch_module, device=lpips_device)

    rows: list[dict[str, Any]] = []
    for idx in range(1, len(aligned)):
        current = aligned[idx]
        current_gray = _to_gray(current)
        rows.append(
            {
                "round": idx,
                "image": image_paths[idx].name,
                "drift_l1_vs_base": float(np.mean(np.abs(current - base))),
                "ssim_vs_base": _ssim_global(base_gray, current_gray),
                "lpips_vs_base": _lpips_vs_base(
                    base_tensor=base_lpips_tensor,
                    current=current,
                    lpips_model=lpips_model,
                    torch_module=torch_module,
                    device=lpips_device,
                ),
            }
        )

    csv_path = output_dir / "metrics.csv"
    plot_path = output_dir / "metrics_l1_ssim_lpips.png"
    summary_path = output_dir / "metrics_summary.json"

    _save_metrics_csv(rows, csv_path)
    _save_plot(rows, plot_path)

    summary = {
        "metrics_csv": str(csv_path),
        "metrics_plot": str(plot_path),
        "rows": rows,
        "final": rows[-1],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
