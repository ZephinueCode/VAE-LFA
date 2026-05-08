from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.metrics import (  # noqa: E402
    _align_to_shape,
    _build_lpips,
    _load_image,
    _lpips_vs_base,
    _ssim_global,
    _to_gray,
    _to_lpips_tensor,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute LPIPS, SSIM, and L1 drift between any two images."
    )
    parser.add_argument("--image-a", type=Path, required=True, help="Reference image path.")
    parser.add_argument("--image-b", type=Path, required=True, help="Comparison image path.")
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Do not resize image-b to image-a. If sizes differ, raise an error.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to save the metric result as JSON.",
    )
    args = parser.parse_args()

    if not args.image_a.exists():
        raise FileNotFoundError(f"Image not found: {args.image_a}")
    if not args.image_b.exists():
        raise FileNotFoundError(f"Image not found: {args.image_b}")

    image_a = _load_image(args.image_a)
    image_b = _load_image(args.image_b)

    target_shape = image_a.shape[:2]
    original_shape_b = image_b.shape[:2]
    resized_b = False

    if image_b.shape[:2] != target_shape:
        if args.no_resize:
            raise ValueError(
                f"Image sizes differ: image-a={target_shape}, image-b={original_shape_b}. "
                "Remove `--no-resize` to align image-b to image-a."
            )
        image_b = _align_to_shape(image_b, target_shape)
        resized_b = True

    lpips_model, torch_module, lpips_device = _build_lpips()
    base_gray = _to_gray(image_a)
    current_gray = _to_gray(image_b)
    base_lpips_tensor = _to_lpips_tensor(image_a, torch_module=torch_module, device=lpips_device)

    l1 = float(abs(image_b - image_a).mean())
    ssim = float(_ssim_global(base_gray, current_gray))
    lpips_value = float(
        _lpips_vs_base(
            base_tensor=base_lpips_tensor,
            current=image_b,
            lpips_model=lpips_model,
            torch_module=torch_module,
            device=lpips_device,
        )
    )

    result = {
        "image_a": str(args.image_a),
        "image_b": str(args.image_b),
        "shape_a": list(target_shape),
        "shape_b_original": list(original_shape_b),
        "shape_b_used": list(image_b.shape[:2]),
        "resized_image_b": resized_b,
        "l1_drift": l1,
        "ssim": ssim,
        "lpips": lpips_value,
    }

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
