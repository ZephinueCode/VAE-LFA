from __future__ import annotations

import argparse
from pathlib import Path

from .config.settings import load_settings
from .inference.pipeline import run_inference


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run iterative inference pipeline.")
    run_parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/config/config.json"),
        help="Path to config JSON.",
    )

    # 这里不重复定义 extract/apply 参数，直接把剩余参数转发给各模块 main
    subparsers.add_parser("extract-pca", help="Extract offline PCA checkpoint from dataset.")
    subparsers.add_parser("apply-pca", help="Apply offline PCA checkpoint to image(s).")

    return parser


def main() -> None:
    parser = _arg_parser()
    args, unknown = parser.parse_known_args()

    if args.command == "run":
        settings = load_settings(args.config)
        result = run_inference(settings)
        print(f"[run] run_dir={result['run_dir']}")
        for res in result['results']:
            print(f"[image] {res['image_name']}: run_dir={res['run_dir']}")
            print(f"[metrics] csv={res['metrics_csv']}")
            print(f"[metrics] plot={res['metrics_plot']}")
        if 'overall_metrics' in result:
            print(f"[overall] csv={result['overall_metrics']['overall_metrics_csv']}")
            print(f"[overall] plot={result['overall_metrics']['overall_metrics_plot']}")
        return

if __name__ == "__main__":
    main()