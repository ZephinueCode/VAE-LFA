from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image

from .config import DataConfig

DEFAULT_NOOP_PROMPT = "Keep the image unchanged."


@dataclass
class ImageSample:
    sample_id: str
    image: Image.Image
    image_path: Path
    meta_path: Path | None
    prompts_by_eval: dict[str, list[str]]


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
        return ImageSample(
            sample_id=image_path.stem,
            image=image.convert("RGB"),
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
    files = sorted(
        p
        for p in globber("*")
        if p.is_file() and _is_supported(p, suffixes)
    )
    for image_path in files:
        meta_path = image_path.with_suffix(".json")
        yield _load_image_sample(image_path, meta_path if meta_path.exists() else None)


def resolve_eval_prompts(sample: ImageSample, eval_mode: str) -> list[str]:
    prompts = sample.prompts_by_eval.get(eval_mode, [])
    if not prompts:
        detail = f"Metadata file: {sample.meta_path}" if sample.meta_path is not None else "No metadata file found."
        raise ValueError(
            f"Sample `{sample.sample_id}` does not provide prompts for eval mode `{eval_mode}`. {detail}"
        )
    return list(prompts)


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
