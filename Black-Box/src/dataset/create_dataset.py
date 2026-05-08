from __future__ import annotations

import argparse
import io
import tarfile
import time
import urllib.error
from pathlib import Path
from typing import Callable, Iterator, TypeVar

from PIL import Image

from src.api import ImageEditClient
from src.config.settings import load_settings
from src.vae import VAEReconstructor

T = TypeVar("T")

VALID_SUFFIXES = {".image", ".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _parse_image_size(image_size: str | None) -> tuple[int, int] | None:
    if image_size is None:
        return None

    text = image_size.strip().lower()
    if "x" in text:
        w_text, h_text = text.split("x", maxsplit=1)
        return int(w_text), int(h_text)
    value = int(text)
    return value, value


def _sample_id_from_name(name: str) -> str:
    return Path(name).name.split(".", maxsplit=1)[0]


def _open_and_prepare_image(image: Image.Image, size: tuple[int, int] | None) -> Image.Image:
    rgb = image.convert("RGB")
    if size is None:
        return rgb
    return rgb.resize(size, Image.Resampling.LANCZOS)


def _save_jpg(image: Image.Image, out_path: Path) -> None:
    image.save(out_path, format="JPEG", quality=95)


def _cleanup_paths(paths: list[Path]) -> None:
    for path in paths:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            # Cleanup should not block dataset creation.
            pass


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError)):
        return True
    if isinstance(exc, OSError):
        text = str(exc).lower()
        network_hints = (
            "network",
            "connection",
            "timed out",
            "timeout",
            "host is down",
            "name or service not known",
            "temporarily unavailable",
            "winerror 53",
            "winerror 64",
            "winerror 121",
        )
        return any(hint in text for hint in network_hints)
    return False


def _run_with_retry(
    task: Callable[[], T],
    *,
    retry_times: int,
    task_name: str,
) -> T:
    if retry_times <= 0:
        raise ValueError("`retry_times` must be > 0.")

    for attempt in range(1, retry_times + 1):
        try:
            return task()
        except Exception as exc:
            is_last_attempt = attempt == retry_times
            if (not _is_retryable_error(exc)) or is_last_attempt:
                raise
            wait_seconds = min(2 ** (attempt - 1), 5)
            print(
                f"[retry] task={task_name} attempt={attempt}/{retry_times} "
                f"error={exc} wait={wait_seconds}s"
            )
            time.sleep(wait_seconds)

    raise RuntimeError(f"Retry loop exited unexpectedly: task={task_name}")


def _iter_flat_samples(data_path: Path, size: tuple[int, int] | None) -> Iterator[tuple[str, Image.Image]]:
    files = sorted(
        p
        for p in data_path.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_SUFFIXES
    )
    for path in files:
        sample_id = _sample_id_from_name(path.name)
        with Image.open(path) as image:
            prepared = _open_and_prepare_image(image, size)
        yield sample_id, prepared


def _iter_tar_samples_from_file(
    tar_path: Path, size: tuple[int, int] | None
) -> Iterator[tuple[str, Image.Image]]:
    with tarfile.open(tar_path, mode="r") as tar:
        members = sorted(
            (
                m
                for m in tar.getmembers()
                if m.isfile() and Path(m.name).suffix.lower() in VALID_SUFFIXES
            ),
            key=lambda member: member.name,
        )
        for member in members:
            stream = tar.extractfile(member)
            if stream is None:
                raise RuntimeError(f"Failed to read member from tar: {member.name}")
            raw = stream.read()
            sample_id = _sample_id_from_name(member.name)
            with Image.open(io.BytesIO(raw)) as image:
                prepared = _open_and_prepare_image(image, size)
            yield sample_id, prepared


def _iter_tar_samples(data_path: Path, size: tuple[int, int] | None) -> Iterator[tuple[str, Image.Image]]:
    tar_files = sorted(p for p in data_path.rglob("*.tar") if p.is_file())
    for tar_path in tar_files:
        yield from _iter_tar_samples_from_file(tar_path, size)


def _iter_samples(data_path: Path, size: tuple[int, int] | None) -> Iterator[tuple[str, Image.Image]]:
    if data_path.is_file():
        if data_path.suffix.lower() != ".tar":
            raise ValueError("When `data_path` is a file, it must be a `.tar` file.")
        yield from _iter_tar_samples_from_file(data_path, size)
        return

    yield from _iter_flat_samples(data_path, size)
    yield from _iter_tar_samples(data_path, size)


def _generate_round_sequence(
    *,
    initial_image: Image.Image,
    sample_id: str,
    api_client: ImageEditClient,
    vae: VAEReconstructor | None,
    prompts: list[str],
    rounds: int,
    image_size: tuple[int, int] | None,
    output_path: Path,
    retry_times: int,
) -> None:
    created_paths: list[Path] = []
    try:
        round0_path = output_path / f"{sample_id}_0.jpg"
        _save_jpg(initial_image, round0_path)
        created_paths.append(round0_path)
        current_input = round0_path

        for round_idx in range(1, rounds + 1):
            prompt = prompts[round_idx - 1]
            image_bytes = _run_with_retry(
                lambda: api_client.edit_image(current_input, prompt),
                retry_times=retry_times,
                task_name=f"api:{sample_id}:{round_idx}",
            )
            with Image.open(io.BytesIO(image_bytes)) as edited_image:
                next_image = _open_and_prepare_image(edited_image, image_size)

            if vae is not None:
                next_image = _open_and_prepare_image(vae.reconstruct(next_image), image_size)

            out_path = output_path / f"{sample_id}_{round_idx}.jpg"
            _save_jpg(next_image, out_path)
            created_paths.append(out_path)
            current_input = out_path
    except Exception:
        _cleanup_paths(created_paths)
        raise


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.dataset.create_dataset",
        description="Create round-indexed jpg training dataset from WebDataset images.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to config JSON.")
    parser.add_argument(
        "--image_size",
        type=str,
        default=None,
        help="Optional resize target: 512 or 512x512. If omitted, keep original size.",
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        required=True,
        help="Number of source samples to generate.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start source sample index for resume (0-based).",
    )
    parser.add_argument("--data_path", type=Path, required=True, help="Input data path (dir or .tar).")
    parser.add_argument("--output_path", type=Path, required=True, help="Output dataset directory.")
    parser.add_argument(
        "--retry_times",
        type=int,
        default=5,
        help="Retry times for network-related API errors.",
    )
    return parser


def main() -> None:
    args = _arg_parser().parse_args()
    settings = load_settings(args.config)
    rounds = settings.run.rounds
    prompts = settings.prompts.per_round
    image_size = _parse_image_size(args.image_size)
    data_path = args.data_path
    output_path = args.output_path
    dataset_size = args.dataset_size
    start = args.start

    if dataset_size <= 0:
        raise ValueError("`dataset_size` must be > 0.")
    if start < 0:
        raise ValueError("`start` must be >= 0.")
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    api_client = ImageEditClient(settings.api)
    vae = VAEReconstructor(settings.vae) if settings.vae.enabled else None

    total_samples = 0
    skipped_samples = 0
    for source_index, (sample_id, prepared) in enumerate(_iter_samples(data_path, image_size)):
        if source_index < start:
            continue
        if total_samples >= dataset_size:
            break
        try:
            _generate_round_sequence(
                initial_image=prepared,
                sample_id=sample_id,
                api_client=api_client,
                vae=vae,
                prompts=prompts,
                rounds=rounds,
                image_size=image_size,
                output_path=output_path,
                retry_times=args.retry_times,
            )
            total_samples += 1
        except Exception as exc:
            skipped_samples += 1
            print(f"[skip] source_index={source_index} sample_id={sample_id} error={exc}")
            continue

    print(f"[dataset] provider={api_client.provider}")
    print(f"[dataset] start={start}")
    print(f"[dataset] requested_dataset_size={dataset_size}")
    print(f"[dataset] generated_samples={total_samples}")
    print(f"[dataset] skipped_samples={skipped_samples}")
    print(f"[dataset] rounds={rounds}")
    print(f"[dataset] output_images={total_samples * (rounds + 1)}")
    print(f"[dataset] retry_times={args.retry_times}")
    print(f"[dataset] output={output_path}")


if __name__ == "__main__":
    main()
