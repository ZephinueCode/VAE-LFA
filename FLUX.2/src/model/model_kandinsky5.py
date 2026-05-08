from __future__ import annotations

import contextlib
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.model.model import FluxKleinRunner, RunTrace
from src.utils.config import AppConfig


KANDINSKY5_BUCKETS: tuple[tuple[int, int], ...] = (
    (1024, 1024),
    (640, 1408),
    (1408, 640),
    (768, 1280),
    (1280, 768),
    (896, 1152),
    (1152, 896),
)


class Kandinsky5Runner(FluxKleinRunner):
    def __init__(
        self,
        config: AppConfig,
        use_lpt: bool = False,
        augmentation: str = "momentum",
        lowpass_filter: str = "avg_pool",
        lowpass_sigma: float = 0.25,
    ):
        super().__init__(
            config=config,
            use_lpt=use_lpt,
            augmentation=augmentation,
            lowpass_filter=lowpass_filter,
            lowpass_sigma=lowpass_sigma,
        )

    def _load_pipeline(self, config: AppConfig) -> Any:
        from diffusers import Kandinsky5I2IPipeline

        kwargs: dict[str, Any] = {
            "torch_dtype": self.dtype,
            "local_files_only": config.model.local_files_only,
            "low_cpu_mem_usage": config.model.low_cpu_mem_usage,
        }
        if config.model.revision:
            kwargs["revision"] = config.model.revision
        if config.model.variant:
            kwargs["variant"] = config.model.variant

        try:
            pipe = Kandinsky5I2IPipeline.from_pretrained(config.model.model_path, **kwargs)
        except TypeError:
            pipe = Kandinsky5I2IPipeline.from_pretrained(
                config.model.model_path,
                torch_dtype=self.dtype,
            )

        if config.model.enable_model_cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(config.model.device)
        return pipe

    @staticmethod
    def _detect_image_key(signature: inspect.Signature) -> str | None:
        for key in ("image", "init_image", "input_image"):
            if key in signature.parameters:
                return key
        return None

    @staticmethod
    def _k5_to_bchw(latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim == 4:
            return latents
        if latents.ndim != 5:
            raise ValueError(f"Unexpected Kandinsky latent shape: {tuple(latents.shape)}")
        if latents.shape[1] != 1:
            raise ValueError(f"Expected Kandinsky temporal dimension 1, got shape {tuple(latents.shape)}")
        return latents[:, 0].permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _bchw_to_k5(latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 4:
            raise ValueError(f"Expected BCHW tensor, got shape {tuple(latents.shape)}")
        return latents.permute(0, 2, 3, 1).unsqueeze(1).contiguous()

    @staticmethod
    def _normalize_output_latents(latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim == 5:
            return latents
        if latents.ndim == 4:
            return Kandinsky5Runner._bchw_to_k5(latents)
        raise RuntimeError(f"Unexpected Kandinsky latent output shape: {tuple(latents.shape)}")

    @staticmethod
    def _select_bucket(width: float, height: float) -> tuple[int, int]:
        target_ratio = float(width) / max(float(height), 1.0)
        target_area = float(width) * float(height)

        def score(bucket: tuple[int, int]) -> tuple[float, float]:
            bucket_h, bucket_w = bucket
            bucket_ratio = float(bucket_w) / float(bucket_h)
            bucket_area = float(bucket_w) * float(bucket_h)
            return (
                abs(bucket_ratio - target_ratio),
                abs(np.log(max(bucket_area, 1.0) / max(target_area, 1.0))),
            )

        return min(KANDINSKY5_BUCKETS, key=score)

    def _resolve_target_size(self, image: Image.Image) -> tuple[int, int]:
        cfg_height = self.config.inference.height
        cfg_width = self.config.inference.width

        width, height = map(float, image.size)
        if cfg_width is not None and cfg_height is not None:
            width = float(cfg_width)
            height = float(cfg_height)
        elif cfg_width is not None:
            scale = float(cfg_width) / max(width, 1.0)
            width = float(cfg_width)
            height = height * scale
        elif cfg_height is not None:
            scale = float(cfg_height) / max(height, 1.0)
            height = float(cfg_height)
            width = width * scale
        else:
            max_side = 1280.0
            long_edge = max(width, height)
            if long_edge > max_side:
                scale = max_side / long_edge
                width = width * scale
                height = height * scale

        return self._select_bucket(width, height)

    @contextlib.contextmanager
    def _encode_context(self, current_latents: torch.Tensor | None):
        orig_encode = self.pipe.vae.encode

        class MockVAEOutput:
            def __init__(self, latents: torch.Tensor):
                self.latents = latents

            @property
            def latent_dist(self):
                latents = self.latents

                class Dist:
                    def sample(self_dist, *args, **kwargs):
                        return latents

                    def mode(self_dist, *args, **kwargs):
                        return latents

                return Dist()

        def patched_encode(*args, **kwargs):
            if current_latents is not None:
                # Kandinsky's prepare_latents() multiplies VAE samples by scaling_factor
                # after encode(), so the mocked encode() must return the unscaled VAE latent.
                mocked_latents = self._k5_to_bchw(current_latents)
                scaling_factor = getattr(self.pipe.vae.config, "scaling_factor", None)
                if scaling_factor is not None:
                    mocked_latents = mocked_latents / scaling_factor
                return MockVAEOutput(mocked_latents)
            return orig_encode(*args, **kwargs)

        self.pipe.vae.encode = patched_encode
        try:
            yield
        finally:
            self.pipe.vae.encode = orig_encode

    def _decode_latents(self, latents: torch.Tensor) -> Image.Image:
        latents = self._normalize_output_latents(latents)
        batch_size, num_frames, height, width, num_channels = latents.shape

        with torch.no_grad():
            decode_input = latents.permute(0, 1, 4, 2, 3).reshape(
                batch_size * num_frames,
                num_channels,
                height,
                width,
            )
            decode_input = decode_input / self.pipe.vae.config.scaling_factor
            image = self.pipe.vae.decode(decode_input).sample
            image = self.pipe.image_processor.postprocess(image, output_type="pil")

        if isinstance(image, list):
            return image[0]
        return image

    def _update_anchor_state(self, latents: torch.Tensor) -> None:
        super()._update_anchor_state(self._k5_to_bchw(latents))

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
        resized_image = current_image_pil.resize((width, height), resample=Image.Resampling.LANCZOS)

        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": "",
            "height": height,
            "width": width,
            "guidance_scale": self.config.inference.guidance_scale,
            "num_inference_steps": self.config.inference.num_inference_steps,
            "generator": self._build_generator(seed),
            "output_type": "latent",
        }
        if self.config.inference.max_sequence_length is not None and "max_sequence_length" in self._call_signature.parameters:
            kwargs["max_sequence_length"] = self.config.inference.max_sequence_length

        if self._supports_image_key is None:
            raise RuntimeError("Current Kandinsky5I2IPipeline call signature does not support image input.")

        kwargs[self._supports_image_key] = resized_image

        with torch.inference_mode():
            with self._encode_context(current_latents if self.use_lpt and current_latents is not None else None):
                output = self.pipe(**kwargs)

        latents_out = getattr(output, "image", None)
        if latents_out is None:
            latents_out = getattr(output, "images", None)
        if latents_out is None:
            latents_out = output[0] if isinstance(output, tuple) else output

        if isinstance(latents_out, (list, tuple)):
            if len(latents_out) == 0:
                raise RuntimeError("Kandinsky pipeline output is empty.")
            denoised_latents = latents_out[0]
        else:
            denoised_latents = latents_out

        pure_latents = self._normalize_output_latents(denoised_latents)

        if self.use_lpt and current_latents is not None:
            before_bchw = self._k5_to_bchw(current_latents)
            after_bchw = self._k5_to_bchw(pure_latents)
            if self.augmentation == "momentum":
                after_bchw = self._restore_moments_momentum(before=before_bchw, after=after_bchw)
            elif self.augmentation == "soft":
                after_bchw = self._soft_augmentation(before=before_bchw, after=after_bchw, strength=0.5)
            pure_latents = self._bchw_to_k5(after_bchw)

        vis_image = self._decode_latents(pure_latents)
        return pure_latents, vis_image

    def run_sample(
        self,
        *,
        sample_id: str,
        image: Image.Image,
        prompts: list[str],
        run_dir: Path,
    ) -> RunTrace:
        sample_dir = run_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        self._reset_anchor_state()
        self._reset_momentum_stats()

        round_images: list[str] = []

        current_image_pil = image.convert("RGB")
        current_latents = None

        input_path = sample_dir / "round_000_input.png"
        current_image_pil.save(input_path)
        round_images.append(str(input_path))

        height, width = self._resolve_target_size(image)

        for round_idx, prompt in enumerate(prompts, start=1):
            seed = self.config.inference.seed + (round_idx - 1) * self.config.inference.seed_stride

            accepted_latents, vis_image = self._run_round(
                prompt=prompt,
                current_image_pil=current_image_pil,
                current_latents=current_latents,
                seed=seed,
                height=height,
                width=width,
            )

            if self.config.output.save_round_images:
                out_path = sample_dir / f"round_{round_idx:03d}.png"
                vis_image.save(out_path)
                round_images.append(str(out_path))

            # Kandinsky has a separate Qwen2.5-VL image branch, so even in latent
            # mode we advance the visible image to keep the RGB-side condition aligned
            # with the accepted latent trajectory.
            current_image_pil = vis_image
            if self.use_lpt:
                self._update_anchor_state(accepted_latents)
                current_latents = accepted_latents
            else:
                current_latents = None

        return RunTrace(sample_id=sample_id, round_images=round_images, collector_outputs={})


__all__ = ["Kandinsky5Runner", "RunTrace"]
