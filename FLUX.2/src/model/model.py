from __future__ import annotations

import contextlib
import inspect
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import numpy as np
from diffusers import Flux2KleinPipeline
from PIL import Image

from src.utils.config import AppConfig


def resolve_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    value = mapping.get(str(name).strip().lower(), torch.float32)
    if value != torch.float32 and not torch.cuda.is_available():
        return torch.float32
    return value


@dataclass
class RunTrace:
    sample_id: str
    round_images: list[str]
    collector_outputs: dict[str, Any]


@dataclass
class LatentAnchorState:
    init_latents: torch.Tensor | None = None
    prev_latents: torch.Tensor | None = None
    ema_latents: torch.Tensor | None = None
    prior_low: torch.Tensor | None = None
    prior_high: torch.Tensor | None = None


class FluxKleinRunner:
    def __init__(
        self,
        config: AppConfig,
        use_lpt: bool = False,
        augmentation: str = "momentum",
        lowpass_filter: str = "avg_pool",
        lowpass_sigma: float = 0.25,
    ):
        self.config = config
        self.use_lpt = use_lpt
        self.augmentation = augmentation
        self.lowpass_filter = str(lowpass_filter).strip().lower()
        self.lowpass_sigma = max(1e-6, float(lowpass_sigma))

        # Alignment anchor: previous + momentum (EMA) + initial + running prior bounds.
        self.anchor_prev_weight = 0.55
        self.anchor_ema_weight = 0.30
        self.anchor_init_weight = 0.15
        self.anchor_ema_decay = 0.92
        self.prior_decay = 0.90
        self._anchor_state = LatentAnchorState()

        # Soft augmentation controls.
        self.soft_low_strength = 0.50
        self.soft_freq_strength = 0.15
        self.soft_residual_strength = 1.00
        self.soft_residual_radius_mult = 2.5
        self.soft_prior_strength = 1.00
        self.soft_prior_margin = 0.20
        self.soft_prior_softness = 0.03
        self.soft_freq_bins = 8
        self.soft_lowpass_kernel = 9
        self.momentum_lowpass_kernel = 9
        self.soft_coral_shrink = 0.05
        self.soft_residual_cov_shrink = 0.10
        self.adaptive_low_std_strength = 0.25
        self.adaptive_high_freq_strength = 0.25
        self.adaptive_residual_strength = 0.35
        self.adaptive_prior_strength = 0.25
        self.momentum_mean_decay = 0.95
        self.momentum_log_std_decay = 0.85
        self.source_low_strength = 1.00
        self.source_momentum_low_strength = 0.50
        self.source_momentum_mean_decay = 0.95
        self.source_momentum_log_std_decay = 0.85
        self.source_momentum_mean_radius = 1.00
        self.source_momentum_log_std_radius = 0.50
        self._momentum_mean: torch.Tensor | None = None
        self._momentum_log_std: torch.Tensor | None = None
        self._source_offset_mean: torch.Tensor | None = None
        self._source_offset_log_std: torch.Tensor | None = None

        self.dtype = resolve_torch_dtype(config.model.torch_dtype)
        self.device = config.model.device
        self.pipe = self._load_pipeline(config)
        self._call_signature = inspect.signature(self.pipe.__call__)
        self._supports_image_key = self._detect_image_key(self._call_signature)

    def _load_pipeline(self, config: AppConfig) -> Flux2KleinPipeline:
        kwargs: dict[str, Any] = {
            "torch_dtype": self.dtype,
            "local_files_only": config.model.local_files_only,
            "low_cpu_mem_usage": config.model.low_cpu_mem_usage,
        }
        if config.model.revision:
            kwargs["revision"] = config.model.revision
        if config.model.variant:
            kwargs["variant"] = config.model.variant

        pipe = Flux2KleinPipeline.from_pretrained(config.model.model_path, **kwargs)
        if config.model.enable_model_cpu_offload:
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

    def _build_generator(self, seed: int) -> torch.Generator:
        gen_device = self.device
        if gen_device.startswith("cuda") and not torch.cuda.is_available():
            gen_device = "cpu"
        generator = torch.Generator(device=gen_device)
        generator.manual_seed(seed)
        return generator

    def _reset_anchor_state(self) -> None:
        self._anchor_state = LatentAnchorState()

    @staticmethod
    def _compute_channel_quantiles(latents: torch.Tensor, low_q: float = 0.005, high_q: float = 0.995) -> tuple[torch.Tensor, torch.Tensor]:
        b, c = latents.shape[:2]
        flat = latents.float().view(b, c, -1)
        low = torch.quantile(flat, low_q, dim=-1)[..., None, None]
        high = torch.quantile(flat, high_q, dim=-1)[..., None, None]
        return low, high

    def _update_anchor_state(self, latents: torch.Tensor) -> None:
        st = self._anchor_state
        lat = latents.detach()
        low, high = self._compute_channel_quantiles(lat)

        if st.init_latents is None:
            st.init_latents = lat
            st.prev_latents = lat
            st.ema_latents = lat.float()
            st.prior_low = low
            st.prior_high = high
            return

        st.prev_latents = lat
        decay = float(self.anchor_ema_decay)
        st.ema_latents = decay * st.ema_latents + (1.0 - decay) * lat.float()

        prior_decay = float(self.prior_decay)
        st.prior_low = prior_decay * st.prior_low + (1.0 - prior_decay) * low
        st.prior_high = prior_decay * st.prior_high + (1.0 - prior_decay) * high

    def _compose_anchor_latents(self, fallback_latents: torch.Tensor, fallback_if_empty: bool = True) -> torch.Tensor | None:
        st = self._anchor_state
        pieces: list[tuple[float, torch.Tensor]] = []

        if st.prev_latents is not None:
            pieces.append((float(self.anchor_prev_weight), st.prev_latents))
        if st.ema_latents is not None:
            pieces.append((float(self.anchor_ema_weight), st.ema_latents))
        if st.init_latents is not None:
            pieces.append((float(self.anchor_init_weight), st.init_latents))

        pieces = [(w, t) for w, t in pieces if w > 0.0]
        if not pieces:
            return fallback_latents if fallback_if_empty else None

        total = sum(w for w, _ in pieces)
        if total <= 1e-8:
            return fallback_latents if fallback_if_empty else None

        out = torch.zeros_like(fallback_latents, dtype=torch.float32)
        for w, t in pieces:
            out = out + (w / total) * t.to(device=fallback_latents.device, dtype=torch.float32)
        return out.to(dtype=fallback_latents.dtype)

    def _resolve_prior_bounds(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        st = self._anchor_state
        b, c = reference.shape[:2]

        if st.prior_low is not None and st.prior_high is not None:
            lo = st.prior_low.to(device=reference.device, dtype=torch.float32)
            hi = st.prior_high.to(device=reference.device, dtype=torch.float32)
            if lo.shape[1] == c and hi.shape[1] == c:
                if lo.shape[0] == 1 and b > 1:
                    lo = lo.expand(b, -1, -1, -1)
                    hi = hi.expand(b, -1, -1, -1)
                if lo.shape[0] == b and hi.shape[0] == b:
                    return lo, hi

        return self._compute_channel_quantiles(reference.float())

    @contextlib.contextmanager
    def _encode_context(self, current_latents: torch.Tensor | None):
        orig_encode = self.pipe.vae.encode

        class MockVAEOutput:
            def __init__(self, latents):
                self.latents = latents
            @property
            def latent_dist(self):
                _latents = self.latents
                class Dist:
                    def sample(self_dist, *args, **kwargs):
                        return _latents
                    def mode(self_dist, *args, **kwargs):
                        return _latents
                return Dist()

        def patched_encode(*args, **kwargs):
            if current_latents is not None:
                return MockVAEOutput(current_latents)
            return orig_encode(*args, **kwargs)

        self.pipe.vae.encode = patched_encode
        try:
            yield
        finally:
            self.pipe.vae.encode = orig_encode

    def _restore_moments(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        std_b = before.std(dim=(2, 3), keepdim=True)
        std_a = after.std(dim=(2, 3), keepdim=True)
        mean_b = before.mean(dim=(2, 3), keepdim=True)
        mean_a = after.mean(dim=(2, 3), keepdim=True)

        out = (after - mean_a) / (std_a + 1e-6)
        out = out * (std_b + 1e-6) + mean_b
        return torch.nan_to_num(out, nan=0.0, posinf=1e3, neginf=-1e3).to(after.dtype)

    def _reset_momentum_stats(self) -> None:
        self._momentum_mean = None
        self._momentum_log_std = None
        self._source_offset_mean = None
        self._source_offset_log_std = None

    @staticmethod
    def _channel_moments(latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        work = latents.detach().float()
        mean = work.mean(dim=(2, 3), keepdim=True)
        std = work.std(dim=(2, 3), unbiased=False, keepdim=True).clamp_min(1e-6)
        return mean, std

    def _low_frequency_moments(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        low_latents = self._lowpass_latent(latents.float())
        mean, std = self._channel_moments(low_latents)
        return low_latents, mean, std

    def _update_momentum_moments(self, mean: torch.Tensor, log_std: torch.Tensor) -> None:
        mean = mean.detach().float()
        log_std = log_std.detach().float()
        if self._momentum_mean is None or self._momentum_log_std is None:
            self._momentum_mean = mean
            self._momentum_log_std = log_std
            return

        if self._momentum_mean.shape != mean.shape:
            self._momentum_mean = mean
            self._momentum_log_std = log_std
            return

        mean_decay = max(0.0, min(0.999, float(self.momentum_mean_decay)))
        std_decay = max(0.0, min(0.999, float(self.momentum_log_std_decay)))
        self._momentum_mean = mean_decay * self._momentum_mean + (1.0 - mean_decay) * mean
        self._momentum_log_std = std_decay * self._momentum_log_std + (1.0 - std_decay) * log_std

    def _source_anchor_latents(self, fallback: torch.Tensor) -> torch.Tensor:
        source = self._anchor_state.init_latents
        if source is None or source.shape != fallback.shape:
            return fallback
        return source.to(device=fallback.device, dtype=fallback.dtype)

    @staticmethod
    def _clip_by_radius(value: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        radius = radius.to(device=value.device, dtype=value.dtype).abs()
        return torch.max(torch.min(value, radius), -radius)

    def _update_source_offset_moments(
        self,
        *,
        source_mean: torch.Tensor,
        source_log_std: torch.Tensor,
        obs_mean: torch.Tensor,
        obs_log_std: torch.Tensor,
    ) -> None:
        source_mean = source_mean.detach().float()
        source_log_std = source_log_std.detach().float()
        obs_mean = obs_mean.detach().float()
        obs_log_std = obs_log_std.detach().float()

        mean_radius = source_log_std.exp().clamp_min(1e-6) * float(self.source_momentum_mean_radius)
        log_std_radius = torch.full_like(source_log_std, float(self.source_momentum_log_std_radius))
        desired_mean_offset = self._clip_by_radius(obs_mean - source_mean, mean_radius)
        desired_log_std_offset = self._clip_by_radius(obs_log_std - source_log_std, log_std_radius)

        if (
            self._source_offset_mean is None
            or self._source_offset_log_std is None
            or self._source_offset_mean.shape != desired_mean_offset.shape
        ):
            self._source_offset_mean = torch.zeros_like(desired_mean_offset)
            self._source_offset_log_std = torch.zeros_like(desired_log_std_offset)

        mean_decay = max(0.0, min(0.999, float(self.source_momentum_mean_decay)))
        std_decay = max(0.0, min(0.999, float(self.source_momentum_log_std_decay)))
        self._source_offset_mean = mean_decay * self._source_offset_mean + (1.0 - mean_decay) * desired_mean_offset
        self._source_offset_log_std = std_decay * self._source_offset_log_std + (1.0 - std_decay) * desired_log_std_offset

    def _align_low_frequency_to_moments(
        self,
        after: torch.Tensor,
        *,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        strength: float,
    ) -> torch.Tensor:
        gate = max(0.0, min(1.0, float(strength)))
        if gate <= 0.0:
            return after

        after_f = after.float()
        low_after, mean_a, std_a = self._low_frequency_moments(after_f)
        high_after = after_f - low_after
        low_aligned = (low_after - mean_a) / (std_a + 1e-6)
        low_aligned = low_aligned * (target_std.to(after.device, torch.float32) + 1e-6)
        low_aligned = low_aligned + target_mean.to(after.device, torch.float32)
        low_out = torch.lerp(low_after, low_aligned, gate)
        out = low_out + high_after
        return torch.nan_to_num(out, nan=0.0, posinf=1e3, neginf=-1e3).to(after.dtype)

    def _restore_moments_momentum(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        _, mean_a, std_a = self._low_frequency_moments(after)
        log_std_a = std_a.log()

        if (
            self._momentum_mean is None
            or self._momentum_log_std is None
            or self._momentum_mean.shape != mean_a.shape
        ):
            _, mean_b, std_b = self._low_frequency_moments(before)
            self._momentum_mean = mean_b.detach().float()
            self._momentum_log_std = std_b.log().detach().float()

        mean_t = self._momentum_mean.to(device=after.device, dtype=torch.float32)
        std_t = self._momentum_log_std.to(device=after.device, dtype=torch.float32).exp()
        out = self._align_low_frequency_to_moments(after, target_mean=mean_t, target_std=std_t, strength=1.0)
        self._update_momentum_moments(mean_a, log_std_a)
        return out

    def _restore_source_low(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        source = self._source_anchor_latents(before)
        _, mean_t, std_t = self._low_frequency_moments(source)
        return self._align_low_frequency_to_moments(
            after,
            target_mean=mean_t,
            target_std=std_t,
            strength=self.source_low_strength,
        )

    def _restore_source_momentum_low(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        source = self._source_anchor_latents(before)
        _, source_mean, source_std = self._low_frequency_moments(source)
        source_log_std = source_std.log()
        _, obs_mean, obs_std = self._low_frequency_moments(after)
        obs_log_std = obs_std.log()

        self._update_source_offset_moments(
            source_mean=source_mean,
            source_log_std=source_log_std,
            obs_mean=obs_mean,
            obs_log_std=obs_log_std,
        )

        if self._source_offset_mean is not None and self._source_offset_log_std is not None:
            mean_t = source_mean + self._source_offset_mean.to(device=after.device, dtype=torch.float32)
            std_t = (source_log_std + self._source_offset_log_std.to(device=after.device, dtype=torch.float32)).exp()
        else:
            mean_t, std_t = source_mean, source_std

        return self._align_low_frequency_to_moments(
            after,
            target_mean=mean_t,
            target_std=std_t,
            strength=self.source_momentum_low_strength,
        )

    def _soft_augmentation(self, before: torch.Tensor, after: torch.Tensor, strength: float = 0.5) -> torch.Tensor:
        gate = max(0.0, min(1.0, float(strength)))
        if gate <= 0.0:
            return after

        b, c, _, _ = after.shape
        orig_dtype = after.dtype
        before_f = before.float()
        after_f = after.float()

        low_a = self._lowpass_latent(after_f)
        low_b = self._lowpass_latent(before_f)
        high_a = after_f - low_a
        high_b = before_f - low_b

        mean_la = low_a.mean(dim=(2, 3), keepdim=True)
        std_la = low_a.std(dim=(2, 3), unbiased=False, keepdim=True).clamp_min(1e-6)
        mean_lb = low_b.mean(dim=(2, 3), keepdim=True)
        std_lb = low_b.std(dim=(2, 3), unbiased=False, keepdim=True).clamp_min(1e-6)
        low_aligned = (low_a - mean_la) / (std_la + 1e-6) * (std_lb + 1e-6) + mean_lb

        std_ha = high_a.std(dim=(2, 3), unbiased=False, keepdim=True).clamp_min(1e-6)
        std_hb = high_b.std(dim=(2, 3), unbiased=False, keepdim=True).clamp_min(1e-6)
        high_aligned = high_a * (std_hb + 1e-6) / (std_ha + 1e-6)

        out_freq = low_aligned + high_aligned
        out = out_freq * gate + after_f * (1.0 - gate)

        mean_b = before_f.mean(dim=(2, 3), keepdim=True)
        std_b = before_f.std(dim=(2, 3), unbiased=False, keepdim=True).clamp_min(1e-6)
        z_score = (out - mean_b) / (std_b + 1e-6)
        z_score = torch.clamp(z_score, -3.5, 3.5)
        out = z_score * (std_b + 1e-6) + mean_b

        flat_before = before_f.view(b, c, -1)
        q_low = torch.quantile(flat_before, 0.005, dim=-1)[..., None, None]
        q_high = torch.quantile(flat_before, 0.995, dim=-1)[..., None, None]
        out = torch.max(torch.min(out, q_high + (out - q_high) * 0.1), q_low + (out - q_low) * 0.1)
        return out.to(orig_dtype)

    def _adaptive_health_alignment(self, reference: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        ref_f = reference.float()
        after_f = after.float()

        low_after = self._lowpass_latent(after_f)
        low_ref = self._lowpass_latent(ref_f)
        high_after = after_f - low_after
        high_ref = ref_f - low_ref

        mean_after = low_after.mean(dim=(2, 3), keepdim=True)
        std_after = low_after.std(dim=(2, 3), unbiased=False, keepdim=True).clamp_min(1e-6)
        std_ref = low_ref.std(dim=(2, 3), unbiased=False, keepdim=True).clamp_min(1e-6)

        low_std_aligned = (low_after - mean_after) / (std_after + 1e-6)
        low_std_aligned = low_std_aligned * (std_ref + 1e-6) + mean_after
        low_out = torch.lerp(
            low_after,
            low_std_aligned,
            max(0.0, min(1.0, float(self.adaptive_low_std_strength))),
        )

        high_out = self._align_high_frequency_energy(
            anchor_high=high_ref,
            after_high=high_after,
            strength=float(self.adaptive_high_freq_strength),
        )

        out = low_out.float() + high_out.float()
        residual_anchor = low_after.float() + high_ref.float()
        out = self._clip_residual_mahalanobis(
            anchor=residual_anchor,
            candidate=out,
            strength=float(self.adaptive_residual_strength),
        )
        out = self._soft_prior_clamp(
            reference=ref_f,
            candidate=out.float(),
            strength=float(self.adaptive_prior_strength),
        )
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out.to(dtype=after.dtype)

    @staticmethod
    def _lowpass_latent_with_kernel(x: torch.Tensor, kernel: int) -> torch.Tensor:
        k = int(kernel)
        if k <= 1:
            return x
        if k % 2 == 0:
            k += 1
        pad = k // 2
        x_pad = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="replicate")
        return torch.nn.functional.avg_pool2d(x_pad, kernel_size=k, stride=1)

    @staticmethod
    def _fft_gaussian_lowpass_latent(x: torch.Tensor, sigma: float) -> torch.Tensor:
        _, _, h, w = x.shape
        if h < 2 or w < 2:
            return x

        sigma = max(1e-6, float(sigma))
        x_f = x.float()
        fft = torch.fft.fftshift(torch.fft.fft2(x_f, dim=(-2, -1)), dim=(-2, -1))
        y, x_grid = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=x.device, dtype=torch.float32),
            torch.linspace(-1.0, 1.0, w, device=x.device, dtype=torch.float32),
            indexing="ij",
        )
        radius = torch.sqrt(x_grid.square() + y.square())
        mask = torch.exp(-0.5 * (radius / sigma).square()).view(1, 1, h, w)
        low_fft = fft * mask
        low = torch.fft.ifft2(torch.fft.ifftshift(low_fft, dim=(-2, -1)), dim=(-2, -1)).real
        return low.to(dtype=x.dtype)

    def _lowpass_latent(self, x: torch.Tensor) -> torch.Tensor:
        if self.lowpass_filter == "fft_gaussian":
            return self._fft_gaussian_lowpass_latent(x, self.lowpass_sigma)
        return self._lowpass_latent_with_kernel(x, self.momentum_lowpass_kernel)

    def _coral_align(self, source: torch.Tensor, target: torch.Tensor, strength: float, eps: float = 1e-5) -> torch.Tensor:
        gate = max(0.0, min(1.0, float(strength)))
        if gate <= 0.0:
            return source

        b, c, h, w = source.shape
        s = h * w
        xs = source.float().flatten(2)
        xt = target.float().flatten(2)

        mu_s = xs.mean(dim=-1, keepdim=True)
        mu_t = xt.mean(dim=-1, keepdim=True)
        xs0 = xs - mu_s
        xt0 = xt - mu_t
        denom = float(max(s - 1, 1))

        cov_s = (xs0 @ xs0.transpose(1, 2)) / denom
        cov_t = (xt0 @ xt0.transpose(1, 2)) / denom

        shrink = max(0.0, min(1.0, float(self.soft_coral_shrink)))
        if shrink > 0.0:
            diag_s = torch.diag_embed(torch.diagonal(cov_s, dim1=-2, dim2=-1))
            diag_t = torch.diag_embed(torch.diagonal(cov_t, dim1=-2, dim2=-1))
            cov_s = (1.0 - shrink) * cov_s + shrink * diag_s
            cov_t = (1.0 - shrink) * cov_t + shrink * diag_t

        eye = torch.eye(c, device=source.device, dtype=torch.float32).unsqueeze(0)
        cov_s = cov_s + eps * eye
        cov_t = cov_t + eps * eye

        try:
            eval_s, evec_s = torch.linalg.eigh(cov_s)
            eval_t, evec_t = torch.linalg.eigh(cov_t)
        except Exception:
            mean_s = source.mean(dim=(2, 3), keepdim=True)
            std_s = source.std(dim=(2, 3), keepdim=True).clamp_min(eps)
            mean_t = target.mean(dim=(2, 3), keepdim=True)
            std_t = target.std(dim=(2, 3), keepdim=True).clamp_min(eps)
            aligned = (source - mean_s) / std_s * std_t + mean_t
            return torch.lerp(source, aligned, gate)

        eval_s = eval_s.clamp_min(eps)
        eval_t = eval_t.clamp_min(eps)
        inv_sqrt_s = evec_s @ torch.diag_embed(eval_s.rsqrt()) @ evec_s.transpose(1, 2)
        sqrt_t = evec_t @ torch.diag_embed(eval_t.sqrt()) @ evec_t.transpose(1, 2)

        aligned = sqrt_t @ inv_sqrt_s @ xs0 + mu_t
        aligned = aligned.view(b, c, h, w)
        return torch.lerp(source, aligned.to(dtype=source.dtype), gate)

    def _align_high_frequency_energy(self, anchor_high: torch.Tensor, after_high: torch.Tensor, strength: float) -> torch.Tensor:
        gate = max(0.0, min(1.0, float(strength)))
        if gate <= 0.0:
            return after_high

        _, _, h, w = after_high.shape
        if h < 2 or w < 2:
            return after_high

        after_fft = torch.fft.rfft2(after_high.float(), dim=(-2, -1), norm="ortho")
        anchor_fft = torch.fft.rfft2(anchor_high.float(), dim=(-2, -1), norm="ortho")
        mag_after = after_fft.abs()
        mag_anchor = anchor_fft.abs()

        wf = mag_after.shape[-1]
        fy = torch.fft.fftfreq(h, device=after_high.device, dtype=torch.float32).abs().view(h, 1)
        fx = torch.fft.rfftfreq(w, device=after_high.device, dtype=torch.float32).abs().view(1, wf)
        radius = torch.sqrt(fy * fy + fx * fx)
        max_radius = float(radius.max().clamp_min(1e-6))

        bins = max(2, int(self.soft_freq_bins))
        edges = torch.linspace(0.0, max_radius, bins + 1, device=after_high.device, dtype=torch.float32)
        gain = torch.ones_like(mag_after)

        for i in range(bins):
            if i == bins - 1:
                mask = (radius >= edges[i]) & (radius <= edges[i + 1])
            else:
                mask = (radius >= edges[i]) & (radius < edges[i + 1])
            if not bool(mask.any()):
                continue

            mask_f = mask.float().view(1, 1, h, wf)
            count = mask_f.sum().clamp_min(1.0)
            rms_after = ((mag_after.square() * mask_f).sum(dim=(-2, -1), keepdim=True) / count).sqrt()
            rms_anchor = ((mag_anchor.square() * mask_f).sum(dim=(-2, -1), keepdim=True) / count).sqrt()
            ratio = (rms_anchor / (rms_after + 1e-6)).clamp(0.67, 1.50)
            band_gain = 1.0 + gate * (ratio - 1.0)
            gain = gain + mask_f * (band_gain - 1.0)

        out = torch.fft.irfft2(after_fft * gain, s=(h, w), dim=(-2, -1), norm="ortho")
        return out.to(dtype=after_high.dtype)

    def _clip_residual_mahalanobis(self, anchor: torch.Tensor, candidate: torch.Tensor, strength: float, eps: float = 1e-5) -> torch.Tensor:
        gate = max(0.0, min(1.0, float(strength)))
        if gate <= 0.0:
            return candidate

        _, c, h, w = anchor.shape
        s = h * w
        x = anchor.float().flatten(2)
        x0 = x - x.mean(dim=-1, keepdim=True)
        denom = float(max(s - 1, 1))
        cov = (x0 @ x0.transpose(1, 2)) / denom

        shrink = max(0.0, min(1.0, float(self.soft_residual_cov_shrink)))
        if shrink > 0.0:
            diag = torch.diag_embed(torch.diagonal(cov, dim1=-2, dim2=-1))
            cov = (1.0 - shrink) * cov + shrink * diag

        eye = torch.eye(c, device=anchor.device, dtype=torch.float32).unsqueeze(0)
        cov = cov + eps * eye

        try:
            evals, evecs = torch.linalg.eigh(cov)
        except Exception:
            return candidate

        evals = evals.clamp_min(eps)
        inv_sqrt = evecs @ torch.diag_embed(evals.rsqrt()) @ evecs.transpose(1, 2)
        sqrt_cov = evecs @ torch.diag_embed(evals.sqrt()) @ evecs.transpose(1, 2)

        delta = (candidate.float() - anchor.float()).flatten(2)
        white_delta = inv_sqrt @ delta
        white_anchor = inv_sqrt @ x0

        delta_norm = white_delta.square().sum(dim=1, keepdim=True).sqrt()
        anchor_norm = white_anchor.square().sum(dim=1, keepdim=True).sqrt()
        radius = torch.quantile(anchor_norm, 0.995, dim=-1, keepdim=True)
        radius = radius * float(self.soft_residual_radius_mult)
        radius = radius.clamp_min(float(c) ** 0.5)

        scale = (radius / (delta_norm + eps)).clamp(max=1.0)
        white_delta = white_delta * scale
        clipped = anchor.float() + (sqrt_cov @ white_delta).view_as(anchor.float())
        return torch.lerp(candidate.float(), clipped, gate).to(dtype=candidate.dtype)

    def _soft_prior_clamp(self, reference: torch.Tensor, candidate: torch.Tensor, strength: float) -> torch.Tensor:
        gate = max(0.0, min(1.0, float(strength)))
        if gate <= 0.0:
            return candidate

        lo, hi = self._resolve_prior_bounds(reference)
        width = (hi - lo).clamp_min(1e-6)
        margin = max(0.0, float(self.soft_prior_margin))
        lo = lo - margin * width
        hi = hi + margin * width

        softness = max(1e-6, float(self.soft_prior_softness))
        scale = (hi - lo).clamp_min(1e-6) * softness

        x = candidate.float()
        clamped = lo + torch.nn.functional.softplus((x - lo) / scale) * scale
        clamped = hi - torch.nn.functional.softplus((hi - clamped) / scale) * scale
        return torch.lerp(x, clamped, gate).to(dtype=candidate.dtype)

    def _soft_augmentation_v2(self, before: torch.Tensor, after: torch.Tensor, strength: float = 0.5) -> torch.Tensor:
        gate = max(0.0, min(1.0, float(strength)))
        if gate <= 0.0:
            return after

        before_f = before.float()
        after_f = after.float()

        low_before = self._lowpass_latent(before_f)
        low_after = self._lowpass_latent(after_f)
        low_strength = float(self.soft_low_strength) * gate
        low_aligned = self._coral_align(source=low_after, target=low_before, strength=low_strength)

        high_before = before_f - low_before
        high_after = after_f - low_after
        high_strength = float(self.soft_freq_strength) * gate
        high_aligned = self._align_high_frequency_energy(anchor_high=high_before, after_high=high_after, strength=high_strength)

        out = low_aligned.float() + high_aligned.float()
        residual_strength = float(self.soft_residual_strength) * gate
        out = self._clip_residual_mahalanobis(anchor=before_f, candidate=out, strength=residual_strength)
        out = torch.lerp(after_f, out.float(), gate)
        out = self._soft_prior_clamp(reference=before_f, candidate=out, strength=float(self.soft_prior_strength) * gate)
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out.to(dtype=after.dtype)

    @staticmethod
    def _round_to_multiple(value: int, multiple: int = 16) -> int:
        value = max(int(value), multiple)
        return max(multiple, (value // multiple) * multiple)

    def _resolve_target_size(self, image: Image.Image) -> tuple[int, int]:
        cfg_height = self.config.inference.height
        cfg_width = self.config.inference.width

        if cfg_height is not None and cfg_width is not None:
            return int(cfg_height), int(cfg_width)

        width, height = map(float, image.size)

        if cfg_width is not None:
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

        resolved_height = self._round_to_multiple(int(round(height)))
        resolved_width = self._round_to_multiple(int(round(width)))
        return resolved_height, resolved_width

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
        anchor_latents = self._compose_anchor_latents(current_latents) if current_latents is not None else None

        with torch.inference_mode():
            with self._encode_context(current_latents):
                output = self.pipe(**kwargs)

        latents_out = getattr(output, "images", None)
        if latents_out is None:
            latents_out = output[0] if isinstance(output, tuple) else output

        if isinstance(latents_out, (list, tuple)):
            if len(latents_out) == 0:
                raise RuntimeError("Pipeline output is empty.")
            denoised_latents = latents_out[0]
        else:
            denoised_latents = latents_out

        denoised_latents = denoised_latents.unsqueeze(0) if denoised_latents.ndim == 3 else denoised_latents
        pure_latents = denoised_latents

        if self.use_lpt and current_latents is not None and anchor_latents is not None:
            if self.augmentation == "momentum":
                pure_latents = self._restore_moments_momentum(before=current_latents, after=pure_latents)
            elif self.augmentation == "soft":
                pure_latents = self._soft_augmentation(before=current_latents, after=pure_latents, strength=0.5)

        with torch.no_grad():
            shift = getattr(self.pipe.vae.config, "shift_factor", 0.0)
            scale = getattr(self.pipe.vae.config, "scaling_factor", 1.0)
            unscaled = (pure_latents / scale) + shift
            dec_tensor = self.pipe.vae.decode(unscaled).sample

        tensor = (dec_tensor / 2 + 0.5).clamp(0, 1)
        tensor = tensor.detach().cpu().permute(0, 2, 3, 1).float().numpy()[0]
        vis_image = Image.fromarray((tensor * 255).astype(np.uint8))

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

            if self.use_lpt:
                self._update_anchor_state(accepted_latents)
                current_latents = accepted_latents
            else:
                current_image_pil = vis_image
                current_latents = None

        return RunTrace(sample_id=sample_id, round_images=round_images, collector_outputs={})

    def close(self) -> None:
        pass
