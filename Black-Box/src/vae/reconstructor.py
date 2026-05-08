from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, AutoencoderTiny
from PIL import Image


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(str(dtype_name).strip().lower(), torch.float32)


def load_diffusers_vae(model_type: str, model_path: Path, dtype_name: str) -> torch.nn.Module:
    torch_dtype = resolve_torch_dtype(dtype_name)
    if not torch.cuda.is_available() and torch_dtype != torch.float32:
        torch_dtype = torch.float32

    kind = model_type.strip().lower()
    if kind == "autoencoder_tiny":
        vae = AutoencoderTiny.from_pretrained(str(model_path), torch_dtype=torch_dtype, local_files_only=True)
    elif kind == "autoencoder_kl":
        vae = AutoencoderKL.from_pretrained(str(model_path), torch_dtype=torch_dtype, local_files_only=True)
    else:
        raise ValueError("`model_type` must be `autoencoder_tiny` or `autoencoder_kl`.")
    return vae


def lowpass_filter(x: torch.Tensor, kernel: int = 9) -> torch.Tensor:
    """使用 AvgPool 进行低通滤波，提取宏观流形"""
    if kernel <= 1:
        return x
    pad = kernel // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    return F.avg_pool2d(x_pad, kernel_size=kernel, stride=1)


class VAEReconstructor:
    def __init__(self, vae_config: Any, intervention_config: Any):
        self.vae_config = vae_config
        self.intervention_config = intervention_config
        self._torch = torch
        
        self.device = "cuda" if torch.cuda.is_available() and vae_config.device in ["cuda", "auto"] else "cpu"
        self.dtype = resolve_torch_dtype(vae_config.dtype)
        
        # 加载分布外的代理 VAE
        self.vae = load_diffusers_vae(vae_config.model_type, vae_config.model_path, vae_config.dtype)
        self.vae.to(self.device, self.dtype)
        self.vae.eval()
        self.vae.requires_grad_(False)
        
        self.inference_backend = "vae_momentum" if getattr(intervention_config, "enabled", False) else "vae"
        
        # 动量状态：只追踪低频分布的统计量
        self._momentum_mean: torch.Tensor | None = None
        self._momentum_log_std: torch.Tensor | None = None

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        x = self._torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return x.to(device=self.device, dtype=self.dtype) * 2.0 - 1.0

    def _tensor_to_image(self, x: torch.Tensor) -> Image.Image:
        x = (x / 2.0 + 0.5).clamp(0.0, 1.0)
        arr = x.squeeze(0).permute(1, 2, 0).detach().float().cpu().numpy()
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.vae.encode(x)
        scale = float(getattr(self.vae.config, "scaling_factor", 1.0))
        if hasattr(encoded, "latent_dist"):
            return encoded.latent_dist.sample() * scale
        elif hasattr(encoded, "latents"):
            return encoded.latents * scale
        raise RuntimeError("Unexpected VAE encode output format.")

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        scale = float(getattr(self.vae.config, "scaling_factor", 1.0))
        decoded = self.vae.decode(z / max(scale, 1e-8))
        if hasattr(decoded, "sample"):
            return decoded.sample
        if isinstance(decoded, tuple) and decoded:
            return decoded[0]
        raise RuntimeError("Unexpected VAE decode output format.")

    def reset_momentum(self):
        """清空动量状态"""
        self._momentum_mean = None
        self._momentum_log_std = None

    def init_momentum(self, image: Image.Image):
        """使用初始图片预热动量状态（仅作为首个锚点，后续会被 EMA 冲淡）"""
        with self._torch.no_grad():
            x = self._image_to_tensor(image)
            z = self.encode_latent(x)
            kernel = getattr(self.intervention_config, "kernel_size", 9)
            low_z = lowpass_filter(z.float(), kernel=kernel)
            
            self._momentum_mean = low_z.mean(dim=(2, 3), keepdim=True).detach()
            self._momentum_log_std = low_z.std(dim=(2, 3), unbiased=False, keepdim=True).clamp_min(1e-6).log().detach()

    def _apply_momentum(self, z: torch.Tensor) -> torch.Tensor:
        """核心干涉：执行低频动量对齐并更新 EMA"""
        if self._momentum_mean is None or self._momentum_log_std is None:
            return z

        kernel = getattr(self.intervention_config, "kernel_size", 9)
        mean_decay = getattr(self.intervention_config, "mean_decay", 0.85)
        std_decay = getattr(self.intervention_config, "std_decay", 0.85)
        
        current = z.float()
        
        # 1. 拆分当前低高频
        low_curr = lowpass_filter(current, kernel=kernel)
        high_curr = current - low_curr
        
        # 2. 计算当前观察到的低频统计量
        mean_curr = low_curr.mean(dim=(2, 3), keepdim=True)
        std_curr = low_curr.std(dim=(2, 3), unbiased=False, keepdim=True).clamp_min(1e-6)
        
        # 3. 提取当前的 EMA 动量目标
        target_mean = self._momentum_mean.to(device=z.device, dtype=torch.float32)
        target_std = self._momentum_log_std.to(device=z.device, dtype=torch.float32).exp()
        
        # 4. 对齐流形：把现在的漂移拉回前 k 轮的平均水位，保留高频细节
        low_aligned = (low_curr - mean_curr) / std_curr * target_std + target_mean
        z_out = low_aligned + high_curr
        z_out = torch.nan_to_num(z_out, nan=0.0, posinf=1e3, neginf=-1e3).to(z.dtype)
        
        # 5. 更新动量 (EMA)：将新的一轮吸收进锚点，逐步忘掉太久远的过去
        self._momentum_mean = mean_decay * self._momentum_mean + (1.0 - mean_decay) * mean_curr.detach()
        self._momentum_log_std = std_decay * self._momentum_log_std + (1.0 - std_decay) * std_curr.log().detach()
        
        return z_out

    def reconstruct(self, image: Image.Image) -> Image.Image:
        with self._torch.no_grad():
            x = self._image_to_tensor(image)
            z = self.encode_latent(x)
            
            if getattr(self.intervention_config, "enabled", False):
                z = self._apply_momentum(z)
                
            image_tensor = self.decode_latent(z)
        return self._tensor_to_image(image_tensor)