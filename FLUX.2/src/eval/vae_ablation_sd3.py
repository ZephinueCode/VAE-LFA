import argparse
import inspect
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from diffusers import Flux2KleinPipeline, StableDiffusion3Img2ImgPipeline
from diffusers.utils import load_image
from PIL import Image
from tqdm import tqdm


def normalize_model_type(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"sd3", "sd3.5", "sd3_5", "sd35"}:
        return "sd3.5"
    if normalized == "flux2":
        return "flux2"
    raise ValueError(f"Unsupported model type: {value}")


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """PIL 转换为 VAE 输入的 [-1, 1] Tensor"""
    transform = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
    return transform(image).unsqueeze(0)


def postprocess_image(tensor: torch.Tensor) -> Image.Image:
    """VAE 输出 Tensor 转换为 PIL"""
    tensor = (tensor / 2 + 0.5).clamp(0, 1)
    tensor = tensor.cpu().permute(0, 2, 3, 1).float().numpy()[0]
    return Image.fromarray((tensor * 255).astype(np.uint8))


class MockVAEOutput:
    def __init__(self, latents):
        self.latents = latents

    @property
    def latent_dist(self):
        _latents = self.latents
        class Dist:
            def sample(self, *args, **kwargs): return _latents
            def mode(self, *args, **kwargs): return _latents
        return Dist()


def lowpass_filter(x: torch.Tensor, kernel: int = 9) -> torch.Tensor:
    """使用 AvgPool 进行低通滤波"""
    if kernel <= 1:
        return x
    pad = kernel // 2
    x_pad = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="replicate")
    return torch.nn.functional.avg_pool2d(x_pad, kernel_size=kernel, stride=1)


def restore_moments_avg_pool(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    """
    复刻 --momentum (avg_pool) 的核心干涉逻辑
    """
    b_f = before.float()
    a_f = after.float()
    
    low_b = lowpass_filter(b_f, kernel=9)
    low_a = lowpass_filter(a_f, kernel=9)
    high_a = a_f - low_a
    
    mean_b = low_b.mean(dim=(2, 3), keepdim=True)
    std_b = low_b.std(dim=(2, 3), unbiased=False, keepdim=True).clamp_min(1e-6)
    mean_a = low_a.mean(dim=(2, 3), keepdim=True)
    std_a = low_a.std(dim=(2, 3), unbiased=False, keepdim=True).clamp_min(1e-6)
    
    low_aligned = (low_a - mean_a) / std_a * std_b + mean_b
    out = low_aligned + high_a
    
    return torch.nan_to_num(out, nan=0.0, posinf=1e3, neginf=-1e3).to(after.dtype)


def compute_frequency_metrics(delta: torch.Tensor, latents: torch.Tensor, bins: int = 20) -> dict:
    """计算低/高频能量占比，高频方差，径向频谱，以及低频部分的每个 channel 的 mean/std"""
    delta_f = delta.float()
    latents_f = latents.float()
    
    # 1. 拆分与残差能量计算
    delta_low = lowpass_filter(delta_f, kernel=9)
    delta_high = delta_f - delta_low
    
    e_low = float((delta_low ** 2).sum().item())
    e_high = float((delta_high ** 2).sum().item())
    var_high = float(delta_high.var().item())
    e_total = max(e_low + e_high, 1e-12)
    
    # 2. 径向频谱计算
    b, c, h, w = delta_f.shape
    fft_delta = torch.fft.fftshift(torch.fft.fft2(delta_f, dim=(-2, -1)), dim=(-2, -1))
    mag_sq = (fft_delta.abs() ** 2).mean(dim=(0, 1))
    
    Y, X = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing='ij')
    R = torch.sqrt(X**2 + Y**2).to(mag_sq.device)
    max_radius = float(R.max().item())
    edges = torch.linspace(0, max_radius, bins + 1, device=mag_sq.device)
    
    spectrum = []
    for i in range(bins):
        mask = (R >= edges[i]) & (R <= edges[i + 1])
        val = float(mag_sq[mask].mean().item()) if mask.any() else 0.0
        spectrum.append(val)
        
    # 3. 低频部分的真实 Mean 和 Std 漂移追踪（按通道）- 绝对潜变量
    low_latents = lowpass_filter(latents_f, kernel=9)
    mean_channels = low_latents.mean(dim=(2, 3)).squeeze(0).cpu().tolist()  # shape [C]
    std_channels = low_latents.std(dim=(2, 3), unbiased=False).squeeze(0).cpu().tolist()  # shape [C]
    
    # 4. 【新增】轮间差值(delta)的低频部分 per-channel 一阶矩/二阶矩
    # 这是核心证据：能量没变，但均值被压平了 -> 证明是空间重排而非全局偏移
    delta_mean_channels = delta_low.mean(dim=(2, 3)).squeeze(0).cpu().tolist()
    delta_std_channels = delta_low.std(dim=(2, 3), unbiased=False).squeeze(0).cpu().tolist()
        
    return {
        "e_low": e_low,
        "e_high": e_high,
        "var_high": var_high,
        "ratio_low": e_low / e_total,
        "ratio_high": e_high / e_total,
        "spectrum": spectrum,
        "mean_channels": mean_channels,
        "std_channels": std_channels,
        "delta_mean_channels": delta_mean_channels,
        "delta_std_channels": delta_std_channels,
    }


def aggregate_metrics(all_sample_metrics: list[dict]) -> dict:
    """对所有图片的数据按 round 求均值"""
    if not all_sample_metrics:
        return {}
    
    num_rounds = len(all_sample_metrics[0])
    mean_metrics = []
    
    for r in range(num_rounds):
        r_metrics = [sample[r] for sample in all_sample_metrics]
        
        e_low_mean = np.mean([m["e_low"] for m in r_metrics])
        e_high_mean = np.mean([m["e_high"] for m in r_metrics])
        var_high_mean = np.mean([m["var_high"] for m in r_metrics])
        r_low_mean = np.mean([m["ratio_low"] for m in r_metrics])
        r_high_mean = np.mean([m["ratio_high"] for m in r_metrics])
        
        spec_array = np.array([m["spectrum"] for m in r_metrics])
        spec_mean = np.mean(spec_array, axis=0).tolist()
        
        # 聚合通道级的 mean/std - 绝对潜变量
        mean_ch_array = np.array([m["mean_channels"] for m in r_metrics])
        std_ch_array = np.array([m["std_channels"] for m in r_metrics])
        mean_ch_mean = np.mean(mean_ch_array, axis=0).tolist()
        std_ch_mean = np.mean(std_ch_array, axis=0).tolist()
        
        # 【新增】聚合 delta 的通道级一阶矩/二阶矩
        delta_mean_ch_array = np.array([m["delta_mean_channels"] for m in r_metrics])
        delta_std_ch_array = np.array([m["delta_std_channels"] for m in r_metrics])
        delta_mean_ch_mean = np.mean(delta_mean_ch_array, axis=0).tolist()
        delta_std_ch_mean = np.mean(delta_std_ch_array, axis=0).tolist()
        
        mean_metrics.append({
            "round": r + 1,
            "e_low": float(e_low_mean),
            "e_high": float(e_high_mean),
            "var_high": float(var_high_mean),
            "ratio_low": float(r_low_mean),
            "ratio_high": float(r_high_mean),
            "spectrum": spec_mean,
            "mean_channels": mean_ch_mean,
            "std_channels": std_ch_mean,
            "delta_mean_channels": delta_mean_ch_mean,
            "delta_std_channels": delta_std_ch_mean,
        })
    return mean_metrics


def plot_results(metrics_A: list[dict], metrics_B: list[dict], outdir: Path, b_label="Exp B (DiT)"):
    """绘制所有结果图表"""
    rounds = [m["round"] for m in metrics_A] if metrics_A else [m["round"] for m in metrics_B]
    if not rounds:
        return

    # -------- 原有图表 1~4 --------
    plt.figure(figsize=(10, 5))
    if metrics_A:
        plt.plot(rounds, [m["e_low"] for m in metrics_A], 'b--', label="Exp A (VAE) - Low Freq")
        plt.plot(rounds, [m["e_high"] for m in metrics_A], 'b-', label="Exp A (VAE) - High Freq")
    if metrics_B:
        plt.plot(rounds, [m["e_low"] for m in metrics_B], 'r--', label=f"{b_label} - Low Freq")
        plt.plot(rounds, [m["e_high"] for m in metrics_B], 'r-', label=f"{b_label} - High Freq")
    plt.yscale('log')
    plt.xlabel("Editing Round")
    plt.ylabel("Absolute Energy (Log)")
    plt.title(f"Absolute Energy Evolution: VAE vs {b_label}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(outdir / "energy_absolute.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    if metrics_A:
        plt.plot(rounds, [m["ratio_low"] for m in metrics_A], 'b--', label="Exp A (VAE) - Low Freq Ratio")
        plt.plot(rounds, [m["ratio_high"] for m in metrics_A], 'b-', label="Exp A (VAE) - High Freq Ratio")
    if metrics_B:
        plt.plot(rounds, [m["ratio_low"] for m in metrics_B], 'r--', label=f"{b_label} - Low Freq Ratio")
        plt.plot(rounds, [m["ratio_high"] for m in metrics_B], 'r-', label=f"{b_label} - High Freq Ratio")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Editing Round")
    plt.ylabel("Energy Ratio")
    plt.title(f"Energy Distribution Ratio: VAE vs {b_label}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(outdir / "energy_ratio.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    bins = len(metrics_A[-1]["spectrum"]) if metrics_A else len(metrics_B[-1]["spectrum"])
    x_axis = np.linspace(0, 1.414, bins)
    if metrics_A:
        plt.plot(x_axis, metrics_A[-1]["spectrum"], 'b-', label="Exp A (VAE) - Final Round Spectrum")
    if metrics_B:
        plt.plot(x_axis, metrics_B[-1]["spectrum"], 'r-', label=f"{b_label} - Final Round Spectrum")
    plt.yscale('log')
    plt.xlabel("Frequency Radius (r)")
    plt.ylabel("Power P(r) (Log)")
    plt.title("Radial Spectrum P(r) of Latent Residuals (Final Round)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(outdir / "radial_spectrum.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    if metrics_A:
        plt.plot(rounds, [m["var_high"] for m in metrics_A], 'b-', marker='o', label="Exp A (VAE) - High Freq Variance")
    if metrics_B:
        plt.plot(rounds, [m["var_high"] for m in metrics_B], 'r-', marker='o', label=f"{b_label} - High Freq Variance")
    plt.yscale('log')
    plt.xlabel("Editing Round")
    plt.ylabel("High Freq Variance (Log)")
    plt.title(f"High Frequency Variance Evolution: VAE vs {b_label}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(outdir / "variance_high_freq.png", dpi=150)
    plt.close()

    # -------- Plot 5: 绝对潜变量的 Low-Frequency Mean Drift --------
    plt.figure(figsize=(12, 6))
    num_channels = len(metrics_A[0]["mean_channels"]) if metrics_A else len(metrics_B[0]["mean_channels"])
    cmap = plt.get_cmap('hsv')
    colors = [cmap(i / num_channels) for i in range(num_channels)]
    
    for c in range(num_channels):
        if metrics_A:
            vals_A = [m["mean_channels"][c] for m in metrics_A]
            plt.plot(rounds, vals_A, linestyle='--', color=colors[c], alpha=0.4, label='Exp A (VAE)' if c == 0 else "")
        if metrics_B:
            vals_B = [m["mean_channels"][c] for m in metrics_B]
            plt.plot(rounds, vals_B, linestyle='-', color=colors[c], alpha=0.7, label=b_label if c == 0 else "")
            
    plt.xlabel("Editing Round")
    plt.ylabel("Low Freq Mean (per channel)")
    plt.title(f"Low Frequency Mean Drift per Channel (All {num_channels} Channels)")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.grid(True, alpha=0.3)
    plt.savefig(outdir / "low_freq_mean_drift.png", dpi=150)
    plt.close()

    # -------- Plot 6: 绝对潜变量的 Low-Frequency Std Drift --------
    plt.figure(figsize=(12, 6))
    for c in range(num_channels):
        if metrics_A:
            vals_A = [m["std_channels"][c] for m in metrics_A]
            plt.plot(rounds, vals_A, linestyle='--', color=colors[c], alpha=0.4, label='Exp A (VAE)' if c == 0 else "")
        if metrics_B:
            vals_B = [m["std_channels"][c] for m in metrics_B]
            plt.plot(rounds, vals_B, linestyle='-', color=colors[c], alpha=0.7, label=b_label if c == 0 else "")
            
    plt.xlabel("Editing Round")
    plt.ylabel("Low Freq Std (per channel)")
    plt.title(f"Low Frequency Std (Contrast) Evolution per Channel (All {num_channels} Channels)")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.grid(True, alpha=0.3)
    plt.savefig(outdir / "low_freq_std_drift.png", dpi=150)
    plt.close()

    # -------- 【新增 Plot 7】轮间差值 Delta 的 Low-Frequency Mean --------
    # 核心证据：能量没变，但差值的均值被压到 0 附近 -> 空间重排而非全局偏移
    plt.figure(figsize=(12, 6))
    num_channels = len(metrics_A[0]["delta_mean_channels"]) if metrics_A else len(metrics_B[0]["delta_mean_channels"])
    colors = [cmap(i / num_channels) for i in range(num_channels)]
    
    for c in range(num_channels):
        if metrics_A:
            vals_A = [m["delta_mean_channels"][c] for m in metrics_A]
            plt.plot(rounds, vals_A, linestyle='--', color=colors[c], alpha=0.4, label='Exp A (VAE)' if c == 0 else "")
        if metrics_B:
            vals_B = [m["delta_mean_channels"][c] for m in metrics_B]
            plt.plot(rounds, vals_B, linestyle='-', color=colors[c], alpha=0.7, label=b_label if c == 0 else "")
            
    plt.axhline(0, color='black', linewidth=1.0, linestyle=':', alpha=0.6, label='Zero bias')
    plt.xlabel("Editing Round")
    plt.ylabel("Inter-turn Delta Mean (Low Freq, per channel)")
    plt.title(f"Inter-turn Delta Mean (Low Freq) per Channel (All {num_channels} Channels)\nEvidence of spatial rearrangement without global bias")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.ylim(-0.5, 0.5)
    plt.legend(by_label.values(), by_label.keys())
    plt.grid(True, alpha=0.3)
    plt.savefig(outdir / "delta_low_freq_mean_drift.png", dpi=150)
    plt.close()

    # -------- 【新增 Plot 8】轮间差值 Delta 的 Low-Frequency Std --------
    plt.figure(figsize=(12, 6))
    for c in range(num_channels):
        if metrics_A:
            vals_A = [m["delta_std_channels"][c] for m in metrics_A]
            plt.plot(rounds, vals_A, linestyle='--', color=colors[c], alpha=0.4, label='Exp A (VAE)' if c == 0 else "")
        if metrics_B:
            vals_B = [m["delta_std_channels"][c] for m in metrics_B]
            plt.plot(rounds, vals_B, linestyle='-', color=colors[c], alpha=0.7, label=b_label if c == 0 else "")
            
    plt.xlabel("Editing Round")
    plt.ylabel("Inter-turn Delta Std (Low Freq, per channel)")
    plt.title(f"Inter-turn Delta Std (Low Freq) per Channel (All {num_channels} Channels)")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.grid(True, alpha=0.3)
    plt.savefig(outdir / "delta_low_freq_std_drift.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["flux2", "sd3.5", "sd3_5", "sd35", "sd3"],
        default="flux2",
        help="Model backend",
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="", 
        help="Comma-separated prompts for each round. If fewer than rounds, the last prompt is repeated."
    )
    parser.add_argument("--rounds", type=int, default=8, help="Number of loops")
    parser.add_argument("--outdir", type=str, default="./vae_ablation_out")
    parser.add_argument("--experiment", type=str, choices=["A", "B", "all"], default="all")
    parser.add_argument(
        "--interfere", 
        action="store_true", 
        help="If set, apply the momentum (avg_pool) constraint to DiT latents in Exp B."
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Number of inference steps. Defaults to 4 for flux2 and 28 for sd3.5.",
    )
    args = parser.parse_args()
    model_type = normalize_model_type(args.model_type)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(args.input_dir)
    image_paths = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg"))
    if not image_paths:
        raise ValueError(f"No images found in {args.input_dir}")

    # 解析 Prompts
    prompts_list = [p.strip() for p in args.prompt.split(",")]
    if len(prompts_list) == 0 or (len(prompts_list) == 1 and prompts_list[0] == ""):
        prompts_list = [""] * args.rounds
    elif len(prompts_list) < args.rounds:
        prompts_list.extend([prompts_list[-1]] * (args.rounds - len(prompts_list)))

    print("Loading pipeline...")
    if model_type == "flux2":
        pipe_cls = Flux2KleinPipeline
        default_steps = 4
    else:
        pipe_cls = StableDiffusion3Img2ImgPipeline
        default_steps = 28
    num_inference_steps = args.num_inference_steps if args.num_inference_steps is not None else default_steps

    pipe = pipe_cls.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    generator = torch.Generator(device=pipe.device).manual_seed(42)

    vae_shift = getattr(pipe.vae.config, "shift_factor", 0.0)
    vae_scale = getattr(pipe.vae.config, "scaling_factor", 1.0)

    call_sig = inspect.signature(pipe.__call__)
    image_key = next((k for k in ("image", "init_image", "input_image") if k in call_sig.parameters), None)
    if not image_key:
        raise RuntimeError("Cannot detect image input key.")

    all_metrics_A = []
    all_metrics_B = []

    original_encode = pipe.vae.encode

    print(f"Found {len(image_paths)} images. Starting ablation...")

    for img_idx, img_path in enumerate(image_paths):
        print(f"\n[{img_idx+1}/{len(image_paths)}] Processing {img_path.name}...")
        init_image = load_image(str(img_path)).convert("RGB").resize((1024, 1024))
        
        sample_outdir = outdir / img_path.stem
        sample_outdir.mkdir(exist_ok=True)

        # ---------------------------------------------------------
        # Experiment A: VAE Loop
        # ---------------------------------------------------------
        if args.experiment in ["A", "all"]:
            metrics_A = []
            latents_prev = pipe.vae.encode(preprocess_image(init_image).to(pipe.device, pipe.dtype)).latent_dist.sample()
            
            for i in tqdm(range(args.rounds), desc="Exp A (VAE)"):
                with torch.no_grad():
                    dec_tensor = pipe.vae.decode(latents_prev).sample
                current_img = postprocess_image(dec_tensor)
                if i == args.rounds - 1 or i == 0:
                    current_img.save(sample_outdir / f"ExpA_R{i+1:02d}.png")
                
                with torch.no_grad():
                    latents_next = pipe.vae.encode(preprocess_image(current_img).to(pipe.device, pipe.dtype)).latent_dist.sample()
                
                delta = latents_next - latents_prev
                # 传入当前轮次产出的绝对潜变量，用于计算 mean/std 漂移
                metrics_A.append(compute_frequency_metrics(delta, latents_next))
                latents_prev = latents_next
                
            all_metrics_A.append(metrics_A)

        # ---------------------------------------------------------
        # Experiment B: DiT Loop (Bypassing VAE, Optional Interfere)
        # ---------------------------------------------------------
        if args.experiment in ["B", "all"]:
            metrics_B = []
            b_label_tqdm = "Exp B (DiT+Interfere)" if args.interfere else "Exp B (DiT)"
            
            with torch.no_grad():
                raw_unscaled = pipe.vae.encode(preprocess_image(init_image).to(pipe.device, pipe.dtype)).latent_dist.sample()
            raw_latents = (raw_unscaled - vae_shift) * vae_scale  # scaled latent for SD3/Flux scheduler space

            # For SD3, bypass image_processor.preprocess so latent tensor passes through
            orig_preprocess = pipe.image_processor.preprocess
            if model_type == "sd3.5":
                def bypass_preprocess(image, height=None, width=None):
                    if isinstance(image, torch.Tensor) and image.shape[1] == pipe.vae.config.latent_channels:
                        return image
                    return orig_preprocess(image, height=height, width=width)
                pipe.image_processor.preprocess = bypass_preprocess

            for i in tqdm(range(args.rounds), desc=b_label_tqdm):
                current_prompt = prompts_list[i]
                
                def mock_encode_fn(x, **kwargs): return MockVAEOutput(raw_latents)
                pipe.vae.encode = mock_encode_fn
                
                kwargs = {
                    "prompt": current_prompt,
                    "height": 1024,
                    "width": 1024,
                    "num_inference_steps": num_inference_steps,
                    "generator": generator,
                    "output_type": "latent",
                }
                if "strength" in call_sig.parameters:
                    kwargs["strength"] = 0.8

                if model_type == "sd3.5":
                    kwargs[image_key] = raw_latents
                else:
                    kwargs[image_key] = init_image
                    
                out = pipe(**kwargs)
                
                latents_out = getattr(out, "images", None)
                if latents_out is None:
                    latents_out = out[0] if isinstance(out, tuple) else out
                    
                if isinstance(latents_out, (list, tuple)):
                    denoised_latents = latents_out[0]
                else:
                    denoised_latents = latents_out
                    
                denoised_latents = denoised_latents.unsqueeze(0) if denoised_latents.ndim == 3 else denoised_latents
                
                if args.interfere:
                    denoised_latents = restore_moments_avg_pool(before=raw_latents, after=denoised_latents)
                
                delta = denoised_latents - raw_latents
                metrics_B.append(compute_frequency_metrics(delta, denoised_latents))
                
                raw_latents = denoised_latents
                
                if i == args.rounds - 1 or i == 0:
                    with torch.no_grad():
                        decode_input = (raw_latents / vae_scale) + vae_shift
                        dec_tensor = pipe.vae.decode(decode_input).sample
                        temp_img = postprocess_image(dec_tensor)
                        prefix = "ExpB_Interfere" if args.interfere else "ExpB"
                        temp_img.save(sample_outdir / f"{prefix}_R{i+1:02d}.png")
                        
            pipe.vae.encode = original_encode
            if model_type == "sd3.5":
                pipe.image_processor.preprocess = orig_preprocess
            all_metrics_B.append(metrics_B)

    # ---------------------------------------------------------
    # Aggregate, Save & Plot
    # ---------------------------------------------------------
    print("\nAggregating metrics across all samples...")
    mean_A = aggregate_metrics(all_metrics_A)
    mean_B = aggregate_metrics(all_metrics_B)

    with open(outdir / "frequency_analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump({"ExpA_VAE": mean_A, "ExpB_DiT": mean_B}, f, indent=2)

    b_label = "Exp B (DiT+Interfere)" if args.interfere else "Exp B (DiT)"
    plot_results(mean_A, mean_B, outdir, b_label=b_label)
    print(f"Done! Results and plots saved to {outdir}")


if __name__ == "__main__":
    main()
