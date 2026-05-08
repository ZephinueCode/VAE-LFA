# VAE-LFA: SD3-UE White-Box Evaluation

This directory implements the **white-box** variant of VAE-LFA for Stable Diffusion 3 (SD3) post-trained on UltraEdit (SD3-UE). Because the internal VAE and DiT are exposed, VAE-LFA is seamlessly integrated into the editing pipeline by eliminating redundant VAE encode–decode operations and aligning low-frequency latent statistics across rounds.

## Overview

- **Latent Pass-Through (LPT):** The pipeline caches the latent tensor between editing rounds and patches the VAE `encode` to return it directly, avoiding repeated encode–decode noise.
- **Low-Frequency Momentum Alignment:** After each DiT denoising step, the latent is split into low/high frequencies via average-pooling-based low-pass filtering. The low-frequency mean and std are aligned to an exponential moving average (EMA) of previous rounds. This suppresses accumulated semantic drift while leaving high-frequency details unconstrained.
- **Training-free:** No retraining, no ground-truth priors, and no modification to the underlying SD3-UE weights.

## Installation

This setup intentionally uses a **separate environment** because it installs a forked `diffusers` from the local UltraEdit checkout.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The requirements install:

```text
-e ./third_party/UltraEdit/diffusers
```

so this environment does not conflict with the main project's `diffusers` installation.

**PyTorch note:** The requirements pin `torch==2.3.0` with CUDA 11.8. If your system CUDA differs, adjust accordingly.

## Dataset Format

Each sample is one image paired with one JSON file of the same stem:

```text
data/creature/
  001.png
  001.json
  002.png
  002.json
```

Example JSON:

```json
{
  "image": "001.png",
  "category": "creature",
  "cycle": ["prompt 1", "prompt 2"],
  "long_chain": ["prompt 1", "prompt 2"]
}
```

Supported `--eval` modes:

- `noop`: automatically repeats `Keep the image unchanged.`
- `cycle`: uses the per-sample `cycle` prompt list from JSON
- `long_chain`: uses the per-sample `long_chain` prompt list from JSON

## Usage

### Baseline (no VAE-LFA)

```bash
python run_ultraedit_metrics.py \
  --config config.ultraedit.yaml \
  --eval cycle \
  --latent false
```

### VAE-LFA enabled

```bash
python run_ultraedit_metrics.py \
  --config config.ultraedit.yaml \
  --eval cycle \
  --latent true \
  --augmentation momentum
```

Other supported eval modes:

- `--eval noop`
- `--eval long_chain`

### Key Options

| Option | Description |
|--------|-------------|
| `--latent true` | Enable latent pass-through (bypass VAE round-trips) |
| `--augmentation momentum` | Low-frequency EMA alignment (VAE-LFA) |
| `--lowpass-filter avg_pool` | AvgPool-based low-pass filter (default) |
| `--lowpass-filter fft_gaussian` | FFT Gaussian low-pass filter |
| `--lowpass-sigma 0.25` | Gaussian sigma for FFT filter |

Auto-size is enabled by default:

- if `height` and `width` are `null`, the original size is used
- if the long edge exceeds `1280`, it is downscaled
- the final size snaps to a multiple of `16`

**Note:** `--strength` is accepted for CLI compatibility but ignored by UltraEdit. `--augmentation soft` is also accepted for compatibility; when `--latent true`, only `momentum` is supported.

## Configuration (`config.ultraedit.yaml`)

| Section | Key | Description |
|---------|-----|-------------|
| `model` | `model_path` | Path to the SD3-UE checkpoint |
| `model` | `torch_dtype` | `float16` or `bfloat16` |
| `model` | `device` | `cuda` or `cpu` |
| `data` | `input_path` | Dataset root directory |
| `data` | `max_samples` | Limit number of samples (optional) |
| `inference` | `guidance_scale` | CFG scale (default: 7.5) |
| `inference` | `image_guidance_scale` | Image CFG scale (default: 1.5) |
| `inference` | `num_inference_steps` | Denoising steps (default: 28) |
| `inference` | `free_form_mask` | Must be `true` for free-form editing |

## Outputs

Each run writes to `results/<run_name>/`:

- `config_snapshot.json`
- `manifest.json`
- `metrics/metrics_summary.json`
- `metrics/metrics_by_group_and_round.csv`
- `metrics/metrics_mean_l1_ssim_lpips.png`
- per-sample images under `samples/<group>_<sample_id>/`

## Metrics

Per-round metrics are computed relative to the original input image:

- **L1 Drift** (lower is better)
- **SSIM** (higher is better)
- **LPIPS** (lower is better)

Aggregated summaries are available in `metrics_summary.json` with both `overall` and `by_group` breakdowns.

## Quick Batch Scripts

```bash
bash run.sh
```

This runs the four standard ablations:

1. No-op baseline
2. No-op + VAE-LFA
3. Cycle baseline
4. Cycle + VAE-LFA

## Project Structure

```text
SD3-UE/
  run_ultraedit_metrics.py      # Standalone runner with latent pass-through & momentum
  config.ultraedit.yaml         # Model & inference config
  requirements.txt              # Isolated deps including forked diffusers
  third_party/UltraEdit/        # Local UltraEdit repo (forked diffusers)
```
