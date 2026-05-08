# VAE-LFA: FLUX.2 White-Box Evaluation

This directory implements the **white-box** variant of VAE-LFA for FLUX.2 Klein 9B. Because internal VAE and DiT components are fully exposed, VAE-LFA is injected directly into the pipeline by bypassing redundant VAE encode–decode cycles and aligning low-frequency latent statistics across editing rounds.

## Overview

- **Latent Pass-Through (LPT):** Instead of decoding latents to pixels and re-encoding every round, the pipeline keeps the latent tensor alive between turns. The VAE `encode` is patched to return the cached latent directly, eliminating round-trip noise.
- **Low-Frequency Momentum Alignment:** After each DiT denoising step, the latent is decomposed into low/high frequencies via average-pooling-based low-pass filtering. The low-frequency mean and std are aligned to an exponential moving average (EMA) of previous rounds, suppressing DiT-induced semantic drift while preserving high-frequency editing details.
- **Training-free:** No retraining, no ground-truth priors, and no access to diffusion parameters beyond the public pipeline.

## Installation

```bash
pip install -r requirements.txt
```

**Note:** `torch==2.8.0` is pinned. Adjust the wheel to match your CUDA version if necessary.

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

- `noop`: automatically repeats `Keep the image unchanged.` for the sample's JSON-defined round count
- `cycle`: uses the per-sample `cycle` prompt list from JSON
- `long_chain`: uses the per-sample `long_chain` prompt list from JSON

`data.input_path` can point to:

- a directory of paired image/JSON samples
- a single image file
- a single JSON file

## Usage

### Baseline (no VAE-LFA)

```bash
python -m src \
  --config config.example.yaml \
  --model-type flux2 \
  --eval cycle \
  --latent false
```

### VAE-LFA enabled (default configuration)

```bash
python -m src \
  --config config.example.yaml \
  --model-type flux2 \
  --eval cycle \
  --latent true \
  --augmentation momentum
```

You can switch to:

- `--model-type kandinsky5`
- `--eval noop`
- `--eval long_chain`

### Key Options

| Option | Description |
|--------|-------------|
| `--latent true` | Enable latent pass-through (bypass VAE round-trips) |
| `--augmentation momentum` | Low-frequency EMA alignment (VAE-LFA) |
| `--augmentation soft` | Softer frequency-wise alignment (alternative ablation) |
| `--lowpass-filter avg_pool` | AvgPool-based low-pass filter (default) |
| `--lowpass-filter fft_gaussian` | FFT Gaussian low-pass filter |
| `--lowpass-sigma 0.25` | Gaussian sigma for FFT filter |
| `--strength 0.8` | Image-to-image strength for DiT editing |

If `inference.height` and `inference.width` are both `null`, the pipeline resolves size automatically:

- `flux2`: uses the original image size, caps the long edge at `1280`, then snaps to a multiple of `16`
- `kandinsky5`: selects the nearest supported Kandinsky 5.0 bucket based on the resized source aspect ratio

## Outputs

Each run writes to `output.output_dir/<run_name>/`:

- `config_snapshot.json`
- `manifest.json`
- `metrics/metrics_summary.json`
- `metrics/metrics_by_group_and_round.csv`
- `metrics/metrics_mean_l1_ssim_lpips.png`
- per-sample images and traces under `samples/<sample_id>/`

## Metrics

The main evaluation computes per round, relative to the original input image:

- **L1 Drift** (lower is better)
- **SSIM** (higher is better)
- **LPIPS** (lower is better)

`metrics_summary.json` contains both:

- `overall`: all samples aggregated together by round
- `by_group`: per-folder aggregated results such as `creature`, `architecture`, `scenery`

## Frequency Analysis Scripts

Controlled ablations for the VAE-only vs. DiT-only drift analysis reported in the paper:

```bash
bash run.sh   # VAE-only loop (Fig. radial spectrum)
bash run_lc.sh   # Latent-only loop
```

Corresponding analysis code:

- `src/eval/freq.py`: radial power spectrum and iterative frequency breakdown
- `src/eval/vae_ablation.py`: FLUX.2 VAE-only ablation
- `src/eval/vae_ablation_sd3.py`: SD3 VAE-only ablation (cross-architecture validation)

## Project Structure

```text
FLUX.2/
  src/
    model/model.py              # FLUX.2 runner with latent pass-through & momentum alignment
    model/model_kandinsky5.py   # Kandinsky 5.0 runner (same LPT path)
    eval/metrics.py             # LPIPS / SSIM / L1 aggregation
    eval/freq.py                # Latent-space frequency analysis
    eval/vae_ablation.py        # VAE-only loop ablation
  config.example.yaml           # Inference & model config
  run.sh / run_lc.sh            # Quick ablation scripts
```
