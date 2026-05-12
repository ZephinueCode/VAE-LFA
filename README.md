# VAE-LFA: Plug-and-Play Low Frequency Alignment for Enhanced Multi-Turn Editing

Official implementation of **VAE-LFA**, a training-free, plug-and-play method that suppresses progressive semantic drift in multi-turn diffusion transformer (DiT) image editing by aligning low-frequency statistics in VAE latent space.

> **TL;DR:** DiT-based editors degrade over multiple editing rounds. VAE-LFA decomposes latent discrepancies via low-pass filtering and aligns low-frequency statistics to an exponential moving average (EMA) of previous rounds—no retraining, no ground-truth priors, and no internal model access required.

## Updates

- May.12 2026: Our paper is released on [Arxiv](https://arxiv.org/abs/2605.08250)!
- May.8 2026: Code base is released.

## Overview

Recent DiT-based editors (FLUX.2, SD3-UE, Qwen Image 2.0, Seedream 4.0) achieve impressive single-turn editing, but suffer from progressive semantic drift and quality degradation under multi-turn (5+) iterative editing.

VAE-LFA operates from a **latent-space frequency perspective**:

- **White-box:** Directly injected into the pipeline by bypassing redundant VAE encode–decode loops and aligning low-frequency latents between DiT denoising steps.
- **Black-box:** Interleaves an off-the-shelf VAE between API-based editing rounds to perform inter-round latent alignment.

### Key Features

- **Training-free:** No fine-tuning or retraining of the base editor.
- **Plug-and-play:** Works with both white-box and black-box DiT editors.
- **Frequency-aware:** Only constrains low-frequency components (semantic structure), leaving high-frequency details untouched.
- **EMA anchor:** Uses exponential moving average of historical low-frequency statistics for adaptive yet stable alignment.

## Repository Structure

```
.
├── Black-Box/          # Black-box API-based editor evaluation
│   ├── src/
│   │   ├── api/image_client.py      # Multi-provider API client
│   │   ├── vae/reconstructor.py     # External VAE + low-frequency EMA alignment
│   │   └── inference/pipeline.py    # Multi-turn editing loop
│   ├── run.py                       # Batch dataset runner
│   ├── dino_vlm_eval.py             # DINO + VLM evaluation
│   └── requirements.txt
├── FLUX.2/             # White-box FLUX.2 Klein 9B evaluation
│   ├── src/
│   │   ├── model/model.py           # FLUX.2 runner with latent pass-through
│   │   └── eval/                    # Frequency analysis & metrics
│   ├── config.example.yaml
│   └── requirements.txt
├── SD3-UE/             # White-box SD3 UltraEdit evaluation
│   ├── run_ultraedit_metrics.py     # Standalone SD3-UE runner
│   ├── config.ultraedit.yaml
│   ├── requirements.txt
│   └── third_party/UltraEdit/       # Forked diffusers (no git history)
└── dataset/            # Evaluation dataset (illustration & photograph)
    ├── illustration/
    └── photograph/
```

## Installation

Each subdirectory has its own environment due to differing PyTorch / diffusers versions. We recommend separate virtual environments.

### FLUX.2 (White-Box)

```bash
cd FLUX.2
pip install -r requirements.txt
```

### SD3-UE (White-Box)

```bash
cd SD3-UE
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: SD3-UE installs a forked `diffusers` from `third_party/UltraEdit/diffusers` to support `StableDiffusion3InstructPix2PixPipeline`.

Diffusers forked from https://github.com/HaozheZhao/UltraEdit. Please clone the repository and save it as third_party in SD3-UE folder.

### Black-Box (API-Based Editors)

```bash
cd Black-Box
pip install -r requirements.txt
```

Download a compatible Diffusers-format VAE (e.g., `stabilityai/sd-vae-ft-ema`) and update `src/config/config.json`.

## Quick Start

### White-Box: FLUX.2 with VAE-LFA

```bash
cd FLUX.2
python -m src \
  --config config.example.yaml \
  --model-type flux2 \
  --eval cycle \
  --latent true \
  --augmentation momentum
```

### White-Box: SD3-UE with VAE-LFA

```bash
cd SD3-UE
python run_ultraedit_metrics.py \
  --config config.ultraedit.yaml \
  --eval cycle \
  --latent true \
  --augmentation momentum
```

### Black-Box: Qwen Image 2.0 with VAE-LFA

```bash
cd Black-Box
python run.py \
  --model qwen \
  --mode noop \
  --use_vae \
  --data_dir ../dataset \
  --qwen_api_key <YOUR_API_KEY>
```

See subdirectory READMEs for detailed configuration and usage:
- [Black-Box/README.md](Black-Box/README.md)
- [FLUX.2/README.md](FLUX.2/README.md)
- [SD3-UE/README.md](SD3-UE/README.md)

## Method

### Low-Frequency Decomposition

Given latent tensor `z`, we decompose it via average-pooling-based low-pass filtering:

```python
low_z = F.avg_pool2d(F.pad(z, pad, mode="replicate"), kernel_size=k, stride=1)
high_z = z - low_z
```

### EMA Alignment

We maintain an exponential moving average of low-frequency mean and standard deviation across editing rounds:

```python
# Align current low-frequency statistics to EMA anchor
low_aligned = (low_z - mean_curr) / std_curr * target_std + target_mean
z_out = low_aligned + high_z

# Update EMA anchor
momentum_mean = decay * momentum_mean + (1 - decay) * mean_curr
momentum_log_std = decay * momentum_log_std + (1 - decay) * std_curr.log()
```

This softly constrains only the macro-statistics of low-frequency latents, preserving editability and generative diversity.

## Evaluation

We support three multi-turn editing protocols:

- **No-op editing:** 10 rounds of identity-preserving prompts (isolates intrinsic drift).
- **Cycle editing:** Alternating inverse instructions for 10 rounds (tests recovery).
- **Long-chain editing:** 10 progressively cumulative instructions (tests creative workflows).

Metrics: LPIPS, SSIM, L1 drift (pixel-level); DINOv3 subject consistency + Qwen-VL scoring (long-chain).

## Dataset

The `dataset/` directory contains 120 images across:
- **Domains:** illustration, photograph
- **Categories:** creature, architecture, scenery

Each image is paired with a JSON file containing `cycle` and `long_chain` prompts.

## Citation

Paper will be released on Arxiv soon.
