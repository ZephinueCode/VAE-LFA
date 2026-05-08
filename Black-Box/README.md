# VAE-LFA: Black-Box Editor Evaluation

This directory implements the **black-box** variant of VAE-LFA for API-based DiT image editors (e.g., Qwen Image 2.0, Seedream 4.0). Since internal model parameters are inaccessible, VAE-LFA interleaves an off-the-shelf VAE between editing rounds to perform low-frequency latent alignment, suppressing accumulated semantic drift without retraining or internal access.

## Overview

- **Training-free & plug-and-play:** No fine-tuning of the base editor is required.
- **External VAE alignment:** Each edited image is encoded into latent space via SD-VAE-ft-ema, low-frequency statistics are aligned to an EMA anchor, and the result is decoded back to pixel space before the next editing round.
- **Supported APIs:** Qwen Image 2.0 (DashScope), Seedream 4.0, Wanx, Doubao, OpenRouter FLUX, Stability AI SD3, and generic OpenAI-compatible endpoints.

## Installation

```bash
pip install -r requirements.txt
```

**Note:** `torch==2.8.0` is pinned in `requirements.txt`. If your CUDA version differs, adjust the PyTorch wheel accordingly.

## Quick Start

### 1. Prepare the External VAE

Download or locate a compatible Diffusers-format VAE (e.g., `stabilityai/sd-vae-ft-ema` or a local copy) and update `vae.model_path` in `src/config/config.json`.

### 2. Configure API Key

Set your API key via the CLI argument (`--qwen_api_key`) or update `api.api_key_env` in `src/config/config.json`.

### 3. Run Single-Image Multi-Turn Editing

```bash
python -m src run --config src/config/config.json
```

The default config runs 10 no-op rounds on the sample image with VAE-LFA enabled.

### 4. Batch Evaluation on a Dataset

```bash
python run.py \
  --model qwen \
  --mode noop \
  --use_vae \
  --data_dir ../dataset \
  --qwen_api_key <YOUR_API_KEY>
```

- `--model`: `qwen` or `seedream`
- `--mode`: `noop`, `cycle`, or `longchain`
- `--use_vae`: enables VAE-LFA (low-frequency alignment); omit for baseline

Results are saved under `results/<model>_<mode>_<vae_status>/` with per-image manifests, round-by-round images, and aggregated LPIPS/SSIM/L1 metrics.

## Dataset Format

Use paired image + JSON files with the same stem:

```text
data/
  creature/
    001.png
    001.json
  architecture/
    002.png
    002.json
```

Example JSON (for `cycle` or `longchain`):

```json
{
  "cycle": ["add a red hat", "remove the red hat"],
  "longchain": ["change the sky to sunset", "add a boat on the lake"]
}
```

For `noop` mode, prompts are auto-generated as identity-preserving instructions.

## Configuration (`src/config/config.json`)

| Section | Key | Description |
|---------|-----|-------------|
| `api` | `base_url` | API endpoint (e.g., DashScope, Seedream, OpenRouter) |
| `api` | `api_key_env` | API key string (or pass via CLI for safety) |
| `api` | `model` | Model name (e.g., `qwen-image-2.0`) |
| `run` | `rounds` | Number of editing turns |
| `run` | `input_image` | Path to a single input image |
| `prompts` | `per_round` | List of prompts, length must equal `rounds` |
| `vae` | `enabled` | Toggle external VAE round-trip |
| `vae` | `model_path` | Path to the off-the-shelf Diffusers VAE |
| `vae` | `device` | `cuda` or `cpu` |
| `vae` | `dtype` | `float16` or `float32` |
| `intervention` | `enabled` | Toggle VAE-LFA low-frequency alignment |
| `intervention` | `kernel_size` | AvgPool kernel for low-pass filtering (default: 9) |
| `intervention` | `mean_decay` | EMA decay for low-frequency mean (default: 0.85) |
| `intervention` | `std_decay` | EMA decay for low-frequency std (default: 0.85) |

## Metrics & Evaluation

Per-round pixel-level metrics (LPIPS, SSIM, L1 drift) are computed automatically and summarized in:

- `overall_metrics.csv` / `overall_metrics.png`
- `manifest.json` per sample

For long-chain editing, run `dino_vlm_eval.py` to compute DINOv3 subject consistency and Qwen-VL overall scores:

```bash
python dino_vlm_eval.py \
  --report results/qwen_longchain_vae_on_report.json \
  --mode longchain \
  --data_dir ../dataset \
  --qwen_api_key <YOUR_API_KEY>
```

## Project Structure

```text
Black-Box/
  src/
    api/image_client.py          # Multi-provider API client
    vae/reconstructor.py          # External VAE + low-frequency EMA alignment
    config/settings.py            # Config loader
    inference/pipeline.py         # Multi-turn editing loop
  run.py                          # Batch dataset runner
  dino_vlm_eval.py                # DINO + VLM long-chain evaluator
  plot.py                         # Visualization utilities
```
