python src/eval/freq.py \
  --config config.example.yaml \
  --image photo_small \
  --model-type flux2 \
  --eval noop \
  --latent false \
  --augmentation none \
  --vae-only \
  --cpu-offload \
  --outdir results/freq_vae_flux2

# python src/eval/freq.py \
#   --config config.example.yaml \
#   --image photo_small \
#   --model-type flux2 \
#   --eval noop \
#   --latent true \
#   --augmentation none \
#   --outdir results/freq_latent_flux2