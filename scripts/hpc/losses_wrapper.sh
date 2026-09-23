# Study 1: one backbone per structural loss, each at its own weight from
# STRUCTURAL_LOSS_WEIGHTS in config.yaml (copied into configs/loss_<L>.yaml below):
#   minkowski 1e-4, spectral 1e-1, ssim 2e-2, wetarea 2e-2, opticalflow 2.5e-2
# (2026-09-23; Adam-share matched to Minkowski at 1e-4, see src/losses/competing.py).
# Previous v2 weights, which left spectral/ssim/opticalflow inactive: 5e-4 / 1.6e-4 / 2e-2 / 4e-5.
# Pass a number instead of "config" to override one run.
for L in minkowski spectral ssim wetarea opticalflow; do
  python src/sweep_util.py patch --base config.yaml --out configs/loss_$L.yaml \
      --set STRUCTURAL_LOSS=$L EXPERIMENT_NAME=UNet_$L
done
bash scripts/hpc/launch_unet_analytical.sh configs/unet_analytical.yaml config 100.0 configs/loss_minkowski.yaml
bash scripts/hpc/launch_unet_analytical.sh configs/unet_analytical.yaml config 100.0 configs/loss_spectral.yaml
bash scripts/hpc/launch_unet_analytical.sh configs/unet_analytical.yaml config 100.0 configs/loss_ssim.yaml
bash scripts/hpc/launch_unet_analytical.sh configs/unet_analytical.yaml config 100.0 configs/loss_wetarea.yaml
bash scripts/hpc/launch_unet_analytical.sh configs/unet_analytical.yaml config 100.0 configs/loss_opticalflow.yaml
