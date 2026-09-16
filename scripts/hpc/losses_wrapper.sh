for L in spectral ssim wetarea opticalflow; do
  python src/sweep_util.py patch --base config.yaml --out configs/loss_$L.yaml \
      --set STRUCTURAL_LOSS=$L EXPERIMENT_NAME=UNet_$L
done
bash scripts/hpc/launch_unet_analytical.sh configs/unet_analytical.yaml 1.6e-4 100.0 configs/loss_ssim.yaml
bash scripts/hpc/launch_unet_analytical.sh configs/unet_analytical.yaml 2e-2 100.0 configs/loss_wetarea.yaml
bash scripts/hpc/launch_unet_analytical.sh configs/unet_analytical.yaml 4e-5 100.0 configs/loss_opticalflow.yaml
bash scripts/hpc/launch_unet_analytical.sh configs/unet_analytical.yaml 5e-4 100.0 configs/loss_spectral.yaml
