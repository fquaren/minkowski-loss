for A in 0.05 0.10 0.20 0.40; do
  python src/sweep_util.py patch --base config.yaml --out /tmp/an_$A.yaml --set REWARD_ANNEAL=$A > /dev/null
  echo "================ REWARD_ANNEAL=$A ================"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  python scripts/evaluate/diagnose_coupling.py /tmp/an_$A.yaml \
      --fm_checkpoint runs/sr_flow_matching/flow_matching_20260716_071518/fm_best.pth \
      --backbone runs/sr_analytical/UNet_Ana_20260623_144958/unet_best.pth \
      --extreme_batch --ensemble 16 --steps 5 2>&1 | sed -n '/TEST 4/,/TEST 5/p'
done
