#!/bin/bash
# One-batch gradient budget for each new FM reward mode: crashes and the parity weight.
cd /work/fquareng/ch2/minkowski-loss/.claude/worktrees/fm-r4
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD
W=/work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion
FMB=$W/runs/sr_flow_matching/Extremes_20261026_092630/fm_best.pth
VAN=$W/runs/sr_analytical/UNet_Ana_20261021_211732/unet_latest.pth
for m in wetarea ssim opticalflow area; do
  $E/bin/python - $m logs/fm_r4/smoke_$m.yaml <<'PY'
import sys, yaml
m, out = sys.argv[1:3]
c = yaml.safe_load(open("/work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion/configs/config_v4_adv_fm_energy_qm.yaml"))
c["REWARD_MODE"] = "energy" if m == "area" else m
if m == "area":
    c["MINKOWSKI_CHANNEL_WEIGHTS"] = [1.0, 0.0, 0.0]
yaml.safe_dump(c, open(out, "w"), sort_keys=False)
PY
  echo "=== $m $(date)"
  env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 taskset -c 4-11 $E/bin/python -u scripts/evaluate/grad_norm_terms.py \
    logs/fm_r4/smoke_$m.yaml --fm_checkpoint $FMB --backbone $VAN --n_batches 2 2>&1 | grep -E "^\[grad\]|^batch|mean \|g\||share|Error|Traceback" 
done
echo "SMOKE_DONE $(date)"
