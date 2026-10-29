#!/bin/bash
# Vanilla backbone (w = 0) on the corrected hourly target, full data, config_v4_adv (25 epochs).
# Serves three purposes: the Study-1 baseline, the checkpoint for the structural-weight calibration
# (DECISIONS §37, step 1b: Adam share at the trained vanilla checkpoint), and the all-data arm of the
# pre-break ablation (§34). Waits for the raw-vs-corrected climatology run so the 8-core budget holds
# (that run uses 4 workers, the national-border pass 3).
cd "$(dirname "$0")/../.."
until grep -q -E "RATE_CLIM_DONE|Traceback" logs/rate_clim.log 2>/dev/null; do sleep 120; done
echo "=== vanilla v4_adv baseline, code $(git rev-parse --short HEAD), $(date)"
bash scripts/hpc/launch_unet_analytical.sh configs/unet_analytical.yaml 0 100.0 configs/config_v4_adv.yaml
echo "=== VANILLA_V4ADV_DONE $(date)"
