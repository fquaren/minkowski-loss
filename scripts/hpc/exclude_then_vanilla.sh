#!/bin/bash
# DECISIONS §36 (researcher, 2026-10-21): exclude the listed radar-years (rsjas 2023, then hrulj 2023), then restart the vanilla v4_adv baseline on
# the final lists. The baseline starts only if the exclusion and the store re-validation pass.
cd "$(dirname "$0")/../.."
CSV=/home/fquareng/work/data/extremes/OPERA/quality_v4/biased_radar_years_all.csv
echo "=== exclusion with $CSV ($(tail -n +2 $CSV | tr '\n' ' ')) $(date)"
bash scripts/hpc/apply_radar_exclusion.sh "$CSV" || { echo "EXCLUSION FAILED: baseline not started"; exit 1; }
echo "=== vanilla v4_adv baseline (restart), code $(git rev-parse --short HEAD), $(date)"
bash scripts/hpc/launch_unet_analytical.sh configs/unet_analytical.yaml 0 100.0 configs/config_v4_adv.yaml
echo "=== VANILLA_V4ADV_DONE $(date)"
