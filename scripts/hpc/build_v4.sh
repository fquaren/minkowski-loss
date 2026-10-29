#!/bin/bash
# v4 hourly dataset, full archive (DECISIONS §21-§31). Every stage is resumable: rerun this script.
cd /work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion
export LD_LIBRARY_PATH=/work/fquareng/.micromamba/envs/dl-stable/lib OMP_NUM_THREADS=1
P="taskset -c 4-11 /work/fquareng/.micromamba/envs/dl-stable/bin/python -u"
Q2=/home/fquareng/work/data/extremes/OPERA/quality_v2
Q4=/home/fquareng/work/data/extremes/OPERA/quality_v4
OUT=/home/fquareng/work/data/extremes/OPERA/patches_v4
set -o pipefail
echo "=== v4 build, code $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD), $(date)"
echo "=== 1. scan_hours, full archive $(date)"
$P scripts/dataset_v2/scan_hours.py config.yaml --quality configs/quality_v4.yaml --out_dir $Q4/hourly \
  --climatology $Q2/clutter_climatology.npz --ring_mask $Q2/ring_mask.npz --workers 7 --skip_existing \
  2>&1 | grep -v "Warn\|warn\|pystepsrc" || { echo "SCAN FAILED"; exit 1; }
echo "=== 2. make_splits_hourly $(date)"
$P scripts/dataset_v2/make_splits_hourly.py config.yaml --hours_dir $Q4/hourly/hours --quality configs/quality_v4.yaml \
  --day_excl $Q4/day_exclusions.csv --out_dir $OUT/meta --tail 20.2 --workers 7 \
  2>&1 | grep -v "Warn\|warn" | tail -40 || { echo "SPLITS FAILED"; exit 1; }
for st in create build verify aux; do
  echo "=== 3. build_store_hourly $st $(date)"
  $P scripts/dataset_v2/build_store_hourly.py config.yaml --meta_dir $OUT/meta --out_dir $OUT \
    --climatology $Q2/clutter_climatology.npz --ring_mask $Q2/ring_mask.npz --stage $st --no_advection --workers 6 \
    2>&1 | grep -v "Warn\|warn\|pystepsrc" | tail -20 || { echo "STORE $st FAILED"; exit 1; }
done
echo "=== 4. build_store_hourly adv (advection-corrected field) $(date)"
$P scripts/dataset_v2/build_store_hourly.py config.yaml --meta_dir $OUT/meta --out_dir $OUT \
  --climatology $Q2/clutter_climatology.npz --ring_mask $Q2/ring_mask.npz --stage adv --workers 6 \
  2>&1 | grep -v "Warn\|warn\|pystepsrc" | tail -20 || { echo "STORE adv FAILED"; exit 1; }
echo "=== V4_BUILD_DONE $(date)"
