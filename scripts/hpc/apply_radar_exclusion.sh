#!/bin/bash
# DECISIONS §36b: exclude the flagged radar-years' tile-hours at the metadata level, then rewrite
# the advection store's lists (no fields recomputed), the post-break lists, the persistence
# thresholds, and re-validate the store.  Usage: apply_radar_exclusion.sh <biased csv>
cd "$(dirname "$0")/../.."
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD
P="taskset -c 4-11 $E/bin/python -u"
CSV=${1:-/home/fquareng/work/data/extremes/OPERA/quality_v4/biased_radar_years_tail.csv}
set -o pipefail
echo "=== filter_meta_v4 --biased $CSV $(date)"
$P scripts/dataset_v2/filter_meta_v4.py --biased "$CSV" || { echo "FILTER FAILED"; exit 1; }
echo "=== build_adv_store --meta_only $(date)"
$P scripts/dataset_v2/build_adv_store.py --meta_only 2>&1 | tail -25 || { echo "META_ONLY FAILED"; exit 1; }
echo "=== post-break lists $(date)"
$P scripts/dataset_v2/postbreak_lists.py || { echo "POSTBREAK FAILED"; exit 1; }
echo "=== persistence thresholds $(date)"
$P scripts/preprocess/compute_persistence_thresholds.py configs/config_v4_adv.yaml 2>&1 | tail -5 || exit 1
echo "=== validate the store $(date)"
$P scripts/validate/validate_stage.py store --config configs/config_v4_adv.yaml --n_days 8 --workers 2 2>&1 \
  | grep -v Warn | tail -25 || { echo "STORE VALIDATION FAILED"; exit 1; }
echo "RADAR_EXCLUSION_DONE $(date)"
