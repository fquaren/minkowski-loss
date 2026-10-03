#!/bin/bash
# v2 dataset, stage 2. Waits for stage 1 (run_scan.sh) to finish, then:
#   splits -> patch store (build, verify, aux) -> config -> gamma targets -> dataset checks.
# Destroys nothing: writes only OPERA/v2 (metadata), OPERA/patches_v2 (store) and
# configs/config_v2.yaml. The old store is left in place for the post-verification step.
#   setsid nohup bash scripts/dataset_v2/run_build.sh > logs/dataset_v2_build.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="taskset -c 4-9 $E/bin/python -u"
D=/home/fquareng/work/data/extremes/OPERA
Q=$D/quality_v2; META=$D/v2; STORE=$D/patches_v2
W=${WORKERS:-6}; BUDGET=${BUDGET:-3000000}
SCANLOG=${SCANLOG:-logs/dataset_v2_scan.log}

echo "=== waiting for the scan ($(date)) ==="
until grep -q "^=== done" "$SCANLOG" 2>/dev/null; do
  grep -q "climatology FAILED" "$SCANLOG" 2>/dev/null && { echo "scan failed, stopping"; exit 1; }
  sleep 300
done
n_days=$(ls -d $D/raw/OPERA/[0-9]*/.zmetadata 2>/dev/null | wc -l)
n_tab=$(ls $Q/tiles/*.csv.gz 2>/dev/null | wc -l)
echo "scan done: $n_tab tile tables for $n_days complete day stores"
grep "^  ! " "$SCANLOG" && echo "(days above failed in the scan)"
[ "$n_tab" -ge $((n_days * 99 / 100)) ] || { echo "fewer than 99% of days scanned, stopping"; exit 1; }

echo "=== splits ($(date)) ==="
$PY scripts/dataset_v2/make_splits.py config.yaml --tiles_dir "$Q/tiles" \
    --events configs/prominent_events.yaml --out_dir "$META" --budget "$BUDGET" \
    || { echo "splits FAILED"; exit 1; }

echo "=== store ($(date)) ==="
$PY scripts/dataset_v2/build_store.py config.yaml --meta_dir "$META" --out_dir "$STORE" \
    --climatology "$Q/clutter_climatology.npz" --stage all --workers "$W" \
    || { echo "store FAILED"; exit 1; }

echo "=== config ($(date)) ==="
$E/bin/python src/sweep_util.py patch --base config.yaml --out configs/config_v2.yaml --set \
    PREPROCESSED_DATA_DIR="$STORE" DEM_STATS="$STORE/dem_stats.json" \
    TRAIN_METADATA_FILE="$META/full_train.txt" VAL_METADATA_FILE="$META/full_val.txt" \
    TEST_METADATA_FILE="$META/full_test.txt" \
    LIGHT_TRAIN_METADATA_FILE="$META/light_train.txt" LIGHT_VAL_METADATA_FILE="$META/light_val.txt" \
    LIGHT_TEST_METADATA_FILE="$META/light_test.txt" \
    EXTREMES_TRAIN_METADATA_FILE="$META/extremes_train.txt" EXTREMES_VAL_METADATA_FILE="$META/extremes_val.txt" \
    EXTREMES_TEST_METADATA_FILE="$META/extremes_test.txt" EVENTS_TEST_METADATA_FILE="$META/events_test.txt" \
    || { echo "config FAILED"; exit 1; }

echo "=== gamma targets ($(date)) ==="
$PY scripts/preprocess/compute_gamma_targets.py configs/config_v2.yaml || { echo "gamma FAILED"; exit 1; }

echo "=== dataset checks ($(date)) ==="
$PY scripts/dataset_v2/check_datasets.py configs/config_v2.yaml
echo "=== done ($(date)) ==="
