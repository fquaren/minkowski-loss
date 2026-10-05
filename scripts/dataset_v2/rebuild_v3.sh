#!/bin/bash
# v3 dataset: the v2 pipeline with the gauge-validated repairs and the fixed event set.
#   - cleaning: clean_frame + footprints of > 500 mm/h cores, rays, range rings (>= 89 mm/h)
#     and cells without temporal support, repaired instead of rejecting the tile
#     (src/data/day_cleaner.py; validation in validation/repair_v3/repair_comparison.md);
#   - events: the 2013 Central European floods and "Andreas" added, HyMeX IOP16 marked
#     unavailable, shifted windows for events without fully covered grid tiles
#     (Emilia-Romagna).
# Writes only new locations: quality_v3/ (tile + window tables), OPERA/v3/ (metadata),
# OPERA/patches_v3/ (store), configs/config_v3.yaml. v2 stays as it is.
# Cores 4-11, 8 workers: run it only when nothing else holds those cores.
#   setsid nohup bash scripts/dataset_v2/rebuild_v3.sh > logs/rebuild_v3.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
W=${WORKERS:-8}
PY="taskset -c 4-11 $E/bin/python -u"
D=/home/fquareng/work/data/extremes/OPERA
Q2=$D/quality_v2; Q=$D/quality_v3; META=$D/v3; STORE=$D/patches_v3
CLIM=$Q2/clutter_climatology.npz; RING=$Q2/ring_mask.npz
BUDGET=${BUDGET:-3000000}
mkdir -p $Q

echo "=== tile scan, v3 cleaning ($(date)) ==="
$PY scripts/dataset_v2/scan_tiles.py config.yaml --climatology $CLIM --ring_mask $RING \
    --out_dir $Q --workers $W --skip_existing || { echo "scan FAILED"; exit 1; }
n_days=$(ls -d $D/raw/OPERA/[0-9]*/.zmetadata | wc -l); n_tab=$(ls $Q/tiles/*.csv.gz | wc -l)
echo "scan: $n_tab tile tables for $n_days day stores"
[ "$n_tab" -ge "$n_days" ] || { echo "not every day scanned, stopping"; exit 1; }

echo "=== event windows ($(date)) ==="
$PY scripts/dataset_v2/scan_event_windows.py config.yaml --events configs/prominent_events.yaml \
    --climatology $CLIM --ring_mask $RING --out_dir $Q --workers $W || { echo "windows FAILED"; exit 1; }

echo "=== splits ($(date)) ==="
$PY scripts/dataset_v2/make_splits.py config.yaml --tiles_dir $Q/tiles \
    --events configs/prominent_events.yaml --event_windows $Q/event_windows \
    --out_dir $META --budget $BUDGET || { echo "splits FAILED"; exit 1; }

echo "=== store ($(date)) ==="
$PY scripts/dataset_v2/build_store.py config.yaml --meta_dir $META --out_dir $STORE \
    --climatology $CLIM --ring_mask $RING --stage all --workers $W || { echo "store FAILED"; exit 1; }

echo "=== config ($(date)) ==="
$E/bin/python src/sweep_util.py patch --base config.yaml --out configs/config_v3.yaml --set \
    PREPROCESSED_DATA_DIR="$STORE" DEM_STATS="$STORE/dem_stats.json" \
    TRAIN_METADATA_FILE="$META/full_train.txt" VAL_METADATA_FILE="$META/full_val.txt" \
    TEST_METADATA_FILE="$META/full_test.txt" \
    LIGHT_TRAIN_METADATA_FILE="$META/light_train.txt" LIGHT_VAL_METADATA_FILE="$META/light_val.txt" \
    LIGHT_TEST_METADATA_FILE="$META/light_test.txt" \
    EXTREMES_TRAIN_METADATA_FILE="$META/extremes_train.txt" EXTREMES_VAL_METADATA_FILE="$META/extremes_val.txt" \
    EXTREMES_TEST_METADATA_FILE="$META/extremes_test.txt" EVENTS_TEST_METADATA_FILE="$META/events_test.txt" \
    MAX_WORKERS=$W || { echo "config FAILED"; exit 1; }

echo "=== gamma targets ($(date)) ==="
$PY scripts/preprocess/compute_gamma_targets.py configs/config_v3.yaml || { echo "gamma FAILED"; exit 1; }

echo "=== dataset checks ($(date)) ==="
$PY scripts/dataset_v2/check_datasets.py configs/config_v3.yaml
echo "=== done ($(date)) ==="
