#!/bin/bash
# v2 rebuild after the 2023-24 refetch, with 2013 and the ODYSSEY/NIMBUS era split.
# Waits for the 2013 fetch (first range of fetch_remaining.sh), then: climatology over every
# day, rescan of 2013 + 2023 + 2024, month-matched era-gap analysis, splits, store, gamma,
# checks (run_build.sh). The 2025-10 -> 2026-09 NIMBUS fetch continues on cores 10-11; days
# complete at scan time join the nimbus split, later ones can be added in a later rebuild.
#   setsid nohup bash scripts/dataset_v2/rebuild_v2.sh > logs/rebuild_v2.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
Q=/home/fquareng/work/data/extremes/OPERA/quality_v2
echo "=== waiting for the 2013 fetch ($(date)) ==="
until grep -q "^=== 2025-10-23" logs/fetch_remaining.log 2>/dev/null; do sleep 300; done
echo "=== 2013 on disk: $(ls -d /home/fquareng/work/data/extremes/OPERA/raw/OPERA/2013*/.zmetadata 2>/dev/null | wc -l) days ($(date)) ==="
[ -f $Q/clutter_climatology.npz ] && mv $Q/clutter_climatology.npz $Q/clutter_climatology_pre_refetch.npz
rm -f $Q/tiles/2013*.csv.gz $Q/tiles/2023*.csv.gz $Q/tiles/2024*.csv.gz
bash scripts/dataset_v2/run_scan.sh > logs/dataset_v2_scan2.log 2>&1
echo "=== era gap: months ($(date)) ==="
taskset -c 4-9 $E/bin/python -u scripts/data_quality/era_gap.py --part months
echo "=== build ($(date)) ==="
SCANLOG=logs/dataset_v2_scan2.log bash scripts/dataset_v2/run_build.sh > logs/dataset_v2_build2.log 2>&1
echo "=== done ($(date)) ==="
