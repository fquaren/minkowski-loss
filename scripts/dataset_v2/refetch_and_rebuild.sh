#!/bin/bash
# Replace the 435 original day stores (2023-08-01..2024-10-30: capped at 150 mm/h upstream,
# no QIND) with archive refetches, then rebuild v2. Gated: a day is swapped only if
# verify_refetch.py passes it; the chain stops if < 95% pass.
#   setsid nohup bash scripts/dataset_v2/refetch_and_rebuild.sh > logs/refetch_rebuild.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLOSC_NTHREADS=1
D=/home/fquareng/work/data/extremes/OPERA
R=$D/raw/OPERA; S=$D/raw/OPERA_refetch; C=$D/raw/OPERA_capped_orig; Q=$D/quality_v2

echo "=== 1. remove the capped-era v2 build ($(date)) ==="
[ -f $D/v2/report.md ] && cp $D/v2/report.md logs/dataset_v2_report_capped_20260929.md
rm -rf $D/patches_v2 $D/v2 && echo "removed patches_v2 and v2 metadata"

echo "=== 2. refetch 2023-08-01..2024-10-30 into staging ($(date)) ==="
taskset -c 4,5 $E/bin/python -u scripts/data/fetch_opera_archive.py \
    --start 2023-08-01 --end 2024-10-30 --out $S --reference $R/20230801 \
    --skip_existing --workers 2 || echo "fetch returned non-zero (checked below)"

echo "=== 3. verify against the originals ($(date)) ==="
taskset -c 4,5 $E/bin/python -u scripts/dataset_v2/verify_refetch.py \
    --orig_dir $R --new_dir $S --out logs/refetch_verify \
    || { echo "VERIFY FAILED: < 95% of days pass; nothing swapped, stopping"; exit 1; }

echo "=== 4. swap passing days; delete their capped originals ($(date)) ==="
mkdir -p $C
n=0
for day in $($E/bin/python -c "import pandas as pd; t=pd.read_csv('logs/refetch_verify.csv',dtype={'day':str}); print(' '.join(t.loc[t['pass'],'day']))"); do
  if [ -f $S/$day/.zmetadata ] && [ ! -d $R/$day/QIND ]; then
    mv $R/$day $C/$day && mv $S/$day $R/$day && n=$((n+1))
  fi
done
echo "swapped $n days; capped originals left in place for failed days: $(ls -d $R/2023* $R/2024* | while read d; do [ -d $d/QIND ] || echo x; done | wc -l)"
echo "deleting $(ls $C | wc -l) capped originals (their verified replacements are in $R)"
rm -rf $C && rmdir $S 2>/dev/null; ls $S 2>/dev/null | head -3

echo "=== 5. climatology + rescan 2023-2024 ($(date)) ==="
mv $Q/clutter_climatology.npz $Q/clutter_climatology_capped2324.npz 2>/dev/null
rm -f $Q/tiles/2023*.csv.gz $Q/tiles/2024*.csv.gz
bash scripts/dataset_v2/run_scan.sh > logs/dataset_v2_scan2.log 2>&1

echo "=== 6. rebuild v2 ($(date)) ==="
SCANLOG=logs/dataset_v2_scan2.log bash scripts/dataset_v2/run_build.sh > logs/dataset_v2_build2.log 2>&1
echo "=== done ($(date)) ==="
