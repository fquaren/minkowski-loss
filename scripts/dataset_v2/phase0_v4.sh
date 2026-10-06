#!/bin/bash
# Phase 0 of the v4 build (EXPERIMENTS §5, DECISIONS §19): the v4 screen on the calibration
# days only, then the gate report. ~1 h on 8 workers. Writes only under quality_v4/calib/.
#   setsid nohup bash scripts/dataset_v2/phase0_v4.sh > logs/phase0_v4.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
W=${WORKERS:-8}
PY="taskset -c 4-11 $E/bin/python -u"
D=/home/fquareng/work/data/extremes/OPERA
C=$D/quality_v4/calib
CLIM=$D/quality_v2/clutter_climatology.npz; RING=$D/quality_v2/ring_mask.npz
mkdir -p $C

echo "=== calibration days ($(date)) ==="
$PY scripts/dataset_v2/calibrate_v4.py days --calib $C || exit 1
sed "s#^data_dir: .*#data_dir: $C#" configs/quality_v4.yaml > $C/quality_calib.yaml

echo "=== radar pass ($(date)) ==="
$PY scripts/dataset_v2/scan_radars.py config.yaml --climatology $CLIM --out_dir $C \
    --days_file $C/days.txt --workers $W --skip_existing || exit 1

echo "=== ceilings + radar frames ($(date)) ==="
$PY scripts/data_quality/radar_screen_v4.py ceilings --quality $C/quality_calib.yaml || exit 1
$PY scripts/data_quality/radar_screen_v4.py frames --quality $C/quality_calib.yaml --keep_frames --workers $W || exit 1

echo "=== v4 tile scan ($(date)) ==="
$PY scripts/dataset_v2/scan_tiles.py config.yaml --climatology $CLIM --ring_mask $RING \
    --out_dir $C --quality $C/quality_calib.yaml --days_file $C/days.txt --workers $W --skip_existing || exit 1

echo "=== v4-off regression: 2018-06-15 must equal the v3 table ($(date)) ==="
$PY scripts/dataset_v2/scan_tiles.py config.yaml --climatology $CLIM --ring_mask $RING \
    --out_dir $C/regression --days 20180615 --workers 1 || exit 1
$E/bin/python - <<PY || exit 1
import pandas as pd
a = pd.read_csv("$C/regression/tiles/20180615.csv.gz"); b = pd.read_csv("$D/quality_v3/tiles/20180615.csv.gz")
ok = a.shape == b.shape and a.equals(b)
print("regression v4-off == v3:", ok, a.shape, b.shape)
raise SystemExit(0 if ok else 1)
PY

echo "=== report ($(date)) ==="
$PY scripts/dataset_v2/calibrate_v4.py report --calib $C || exit 1
$PY scripts/dataset_v2/calibrate_v4.py gallery --calib $C || exit 1
echo "=== done ($(date)) ==="
