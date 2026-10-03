#!/bin/bash
# v2: gamma targets for the nimbus group (compute_gamma_targets.py covered only
# train/validation/test on the first build), then the dataset checks again.
#   setsid nohup bash scripts/dataset_v2/finish_nimbus_gamma.sh > logs/finish_nimbus_gamma.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CFG=${CFG:-configs/config_v2.yaml}
echo "=== gamma targets: nimbus ($(date)) ==="
taskset -c 4-9 $E/bin/python -u scripts/preprocess/compute_gamma_targets.py "$CFG" || { echo "gamma FAILED"; exit 1; }
echo "=== dataset checks ($(date)) ==="
taskset -c 4-9 $E/bin/python -u scripts/dataset_v2/check_datasets.py configs/config_v2.yaml
echo "=== done ($(date)) ==="
