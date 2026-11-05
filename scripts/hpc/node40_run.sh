#!/bin/bash
# Run one python script on node40 (shared node, borrowed until ~2026-11-09), detached under setsid.
#   node40_run.sh <gpu 0-3> <label> <script.py> [args...]      -> prints the log path
# Limits (researcher, 2026-11-05): GPUs 0-3 only; other people use this node, so
#   - refuse if the GPU already holds any compute process (ours or anyone's);
#   - all our jobs together use at most 6 CPUs (researcher, 2026-11-05; first 4): CPUs 0-5, six separate physical
#     cores (0-3 on NUMA 0, local to GPUs 0-3; 4-5 on NUMA 1, since NUMA 0 has only 4 physical cores); every
#     hyperthread sibling (8-13) stays free. 1 thread per process, nice 10.
#     DataLoader workers inherit the affinity: they wait on NFS most of the time, so more workers add I/O
#     parallelism, not CPU beyond these 6.
# src/node_limits.py is a no-op here (our node32 card is not in this host), so these guards are the only ones.
# RUN_CWD (default: project root) is the working directory (runs/ is created relative to it).
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[ "$(hostname -s)" = node40 ] || { echo "not node40"; exit 1; }
[ $# -ge 3 ] || { sed -n 2,3p "$0"; exit 1; }
GPU=$1; LABEL=$2; shift 2
case $GPU in
  0) UUID=GPU-65c61152-9058-a31d-fabc-a5959e64ed32 ;;
  1) UUID=GPU-d37a33f4-2960-d6d2-c931-85654eb102b0 ;;
  2) UUID=GPU-9387ca5a-7372-7069-1cf9-8b59b721b62a ;;
  3) UUID=GPU-f42e5605-a5c7-be28-d3b3-36afa0e2eaf9 ;;
  *) echo "GPU $GPU not allowed (0-3 only)"; exit 1 ;;
esac
nvidia-smi -i "$UUID" --query-gpu=uuid --format=csv,noheader | grep -qi "$UUID" || { echo "UUID mismatch for GPU $GPU"; exit 1; }
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -qi "$UUID"; then
  echo "GPU $GPU ($UUID) is busy, refusing"; exit 1
fi
# A job still importing has no CUDA context yet, so nvidia-smi cannot see it: our own jobs also hold a pid lock.
LOCK=/tmp/fquareng_node40_gpu$GPU.pid
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
  echo "GPU $GPU held by our pid $(cat "$LOCK"), refusing"; exit 1
fi
E=/work/fquareng/.micromamba/envs/dl-stable
LOG="$PROJECT_ROOT/logs/node40/${LABEL}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"
cd "${RUN_CWD:-$PROJECT_ROOT}"
echo "node40 run: label $LABEL gpu $GPU $UUID cwd $PWD code $(git -C "$PROJECT_ROOT" rev-parse --short HEAD) cmd $*" > "$LOG"
setsid nohup env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$UUID" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  LD_LIBRARY_PATH=$E/lib PYTHONPATH="$PROJECT_ROOT" PYTHONUNBUFFERED=1 \
  nice -n 10 taskset -c 0-5 $E/bin/python -u "$@" >> "$LOG" 2>&1 < /dev/null &
echo $! > "$LOCK"
echo "$LOG"
