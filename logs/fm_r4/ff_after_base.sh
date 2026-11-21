#!/bin/bash
# Fast-forward the fm-r4 worktree to main (packed read path) between the FM base (cut) training and the
# queue's next step, with no race: the queue shell is paused (SIGSTOP; the training child keeps running),
# and is continued once the training has exited and the fast-forward is done. The trap continues the queue
# on any exit of this script, including failure.
Q=46287            # bash scripts/hpc/run_fm_r4_queue.sh
T=386294           # train_flow_matching.py fm_base_cut.yaml
W=/work/fquareng/ch2/minkowski-loss/.claude/worktrees/fm-r4
TARGET=46e0a4c     # main with the packed store
log() { echo "$(date '+%F %T') $*"; }
trap 'kill -CONT $Q 2>/dev/null; log "queue $Q continued"' EXIT
trap 'exit 1' TERM INT HUP

tr '\0' ' ' < /proc/$Q/cmdline | grep -q run_fm_r4_queue.sh || { log "queue pid $Q is not the queue"; exit 1; }
tr '\0' ' ' < /proc/$T/cmdline | grep -q fm_base_cut.yaml || { log "training pid $T is not fm_base_cut"; exit 1; }
kill -STOP $Q && log "queue $Q paused; waiting for training $T"
# with the parent paused, the finished training stays a zombie until the queue reaps it
while [ -e /proc/$T ] && ! grep -q '^State:[[:space:]]*Z' /proc/$T/status 2>/dev/null; do sleep 30; done
log "training $T exited"
if git -C $W merge --ff-only $TARGET; then
  log "fm-r4 now at $(git -C $W rev-parse --short HEAD); packed_store present: $(test -f $W/src/data/packed_store.py && echo yes)"
else
  log "FAST-FORWARD FAILED, the queue continues on the old code"
fi
