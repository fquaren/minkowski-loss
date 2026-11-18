#!/bin/bash
# NFS operations per second on /work (server1) for the whole node, over <seconds> (default 20).
#   ssh <node> bash /work/fquareng/ch2/minkowski-loss/scripts/hpc/nfs_rate.sh [seconds]
# A run on the packed read path shows CLOSE and DELEGRETURN near 0/s. Hundreds per second is the per-file
# pattern that preceded the state-manager stalls on node32 (2026-11-05) and node40 (2026-11-16).
get() { awk '/server1i:\/export\/data1 mounted on \/work /{f=1} f&&/^\s+(OPEN|CLOSE|DELEGRETURN|READ|GETATTR|LOOKUP|ACCESS):/{gsub(":","",$1); print $1, $2} f&&/^device /&&!/server1i/{f=0}' /proc/self/mountstats | sort -u -k1,1; }
d=${1:-20}; a=$(get); sleep "$d"; b=$(get)
join <(echo "$a") <(echo "$b") | awk -v d="$d" -v h="$(hostname -s)" '{printf "%s %s %.0f/s\n", h, $1, ($3-$2)/d}'
