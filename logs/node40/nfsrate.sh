#!/bin/bash
# NFS client rate on this host over $1 seconds (default 60): total RPC calls/s and NFSv4 OPEN_NOATTR/s.
snap() { nfsstat -c -4 | awk '/open_noat/{getline; print $3}'; nfsstat -rc | sed -n 3p | awk '{print $1}'; }
read o1 c1 < <(snap | tr '\n' ' '); sleep ${1:-60}; read o2 c2 < <(snap | tr '\n' ' ')
echo "$(date +%H:%M:%S) nfs: $(( (c2-c1)/${1:-60} )) calls/s, $(( (o2-o1)/${1:-60} )) opens/s"
