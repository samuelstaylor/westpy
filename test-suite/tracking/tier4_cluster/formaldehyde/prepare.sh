#!/bin/bash
# Build the formaldehyde run directories from the downloaded tutorial files.
#   bash prepare.sh [BASE]     (default BASE=$SCRATCH/tracking_tests/formaldehyde)
# BASE/downloads must contain pw.in and the C/H/O ONCV UPFs (tutorial 12, es/).
# The tutorial wbse.in is replaced by ./wbse.in (n_liouville_eigen: 4).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=${1:-$SCRATCH/tracking_tests/formaldehyde}
DL=$BASE/downloads
for f in pw.in C_ONCV_PBE-1.2.upf H_ONCV_PBE-1.2.upf O_ONCV_PBE-1.2.upf; do
    [ -f "$DL/$f" ] || { echo "missing $DL/$f"; exit 1; }
done
for run in 0_plain 1_off 2_on; do
    d=$BASE/$run
    if [ -e "$d" ]; then echo "exists, not touched: $d"; continue; fi
    mkdir -p "$d"
    cp "$DL"/pw.in "$DL"/*_ONCV_PBE-1.2.upf "$d"/
    cp "$HERE"/wbse.in "$d"/
    cp "$HERE"/bfgs_${run}.py "$d"/bfgs.py
    sed "s/RUNNAME/${run}/" "$HERE"/job.sh > "$d"/job_${run}.sh
    [ "$run" = 2_on ] && cp "$HERE"/westpp.in "$d"/
    echo "prepared $d: $(ls "$d" | tr '\n' ' ')"
done
