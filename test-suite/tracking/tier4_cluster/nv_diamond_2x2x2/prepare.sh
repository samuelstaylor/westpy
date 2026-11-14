#!/bin/bash
# Build the NV- diamond 2x2x2 run directories from a copy of the source run.
#   bash prepare.sh [BASE]   (default BASE=/global/cfs/cdirs/m4382/sstaylor/tracking_tests/nv_diamond_2x2x2)
# BASE/source = copy of /global/cfs/cdirs/m4382/sstaylor/projects/NV-center-diamond/2x2x2/tddft
# (initial C3v pw.in is used; final_geo is not). wbse.in is replaced (n_liouville_eigen: 4).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=${1:-/global/cfs/cdirs/m4382/sstaylor/tracking_tests/nv_diamond_2x2x2}
SRC=$BASE/source
for f in pw.in C_ONCV_PBE-1.2.upf N_ONCV_PBE-1.2.upf; do
    [ -f "$SRC/$f" ] || { echo "missing $SRC/$f"; exit 1; }
done
for run in fixture relax; do
    d=$BASE/$run
    if [ -e "$d" ]; then echo "exists, not touched: $d"; continue; fi
    mkdir -p "$d"
    cp "$SRC"/pw.in "$SRC"/C_ONCV_PBE-1.2.upf "$SRC"/N_ONCV_PBE-1.2.upf "$d"/
    cp "$HERE"/wbse.in "$HERE"/westpp.in "$d"/
    cp "$HERE"/job_${run}.sh "$d"/
    [ "$run" = relax ] && cp "$HERE"/bfgs_tracked.py "$d"/bfgs.py
    echo "prepared $d: $(ls "$d" | tr '\n' ' ')"
done
