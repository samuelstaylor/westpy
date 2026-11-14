#!/bin/bash
#SBATCH -A m542                 # CPU account (no _g suffix)
#SBATCH --qos=debug             # regular, debug
#SBATCH --nodes=1               # 8 max on debug
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=8       # 2*floor(128/32) per NERSC guidance
#SBATCH -C cpu
#SBATCH -t 00:30:00             # 30 min max for debug
#SBATCH -J nvd-fixture
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

###############################################################################
# Job Information
###############################################################################

print_divider () {
    printf '%*s\n' "${COLUMNS:-80}" '' | tr ' ' '='
}

# Print a value, falling back to N/A if the Slurm env var is empty/unset.
# This keeps the header honest instead of silently printing blank lines.
val () {
    local v="$1"
    if [ -z "$v" ]; then
        echo "N/A"
    else
        echo "$v"
    fi
}

print_job_info () {

    # Derived values: Slurm doesn't reliably populate SLURM_NTASKS when
    # resources are requested per-node (--ntasks-per-node), so compute it.
    local total_tasks=$(( SLURM_JOB_NUM_NODES * ${SLURM_NTASKS_PER_NODE:-0} ))

    print_divider
    echo "SLURM JOB INFORMATION"
    print_divider

    echo "Job Name            : $(val "${SLURM_JOB_NAME}")"
    echo "Job ID              : $(val "${SLURM_JOB_ID}")"
    echo "Cluster             : $(val "${SLURM_CLUSTER_NAME}")"
    echo "Account             : $(val "${SLURM_JOB_ACCOUNT}")"
    echo "Partition           : $(val "${SLURM_JOB_PARTITION}")"
    echo "QOS                 : $(val "$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/.* QOS=\([^ ]*\).*/\1/p')")"
    echo
    echo "Resources Requested"
    echo "-------------------"
    echo "Nodes               : $(val "${SLURM_JOB_NUM_NODES}")"
    echo "Node List           : $(val "${SLURM_JOB_NODELIST}")"
    echo "Tasks per Node      : $(val "${SLURM_NTASKS_PER_NODE}")"
    echo "Total Tasks (calc)  : $(val "${total_tasks}")"
    echo "CPUs per Task       : $(val "${SLURM_CPUS_PER_TASK}")"
    echo "Memory per Node     : $(val "${SLURM_MEM_PER_NODE}")"
    echo "Memory per CPU      : $(val "${SLURM_MEM_PER_CPU}")"

    echo
    echo "Timing"
    echo "------"
    echo "Submit Directory    : $(val "${SLURM_SUBMIT_DIR}")"
    echo "Submit Host         : $(val "${SLURM_SUBMIT_HOST}")"
    echo "Working Directory   : $(pwd)"
    echo "Start Time          : $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "Requested Walltime  : $(val "$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/.* TimeLimit=\([^ ]*\).*/\1/p')")"

    echo
    echo "Environment"
    echo "-----------"
    echo "User                : $(val "${USER}")"
    echo "Hostname            : $(hostname)"
    echo "SLURM Version       : $(srun --version 2>/dev/null | head -1)"

    print_divider
    echo
}

print_job_summary () {

    echo
    print_divider
    echo "FINAL JOB SUMMARY"
    print_divider

    # sacct can lag a few seconds behind the job actually finishing;
    # give the Slurm DB a moment before pulling final accounting data.
    sleep 10

    sacct -j "${SLURM_JOB_ID}" \
        --format=JobID,JobName,Partition,QOS,Account,AllocNodes,NTasks,State,ExitCode,Elapsed,TotalCPU,CPUTime,MaxRSS,NodeList \
        --parsable2

    print_divider
}

###############################################################################
# Print job information at startup
###############################################################################

print_job_info

echo
echo "Queue Information"
echo "-----------------"

sacct -j "${SLURM_JOB_ID}" \
    --format=JobID,Submit,Eligible,Start,State \
    --noheader

###############################################################################
# Automatically print job summary when the script exits
###############################################################################

trap print_job_summary EXIT

###############################################################################
# YOUR ACTUAL JOB STARTS HERE
###############################################################################

# ================================
# Environment Setup
# ================================
module unload darshan
module load cpu
# QEdir_cpu was built with PrgEnv-gnu + cray-fftw + cray-libsci (the cpu defaults)
# and its embedded Python (forpy, used by wbse.x/westpp.x) with cray-python 3.11.7
# (numpy 1.x): WEST must see this Python, exactly as in the production scripts.
module load cray-python/3.11.7

export LD_LIBRARY_PATH=/opt/cray/pe/python/3.11.7/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export SLURM_CPU_BIND=cores
# srun no longer inherits --cpus-per-task from sbatch
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
# pw.in has pseudo_dir commented out: read the UPFs from the run/step directory
export ESPRESSO_PSEUDO=./

unset PYTHONPATH
unset PYTHONHOME

# ---- westpy: run bfgs.py with the westpy-dev env's Python, by full path ----
# Do NOT `conda activate`: that puts the env's python3 (numpy 2.x) first on PATH,
# and WEST's embedded interpreter then fails with "cannot init forpy".
# PYTHONNOUSERSITE=1 hides ~/.local (production westpy 6.3.0) from both Pythons;
# cray-python ships the numpy/yaml/h5py that WEST needs.
DEV_PY=/global/homes/s/sstaylor/.conda/envs/westpy-dev/bin/python
export PYTHONNOUSERSITE=1

# ================================
# Paths
# ================================
QEDIR=/global/homes/s/sstaylor/software/QEdir_cpu/bin
WESTPY_REPO=/global/homes/s/sstaylor/testing/westpy

# Sanity check: fail fast (before burning node-hours) if the binaries
# this script depends on aren't where we expect.
if [ ! -d "$QEDIR" ]; then
    echo "ERROR: QEDIR does not exist: $QEDIR"
    exit 1
fi
for exe in pw.x westpp.x wbse.x; do
    if [ ! -x "$QEDIR/$exe" ]; then
        echo "ERROR: $QEDIR/$exe not found or not executable"
        exit 1
    fi
done

# Sanity check: westpy must be the editable dev copy, not the production one.
# (find_spec locates the package without importing it; -P keeps the current
# directory off sys.path, so a folder named westpy/ there cannot shadow it.)
WESTPY_INIT=$($DEV_PY -P -c "import importlib.util as u; print(u.find_spec('westpy').origin)")
echo "Python (westpy)     : ${DEV_PY} ($($DEV_PY --version 2>&1))"
echo "Python (WEST forpy) : $(which python3) ($(python3 --version 2>&1))"
echo "westpy              : ${WESTPY_INIT}"
echo "westpy commit       : $(git -C "$WESTPY_REPO" log -1 --format='%h %s' 2>/dev/null)"
case "$WESTPY_INIT" in
    "$WESTPY_REPO"/westpy/__init__.py|/global/u2/s/sstaylor/testing/westpy/westpy/__init__.py) ;;
    *) echo "ERROR: westpy is not the dev copy in $WESTPY_REPO"; exit 1 ;;
esac

# ================================
# Job Info Header
# ================================
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Submit Dir: $SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR" || exit 1
echo "========================================"

# ================================
# Helper function (for ALL commands)
# ================================
run_cmd () {
    echo "----------------------------------------"
    echo "Running:"
    echo "$1"
    echo "Start: $(date)"

    local t0=$SECONDS
    bash -c "$1"
    status=$?
    local t1=$SECONDS
    local dt=$(( t1 - t0 ))

    echo "End: $(date)"
    printf 'Elapsed: %02d:%02d:%02d\n' $((dt/3600)) $(((dt%3600)/60)) $((dt%60))

    if [ $status -ne 0 ]; then
        echo "ERROR: command failed"
        exit 1
    fi
}

# ================================
# Workflow
# ================================

# --== NV- DIAMOND 2x2x2: SINGLE-POINT FIXTURE (pw -> wbse -> westpp) ==--
# Initial C3v geometry, 4 TDDFT states with forces on state 1, CM decomposition 1-4.

run_cmd "srun -n 32 $QEDIR/pw.x -i pw.in > pw.out"
run_cmd "srun -n 32 $QEDIR/wbse.x -i wbse.in > wbse.out"
run_cmd "srun -n 32 $QEDIR/westpp.x -i westpp.in > westpp.out"

# identification against itself (no executables): identity, O_ii ~ 1
run_cmd "$DEV_PY -P -m westpy.tracking_cli selfcompare . > selfcompare.out"

# ================================
# Done
# ================================
echo "========================================"
echo "All steps completed successfully"
echo "End Time: $(date)"
echo "========================================"
