# Tier 4a: formaldehyde smoke test

WEST tutorial 12 excited-state relaxation (https://west-code.org/doc/training/formaldehyde/es/),
with `n_liouville_eigen: 4` and westpp (`C`, nspin = 1) over states 1-4.
BASE = /global/cfs/cdirs/m4382/sstaylor/tracking_tests/formaldehyde (CFS; Slurm
scratch licenses were held on 2026-11-12). BASE/downloads holds the tutorial
pw.in/wbse.in and the C/H/O ONCV UPFs.

    bash prepare.sh BASE      # builds BASE/0_plain, 1_off, 2_on (never overwrites)
    cd BASE/<run> && sbatch job_<run>.sh

| Run | Class | Settings |
|---|---|---|
| `0_plain` | `bfgs_iter` | maxiter 5 (reference) |
| `1_off` | `bfgs_iter_tracked` | `l_track_state=False`, maxiter 5: must match 0_plain |
| `2_on` | `bfgs_iter_tracked` | tracking + `l_test_mode=True`, maxiter 3 |

CPU build `QEdir_cpu`, debug queue, 1 node x 16 MPI x 16 CPUs, 30 min. The job
activates `westpy-dev` and exits if westpy is not the dev copy.

    conda run -n westpy-dev python check_outputs.py BASE
