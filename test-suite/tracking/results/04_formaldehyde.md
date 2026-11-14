# 4. Formaldehyde smoke test (tier 4, QE/WEST on CPU)

**Inputs:** WEST tutorial 12, excited-state part
(https://west-code.org/doc/training/formaldehyde/es/), with `n_liouville_eigen: 4`
and `n_liouville_maxiter: 1000` (diff vs the tutorial `wbse.in`: exactly these two
lines) and westpp `C` over states 1-4 (not `CM`: the multiplicity needs nspin = 2).
**Runs:** `/global/cfs/cdirs/m4382/sstaylor/tracking_tests/formaldehyde/`
(`0_plain`, `1_off`, `2_on`; failed attempts in `failed_attempt1/`, `failed_attempt2/`,
the capped run in `2_on_maxiter3/`). Templates in `tier4_cluster/formaldehyde/`.
**Machine:** CPU debug queue, `QEdir_cpu` (no CUDA), 1 node x 16 MPI x 16 CPUs.
**Check:** `conda run -n westpy-dev python tier4_cluster/formaldehyde/check_outputs.py BASE`

| Run | Class | Settings |
|---|---|---|
| `0_plain` | `bfgs_iter` | maxiter 5, reference |
| `1_off` | `bfgs_iter_tracked`, `l_track_state=False` | maxiter 5 |
| `2_on` | `bfgs_iter_tracked`, tracking + test mode | until convergence |

## Results

**Tracking off == plain `bfgs_iter`, exactly.** All 39 logged numbers, the 5
trajectory frames (energies and positions) and the 15 output files are identical
(max difference 0.0), and run 1 wrote no tracking files.

**Tracking on (run 2) converged like the tutorial:** 15 scf cycles / 14 BFGS steps,
E = -45.2741678787 Ry (tutorial: 15 cycles, -45.2741969921 Ry; 3e-5 Ry from 4
states vs 1 and a different version/machine). Its first 3 cycles are identical to
run 0. No swap, no warnings.

| | this run | tutorial | literature |
|---|---|---|---|
| C-O (A) | 1.3108 | 1.3109 | 1.308 |
| C-H (A) | 1.1017 | 1.1019 | 1.103 |
| H-C-H (deg) | 117.04 | 117.12 | 116.8 |
| out-of-plane (deg) | 31.25 | 31.11 | 30.0 |

Overlaps along the path: 0.9765 (step 2), 0.9963 (step 3), then >= 0.999. The
tracked state is band 6 -> 7 at 99.97%, the n -> pi* (1A'') state. Seen on real
data for the first time:

* **the exciton's overall sign flipped** at step 3 (X = -0.9997 vs +0.9997), O stayed 0.996;
* **the automatic band window moved** from 6-10 to 3-10, so two geometries with
  different windows were compared;
* the **nspin = 1** path (`wfc1.dat`) works.

Cost: westpp about 3 s per step; test mode 155 MB for 3 steps, stash 18 MB.

## Problems found on the way (three attempts)

1. **Scratch held by NERSC.** Since 2026-11-12 12:50 a root reservation holds all
   Slurm `scratch` licenses, so jobs submitted from `$SCRATCH` pend with
   `(Licenses)`. The runs were moved to CFS (`cfs:1` license).
2. **`cannot init forpy` in wbse.x (my mistake).** WEST embeds Python and was built
   against cray-python 3.11.7 (NumPy 1.24). The first job script did
   `conda activate westpy-dev`, which put the env's python3 (NumPy 2.4) first on
   PATH and dropped cray-python's `LD_LIBRARY_PATH`. Fix: keep cray-python for
   WEST and run only `bfgs.py` with the dev Python by full path, with
   `PYTHONNOUSERSITE=1` so neither Python sees production westpy in `~/.local`.
   A guard in the job script stops if westpy is not the dev copy.
3. **Davidson not converged at step 3.** The tutorial works with the default 100
   iterations because it computes 1 state. With 4 states it needed 29, then 70,
   then more than 100 iterations. Fix: `n_liouville_maxiter: 1000`.

Also found:

* `history.json` stored band indices as floats (`json.dump(default=float)`).
  Fixed, with regression tests.
* A failed executable makes `bfgs_iter` call `exit()` with status 0, so Slurm
  reports `COMPLETED 0:0`. Check `bfgs.out` for `failed:`.
* In attempt 2, run 0's first wbse.x took 6 min against 14 s for the identical
  computation elsewhere, with identical results: an I/O stall on CFS, not the code.
* The first run 2 had `maxiter=3` (from the plan), so it stopped unconverged; it
  was rerun without the cap.
