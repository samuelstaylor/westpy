# 2. Full BFGS loop with fake executables (tier 2)

**Directory:** `test-suite/tracking/tier2_mockwest/` (11 tests, about 1 min)
**Run:** `cd test-suite/tracking && conda run -n westpy-dev python -m pytest -v tier2_mockwest`
**Result:** 11 passed.

## What is tested

The real `bfgs_iter` / `bfgs_iter_tracked` loop end to end, with `fake_west.py`
standing in for pw.x, wbse.x and westpp.x. The fakes are small Python scripts
(not QE/WEST) that write files with the real layouts: `data-file-schema.xml`,
Fortran `wfc*.dat` read by `read_wfc`, `wbse.json` and `westpp*.json`, including
WEST's habit of writing `wbse_1.json` / `westpp_1.json` instead of overwriting.

The model has a 1D coordinate q (x displacement of atom 1). Excitons A and B
**cross at q = 0.05 Bohr**, and the virtual bands swap order at the crossing.
Along the way the degenerate occupied pair rotates and some band signs flip. The
run starts at q = 0 on state 1 = A; the first BFGS step lands at q of about 0.1,
past the crossing.

* Tracking on: A must be followed, so state 1 -> 2 at step 2, converging at A's
  minimum q = 0.0625 Bohr.
* Plain BFGS follows "state 1", which is B after the crossing, and ends at q = 0.1875 Bohr.

## Important results

| Test | Result |
|---|---|
| Pass-through (`l_track_state=False`) | **Same commands, same files (byte for byte), same log** as `bfgs_iter`; no `tracking/`, no `wbse.in.orig` |
| Planted crossing, test mode | `STATE SWAP: 1 -> 2` at step 2; step 2 ran pw, wbse, westpp, **wbse again** (rerun only); `wbse_state-1.out` and `wbse_state-1.json` kept; trajectory energy of step 2 corrected; converged at A's minimum |
| Orbitals off | the swap is invisible by band index, so the run **ends on the wrong state** (B's minimum) |
| Restart | stopped after 2 steps, restarted with `l_restart=True`: label 2 and the stash were restored, the history is continuous, one swap in total |
| Low overlap | random new orbitals after the crossing: warning by default, `SystemExit` with `stop_on_low_overlap=True` |
| Replay (dry run) | `replay()` over `tracking/test/` reproduces the swap and the recorded overlaps; **no executable is launched** |
| Others | startup warnings reach the log; `run_westpp` is required; the `step-1` mkdir fix works |

## Bug found

**Trajectory rewrite wrote `energy=np.float64(0.298)`.** After a swap the last
frame of `bfgs_traj.xyz` is rewritten with the new energy. The code used
`repr(energy)`, and in NumPy 2 the repr of an `np.float64` is `np.float64(...)`,
which would have corrupted `bfgs_traj.xyz` in production. Tier 1 had missed it
because that test passed a plain Python float. Fixed with `repr(float(energy))`;
the tier-1 test now uses `np.float64`.
