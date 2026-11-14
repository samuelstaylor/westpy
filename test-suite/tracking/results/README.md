# Test results: excited-state tracking (`bfgs_iter_tracked`)

One file per test, with what it checks, how it was run, the numbers, and what
we learned. Dates are when the runs were done (Nov 2026, Perlmutter).

| # | Test | Tier | Data | Outcome |
|---|---|---|---|---|
| 1 | [Synthetic unit tests](01_unit_tests.md) | 1 | synthetic arrays | 51/51 pass |
| 2 | [Full BFGS loop with fake executables](02_mockwest_loop.md) | 2 | fake pw/wbse/westpp | 11/11 pass; found the trajectory-energy bug |
| 3 | [MgO VEE fixture](03_mgo_vee_fixture.md) | 3 | real WEST output (NV- in MgO) | westpp format established; self-comparison = identity |
| 4 | [Formaldehyde smoke test](04_formaldehyde.md) | 4 | QE/WEST on CPU, tutorial 12 | tracking off = `bfgs_iter` exactly; tracking on converges like the tutorial |
| 5 | [NV- diamond single-point fixture](05_nv_diamond_fixture.md) | 3/4 | QE/WEST, spin-polarized | format confirmed for nspin = 2; self-comparison = identity |
| 6 | [NV- diamond tracked relaxation](06_nv_diamond_relaxation.md) | 4 | QE/WEST, 3E Jahn-Teller | converges; **e-orbital band swap: band-index tracking would have failed** |

Bugs and pitfalls found along the way are listed in each file. In short:

* a NumPy 2 repr bug in the trajectory rewrite (`energy=np.float64(...)`), found by test 2;
* band indices written as floats in `history.json`, found by test 4;
* WEST's embedded Python (forpy) needs the cray-python it was built with, not the conda env (test 4);
* with 4 TDDFT states the Davidson solver needs `n_liouville_maxiter: 1000` (formaldehyde) and,
  at an exactly degenerate C3v geometry, also `n_liouville_times: 10` (diamond);
* `bfgs_iter` exits with status 0 when an executable fails, and a walltime kill is
  logged as "<exe> failed": check `bfgs.out`, not only the Slurm state.

How to rerun everything: see `../README.md`.
