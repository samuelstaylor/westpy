# 6. NV- diamond 2x2x2: tracked 3E relaxation (the real-data test)

**Setup:** same system as test 5. Start from the **symmetric C3v geometry** (where
the 3E pair is degenerate), `forces_state: 1`, `n_liouville_eigen: 4`,
`n_liouville_maxiter: 1000`, `n_liouville_times: 10`, westpp CM over 1-4,
`bfgs_iter_tracked` with tracking and test mode on, automatic restart.
**Run:** `/global/cfs/cdirs/m4382/sstaylor/tracking_tests/nv_diamond_2x2x2/relax/`,
CPU debug, 1 node x 48 MPI x 4 CPUs. Jobs 61297671 (stopped by the 30-min
walltime after 4 cycles) and 61299769 (restarted from cycle 4, converged).
**Check:** `cat relax/tracking/test/swap_summary.txt` and
`python -m westpy.tracking_cli replay relax/tracking/test [--no-orbitals]`

## Result

**Converged in 7 scf cycles / 6 BFGS steps**, E = -724.3058136701 Ry, force
0.0027 Ry/Bohr. The restart worked (`tracking restarted: label = 1`). No swap,
no warnings.

Comparison with the user's earlier plain-BFGS run (Midway, QE 6.2.1, 1 state, 8 cycles):

| | this run | Midway |
|---|---|---|
| final energy (Ry) | -724.30581 | -724.30567 (diff 1.4e-4 Ry, about 2 meV) |
| atomic positions | | rms 0.015 A, max 0.035 A apart |
| N-C distances (A) | 1.478 / 1.478 / 1.480 | 1.478 / 1.479 / 1.480 (initial C3v: 1.469 x 3) |

Same Jahn-Teller distortion. Timing: the first wbse.x took 18.5 min at the
degenerate C3v geometry; once the symmetry was broken, about 2 min per cycle.

## The state along the path

| Step | 3E pair (eV) | splitting | tracked state | O(1,1) |
|---|---|---|---|---|
| 1 | 1.976 / 1.976 | 0.08 meV | 126 -> **128** (weight 0.95) | reference |
| 2 | 1.745 / 1.833 | 88 meV | 126 -> **127** (weight 0.97) | 0.997 |
| 3 | 1.742 / 1.839 | 98 meV | 126 -> 127 | 1.000 |
| 4 | 1.717 / 1.832 | 115 meV | 126 -> 127 | 0.999 |
| 5 | 1.698 / 1.824 | 126 meV | 126 -> 127 | 1.000 |
| 6 | 1.691 / 1.822 | 131 meV | 126 -> 127 | 1.000 |
| 7 | 1.691 / 1.822 | 130 meV | 126 -> 127 | 1.000 |

The tracked state is the **lower Jahn-Teller branch**: it drops by 0.29 eV while
the splitting opens from 0 to 130 meV. States 3/4 (spin-up e -> CB) stay at
2.56-2.61 eV. State 1 remains state 1 throughout, which is physically correct.

## Key finding: the e orbitals swapped band numbers, and band-index tracking fails

The first BFGS step broke the symmetry, and the **spin-down e orbitals swapped
band numbers**: the orbital that was band 128 at the symmetric geometry dropped to
16.014 eV and became **band 127**.

| abs(<psi_m(step 1) / psi_n(step 2)>) | step-2 band 127 | step-2 band 128 |
|---|---|---|
| step-1 band 127 | 0.111 | **0.993** |
| step-1 band 128 | **0.993** | 0.111 |

So the tracked state, built on that orbital (126 -> 128 at step 1), *is*
126 -> 127 at step 2, i.e. state 1. I replayed the same 7 steps both ways
(`tracking_cli replay`, which runs nothing):

* **With orbital overlaps (`l_track_orbitals=True`):** correctly stays on state 1,
  O = 0.997.
* **By band index only (`l_track_orbitals=False`):** a **false "STATE SWAP 1 -> 2"
  with O = 0.973**, after which it follows the **upper** Jahn-Teller branch
  (state 2) for the rest of the run. With confidence that high, nothing would
  have warned you that it was wrong. In a real tracked run it would have put the
  forces on the wrong state.

```
orbitals on                          orbitals off (band index only)
step-2  1 -> 1  1:0.997 2:0.001      step-2  1 -> 2  1:0.112 2:0.973  STATE SWAP
step-3  1 -> 1  1:1.000 2:0.000      step-3  2 -> 2  1:0.000 2:0.977
...                                  ...
step-7  1 -> 1  1:1.000 2:0.000      step-7  2 -> 2  1:0.000 2:0.972
```

This is the real-data version of the synthetic band-swap test (test 1): wherever
degenerate orbitals split, which is exactly the Jahn-Teller situation of NV
centers, band indices are not a reliable label and `l_track_orbitals=True` is needed.

## What this test did not show

There was **no genuine crossing of excited states**: the tracked state stayed the
lowest in energy, so plain `bfgs_iter` with `forces_state: 1` would have followed
it too. Confirmed on real spin-polarized data: the wbse -> westpp pipeline at
every step, restart with tracking, the degenerate-orbital reordering, and that
band-index tracking gives the wrong answer here. A forced wbse rerun after a
genuine swap has so far only been tested with the fake executables (test 2); the
1B1/1B2 MgO relaxations are the natural real-data test.

## Practical notes

* A walltime kill interrupts wbse.x, which `bfgs_iter` logs as "wbse.x failed".
  That is not a WEST error (Slurm: `CANCELLED ... DUE TO TIME LIMIT`, no `CRASH`
  file). Resubmitting continues from `bfgs_tmp.json` / `bfgs_tracking.json`; the
  interrupted step is recomputed.
* Test mode kept all 7 step directories under `tracking/test/`, which is what made
  the replay comparison above possible.
