# 5. NV- diamond 2x2x2: single-point fixture

**What a "fixture" is here:** a fixed set of real output from one calculation,
kept on disk so the tier-3 tests can check the code against real files. This job
computed **one geometry only** (no BFGS).

**Source:** `/global/cfs/cdirs/m4382/sstaylor/projects/NV-center-diamond/2x2x2/tddft`
(copied, not moved). 63 atoms, 152 bands, ecutwfc 85 Ry, nspin = 2, charge -1,
magnetization 2, gamma point, PBE. Initial C3v geometry from `pw.in`.
**Run:** `/global/cfs/cdirs/m4382/sstaylor/tracking_tests/nv_diamond_2x2x2/fixture/`
(failed attempt in `fixture_failed1/`). Job: pw.x -> wbse.x (4 states, forces on
state 1) -> westpp.x (CM, 1-4) -> `tracking_cli selfcompare`. CPU debug,
1 node x 48 MPI x 4 CPUs. `ESPRESSO_PSEUDO=./` because `pw.in` has `pseudo_dir`
commented out.
**Test:** `pytest -m fixture tier3_fixture -k diamond` passes.

## Results

* **The westpp format is the same as for MgO:** one `westpp.json` (fresh
  directory), signed, blocks `(1,1)` up 128 x 24 and `(2,2)` down 126 x 26, full
  occupied x virtual. No code change was needed.
* **Self-comparison = identity**, min O_ii = 0.9999999999999962.
* Timing: pw.x 22 s, wbse.x 9 min 28 s (693 Davidson steps), westpp.x 9 s.

| States | Energy | Character |
|---|---|---|
| 1, 2 | 1.976 eV (6 uRy apart) | 97% spin-down 126 -> 127/128: a1 -> e in the minority channel = **3E** |
| 3, 4 | 2.604 eV (pair) | 99.9% spin-up, occupied e (127/128) -> conduction band |

The KS spin-down e pair (bands 127/128) is degenerate at 16.129 eV. The 3E pair
already shows an **arbitrary rotation inside the e pair**: state 1 is mostly
126 -> 128 (-0.974) with 0.110 of 126 -> 127, and state 2 the reverse.

## First attempt failed: Davidson at an exact degeneracy

The first run stopped after 1000 Davidson iterations (11 min) with a residual of
1.4e-7 against the 1e-8 target, and 2 eigenvalues unconverged. At the C3v
geometry all 4 states form **two exactly degenerate E doublets** (0.14525 x 2 and
0.19146 x 2 Ry), which converge slowly with the default subspace. The user's
original 2x2x2 input asked for 1 state only. Fix, taken from the user's 3x3x3
multi-state input: **`n_liouville_times: 10`** (search space of 10 vectors per
state instead of 4). pw.x was fine in both attempts (14 SCF iterations,
E = -724.44028821 Ry).
