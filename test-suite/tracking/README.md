# Tests for excited-state tracking (`bfgs_iter_tracked`)

Four tiers, from fast and synthetic to real cluster runs. Run Python through the
dev environment (`westpy-dev`, with `PYTHONNOUSERSITE=1`).

| Tier | Directory | What it tests | Data | Run |
|---|---|---|---|---|
| 1 | `tier1_unit/` | westpp JSON parsing, band window, overlaps and identification (band swap, sign flip, rotated degenerate pair; orbitals on/off), validation warnings, wbse.in edit, stash round trip, trajectory rewrite | synthetic (`tracking_model.py`) | always, ~2 s |
| 2 | `tier2_mockwest/` | the whole BFGS loop with fake `pw.x`/`wbse.x`/`westpp.x` (`fake_west.py`): pass-through identical to `bfgs_iter`, planted crossing (swap, wbse-only rerun, test-mode files, trajectory), orbitals off missing the swap, restart, stop on low overlap, replay, `step-1` mkdir fix | synthetic files | always, ~1 min |
| 3 | `tier3_fixture/` | real WEST output: JSON vs `westpp-CM.out`, self-comparison = identity with O_ii ~ 1, real-size stash | `$SCRATCH/tracking_fixture` (MgO VEE), `/global/cfs/cdirs/m4382/sstaylor/tracking_tests/nv_diamond_2x2x2/fixture` (NV- diamond) | `-m fixture`, ~40 s |
| 4 | `tier4_cluster/` | job templates you submit: formaldehyde smoke test, NV- diamond 2x2x2 crossing test, MgO example `bfgs.py` | Perlmutter | by hand |

The fake executables are small Python scripts, not QE/WEST: no real code is run by the tests.

**Results of every test, with what was found:** [`results/`](results/README.md).

## Commands

```bash
cd ~/testing/westpy/test-suite/tracking
conda run -n westpy-dev python -m pytest -v                 # tiers 1 + 2
conda run -n westpy-dev python -m pytest -v -m fixture      # tier 3 (real data)
# other fixture locations:
TRACKING_FIXTURE=/path/to/run conda run -n westpy-dev python -m pytest -m fixture
```

Dry-run tools on real directories (no executables are launched):

```bash
conda run -n westpy-dev python -m westpy.tracking_cli selfcompare $SCRATCH/tracking_fixture
conda run -n westpy-dev python -m westpy.tracking_cli replay <run_dir>/tracking/test
```

## The synthetic model

`tracking_model.py` defines orbitals and excitons in a fixed orbital basis and
expresses them in the band basis a calculation would report (`X = C A C^T`,
`psi_b = sum_o C[b, o] phi_o`), with controllable virtual-band swaps, band sign
flips and rotations inside a degenerate occupied pair. `fake_west.py` uses the
same model along a 1D coordinate q in which excitons A and B cross at
q = 0.05 Bohr and the virtual bands swap order at the crossing. With tracking,
BFGS follows A to q = 0.0625 Bohr; plain BFGS (and tracking with
`l_track_orbitals=False`) ends on B at q = 0.1875 Bohr.
