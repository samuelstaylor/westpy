# Excited-state tracking in BFGS relaxations (`bfgs_iter_tracked`)

TDDFT (wbse.x) labels excited states by energy. When states cross during an
excited-state relaxation, `forces_state = i` silently switches to a different
electronic state. `bfgs_iter_tracked` (in `westpy/relaxation_tracked.py`) is a
subclass of `bfgs_iter` that can follow a state by its *character* instead.

```python
from westpy import *

bfgs = bfgs_iter_tracked(run_pw=run_pw, run_wbse=run_wbse, pp=pp,   # as for bfgs_iter
                         l_track_state=True,
                         run_westpp="srun -n 16 /path/to/westpp.x",
                         westpp_input="westpp.in")
bfgs.solve()
```

A complete example (NV- in MgO with nscf and wbse_init) is in
`test-suite/tracking/tier4_cluster/examples/bfgs_tracked_mgo.py`, with a
westpp input in `westpp_tracking.in`.

## Flags

All `bfgs_iter` arguments are passed through unchanged.

| Argument | Default | Meaning |
|---|---|---|
| `l_track_state` | `False` | Master switch. If False the class behaves exactly like `bfgs_iter` (no westpp runs, no wbse.in edits, no extra files). |
| `l_track_orbitals` | `True` | Fold KS orbital overlaps between geometries into the excitation overlap. If False, decompositions are compared by band index (S = identity), which misses band reorderings. |
| `l_test_mode` | `False` | Keep every step directory and stash under `tracking/test/` (uses a lot of disk: wavefunction files are copied). |
| `run_westpp` | `None` | Full command to run westpp.x. Required if tracking is on. |
| `westpp_input` | `"westpp.in"` | Your westpp input. Copied into every step directory and used as is. It needs `westpp_calculation` containing `C`, a `westpp_range` that covers the tracked states, and preferably `westpp_sign: true`. |
| `band_window` | `None` | KS bands used in the overlaps: `None` (automatic), `(lo, hi)` for all spins, or `{spin: (lo, hi)}`; 1-based. |
| `band_window_weight` | `0.999` | Weight of the decomposition (per spin, all decomposed states) kept by the automatic window. |
| `overlap_thr` | `0.5` | Warn if the best overlap is below this value. |
| `ambiguity_thr` | `0.1` | Warn if the two best candidates differ by less than this. |
| `stop_on_low_overlap` | `False` | Stop (instead of warning) on low overlap or ambiguity. |
| `degeneracy_thr` | `1e-3` | Eigenvalue gap (Ry) defining near-degenerate KS blocks for the diagnostic. |

At startup the westpp input is checked against wbse.in and pw.in, and problems
are **reported, never fixed**: decomposed states not covering `forces_state` or
exceeding `n_liouville_eigen`, `westpp_n_liouville_to_use` smaller than the range,
`westpp_l_spin_flip` inconsistent with `wbse_control.l_spin_flip` (nspin = 2;
for nspin = 1 it must be False, and `spin_excitation` is not involved), differing
`qe_prefix`/`west_prefix`/`outdir`, unknown `westpp_control` keys (e.g. the
non-existent `westpp_l_compute`), missing `C` calculation, `l_forces` not set, and
unsigned output.

## Algorithm (each BFGS step, tracking on)

1. Set `wbse_control.forces_state` in the root wbse.in to the tracked label i
   (YAML load/dump, only that key; the original is kept as `wbse.in.orig`).
2. Run the normal step (`bfgs_iter._run_calc`: pw, nscf, wbse_init, wbse with forces on i).
3. Run westpp.x in the step directory with your westpp input.
4. Read the decomposition from the newest `west.westpp.save/westpp*.json` with a
   `C` calculation (westpp.x never overwrites its JSON). Coefficients are
   `X[v, c] = vals[(v - v0) * nc + (c - c0)]` per spin block `K00000s`, with
   1-based global band indices.
5. If `l_track_orbitals`, compute the KS overlaps `S_up`, `S_dw` between the
   previous and current geometry with `westpy.qe_wfc_overlap.overlap`, using
   `<prefix>.save` as it is when wbse.x runs (bfgs_iter restores the SCF
   wavefunctions after nscf; only the ACE operators come from nscf).
6. Overlap of the tracked state i (previous geometry) with every decomposed state j:
   * signed: `O_ij = |sum X_vc^i S_vv' S_cc' X_v'c'^j| / (|X^i| |X^j|)`,
   * weights only: `sum w_vc^i |S_vv'|^2 |S_cc'|^2 w_v'c'^j / (|w^i| |w^j|)`,
   combining only blocks of the same (from_spin, to_spin), inside the band window.
7. `j* = argmax_j O_ij`. If `j* = i` nothing else happens. If `j* != i`
   (**STATE SWAP**): `wbse.out` is renamed `wbse_state-i.out` (and `wbse.json`
   `wbse_state-i.json`), `forces_state = j*`, **only wbse.x** is rerun in the same
   step directory, the energy and gradient are re-read, and the energy of the
   last frame of `bfgs_traj.xyz` is corrected.
8. The current geometry is stashed as the new reference, and the label becomes j*.

On the first step the initial `forces_state` is the label and the step is the
reference.

**Reference geometry.** The comparison is always with the most recently
*computed* geometry, also when BFGS rejects a step and goes back to the previous
positions: it is the closest one in configuration space.

**Degenerate orbitals.** Within a degenerate KS block (e.g. the NV- *e* pair)
QE returns an arbitrary rotation. The S-matrix formulation is invariant under
it; `sum |S|^2 / n` over each near-degenerate block is logged as a diagnostic
(about 1 means the subspace is preserved). The weights-only formula is not
rotation invariant, so use `westpp_sign: true`.

**Known limitation.** When the wbse-only rerun targets one member of a
degenerate pair of *excitations*, wbse.x may return a differently rotated pair.
The rerun's excitation energy is logged next to the first run's and the
westpp energy of j*, as a check.

## Outputs

| File | Content |
|---|---|
| `bfgs_tracking.json` | Restart file: tracked label, history, path of the reference stash (`bfgs_tmp.json` is not touched). |
| `tracking/history.json` | One record per step: labels before/after, swap, overlap row, warnings, band windows, retained weight, degenerate-block diagnostics, dominant transitions, energies, total force. |
| `tracking/stash/step-N/` | Reference geometry: `ks_s<spin>.npz` (exactly the `read_wfc` keys `miller, evc, gamma_only, npol, bands`), `decomp.npz`, `eigenvalues.npz`, `meta.json`. Only the latest is kept unless test mode. (NV- MgO: about 0.5 GB, 20 s to write.) |
| `tracking/test/step-N/` | Test mode: full copy of the step directory (both wbse outputs after a swap) and `tracking_summary.txt`. |
| `tracking/test/swap_summary.txt` | Test mode: one line per step (step, forces_state, swap, max overlap). |

The log shows, per step, the label before and after, the overlap row
`O(i,j)`, the band windows, warnings, and a `STATE SWAP: i -> j` line.

Restart works with `l_restart=True` as for `bfgs_iter`; the label and the
reference stash are read from `bfgs_tracking.json`.

## Dry-run tools

No executables are launched:

```bash
python -m westpy.tracking_cli selfcompare <dir>   # identification of a run against itself
python -m westpy.tracking_cli replay <run>/tracking/test   # replay a test-mode relaxation
```

## Tests

`test-suite/tracking/` (see its README): tier 1 synthetic unit tests, tier 2 the
full loop with fake executables, tier 3 real WEST output, tier 4 cluster templates.
