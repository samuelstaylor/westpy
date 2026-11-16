# Excited-state tracking for westpy BFGS relaxations

**Branch:** `state-tracking` on `samuelstaylor/westpy` (based on upstream westpy v6.4.0, commit `3d965f3`)
**New class:** `bfgs_iter_tracked` in `westpy/relaxation_tracked.py`
**Status:** implemented, tested, pushed to the fork; not proposed upstream.

---

## Contents

1. [The problem](#1-the-problem)
2. [Summary of changes](#2-summary-of-changes)
3. [Changes to existing westpy files](#3-changes-to-existing-westpy-files)
4. [The new class: `bfgs_iter_tracked`](#4-the-new-class-bfgs_iter_tracked)
5. [The physics and math](#5-the-physics-and-math)
6. [What happens in one BFGS step](#6-what-happens-in-one-bfgs-step)
7. [Output files](#7-output-files)
8. [Command-line tools](#8-command-line-tools)
9. [Tests](#9-tests)
10. [Results on real systems](#10-results-on-real-systems)
11. [Bugs found and fixed](#11-bugs-found-and-fixed)
12. [Limitations and next steps](#12-limitations-and-next-steps)

---

## 1. The problem

In a TDDFT excited-state relaxation, `wbse.x` computes the forces on the excited
state selected by `forces_state: i` in `wbse.in`. The states are numbered by
**energy order**, so `i` means "the i-th lowest state at this geometry".

When two excited states **cross** during the relaxation, their order changes. With a
fixed `forces_state: i`, the relaxation then **silently switches to a different
electronic state**: from then on the forces belong to a different state, and the
final geometry is not the minimum of the state you started on.

`bfgs_iter_tracked` labels the state by its **character** instead of its energy
order. At every step it finds which state at the new geometry is the continuation
of the tracked state, and puts the forces on that state.

---

## 2. Summary of changes

5 commits, 42 files, +4363 / -1 lines relative to upstream.

| Commit | Description |
|---|---|
| `4976a21` | `relaxation.py`: reuse an existing `step-1` directory instead of failing |
| `5774607` | Add optional excited-state tracking to BFGS relaxation (`bfgs_iter_tracked`) |
| `811ba6c` | Keep integers in `history.json`; add formaldehyde and NV- diamond cluster tests |
| `cfadb35` | Test write-ups; converged formaldehyde run; diamond solver settings |
| `f21e978` | Tests README: point to the actual NV- diamond fixture location |

| File | Lines | Kind |
|---|---|---|
| `westpy/relaxation.py` | +2 / -1 | bug fix (existing file) |
| `westpy/__init__.py` | +1 | one import line (existing file) |
| `westpy/relaxation_tracked.py` | 1207 | **new**: the feature |
| `westpy/qe_wfc_overlap.py` | 112 | **new**: KS wavefunction reader and overlaps |
| `westpy/tracking_cli.py` | 12 | **new**: command-line entry point |
| `STATE_TRACKING.md` | 120 | **new**: user documentation |
| `test-suite/tracking/` | ~2900 | **new**: four tiers of tests, job templates, results |

---

## 3. Changes to existing westpy files

Only two existing files were touched.

### `westpy/relaxation.py`: the `step-1` fix

```python
if self.scf_iter == 1:
    if not os.path.exists(work_dir):   # added
        os.mkdir(work_dir)
```

Rerunning a relaxation in a directory where `step-1/` already existed crashed with
`FileExistsError`. This is a bug in plain `bfgs_iter`, independent of tracking, so
it is its own commit and could be its own pull request.

### `westpy/__init__.py`

```python
from westpy.relaxation_tracked import *
```

so that `from westpy import *` exposes `bfgs_iter_tracked`.

---

## 4. The new class: `bfgs_iter_tracked`

`bfgs_iter_tracked` is a **subclass of `bfgs_iter`**. It accepts every `bfgs_iter`
argument unchanged, plus:

| Argument | Default | Meaning |
|---|---|---|
| `l_track_state` | `False` | Master switch. When `False` the class is **exactly** `bfgs_iter`: nothing extra is read, written or run. |
| `l_track_orbitals` | `True` | Use KS orbital overlaps between geometries. When `False`, orbitals are matched by band index only. |
| `l_test_mode` | `False` | Keep every step directory and stash under `tracking/test/` (uses a lot of disk). |
| `run_westpp` | `None` | Full command for `westpp.x` (required when tracking). |
| `westpp_input` | `"westpp.in"` | The user's westpp input, copied as-is into every step. |
| `band_window` | `None` | `None` = automatic; `(lo, hi)` or `{spin: (lo, hi)}` with 1-based band indices. |
| `band_window_weight` | `0.999` | Weight fraction kept by the automatic band window. |
| `overlap_thr` | `0.5` | Warn if the best overlap is below this value. |
| `ambiguity_thr` | `0.1` | Warn if the two best overlaps differ by less than this. |
| `stop_on_low_overlap` | `False` | Stop instead of warning on low or ambiguous overlap. |
| `degeneracy_thr` | `1e-3` Ry | Eigenvalue gap defining a degenerate orbital block (diagnostic only). |

### Example

```python
from westpy import *

bfgs = bfgs_iter_tracked(
    run_pw="srun -n 48 pw.x",
    run_wbse="srun -n 48 wbse.x",
    run_westpp="srun -n 48 westpp.x",
    pw_input="pw.in",
    wbse_input="wbse.in",
    westpp_input="westpp.in",
    l_track_state=True,
    l_test_mode=False,
    l_restart=os.path.exists("bfgs_tmp.json"),
)
bfgs.solve()
```

`westpp.in` must ask for the excitation decomposition, with signed coefficients:

```yaml
westpp_control:
  westpp_calculation: CM          # C for nspin = 1
  westpp_range: [1, 4]            # must contain the tracked state and its neighbours
  westpp_n_liouville_to_use: 4    # >= upper end of westpp_range
  westpp_format: C
  westpp_sign: true               # signed coefficients
  westpp_l_spin_flip: false       # must match wbse_control.l_spin_flip (nspin = 2)
```

### Which methods are overridden

| Method | What the override adds | `super()` call |
|---|---|---|
| `__init__` | validate inputs, back up `wbse.in`, set up `tracking/`, restore state on restart | first |
| `_run_calc` | set `forces_state`, run westpp, identify the state, rerun wbse.x on a swap | in the middle |
| `_write_new_pw_input` | stash this geometry before the step directory is deleted | last |
| `_save_final_geo` | copy the final step (test mode), write the tracking state | last |
| `_terminate_bfgs` | log a summary of all swaps | last |

Every override starts with `if not self.l_track_state: return super()...()`, which is
why tracking-off is identical to `bfgs_iter` (verified to the last digit, see §10).

---

## 5. The physics and math

### 5.1 Excitations in the transition basis

In linear-response TDDFT (as solved by `wbse.x`), excited state $i$ is described by
its amplitudes on KS transitions from an occupied band $v$ to a virtual band $c$,
for each spin channel $\sigma$:

$$
|\,i\,\rangle \;\simeq\; \sum_{\sigma} \sum_{v\in\text{occ}_\sigma} \sum_{c\in\text{vir}_\sigma} X^{i}_{\sigma,vc}\; |\,v\sigma \rightarrow c\sigma\,\rangle .
$$

`westpp.x` with `westpp_calculation: C` (`CM` for `nspin = 2`) writes these
coefficients to `west.westpp.save/westpp*.json`:

```
output.E00000i.K00000s.projection = {from_spin, to_spin, from_bandrange, to_bandrange, vals}
```

* `K000001` = spin up, `K000002` = spin down; band indices are 1-based KS indices.
* `vals` is **occupied-major**:

$$
X^i_{\sigma,vc} = \texttt{vals}\big[(v - v_0)\,n_c + (c - c_0)\big],
\qquad n_c = c_1 - c_0 + 1 .
$$

  This ordering reproduces every coefficient printed in `westpp-CM.out`, signs
  included; the transposed order does not.
* With `westpp_sign: true` the coefficients are **signed** (needed for the signed
  overlap below). Without it, westpp gives weights $w_{vc} = |X_{vc}|^2$.
* The printed coefficients sum to $\sum |X|^2 \approx 0.987$ - $0.998$, so the code
  normalizes explicitly (§5.4).
* `westpp.x` never overwrites its JSON (`westpp.json`, `westpp_1.json`, ...). The code
  picks the **newest** file whose `westpp_calculation` contains `C`.

### 5.2 KS orbital overlaps between two geometries

Let geometry $A$ be the previous step and $B$ the current one. For each spin, the
overlap between KS orbital $m$ at $A$ and $n$ at $B$ is computed from the plane-wave
coefficients $c_m(\mathbf G)$ (`qe_wfc_overlap.overlap`):

$$
S^{\sigma}_{mn} \;=\; \langle \psi^{A}_{m\sigma} \,|\, \psi^{B}_{n\sigma} \rangle
\;=\; \sum_{\mathbf G} c^{A*}_{m\sigma}(\mathbf G)\, c^{B}_{n\sigma}(\mathbf G),
$$

summed over the G-vectors common to both (matched by Miller indices). At the Γ point
QE stores only half the G-sphere ($\psi(-\mathbf G) = \psi(\mathbf G)^*$), so

$$
S_{mn} \;=\; 2\,\mathrm{Re}\!\sum_{\mathbf G \in \text{half}} c^{A*}_{m}(\mathbf G)\,c^{B}_{n}(\mathbf G) \;-\; c^{A}_{m}(\mathbf 0)\,c^{B}_{n}(\mathbf 0).
$$

This compares orbitals between two different atomic geometries in the same cell and
basis; it is a good measure of orbital continuity for the small steps BFGS takes.

With `l_track_orbitals=False` the orbitals are matched **by band index only**:

$$
S^{\sigma}_{mn} = \delta_{mn}.
$$

### 5.3 Overlap between excited states at two geometries

The overlap between state $i$ at $A$ and state $j$ at $B$ is the overlap of the two
transition expansions, with each occupied and each virtual orbital mapped through
$S$:

$$
\boxed{\;
O_{ij} \;=\; \frac{\Big|\displaystyle\sum_{\sigma}\sum_{v,c}\sum_{v',c'}
X^{i}_{\sigma,vc}\; S^{\sigma}_{vv'}\; S^{\sigma}_{cc'}\; X^{j}_{\sigma,v'c'}\Big|}
{\big\|X^{i}\big\|\,\big\|X^{j}\big\|}\;},
\qquad
\big\|X\big\|^2 = \sum_{\sigma,v,c} \big|X_{\sigma,vc}\big|^2 .
$$

In matrix form, per spin block, the numerator is
$\sum_{\sigma}\mathrm{Tr}\!\left[(S^{\sigma}_{\text{occ}})^{T} X^{i}_\sigma\, S^{\sigma}_{\text{vir}}\,(X^{j}_\sigma)^{T}\right]$,
which is what the code evaluates (`excitation_overlap`).

Properties:

* $0 \le O_{ij} \le 1$; $O_{ij} = 1$ when state $j$ is state $i$ carried smoothly to the new geometry.
* The absolute value makes it insensitive to the **arbitrary overall sign** of an
  eigenvector (seen in formaldehyde, §10).
* It is insensitive to **sign flips of individual KS orbitals**: a flip of orbital $m$
  changes the sign of both $X_{m\cdot}$ and row $m$ of $S$, which cancel.
* It is insensitive to **reordering of KS bands** and to an arbitrary **rotation inside
  a degenerate orbital pair**, because $S$ carries the orbitals to their new labels.
  This is exactly what band-index matching ($S = \delta$) gets wrong (§10, NV- diamond).
* Only blocks with the same (from-spin, to-spin) are combined.

If the decomposition is unsigned (weights $w = |X|^2$), a weaker weight overlap is used:

$$
O^{(w)}_{ij} = \frac{\sum w^{i}_{vc}\,|S_{vv'}|^2\,|S_{cc'}|^2\,w^{j}_{v'c'}}{\|w^i\|\,\|w^j\|}.
$$

### 5.4 Band windows

Reading every KS orbital at every step would be expensive, so each spin is restricted
to a **band window** $[\ell_\sigma, h_\sigma]$. The weight carried by band $b$ of spin
$\sigma$, summed over all decomposed states (each normalized to weight 1), is

$$
W_\sigma(b) = \sum_{i} \frac{1}{N_i}\left[\;\sum_{c} |X^{i}_{\sigma,bc}|^2 \;+\; \sum_{v} |X^{i}_{\sigma,vb}|^2\right],
\qquad N_i = \sum_{\sigma,v,c}|X^i_{\sigma,vc}|^2 .
$$

The automatic window is the **smallest contiguous range** holding at least a fraction
$f$ (`band_window_weight`, default 0.999) of that spin's total weight:

$$
[\ell_\sigma, h_\sigma] = \arg\min_{[\ell,h]} (h-\ell)
\quad\text{such that}\quad
\sum_{b=\ell}^{h} W_\sigma(b) \;\ge\; f \sum_b W_\sigma(b) ,
$$

found with a sliding window in $O(N_{\text{bands}})$. A user-supplied `band_window`
overrides it. The norms in §5.3 use only the part of each state inside the window,
and the **retained weight** (the fraction of $\|X^i\|^2$ inside the window) is logged
so a too-narrow window is visible.

### 5.5 Identifying the tracked state

With tracked label $i$ at the previous geometry, the overlap row $O_{ij}$ is computed
for every decomposed state $j$ at the current geometry, and

$$
j^{*} = \arg\max_j \; O_{ij}.
$$

Warnings (or a stop, with `stop_on_low_overlap=True`) are raised when

| Condition | Meaning |
|---|---|
| $O_{ij^*} < $ `overlap_thr` | no state at the new geometry looks like the tracked one |
| $O_{ij^*} - O_{ij_2} < $ `ambiguity_thr` | two states match about equally well ($j_2$ = second best) |
| $j^* = \max(\texttt{westpp\_range})$ | the tracked state is at the top of the computed range and may be leaving it |

If $j^* \ne i$, this is a **state swap**: the forces are recomputed on $j^*$ and the
label becomes $j^*$.

The comparison is always against the **most recently computed** geometry, also when
BFGS rejects a step and returns to an earlier one, because it is the closest in
configuration space to the next geometry.

### 5.6 Degenerate-orbital diagnostic

For each group $D$ of consecutive bands at the previous geometry with eigenvalue gaps
below `degeneracy_thr`, the code logs

$$
\Delta_D = \frac{1}{|D|}\sum_{m,n\in D} |S_{mn}|^2 .
$$

$\Delta_D \approx 1$ means the degenerate subspace is preserved (an arbitrary rotation
inside it is harmless for $O_{ij}$); $\Delta_D$ clearly below 1 means the block mixed
with other bands. It is a diagnostic only and does not change the identification.

### 5.7 Self-comparison check

Comparing a geometry with itself ($A = B$) must give the identity, $O_{ij} = \delta_{ij}$.
`self_compare` computes the full matrix; on real data the diagonal is
$1 - O_{ii} < 10^{-14}$ (§10). This checks the parsing, the band windows and the
overlap code end to end.

---

## 6. What happens in one BFGS step

```
 tracked label i (from forces_state at start, then from the previous step)
        |
  1.  write forces_state: i into wbse.in
  2.  bfgs_iter step: pw.x  [-> nscf -> wbse_init.x]  -> wbse.x (forces on i)
  3.  westpp.x  -> X^j_{vc} for every state j in westpp_range
  4.  load: decomposition, band windows, KS orbitals in the windows, eigenvalues
  5.  O_ij for all j against the stash of the previous geometry; j* = argmax
        |-- warnings: low overlap / ambiguous / top of range
        |-- degenerate-block diagnostic
  6.  if j* != i:  STATE SWAP
        - rename wbse.out, wbse.json -> *_state-i.*
        - forces_state: j*, rerun ONLY wbse.x in the same step directory
        - read the new energy and forces
        - rewrite the energy of the last trajectory frame
  7.  label := j*; append to history; write bfgs_tracking.json
  8.  BFGS update (bfgs_iter); before the step directory is deleted,
      stash this geometry as the next reference
```

* **First step:** there is no reference, so it is only stored.
* **Restart** (`l_restart=True`): the label, history and stash path are read from
  `bfgs_tracking.json`; tracking continues where it stopped. The interrupted step is
  recomputed by `bfgs_iter` as usual.
* **The user's `wbse.in`** is saved once as `wbse.in.orig`; only `forces_state` is
  edited afterwards.
* **Input validation at startup** (reports, never fixes), as warnings in `bfgs.out`:
  * no `westpp_control` section, or unknown keys in it (e.g. `westpp_l_compute`, which
    is not a WEST key; the real one is `westpp_l_compute_tdm`);
  * `westpp_calculation` without `C` (no decomposition would be written);
  * `westpp_range` not covering the initial `forces_state`, or exceeding
    `westpp_n_liouville_to_use` or `n_liouville_eigen`;
  * `wbse_control.l_forces` not `True`;
  * `westpp_l_spin_flip` different from `wbse_control.l_spin_flip` (nspin = 2), or set with nspin = 1;
  * `westpp_sign` not `True` (only weights can be compared);
  * `qe_prefix`, `west_prefix` or `outdir` different between `wbse.in` and `westpp.in`.

---

## 7. Output files

| File | Content |
|---|---|
| `bfgs.out` | the usual log, plus `tracking:` lines: windows, overlap row, label before/after, `STATE SWAP`, warnings |
| `bfgs_tracking.json` | restart state: label, history, stash path |
| `tracking/history.json` | one record per step: labels, overlap row, max overlap, windows, retained weight, degenerate blocks, energies, total force, dominant transitions |
| `tracking/stash/step-N/` | reference for the next step: `ks_spin*.npz` (windowed wavefunctions in `read_wfc` format: `miller, evc, gamma_only, npol, bands`), decomposition, eigenvalues. Only the latest is kept unless test mode is on. |
| `tracking/test/step-N/` | **test mode only:** full copy of every step directory, plus `tracking_summary.txt` |
| `tracking/test/swap_summary.txt` | **test mode only:** one line per step |
| `step-N/wbse_state-i.out`, `.../wbse_state-i.json` | after a swap: the first `wbse.x` run, kept |

---

## 8. Command-line tools

These run nothing (no QE/WEST executables); they only read existing files.

```bash
# a geometry against itself: the overlap matrix should be the identity
python -m westpy.tracking_cli selfcompare <step_dir>

# redo the tracking over the saved steps of a test-mode run
python -m westpy.tracking_cli replay <run_dir>/tracking/test
python -m westpy.tracking_cli replay <run_dir>/tracking/test --no-orbitals   # band index only
```

`replay` is how the orbital-tracking result on NV- diamond (§10) was obtained.

---

## 9. Tests

All in `test-suite/tracking/`. Run with the dev environment:

```bash
cd test-suite/tracking
conda run -n westpy-dev python -m pytest -v               # tiers 1 + 2
conda run -n westpy-dev python -m pytest -v -m fixture    # tier 3 (real data)
```

**Result:** 75 passed, 6 skipped for the whole test suite (13 existing + 62 tracking);
the 6 tier-3 tests pass with `-m fixture`.

| Tier | Directory | What it tests | Tests |
|---|---|---|---|
| 1 | `tier1_unit/` | westpp JSON parsing, band windows, overlaps and identification (band swap, sign flips, rotated degenerate pair; orbitals on and off), input validation, `wbse.in` edit, stash round trip, trajectory rewrite | 51 |
| 2 | `tier2_mockwest/` | full BFGS loop with fake `pw.x` / `wbse.x` / `westpp.x`: tracking-off identical to `bfgs_iter`; a planted crossing is followed (swap, wbse-only rerun, trajectory fix); orbitals-off misses it; restart; stop on low overlap; replay; `step-1` fix | 11 |
| 3 | `tier3_fixture/` | real WEST output (MgO VEE, NV- diamond): JSON vs `westpp-CM.out`, self-comparison = identity, real-size stash | 6 |
| 4 | `tier4_cluster/` | Perlmutter CPU job templates: formaldehyde, NV- diamond 2x2x2 fixture and relaxation, MgO example | by hand |

### The synthetic model (tiers 1 and 2)

`tracking_model.py` builds orbitals $\phi_o$ and excitons $A$ in a fixed orbital basis,
then expresses them in the band basis a calculation would report:

$$
\psi_b = \sum_o C_{bo}\,\phi_o, \qquad X = C\,A\,C^{T},
$$

where $C$ contains controllable band swaps, sign flips and rotations inside a
degenerate pair. `fake_west.py` uses the same model along a 1D coordinate $q$ with
two excitons that cross at $q = 0.05$ Bohr while the virtual bands swap order. With
tracking, BFGS follows exciton A to $q = 0.0625$ Bohr; plain BFGS (and tracking with
`l_track_orbitals=False`) ends on exciton B at $q = 0.1875$ Bohr.

Detailed write-ups of every test: `test-suite/tracking/results/01` - `06`.

---

## 10. Results on real systems

| System | Result |
|---|---|
| **MgO NV- VEE** (single point, 10 states, DDH) | westpp format confirmed; self-comparison = identity, $\min O_{ii} = 0.99999999999999957$ |
| **Formaldehyde, tracking off** | identical to plain `bfgs_iter`: all 39 logged numbers, 5 trajectory frames and 15 output files, max difference 0.0 |
| **Formaldehyde, tracking on** (WEST tutorial 12, 4 states) | converged in 15 cycles like the tutorial, E = -45.2741678787 Ry (tutorial -45.2741969921); geometry within 0.0002 A / 0.14 deg of the tutorial; tracked n -> pi* state throughout; overall eigenvector sign flipped at step 3 with O = 0.996 (handled) |
| **NV- diamond 2x2x2 fixture** (63 atoms, nspin = 2) | self-comparison = identity, $\min O_{ii} = 0.9999999999999962$; states 1/2 = 3E (spin-down a1 -> e) at 1.976 eV |
| **NV- diamond 2x2x2 relaxation of 3E** | converged in 7 cycles, E = -724.3058136701 Ry (earlier plain run -724.3056687317); positions rms 0.015 A from the earlier run; same Jahn-Teller distortion; no swap |

### Key finding: band indices are not a reliable label

At the first BFGS step of NV- diamond, the symmetry broke and the spin-down $e$
orbitals **swapped band numbers**:

| $\lvert S_{mn}\rvert$ | step-2 band 127 | step-2 band 128 |
|---|---|---|
| step-1 band 127 | 0.111 | **0.993** |
| step-1 band 128 | **0.993** | 0.111 |

The tracked state was $126 \to 128$ at step 1 and is therefore $126 \to 127$ at step 2,
which is still state 1. Replaying the same 7 steps:

* **with orbital overlaps:** stays on state 1, $O = 0.997$ (correct);
* **by band index only:** a false **STATE SWAP 1 -> 2** with $O = 0.973$, then follows
  the upper Jahn-Teller branch. The overlap is high, so nothing would have warned
  that it was wrong.

Wherever degenerate orbitals split (the Jahn-Teller situation typical of NV centers),
`l_track_orbitals=True` is needed.

---

## 11. Bugs found and fixed

| Bug | Fix |
|---|---|
| `bfgs_iter` crashed when `step-1/` already existed | `os.path.exists` check in `relaxation.py` (§3) |
| trajectory energy written as `np.float64(...)` under NumPy 2 | `repr(float(energy))` |
| `history.json` stored band indices as floats | JSON encoder that keeps NumPy integers as integers; regression tests |
| `python -m westpy.relaxation_tracked` gave a runpy warning | separate `westpy/tracking_cli.py` |
| CLI printed `np.float64(...)` values | cast to `float` |
| `wbse.x`: "cannot init forpy" in the job scripts | run WEST under cray-python 3.11.7 (WEST's embedded Python, NumPy 1.24); only `bfgs.py` uses the dev Python, by full path, with `PYTHONNOUSERSITE=1` |
| Davidson not converged with 4 states | `n_liouville_maxiter: 1000`; for NV- diamond at the exactly degenerate C3v start, also `n_liouville_times: 10` |
| example input used `westpp_l_compute` (not a WEST key, silently ignored) | startup warning for unknown `westpp_control` keys |

Also noted (not changed, since they are in upstream `bfgs_iter`): a failed executable
makes `bfgs_iter` call `exit()` with status 0, so Slurm reports `COMPLETED`; and a
walltime kill is logged as "wbse.x failed". Check `bfgs.out` for `failed:`.

---

## 12. Limitations and next steps

* **A genuine crossing on real data has not happened yet.** The forced `wbse.x` rerun
  after a swap is tested only with the fake executables. A tracked 1B1 or 1B2 MgO
  relaxation is the natural real-data test.
* **Band-window size.** The per-spin 99.9 % rule can be wasteful: in MgO it keeps 396
  spin-up bands for about 0.2 % of the weight. A single weight budget over both spins
  would make the stash much smaller.
* **Cost.** One extra `westpp.x` per step (seconds), reading the windowed wavefunctions,
  and on a swap one extra `wbse.x`. Test mode copies every step directory.
* **Upstreaming** (after discussing with Giulia): submit the `step-1` fix as its own pull
  request; keep the NERSC-specific job scripts and results (`tier4_cluster/`, `results/`)
  on the fork; check whether the maintainers want pytest or `unittest`; credit the
  origin of `qe_wfc_overlap.py`.

### Keeping the branch up to date

```bash
git fetch upstream
git rebase upstream/master
cd test-suite/tracking && conda run -n westpy-dev python -m pytest -q
git push --force-with-lease origin state-tracking
```
