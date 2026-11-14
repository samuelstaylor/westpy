# 3. MgO VEE fixture: westpp format and self-comparison (tier 3)

**Data:** `$SCRATCH/tracking_fixture`, a slim copy (1.5 GB, 81 files) of
`/global/cfs/cdirs/m4382/sstaylor/projects/nv_mgo/tddft-ddh/vees/spin-conserving/`
(the original was only read). Excluded: `west.wbse_init.save/`, the cube files,
`west.wbse.save/E*.dat`, `pwscf.save/ace*.dat`.
**Run:** `cd test-suite/tracking && conda run -n westpy-dev python -m pytest -v -m fixture tier3_fixture`
and `python -m westpy.tracking_cli selfcompare $SCRATCH/tracking_fixture`
**Result:** 5/5 MgO tests pass.

This directory is a **vertical-excitation single point** (NV- in MgO, DDH,
spin-conserving, 10 states), not a relaxation: there is no `forces_state` and no
BFGS output. It was used to learn the westpp output format and for the first
real self-comparison.

## The westpp output format (confirmed with the user)

* The decomposition is in `west.westpp.save/westpp*.json`, **not** in
  `westpp-CM.out`: the text table only lists the main components (|X| > about 0.1),
  while the JSON holds the full occupied x virtual block.
* **westpp.x never overwrites its JSON**: `westpp.json`, `westpp_1.json`, ... The
  fixture has six; the CM one is `westpp_3.json`. The code picks the newest file
  whose `westpp_calculation` contains `C`. wbse.x does the same (`wbse.json`,
  `wbse_1.json`), which is why `bfgs_iter` deletes `wbse.json` before each run and
  why the swap rerun must move it away.
* Layout: `output.E00000n.K00000s.projection` with `from_spin`, `to_spin`,
  `from_bandrange`, `to_bandrange`, `vals`. `K000001` = spin up, `K000002` = spin down.
* `vals` are **occupied-major**: `X[v, c] = vals[(v - v0) * nc + (c - c0)]`. This
  reproduces every coefficient printed in `westpp-CM.out`, signs included; the
  transposed order gives zeros.
* **Signed** with `westpp_sign: true` (about half the values are negative).
* Band indices are **1-based global KS indices**, the same as `wfc*.dat` and the XML.
* Sum of X^2 over both spins is 0.987-0.998, so the code normalizes explicitly.
* Energies (Ry) equal `wbse_1.json` `davitr[-1].ev`.

## Other findings

* `bfgs_iter` restores the SCF `pwscf.save` after nscf and keeps only the nscf ACE
  operators, so "the wavefunctions when wbse runs" are `pwscf.save` as it is.
* The spec's example input (and the user's commented CM block) uses
  `westpp_l_compute`, **which is not a WEST key**; the real key is
  `westpp_l_compute_tdm`, and WEST silently ignored it. The startup check now warns
  about unknown `westpp_control` keys.
* States (1,2), (3,4), (5,6) are near-degenerate pairs, 98-99% spin down; E1/E2 are
  mostly dw 861 -> 862/863 (the degenerate e pair at 14.172 eV) with tails down to band 540.

## Self-comparison

| | |
|---|---|
| Mapping | identity for all 6 states |
| min O_ii | 0.99999999999999957 |
| largest off-diagonal | 0.0056 |
| automatic windows | up 505-900, dw 539-864; retained weight about 0.998 |
| cost | 8.6 s, 1.7 GB |
| stash | 535 MiB, 20 s to write, 3 s to read |

Observation: the per-spin 99.9% rule spends 396 spin-up bands on about 0.2% of
the weight. A global weight budget would make the stash much smaller (not done).
