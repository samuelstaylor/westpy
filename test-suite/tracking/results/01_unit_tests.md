# 1. Synthetic unit tests (tier 1)

**Directory:** `test-suite/tracking/tier1_unit/` (51 tests, about 2 s)
**Run:** `cd test-suite/tracking && conda run -n westpy-dev python -m pytest -v tier1_unit`
**Result:** 51 passed.

## What is tested

The pure functions of `westpy/relaxation_tracked.py`, on synthetic data from
`tracking_model.py`. The model defines orbitals and excitons in a fixed orbital
basis and expresses them in the band basis a calculation would report
(`psi_b = sum_o C[b,o] phi_o`, `X = C A C^T`). `C` can contain a swap of the
virtual bands, sign flips of single bands, and a rotation inside a degenerate
occupied pair, which are the things that change between two real geometries.

| File | Tests | Checks |
|---|---|---|
| `test_overlap_identification.py` | 16 | identification with `l_track_orbitals` on/off for: same geometry, band swap, state crossing, crossing + band swap, sign flips, rotated degenerate pair (0.3, pi/4, pi/2), everything at once, weights-only path, spin blocks never mixed, warning flags, degenerate-block diagnostic, self-comparison |
| `test_validation.py` | 17 | every startup warning (range vs forces_state, n_liouville_to_use, n_liouville_eigen, spin flip for nspin 1/2, unknown key, missing `C`, l_forces, unsigned output, prefixes/outdir); the spec's example westpp input |
| `test_westpp_parsing.py` | 5 | newest CM JSON chosen by content, occupied-major layout, unsigned values converted to weights, dominant transitions |
| `test_band_window.py` | 5 | automatic window keeps a deep tail, explicit windows, retained weight |
| `test_io_helpers.py` | 8 | `wbse.in` edit changes only `forces_state`, stash round trip with exactly the `read_wfc` keys, trajectory energy rewrite, `nspin` and XML eigenvalue readers, integers kept in JSON |

## Important results

* **The band swap fails with orbitals off, as the spec requires.** When the two
  virtual bands swap order but the states keep their energy order, comparing by
  band index picks the wrong state (O(1,1) < 0.1, O(1,2) high). With the KS
  overlaps folded in, O(1,1) = 1 to 1e-10. This is the synthetic version of what
  test 6 later found on real data.
* **Crossing + band swap together** (the realistic case): orbitals on detects the
  swap 1 -> 2; orbitals off sees no swap at all.
* **Rotated degenerate pair:** orbitals on gives O = 1 for any angle. By band
  index the overlap drops to about cos(theta). For small angles the drop is small
  (0.955 at 0.3 rad), so a band-index comparison would not even warn.
* **Weights-only formula** (no signs) still follows swaps and crossings, but is
  only approximate for a rotated degenerate pair (0.8 < O < 1 at 0.3 rad). Hence
  the recommendation `westpp_sign: true`.

## Corrections made

* My first rotated-pair test asserted that band-index overlap drops by more than
  0.05; for 0.3 rad it drops by 0.045. The code was right, the assertion was too
  strict; now it asserts "strictly lower".
* `test_json_keeps_integers` was added after test 4 found band indices written as
  floats in `history.json`.
