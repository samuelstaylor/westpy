"""Tier 1: overlap and identification on synthetic data (no files).

Spec cases: a known band swap, a sign flip, a rotated degenerate pair, each
with l_track_orbitals on and off. The band-swap case must FAIL with orbitals
off, which shows why the flag matters.
"""

import numpy as np
import pytest

import tracking_model as tm
from westpy.relaxation_tracked import (
    TrackingGeometry,
    compare_geometries,
    degenerate_diagnostics,
    excitation_overlap,
    identify,
    ks_overlaps,
    self_compare,
)


def make_geo(order=("A", "B", "C"), swap_virtual=False, flip=(), theta=0.0, exc_sign=None, weights=False):
    C = {s: tm.band_transform(swap_virtual, flip, theta) for s in (1, 2)}
    decomp = tm.geometry_state(list(order), C, exc_sign)
    if weights:
        for st in decomp["states"].values():
            for b in st["blocks"].values():
                b["X"] = b["X"] ** 2
        decomp["signed"] = False
    windows = {1: (1, tm.NBND), 2: (1, tm.NBND)}
    ks = {s: tm.wavefunctions(C[s], s) for s in (1, 2)}
    eig = {s: tm.eigenvalues() for s in (1, 2)}
    return TrackingGeometry(decomp, windows, ks, eig, nspin=2)


def pick(prev, curr, label=1, orbitals=True):
    row, _, _ = compare_geometries(prev, curr, label, l_track_orbitals=orbitals)
    return identify(row)["j_star"], row


REF = make_geo()


@pytest.mark.parametrize("orbitals", [True, False])
def test_same_geometry_is_identity(orbitals):
    for label in (1, 2, 3):
        j, row = pick(REF, make_geo(), label, orbitals)
        assert j == label
        assert row[label] == pytest.approx(1.0, abs=1e-10)


def test_band_swap_needs_orbitals():
    """Virtual bands 5/6 swap order, state order unchanged: A stays state 1."""
    curr = make_geo(swap_virtual=True)
    j_on, row_on = pick(REF, curr, 1, orbitals=True)
    j_off, row_off = pick(REF, curr, 1, orbitals=False)
    assert j_on == 1 and row_on[1] == pytest.approx(1.0, abs=1e-10)
    # by band index, A (4->5) now looks like B (which occupies band 5): wrong pick
    assert j_off == 2
    assert row_off[1] < 0.1


@pytest.mark.parametrize("orbitals", [True, False])
def test_state_crossing_detected(orbitals):
    """Energy order of A and B swaps, orbitals unchanged: A is now state 2."""
    j, row = pick(REF, make_geo(order=("B", "A", "C")), 1, orbitals)
    assert j == 2
    assert row[2] == pytest.approx(1.0, abs=1e-10)


def test_crossing_with_band_swap():
    """Realistic crossing: states and virtual bands swap together."""
    curr = make_geo(order=("B", "A", "C"), swap_virtual=True)
    assert pick(REF, curr, 1, orbitals=True)[0] == 2
    # by band index the swap is invisible: the tracked state looks unchanged
    assert pick(REF, curr, 1, orbitals=False)[0] == 1


def test_sign_flips():
    """Band signs and the overall exciton sign are arbitrary."""
    curr = make_geo(flip=(2, 4, 5), exc_sign={"A": -1.0})
    j_on, row_on = pick(REF, curr, 1, orbitals=True)
    j_off, row_off = pick(REF, curr, 1, orbitals=False)
    assert j_on == 1 and row_on[1] == pytest.approx(1.0, abs=1e-10)
    assert j_off == 1  # still found, but the partial cancellation lowers O
    assert row_off[1] < row_on[1] - 0.05


@pytest.mark.parametrize("theta", [0.3, np.pi / 4, np.pi / 2])
def test_rotated_degenerate_pair(theta):
    """QE returns an arbitrary rotation inside the degenerate occupied pair 3/4."""
    curr = make_geo(theta=theta)
    j_on, row_on = pick(REF, curr, 1, orbitals=True)
    assert j_on == 1 and row_on[1] == pytest.approx(1.0, abs=1e-10)
    # by band index the rotation is seen as a loss of overlap (about cos(theta)
    # for small angles), never as a perfect match
    _, row_off = pick(REF, curr, 1, orbitals=False)
    assert row_off[1] < row_on[1] - 1e-3


def test_everything_at_once():
    curr = make_geo(order=("B", "A", "C"), swap_virtual=True, flip=(1, 6), theta=1.1, exc_sign={"B": -1.0})
    for label, expected in ((1, 2), (2, 1), (3, 3)):
        j, row = pick(REF, curr, label, orbitals=True)
        assert j == expected
        assert row[expected] == pytest.approx(1.0, abs=1e-10)


def test_weights_only_path():
    """Without signs, the phase-free similarity still follows band swaps and crossings."""
    prev = make_geo(weights=True)
    curr = make_geo(order=("B", "A", "C"), swap_virtual=True, flip=(2, 5), weights=True)
    j, row = pick(prev, curr, 1, orbitals=True)
    assert j == 2
    assert row[2] == pytest.approx(1.0, abs=1e-10)
    # a rotated degenerate pair is only approximate with weights
    j, row = pick(prev, make_geo(theta=0.3, weights=True), 1, orbitals=True)
    assert j == 1 and 0.8 < row[1] < 1.0


def test_spin_blocks_are_not_mixed():
    """An up->up block never overlaps with a down->down block."""
    prev, curr = make_geo(), make_geo()
    S, bands = ks_overlaps(prev, curr)
    up_only = {"blocks": {(1, 1): prev.decomp["states"][1]["blocks"][(1, 1)]}}
    dw_only = {"blocks": {(2, 2): curr.decomp["states"][1]["blocks"][(2, 2)]}}
    assert excitation_overlap(up_only, prev, dw_only, curr, S, bands) == 0.0
    assert excitation_overlap(up_only, prev, up_only, curr, S, bands) == pytest.approx(1.0)


def test_identify_flags():
    r = identify({1: 0.95, 2: 0.10, 3: 0.02})
    assert r["j_star"] == 1 and not (r["low_overlap"] or r["ambiguous"] or r["top_state"])
    r = identify({1: 0.40, 2: 0.10, 3: 0.02}, overlap_thr=0.5)
    assert r["low_overlap"]
    r = identify({1: 0.70, 2: 0.65, 3: 0.02}, ambiguity_thr=0.1)
    assert r["ambiguous"] and r["second"] == 2
    r = identify({1: 0.10, 2: 0.20, 3: 0.90})
    assert r["j_star"] == 3 and r["top_state"]


def test_degenerate_diagnostic():
    """A rotation inside the degenerate pair keeps sum|S|^2/n = 1 for that block."""
    prev, curr = make_geo(), make_geo(theta=0.7)
    S, bands = ks_overlaps(prev, curr)
    diag = degenerate_diagnostics(prev, S, bands, thr=1e-3)
    assert {(s, b0, b1) for s, b0, b1, _ in diag} == {(1, 3, 4), (2, 3, 4)}
    for *_, val in diag:
        assert val == pytest.approx(1.0, abs=1e-10)


def test_self_compare_identity():
    states, O = self_compare(make_geo(swap_virtual=True, theta=0.4))
    assert states == [1, 2, 3]
    assert np.allclose(np.diag(O), 1.0)
    assert np.argmax(O, axis=1).tolist() == [0, 1, 2]
