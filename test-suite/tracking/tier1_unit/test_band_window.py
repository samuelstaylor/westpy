"""Tier 1: automatic and explicit band windows."""

import numpy as np
import pytest

from westpy.relaxation_tracked import auto_band_window, resolve_windows, retained_weight


def _decomp(tail=0.1):
    """One state, spin down: occ 1-10, vir 11-14; main 10->11 plus a deep tail 3->12."""
    X = np.zeros((10, 4))
    X[9, 0] = np.sqrt(1 - tail**2)
    X[2, 1] = tail
    st = {"energy": 0.3, "blocks": {(2, 2): {"occ": (1, 10), "vir": (11, 14), "X": X}}}
    return {"signed": True, "path": "x", "states": {1: st}}


def test_auto_window_keeps_deep_tail():
    d = _decomp(tail=0.1)  # tail weight 1% > 0.1% budget
    assert auto_band_window(d, 2, 14, 0.999) == (3, 12)


def test_auto_window_drops_tail_when_allowed():
    d = _decomp(tail=0.1)
    assert auto_band_window(d, 2, 14, 0.95) == (10, 11)


def test_spin_without_weight_has_no_window():
    assert auto_band_window(_decomp(), 1, 14) is None
    assert resolve_windows(_decomp(), {1: 14, 2: 14}) == {2: (3, 12)}


def test_explicit_windows():
    d = _decomp()
    assert resolve_windows(d, {1: 14, 2: 14}, band_window=(5, 13)) == {1: (5, 13), 2: (5, 13)}
    assert resolve_windows(d, {1: 14, 2: 14}, band_window={2: (0, 99)}) == {2: (1, 14)}


def test_retained_weight():
    st = _decomp(tail=0.1)["states"][1]
    assert retained_weight(st, {2: (3, 12)}, True) == pytest.approx(1.0)
    assert retained_weight(st, {2: (10, 11)}, True) == pytest.approx(0.99)
    assert retained_weight(st, {}, True) == 0.0
