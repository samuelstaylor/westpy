"""Tier 1: reading westpp JSON files (layout confirmed on the MgO VEE run)."""

import json
import os

import numpy as np
import pytest

import tracking_model as tm
from westpy.relaxation_tracked import (
    dominant_transitions,
    find_westpp_cm_json,
    read_westpp_decomposition,
)


def _decomp():
    C = {s: tm.band_transform() for s in (1, 2)}
    return tm.geometry_state(["A", "B", "C"], C)


def _write(path, doc):
    with open(path, "w") as f:
        json.dump(doc, f)


def test_newest_cm_json_is_selected(tmp_path):
    """westpp.x never overwrites: pick by content, highest index wins."""
    doc = tm.westpp_json(_decomp())
    w_run = {"input": {"westpp_control": {"westpp_calculation": "W"}}}
    r_run = {"input": {"westpp_control": {"westpp_calculation": "R"}}}
    _write(tmp_path / "westpp.json", w_run)
    _write(tmp_path / "westpp_1.json", doc)
    _write(tmp_path / "westpp_2.json", tm.westpp_json(_decomp(), calc="C"))
    _write(tmp_path / "westpp_3.json", r_run)
    assert os.path.basename(find_westpp_cm_json(str(tmp_path))) == "westpp_2.json"


def test_no_decomposition_raises(tmp_path):
    _write(tmp_path / "westpp.json", {"input": {"westpp_control": {"westpp_calculation": "W"}}})
    with pytest.raises(FileNotFoundError):
        find_westpp_cm_json(str(tmp_path))


def test_occupied_major_layout(tmp_path):
    ref = _decomp()
    _write(tmp_path / "westpp.json", tm.westpp_json(ref))
    got = read_westpp_decomposition(str(tmp_path / "westpp.json"))
    assert got["signed"] is True
    assert sorted(got["states"]) == [1, 2, 3]
    for n in (1, 2, 3):
        for key, b in ref["states"][n]["blocks"].items():
            g = got["states"][n]["blocks"][key]
            assert g["occ"] == b["occ"] and g["vir"] == b["vir"]
            np.testing.assert_allclose(g["X"], b["X"])
    # explicit element check: X[v, c] = vals[(v - v0) * nc + (c - c0)]
    raw = json.load(open(tmp_path / "westpp.json"))["output"]["E000001"]["K000002"]["projection"]
    X = got["states"][1]["blocks"][(2, 2)]["X"]
    v0, c0, nc = raw["from_bandrange"][0], raw["to_bandrange"][0], X.shape[1]
    assert X[4 - v0, 5 - c0] == raw["vals"][(4 - v0) * nc + (5 - c0)]


def test_unsigned_values_become_weights(tmp_path):
    ref = _decomp()
    # |X| stored (amplitudes): converted to |X|^2
    doc = tm.westpp_json(ref, sign=False)
    for e in doc["output"].values():
        for k in e:
            if k.startswith("K"):
                e[k]["projection"]["vals"] = [abs(v) for v in e[k]["projection"]["vals"]]
    _write(tmp_path / "westpp.json", doc)
    got = read_westpp_decomposition(str(tmp_path / "westpp.json"))
    assert got["signed"] is False
    np.testing.assert_allclose(got["states"][1]["blocks"][(2, 2)]["X"], ref["states"][1]["blocks"][(2, 2)]["X"] ** 2)
    # |X|^2 stored (weights): kept as is
    _write(tmp_path / "westpp_1.json", tm.westpp_json(ref, sign=False, weights=True))
    got = read_westpp_decomposition(str(tmp_path / "westpp_1.json"))
    np.testing.assert_allclose(got["states"][1]["blocks"][(2, 2)]["X"], ref["states"][1]["blocks"][(2, 2)]["X"] ** 2)


def test_dominant_transitions():
    top = dominant_transitions(_decomp()["states"][1], n=3)
    assert top[0][:4] == (2, 4, 2, 5)  # A: down 4 -> 5
    assert abs(top[0][4]) > abs(top[1][4]) >= abs(top[2][4])
