"""Tier 3: real WEST output. Opt-in (slow, needs data): pytest -m fixture

MgO VEE fixture (default $SCRATCH/tracking_fixture, override TRACKING_FIXTURE):
slim copy of /global/cfs/cdirs/m4382/sstaylor/projects/nv_mgo/tddft-ddh/vees/spin-conserving.
NV- diamond 2x2x2 fixture (TRACKING_FIXTURE_DIAMOND): a single-point
pw + wbse + westpp run, prepared in tier4_cluster/nv_diamond_2x2x2 (skipped until it exists).
"""

import os
import re
import time

import numpy as np
import pytest

from conftest import DIAMOND_FIXTURE, MGO_VEE_FIXTURE
from westpy.qe_wfc_overlap import overlap
from westpy.relaxation_tracked import (
    find_westpp_cm_json,
    load_geometry,
    load_stash,
    read_westpp_decomposition,
    save_stash,
    self_compare,
)

pytestmark = pytest.mark.fixture


def need(path, *files):
    for f in files:
        if not os.path.exists(os.path.join(path, f)):
            pytest.skip(f"fixture file missing: {os.path.join(path, f)}")


@pytest.fixture(scope="module")
def mgo():
    need(MGO_VEE_FIXTURE, "pwscf.save/wfcdw1.dat", "west.westpp.save")
    return load_geometry(MGO_VEE_FIXTURE)


def test_mgo_cm_json_and_layout():
    need(MGO_VEE_FIXTURE, "west.westpp.save", "westpp-CM.out")
    path = find_westpp_cm_json(os.path.join(MGO_VEE_FIXTURE, "west.westpp.save"))
    assert os.path.basename(path) == "westpp_3.json"
    d = read_westpp_decomposition(path)
    assert d["signed"] and sorted(d["states"]) == [1, 2, 3, 4, 5, 6]
    # every coefficient printed in westpp-CM.out matches the JSON (occupied-major)
    text = open(os.path.join(MGO_VEE_FIXTURE, "westpp-CM.out")).read()
    n_checked = 0
    for block in re.split(r"#\s+Exciton :", text)[1:7]:
        n = int(block.split()[0])
        for fs, v, ts, c, x in re.findall(r"^\s+(\d)\s+(\d+)\s+\|\s+(\d)\s+(\d+)\s+\|\s+(\S+)\s*$", block, re.M):
            b = d["states"][n]["blocks"][(int(fs), int(ts))]
            got = b["X"][int(v) - b["occ"][0], int(c) - b["vir"][0]]
            assert got == pytest.approx(float(x), abs=1e-6)
            n_checked += 1
    assert n_checked > 30


@pytest.mark.parametrize("orbitals", [True, False])
def test_mgo_self_compare_identity(mgo, orbitals):
    """Spec test 2: the identification against itself is the identity, O_ii ~ 1."""
    states, O = self_compare(mgo, l_track_orbitals=orbitals)
    assert states == [1, 2, 3, 4, 5, 6]
    assert np.argmax(O, axis=1).tolist() == list(range(6))
    assert np.min(np.diag(O)) > 0.999999
    assert np.max(O - np.diag(np.diag(O))) < 0.05


def test_mgo_window_and_gamma(mgo):
    assert all(w["gamma_only"] for w in mgo.ks.values())
    assert mgo.windows[2][1] >= 863  # down e pair 862/863 inside
    assert min(mgo.retained().values()) > 0.99


def test_mgo_stash_round_trip(mgo, tmp_path):
    """The real-size stash (what production writes every step) gives the same S."""
    t0 = time.time()
    save_stash(mgo, str(tmp_path / "stash"))
    t_save = time.time() - t0
    size = sum(f.stat().st_size for f in (tmp_path / "stash").iterdir()) / 2**20
    t0 = time.time()
    back = load_stash(str(tmp_path / "stash"))
    t_load = time.time() - t0
    print(f"\nstash: {size:.0f} MiB, save {t_save:.1f} s, load {t_load:.1f} s")
    for s in mgo.ks:
        S = overlap(back.ks[s], mgo.ks[s])
        assert np.allclose(np.abs(np.diag(S)), 1.0, atol=1e-10)


def test_diamond_self_compare_identity():
    need(DIAMOND_FIXTURE, "pwscf.save/wfcdw1.dat", "west.westpp.save")
    geo = load_geometry(DIAMOND_FIXTURE)
    states, O = self_compare(geo)
    assert np.argmax(O, axis=1).tolist() == list(range(len(states)))
    assert np.min(np.diag(O)) > 0.999999
