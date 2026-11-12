"""Tier 1: wbse.in editing, stash round trip, trajectory rewrite, XML/pw.in readers."""

import os

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.io import read, write

import tracking_model as tm
from westpy.qe_wfc_overlap import overlap, read_wfc
from westpy.relaxation_tracked import (
    TrackingGeometry,
    load_stash,
    read_eigenvalues_xml,
    read_nspin_pw_input,
    rewrite_last_traj_energy,
    save_stash,
    set_forces_state,
)

WBSE_TEXT = """input_west:
  outdir: ./
wbse_init_control:
  solver: TDDFT
wbse_control:
  wbse_calculation: D
  n_liouville_eigen: 4
  trev_liouville: 0.00000001
  spin_excitation: S
  l_forces: True
  forces_state: 1
"""


def test_set_forces_state_changes_only_that_key(tmp_path):
    p = tmp_path / "wbse.in"
    p.write_text(WBSE_TEXT)
    before = yaml.safe_load(WBSE_TEXT)
    assert set_forces_state(str(p), 3) is True
    after = yaml.safe_load(p.read_text())
    assert after["wbse_control"].pop("forces_state") == 3
    before["wbse_control"].pop("forces_state")
    assert after == before
    assert list(after["wbse_control"]) == list(before["wbse_control"])  # order kept


def test_set_forces_state_noop_leaves_file_untouched(tmp_path):
    p = tmp_path / "wbse.in"
    p.write_text(WBSE_TEXT)
    assert set_forces_state(str(p), 1) is False
    assert p.read_text() == WBSE_TEXT


def _geo():
    C = {s: tm.band_transform(theta=0.2) for s in (1, 2)}
    ks = {s: tm.wavefunctions(C[s], s) for s in (1, 2)}
    for s in ks:  # keep a window of bands 2-6, like read_wfc(bands=...)
        ks[s]["evc"] = ks[s]["evc"][1:]
        ks[s]["bands"] = ks[s]["bands"][1:]
    decomp = tm.geometry_state(["A", "B", "C"], C)
    return TrackingGeometry(decomp, {1: (2, 6), 2: (2, 6)}, ks, {s: tm.eigenvalues() for s in (1, 2)}, 2, 7, "/x")


def test_stash_round_trip(tmp_path):
    geo = _geo()
    save_stash(geo, str(tmp_path / "stash"))
    back = load_stash(str(tmp_path / "stash"))
    for s in (1, 2):
        # exactly the read_wfc keys, so the dict goes straight into overlap()
        assert set(back.ks[s]) == {"miller", "evc", "gamma_only", "npol", "bands"}
        assert isinstance(back.ks[s]["gamma_only"], bool) and isinstance(back.ks[s]["npol"], int)
        assert back.ks[s]["bands"] == [2, 3, 4, 5, 6]
        np.testing.assert_allclose(overlap(back.ks[s], geo.ks[s]), overlap(geo.ks[s], geo.ks[s]))
        np.testing.assert_allclose(back.eig[s], geo.eig[s])
    assert back.windows == geo.windows and back.scf_iter == 7 and back.nspin == 2
    for n, st in geo.decomp["states"].items():
        for key, b in st["blocks"].items():
            np.testing.assert_array_equal(back.decomp["states"][n]["blocks"][key]["X"], b["X"])


def test_stash_matches_read_wfc_dict(tmp_path):
    """A real wfc file read with read_wfc survives the stash unchanged."""
    import fake_west

    geo_dir = tmp_path / "save"
    geo_dir.mkdir()
    fake_west.write_wfc(str(geo_dir / "wfcdw1.dat"), tm.wavefunctions(tm.band_transform(), 2), ispin=2)
    wfc = read_wfc(str(geo_dir / "wfcdw1.dat"), bands=[3, 4, 5])
    geo = TrackingGeometry(tm.geometry_state(["A"], {s: tm.band_transform() for s in (1, 2)}),
                           {2: (3, 5)}, {2: wfc}, {2: tm.eigenvalues()}, 2)
    save_stash(geo, str(tmp_path / "stash"))
    back = load_stash(str(tmp_path / "stash")).ks[2]
    assert back.keys() == wfc.keys()
    for k in wfc:
        np.testing.assert_array_equal(np.asarray(back[k]), np.asarray(wfc[k]))


def test_rewrite_last_traj_energy(tmp_path):
    traj = str(tmp_path / "bfgs_traj.xyz")
    for step, e in ((1, -10.0), (2, -10.5)):  # same calls as bfgs_iter._save_traj
        atoms = Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.2]], cell=[5, 5, 5], pbc=True)
        atoms.info["step"] = step
        atoms.info["energy"] = e
        write(traj, atoms, format="extxyz", append=True)
    # bfgs_iter energies are np.float64 (repr "np.float64(...)" in NumPy 2)
    assert rewrite_last_traj_energy(traj, np.float64(-11.25))
    frames = read(traj, index=":")

    def energy(a):
        return a.info["energy"] if "energy" in a.info else a.get_potential_energy()

    assert [energy(a) for a in frames] == [-10.0, -11.25]
    assert [a.info["step"] for a in frames] == [1, 2]


def test_read_nspin(tmp_path):
    p = tmp_path / "pw.in"
    p.write_text("&SYSTEM\n  nat = 2\n  ! nspin = 4\n  nspin            = 2\n/\n")
    assert read_nspin_pw_input(str(p)) == 2
    p.write_text("&SYSTEM\n  nat = 2\n/\n")
    assert read_nspin_pw_input(str(p)) == 1


def test_read_eigenvalues_xml(tmp_path):
    import fake_west

    xml = tmp_path / "data-file-schema.xml"
    e = {1: tm.eigenvalues(), 2: tm.eigenvalues() + 0.01}
    fake_west.write_band_structure_only(str(xml), e)
    got = read_eigenvalues_xml(str(xml))
    np.testing.assert_allclose(got[1], e[1])
    np.testing.assert_allclose(got[2], e[2])
