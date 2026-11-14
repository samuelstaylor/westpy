"""Tier 2: the whole BFGS loop with fake pw.x / wbse.x / westpp.x (see fake_west.py).

Model: excitons A and B cross at q = 0.05 Bohr. The relaxation starts at q = 0
on state 1 (= A). The first BFGS step lands at q ~ 0.1, past the crossing, so
from step 2 on A is state 2.
  * tracking on:  swap 1 -> 2 at step 2, converge at A's minimum q = 0.1/1.6
  * tracking off: follows "state 1" (= B after the crossing), converge at q = 0.3/1.6
"""

import glob
import json
import os
import re
import shutil
import sys

import numpy as np
import pytest
import yaml

import fake_west
from westpy.relaxation import bfgs_iter
from westpy.relaxation_tracked import bfgs_iter_tracked, replay

pytestmark = pytest.mark.mockwest

FAKE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fake_west.py")
PY = sys.executable
CMD = {name: f"{PY} {FAKE} {name}" for name in ("pw", "wbse", "westpp")}
Q_A = 0.10 / 1.6  # minimum of the tracked exciton A
Q_B = 0.30 / 1.6  # minimum of exciton B

PW_IN = """&CONTROL
  calculation = 'scf'
  tprnfor = .true.
/
&SYSTEM
  ibrav = 0
  nat = 2
  ntyp = 2
  ecutwfc = 30
  nspin = 2
/
&ELECTRONS
/
ATOMIC_SPECIES
C 12.011 C.upf
O 15.999 O.upf
K_POINTS gamma
CELL_PARAMETERS angstrom
5.0 0.0 0.0
0.0 5.0 0.0
0.0 0.0 5.0
ATOMIC_POSITIONS angstrom
C 1.000000000000 2.500000000000 2.500000000000
O 2.200000000000 2.500000000000 2.500000000000
"""

WBSE_IN = """input_west:
  outdir: ./
wbse_init_control:
  solver: TDDFT
wbse_control:
  wbse_calculation: D
  n_liouville_eigen: 3
  l_forces: True
  forces_state: 1
"""

WESTPP_IN = """westpp_control:
  westpp_calculation: CM
  westpp_range:
  - 1
  - 3
  westpp_n_liouville_to_use: 3
  westpp_format: C
  westpp_sign: true
"""


def setup_run(path, westpp_in=WESTPP_IN, model=None):
    path.mkdir(parents=True, exist_ok=True)
    (path / "pw.in").write_text(PW_IN)
    (path / "wbse.in").write_text(WBSE_IN)
    (path / "westpp.in").write_text(westpp_in)
    for pp in ("C.upf", "O.upf"):
        (path / pp).write_text("fake pseudopotential\n")
    if model:
        (path / "fake_west_model.json").write_text(json.dumps(model))
    return path


def base_kwargs(**extra):
    kw = dict(run_pw=CMD["pw"], run_wbse=CMD["wbse"], pp=["C.upf", "O.upf"], maxiter=30)
    kw.update(extra)
    return kw


def tracked_kwargs(**extra):
    kw = base_kwargs(l_track_state=True, run_westpp=CMD["westpp"])
    kw.update(extra)
    return kw


def final_q(run_dir):
    _, pos, _ = fake_west.parse_pw_in(str(run_dir / "final_geo" / "pw.in"))
    _, ref, _ = fake_west.parse_pw_in(str(run_dir / "pw.in"))
    return pos[0, 0] - ref[0, 0]


def calls(run_dir):
    return (run_dir / "fake_calls.log").read_text().splitlines()


def tree(run_dir):
    out = {}
    for p in sorted(run_dir.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(run_dir))] = p.read_bytes()
    return out


def traj_energies(run_dir):
    text = (run_dir / "bfgs_traj.xyz").read_text()
    return [float(x) for x in re.findall(r"(?<![\w])energy=(\S+)", text)]


# --------------------------------------------------------------------------


def test_passthrough_identical_to_bfgs_iter(tmp_path, monkeypatch, capsys):
    """l_track_state=False: same calls, same files, same log as bfgs_iter (spec test 3)."""
    logs = {}
    for name, cls in (("plain", bfgs_iter), ("tracked_off", bfgs_iter_tracked)):
        d = setup_run(tmp_path / name)
        monkeypatch.chdir(d)
        cls(**base_kwargs()).solve()
        logs[name] = capsys.readouterr().out.replace(str(d), "<RUN>")
    plain, off = tmp_path / "plain", tmp_path / "tracked_off"
    assert calls(plain) == calls(off)
    assert tree(plain) == tree(off)
    assert logs["plain"] == logs["tracked_off"]
    assert not (off / "tracking").exists() and not (off / "bfgs_tracking.json").exists()
    assert not (off / "wbse.in.orig").exists()
    assert not any("westpp" in c for c in calls(off))
    # and plain BFGS follows "state 1", i.e. B after the crossing
    assert final_q(plain) == pytest.approx(Q_B, abs=5e-3)


def test_planted_crossing_with_test_mode(tmp_path, monkeypatch, capsys):
    d = setup_run(tmp_path / "run")
    monkeypatch.chdir(d)
    bfgs = bfgs_iter_tracked(**tracked_kwargs(l_test_mode=True))
    bfgs.solve()
    log = capsys.readouterr().out

    assert bfgs.conv_bfgs
    assert final_q(d) == pytest.approx(Q_A, abs=5e-3)  # followed A, not B

    hist = json.loads((d / "tracking" / "history.json").read_text())
    assert [r["scf_iter"] for r in hist] == list(range(1, len(hist) + 1))
    assert hist[0]["overlap_row"] is None  # reference step
    assert hist[1]["swap"] and (hist[1]["label_before"], hist[1]["label_after"]) == (1, 2)
    assert all(not r["swap"] for r in hist[2:])
    assert all(r["label_after"] == 2 for r in hist[1:])
    for r in hist[1:]:
        assert r["max_overlap"] == pytest.approx(1.0, abs=1e-8)
        assert r["total_force"] is not None
    for r in hist:  # band indices stay integers in history.json
        assert all(isinstance(k, int) for t in r["dominant_transitions"] for k in t[:4])
    assert "STATE SWAP: 1 -> 2" in log
    assert "tracking: 1 state swap(s), final tracked state = 2" in log

    # rerun of wbse.x only, in step 2
    c = calls(d)
    step2 = [x.split()[0] for x in c if x.split()[1] == "step-2"]
    assert step2 == ["pw", "wbse", "westpp", "wbse"]
    for line in c:  # every other step: pw, wbse, westpp once
        assert line.split()[1] == "step-2" or c.count(line) == 1

    t2 = d / "tracking" / "test" / "step-2"
    assert (t2 / "wbse_state-1.out").exists() and (t2 / "wbse.out").exists()
    assert "forces_state = 2" in (t2 / "wbse.out").read_text()
    assert "forces_state = 1" in (t2 / "wbse_state-1.out").read_text()
    assert (t2 / "west.wbse.save" / "wbse_state-1.json").exists()
    assert yaml.safe_load((t2 / "wbse.in").read_text())["wbse_control"]["forces_state"] == 2
    summary = (t2 / "tracking_summary.txt").read_text()
    assert "state swap           : yes" in summary and "label after          : 2" in summary
    assert (t2 / "pwscf.save" / "wfcdw1.dat").exists()  # full step dir kept

    # every step kept in test mode; the converged one too
    nsteps = len(hist)
    assert len(glob.glob(str(d / "tracking" / "test" / "step-*"))) == nsteps
    assert len(glob.glob(str(d / "tracking" / "stash" / "step-*"))) == nsteps - 1
    swap_table = (d / "tracking" / "test" / "swap_summary.txt").read_text().splitlines()
    assert len(swap_table) == nsteps + 1 and "yes" in swap_table[2] and swap_table[2].split()[0] == "2"

    # trajectory: last frame of step 2 rewritten with the energy of state 2 (= A)
    e = traj_energies(d)
    assert len(e) == nsteps
    assert e[1] == pytest.approx(hist[1]["energy"], abs=1e-12)

    # root wbse.in carries the label; the original is kept
    assert yaml.safe_load((d / "wbse.in").read_text())["wbse_control"]["forces_state"] == 2
    assert yaml.safe_load((d / "wbse.in.orig").read_text())["wbse_control"]["forces_state"] == 1


def test_only_latest_stash_without_test_mode(tmp_path, monkeypatch):
    d = setup_run(tmp_path / "run")
    monkeypatch.chdir(d)
    bfgs_iter_tracked(**tracked_kwargs()).solve()
    assert len(glob.glob(str(d / "tracking" / "stash" / "step-*"))) == 1
    assert not (d / "tracking" / "test").exists()
    stash = glob.glob(str(d / "tracking" / "stash" / "step-*"))[0]
    assert sorted(os.listdir(stash)) == ["decomp.npz", "eigenvalues.npz", "ks_s1.npz", "ks_s2.npz", "meta.json"]


def test_orbitals_off_misses_the_swap(tmp_path, monkeypatch):
    """The virtual bands swap with the states: by band index the swap is invisible."""
    d = setup_run(tmp_path / "run")
    monkeypatch.chdir(d)
    bfgs_iter_tracked(**tracked_kwargs(l_track_orbitals=False)).solve()
    hist = json.loads((d / "tracking" / "history.json").read_text())
    assert not any(r["swap"] for r in hist)
    assert final_q(d) == pytest.approx(Q_B, abs=5e-3)  # ended on the wrong state
    assert not glob.glob(str(d / "tracking" / "stash" / "step-*" / "ks_s*.npz"))


def test_restart_keeps_label_and_history(tmp_path, monkeypatch, capsys):
    d = setup_run(tmp_path / "run")
    monkeypatch.chdir(d)
    first = bfgs_iter_tracked(**tracked_kwargs(maxiter=2))
    first.solve()
    assert "maximum number of steps" in capsys.readouterr().out
    t = json.loads((d / "bfgs_tracking.json").read_text())
    assert t["label"] == 2 and len(t["history"]) == 2 and os.path.isdir(t["stash"])

    second = bfgs_iter_tracked(**tracked_kwargs(l_restart=True))
    assert second.track_label == 2
    second.solve()
    assert "tracking restarted: label = 2" in capsys.readouterr().out
    hist = json.loads((d / "tracking" / "history.json").read_text())
    assert [r["scf_iter"] for r in hist] == list(range(1, len(hist) + 1))
    assert sum(r["swap"] for r in hist) == 1
    assert hist[2]["compared_with_scf_iter"] == 2  # compared with the stash from before the restart
    assert final_q(d) == pytest.approx(Q_A, abs=5e-3)


def test_stop_on_low_overlap(tmp_path, monkeypatch, capsys):
    """New random orbitals after the crossing: no state matches, the run stops."""
    d = setup_run(tmp_path / "run", model={"scramble_after_crossing": True})
    monkeypatch.chdir(d)
    with pytest.raises(SystemExit):
        bfgs_iter_tracked(**tracked_kwargs(stop_on_low_overlap=True)).solve()
    out = capsys.readouterr().out
    assert "stop_on_low_overlap is True: stopping" in out
    hist = json.loads((d / "tracking" / "history.json").read_text())
    assert any("< overlap_thr" in w for w in hist[-1]["warnings"])


def test_low_overlap_only_warns_by_default(tmp_path, monkeypatch, capsys):
    d = setup_run(tmp_path / "run", model={"scramble_after_crossing": True})
    monkeypatch.chdir(d)
    bfgs_iter_tracked(**tracked_kwargs(maxiter=3)).solve()
    assert "WARNING (tracking): max overlap" in capsys.readouterr().out


def test_startup_warnings_are_reported(tmp_path, monkeypatch, capsys):
    bad = WESTPP_IN.replace("  westpp_sign: true\n", "  westpp_sign: true\n  westpp_l_compute: false\n")
    d = setup_run(tmp_path / "run", westpp_in=bad)
    monkeypatch.chdir(d)
    bfgs_iter_tracked(**tracked_kwargs(maxiter=1))
    out = capsys.readouterr().out
    assert "WARNING (tracking): unknown westpp_control key(s) ['westpp_l_compute']" in out


def test_run_westpp_required(tmp_path, monkeypatch):
    d = setup_run(tmp_path / "run")
    monkeypatch.chdir(d)
    with pytest.raises(ValueError):
        bfgs_iter_tracked(**base_kwargs(l_track_state=True))


def test_existing_step1_no_longer_crashes(tmp_path, monkeypatch):
    """relaxation.py mkdir fix: a leftover step-1/ is reused instead of crashing."""
    d = setup_run(tmp_path / "run")
    (d / "step-1").mkdir()
    monkeypatch.chdir(d)
    bfgs_iter(**base_kwargs(maxiter=1)).solve()
    assert (d / "step-2" / "pw.in").exists()


def test_replay_reproduces_identification(tmp_path, monkeypatch):
    """Dry run over tracking/test: same swap, no executable is launched."""
    d = setup_run(tmp_path / "run")
    monkeypatch.chdir(d)
    bfgs_iter_tracked(**tracked_kwargs(l_test_mode=True)).solve()
    n_calls = len(calls(d))
    steps = glob.glob(str(d / "tracking" / "test" / "step-*"))
    res = replay(steps, label=1)
    assert [r["label_after"] for r in res] == [1] + [2] * (len(res) - 1)
    assert len(calls(d)) == n_calls
    hist = json.loads((d / "tracking" / "history.json").read_text())
    for r, h in zip(res[1:], hist[1:]):
        assert r["max_overlap"] == pytest.approx(h["max_overlap"], abs=1e-10)
