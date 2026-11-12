#!/usr/bin/env python3
"""Fake pw.x / wbse.x / westpp.x for the tier-2 tracking tests.

These are NOT the WEST/QE codes. They write tiny files with the same layout as
the real ones (data-file-schema.xml, wfc*.dat, wbse.json, westpp*.json), from
the synthetic model in tracking_model.py, so the whole bfgs_iter loop can run.

Usage (as bfgs_iter calls them, from inside a step directory):
    python fake_west.py pw     -i pw.in     > pw.out
    python fake_west.py wbse   -i wbse.in   > wbse.out
    python fake_west.py westpp -i westpp.in > westpp.out

Model (per geometry, q = x displacement of atom 1 from the root pw.in, Bohr):
    E_gs = 0.5 k sum_atoms |r - r_ref|^2                (Ry)
    E_X(q) = e0_X + slope_X q   for excitons X = A, B, C (Ry)
States are ordered by energy. With the defaults A and B cross at q = 0.05;
the virtual bands 5/6 swap order at the crossing, the degenerate occupied pair
rotates by theta_per_bohr * q, and bands 2 and 5 flip sign for q > 0.02.
Every call is appended to <root>/fake_calls.log.
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import yaml
from scipy.io import FortranFile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tracking_model as tm  # noqa: E402

BOHR_PER_ANG = 1.0 / 0.529177210903

DEFAULT_MODEL = {
    "k": 1.6,
    "excitons": {
        "A": {"e0": 0.30, "slope": -0.10},
        "B": {"e0": 0.31, "slope": -0.30},
        "C": {"e0": 0.45, "slope": 0.00},
    },
    "swap_bands_at_crossing": True,
    "theta_per_bohr": 6.0,
    "flip_after": 0.02,
    "scramble_after_crossing": False,  # new random orbitals after the crossing
}


# --------------------------------------------------------------- model


def load_model(root):
    model = json.loads(json.dumps(DEFAULT_MODEL))
    path = os.path.join(root, "fake_west_model.json")
    if os.path.exists(path):
        with open(path) as f:
            user = json.load(f)
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(model.get(k), dict):
                model[k].update(v)
            else:
                model[k] = v
    return model


def parse_pw_in(path):
    """Return (symbols, positions in Bohr, cell in Bohr) from a pw.in (angstrom units)."""
    with open(path) as f:
        lines = f.readlines()
    nat = int(re.search(r"\bnat\s*=\s*(\d+)", "".join(lines)).group(1))
    for i, line in enumerate(lines):
        if line.strip().startswith("CELL_PARAMETERS"):
            cell = np.array([lines[i + k].split()[:3] for k in (1, 2, 3)], float) * BOHR_PER_ANG
        if line.strip().startswith("ATOMIC_POSITIONS"):
            rows = [lines[i + 1 + k].split() for k in range(nat)]
    symbols = [r[0] for r in rows]
    pos = np.array([r[1:4] for r in rows], float) * BOHR_PER_ANG
    return symbols, pos, cell


def geometry(step_dir):
    root = os.path.dirname(os.path.abspath(step_dir.rstrip("/")))
    model = load_model(root)
    _, ref, _ = parse_pw_in(os.path.join(root, "pw.in"))
    symbols, pos, cell = parse_pw_in(os.path.join(step_dir, "pw.in"))
    q = float(pos[0, 0] - ref[0, 0])
    return root, model, symbols, pos, ref, cell, q


def exciton_energies(model, q):
    return {name: p["e0"] + p["slope"] * q for name, p in model["excitons"].items()}


def state_order(model, q):
    e = exciton_energies(model, q)
    return sorted(e, key=lambda n: e[n]), e


def crossed(model, q):
    e = exciton_energies(model, q)
    return e["A"] > e["B"]


def transforms(model, q):
    swap = bool(model["swap_bands_at_crossing"]) and crossed(model, q)
    flip = (2, 5) if q > model["flip_after"] else ()
    theta = model["theta_per_bohr"] * q
    return {s: tm.band_transform(swap, flip, theta) for s in (1, 2)}


def ks_eigenvalues(model, q):
    e = exciton_energies(model, q)
    occ = [-1.0, -0.8, -0.5, -0.5]
    vir = sorted([0.2 + (e["A"] - 0.30), 0.2 + (e["B"] - 0.30)])
    ev = np.array(occ + vir)
    return {1: ev, 2: ev.copy()}


# --------------------------------------------------------------- writers


def write_wfc(path, wfc, ispin=1):
    """Write a QE-style Fortran-unformatted wfc file readable by read_wfc."""
    evc = np.asarray(wfc["evc"], dtype="<c16")
    miller = np.asarray(wfc["miller"], dtype="<i4")
    nbnd, igwx = evc.shape
    head = np.zeros(1, dtype=np.dtype([("ik", "<i4"), ("xk", "<f8", 3), ("ispin", "<i4"),
                                       ("gamma_only", "<i4"), ("scalef", "<f8")]))
    head["ik"], head["ispin"] = 1, ispin
    head["gamma_only"], head["scalef"] = int(bool(wfc["gamma_only"])), 1.0
    f = FortranFile(path, "w")
    f.write_record(head)
    f.write_record(np.array([igwx, igwx, int(wfc["npol"]), nbnd], dtype="<i4"))
    f.write_record(np.eye(3, dtype="<f8").ravel())
    f.write_record(miller.ravel())
    for b in range(nbnd):
        f.write_record(evc[b])
    f.close()


def _band_structure_xml(eig):
    ev = " ".join(f"{x / 2.0:.12e}" for s in sorted(eig) for x in eig[s])  # Ry -> Ha
    lsda = len(eig) == 2
    nb = len(eig[1])
    parts = [f"<lsda>{'true' if lsda else 'false'}</lsda>"]
    if lsda:
        parts.append(f"<nbnd_up>{nb}</nbnd_up><nbnd_dw>{nb}</nbnd_dw>")
    else:
        parts.append(f"<nbnd>{nb}</nbnd>")
    parts.append(f"<ks_energies><k_point weight=\"2.0\">0 0 0</k_point>"
                 f"<eigenvalues size=\"{nb * len(eig)}\">{ev}</eigenvalues></ks_energies>")
    return "<band_structure>" + "".join(parts) + "</band_structure>"


def write_band_structure_only(path, eig):
    with open(path, "w") as f:
        f.write(f"<espresso><output>{_band_structure_xml(eig)}</output></espresso>\n")


def write_pw_xml(path, symbols, pos, cell, etot_ry, forces_ry, eig):
    alat = float(np.linalg.norm(cell[0]))
    at = cell / alat
    bg = np.linalg.inv(at).T
    atoms = "".join(f'<atom name="{s}" index="{i + 1}">{p[0]:.12e} {p[1]:.12e} {p[2]:.12e}</atom>'
                    for i, (s, p) in enumerate(zip(symbols, pos)))
    cellx = "".join(f"<a{i + 1}>{a[0]:.12e} {a[1]:.12e} {a[2]:.12e}</a{i + 1}>" for i, a in enumerate(cell))
    recip = "".join(f"<b{i + 1}>{b[0]:.12e} {b[1]:.12e} {b[2]:.12e}</b{i + 1}>" for i, b in enumerate(bg))
    forces = " ".join(f"{x / 2.0:.12e}" for x in forces_ry.ravel())  # Ry/Bohr -> Ha/Bohr
    xml = (
        "<espresso><input>"
        "<control_variables><calculation>scf</calculation></control_variables>"
        f'<atomic_structure nat="{len(symbols)}" alat="{alat:.12e}">'
        f"<atomic_positions>{atoms}</atomic_positions><cell>{cellx}</cell></atomic_structure>"
        "<dft><functional>PBE</functional></dft>"
        "</input><output>"
        f"<basis_set><reciprocal_lattice>{recip}</reciprocal_lattice></basis_set>"
        f"{_band_structure_xml(eig)}"
        f"<total_energy><etot>{etot_ry / 2.0:.12e}</etot></total_energy>"
        f'<forces rank="2" dims="3 {len(symbols)}">{forces}</forces>'
        "</output></espresso>\n"
    )
    with open(path, "w") as f:
        f.write(xml)


def _next_json(save_dir, stem):
    """WEST-style naming: stem.json, then stem_1.json, stem_2.json, ..."""
    path = os.path.join(save_dir, f"{stem}.json")
    n = 0
    while os.path.exists(path):
        n += 1
        path = os.path.join(save_dir, f"{stem}_{n}.json")
    return path


# --------------------------------------------------------------- executables


def run_pw(step_dir):
    root, model, symbols, pos, ref, cell, q = geometry(step_dir)
    k = model["k"]
    etot = 0.5 * k * float(((pos - ref) ** 2).sum())
    forces = -k * (pos - ref)
    eig = ks_eigenvalues(model, q)
    save = os.path.join(step_dir, "pwscf.save")
    os.makedirs(save, exist_ok=True)
    write_pw_xml(os.path.join(save, "data-file-schema.xml"), symbols, pos, cell, etot, forces, eig)
    C = transforms(model, q)
    seed = 777 if (model["scramble_after_crossing"] and crossed(model, q)) else 1234
    for spin, name in ((1, "wfcup1.dat"), (2, "wfcdw1.dat")):
        U = tm.orbital_basis(spin, seed=seed)
        wfc = {"miller": tm.miller_indices(), "evc": C[spin] @ U.T, "gamma_only": False, "npol": 1}
        write_wfc(os.path.join(save, name), wfc, ispin=spin)
    print(f"fake pw.x: q = {q:.6f} Bohr, etot = {etot:.10f} Ry")


def run_wbse(step_dir, inp):
    root, model, symbols, pos, ref, cell, q = geometry(step_dir)
    with open(os.path.join(step_dir, inp)) as f:
        data = yaml.safe_load(f)
    wc = data.get("wbse_control", {})
    fs = int(wc.get("forces_state", 1))
    neig = int(wc.get("n_liouville_eigen", 1))
    order, e = state_order(model, q)
    ev = [e[n] for n in order][:neig]
    force = np.zeros_like(pos)
    force[0, 0] = -model["excitons"][order[fs - 1]]["slope"]
    save = os.path.join(step_dir, "west.wbse.save")
    os.makedirs(save, exist_ok=True)
    doc = {
        "input": {"wbse_control": {"forces_state": fs, "n_liouville_eigen": neig}},
        "exec": {"ndav": 1, "davitr": [{"ev": ev}]},
        "output": {"forces": {"forces_corrected": force.ravel().tolist()}},
    }
    with open(_next_json(save, "wbse"), "w") as f:
        json.dump(doc, f)
    print(f"fake wbse.x: q = {q:.6f}, forces_state = {fs} ({order[fs - 1]}), ev = {ev}")


def run_westpp(step_dir, inp):
    root, model, symbols, pos, ref, cell, q = geometry(step_dir)
    with open(os.path.join(step_dir, inp)) as f:
        pc = (yaml.safe_load(f) or {}).get("westpp_control", {})
    lo, hi = pc.get("westpp_range", [1, 2])
    sign = bool(pc.get("westpp_sign", False))
    order, e = state_order(model, q)
    exc_sign = {"A": -1.0} if q > 0.08 else {}
    decomp = tm.geometry_state(order, transforms(model, q), exc_sign)
    decomp["states"] = {n: st for n, st in decomp["states"].items() if lo <= n <= hi}
    for n, st in decomp["states"].items():
        st["energy"] = e[order[n - 1]]
    doc = tm.westpp_json(decomp, calc=pc.get("westpp_calculation", "R"), sign=sign,
                         weights=not sign, nliou=pc.get("westpp_n_liouville_to_use", 1))
    save = os.path.join(step_dir, "west.westpp.save")
    os.makedirs(save, exist_ok=True)
    with open(_next_json(save, "westpp"), "w") as f:
        json.dump(doc, f)
    print(f"fake westpp.x: q = {q:.6f}, states {lo}-{hi} = {order[lo - 1:hi]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("exe", choices=["pw", "wbse", "westpp"])
    p.add_argument("-i", dest="inp", required=True)
    a = p.parse_args()
    step_dir = os.getcwd()
    root = os.path.dirname(step_dir)
    with open(os.path.join(root, "fake_calls.log"), "a") as f:
        f.write(f"{a.exe} {os.path.basename(step_dir)} -i {a.inp}\n")
    if a.exe == "pw":
        run_pw(step_dir)
    elif a.exe == "wbse":
        run_wbse(step_dir, a.inp)
    else:
        run_westpp(step_dir, a.inp)


if __name__ == "__main__":
    main()
