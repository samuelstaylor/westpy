"""Excited-state tracking for westpy's BFGS relaxation.

``bfgs_iter_tracked`` subclasses :class:`westpy.relaxation.bfgs_iter`. With
``l_track_state=False`` (default) it behaves exactly like ``bfgs_iter``. With
``l_track_state=True`` it follows an excited state by its *character*: at every
step the excitation is decomposed with westpp.x, overlapped with the tracked
state of the previous geometry, and ``forces_state`` is changed if the tracked
state changed its energy index.

See STATE_TRACKING.md in the repository root for a full description.
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from westpy.relaxation import bfgs_iter
from westpy.qe_wfc_overlap import read_wfc, overlap

__all__ = ["bfgs_iter_tracked"]

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

# valid westpp_control keys (WEST manual, westpp_control section)
VALID_WESTPP_KEYS = {
    "westpp_calculation",
    "westpp_range",
    "westpp_format",
    "westpp_sign",
    "westpp_n_pdep_eigen_to_use",
    "westpp_r0",
    "westpp_nr",
    "westpp_rmax",
    "westpp_epsinfty",
    "westpp_box",
    "westpp_n_liouville_to_use",
    "westpp_l_spin_flip",
    "westpp_l_compute_tdm",
    "westpp_wannier_tr_rel",
    "westpp_l_dipole_realspace",
}

HARTREE_TO_RY = 2.0
SPIN_NAME = {1: "up", 2: "dw"}


# ---------------------------------------------------------------------------
# small I/O helpers
# ---------------------------------------------------------------------------


def _load_yaml(path):
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    return data if data else {}


def _norm_outdir(outdir):
    outdir = outdir if outdir else "./"
    return outdir if outdir.endswith("/") else outdir + "/"


def read_nspin_pw_input(pw_input):
    """Return nspin from a pw.x input file (default 1)."""
    with open(pw_input, "r") as f:
        for line in f:
            line = line.split("!")[0]
            m = re.search(r"\bnspin\s*=\s*(\d+)", line, flags=re.IGNORECASE)
            if m:
                return int(m.group(1))
    return 1


def get_forces_state(wbse_input):
    data = _load_yaml(wbse_input)
    return int(data.get("wbse_control", {}).get("forces_state", 1))


def set_forces_state(wbse_input, state):
    """Set wbse_control.forces_state via YAML load/dump; nothing else changes.

    The file is not rewritten if forces_state already has this value.
    """
    data = _load_yaml(wbse_input)
    wc = data.setdefault("wbse_control", {})
    if "forces_state" in wc and int(wc["forces_state"]) == int(state):
        return False
    wc["forces_state"] = int(state)
    with open(wbse_input, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
    return True


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------


def validate_inputs(wbse_data, westpp_data, nspin, forces_state=None):
    """Check a westpp input against wbse.in. Returns a list of warning strings.

    Nothing is fixed here; problems are only reported.
    """
    warnings = []
    wc = wbse_data.get("wbse_control", {}) or {}
    pc = westpp_data.get("westpp_control", {}) or {}

    if not pc:
        warnings.append("westpp input has no westpp_control section")

    unknown = sorted(set(pc) - VALID_WESTPP_KEYS)
    if unknown:
        warnings.append(
            f"unknown westpp_control key(s) {unknown} (westpp.x ignores them); "
            f"did you mean e.g. westpp_l_compute_tdm?"
        )

    calc = str(pc.get("westpp_calculation", "R")).upper()
    if "C" not in calc:
        warnings.append(
            f"westpp_calculation = '{calc}' does not contain 'C': "
            f"no excited-state decomposition will be written"
        )

    rng = pc.get("westpp_range", [1, 2])
    n_use = int(pc.get("westpp_n_liouville_to_use", 1))
    n_eigen = wc.get("n_liouville_eigen")
    lo, hi = int(rng[0]), int(rng[1])
    if forces_state is None:
        forces_state = int(wc.get("forces_state", 1))
    if not (lo <= forces_state <= hi):
        warnings.append(
            f"westpp_range {[lo, hi]} does not cover the initial forces_state {forces_state}"
        )
    if hi > n_use:
        warnings.append(
            f"westpp_range upper bound {hi} exceeds westpp_n_liouville_to_use {n_use}"
        )
    if n_eigen is not None:
        if hi > int(n_eigen) or n_use > int(n_eigen):
            warnings.append(
                f"decomposed states (range {[lo, hi]}, n_liouville_to_use {n_use}) "
                f"exceed n_liouville_eigen {n_eigen}"
            )
    if not bool(wc.get("l_forces", False)):
        warnings.append("wbse_control.l_forces is not True: no excited-state forces")

    # spin flip: spin_excitation is used only for nspin = 1; for nspin = 2
    # l_spin_flip chooses spin-conserving vs spin-flip (WEST manual)
    pp_sf = bool(pc.get("westpp_l_spin_flip", False))
    if nspin == 2:
        wbse_sf = bool(wc.get("l_spin_flip", False))
        if pp_sf != wbse_sf:
            warnings.append(
                f"westpp_l_spin_flip = {pp_sf} but wbse_control.l_spin_flip = {wbse_sf}"
            )
    elif pp_sf:
        warnings.append(
            "westpp_l_spin_flip = True but nspin = 1 (it is used only when nspin = 2)"
        )

    if not bool(pc.get("westpp_sign", False)):
        warnings.append(
            "westpp_sign is False: only weights are compared (phase-free similarity), "
            "which is less reliable for rotated degenerate orbitals"
        )

    wi = wbse_data.get("input_west", {}) or {}
    pi = westpp_data.get("input_west", {}) or {}
    defaults = {"qe_prefix": "pwscf", "west_prefix": "west", "outdir": "./"}
    for key, default in defaults.items():
        a, b = wi.get(key, default), pi.get(key, default)
        if key == "outdir":
            a, b = _norm_outdir(a), _norm_outdir(b)
        if a != b:
            warnings.append(f"input_west.{key} differs: wbse.in '{a}' vs westpp '{b}'")

    return warnings


# ---------------------------------------------------------------------------
# westpp decomposition
# ---------------------------------------------------------------------------


def _westpp_json_index(path):
    m = re.search(r"westpp(?:_(\d+))?\.json$", os.path.basename(path))
    return int(m.group(1)) if (m and m.group(1)) else 0


def find_westpp_cm_json(westpp_save_dir):
    """Newest westpp*.json in westpp_save_dir that contains a decomposition.

    westpp.x never overwrites its JSON (westpp.json, westpp_1.json, ...), so the
    file is chosen by content (westpp_calculation contains 'C' and an output
    section exists) and the highest index wins.
    """
    candidates = []
    for path in glob.glob(os.path.join(westpp_save_dir, "westpp*.json")):
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        pc = data.get("input", {}).get("westpp_control", {})
        calc = str(pc.get("westpp_calculation", "")).upper()
        if "C" in calc and data.get("output"):
            candidates.append((_westpp_json_index(path), os.path.getmtime(path), path))
    if not candidates:
        raise FileNotFoundError(
            f"no westpp decomposition (westpp_calculation with 'C') in {westpp_save_dir}"
        )
    return max(candidates)[2]


def read_westpp_decomposition(path):
    """Read the excited-state decomposition from a westpp JSON file.

    Returns a dict
      {"signed": bool, "path": str,
       "states": {n: {"energy": Ry, "blocks": {(from_spin, to_spin):
                      {"occ": (v0, v1), "vir": (c0, c1), "X": array (nv, nc)}}}}}
    vals are stored occupied-major: X[v - v0, c - c0] = vals[(v - v0) * nc + (c - c0)].
    Band indices are 1-based global Kohn-Sham indices.

    Signed coefficients are expected with westpp_sign: true. If the file holds
    non-negative values only, they are treated as weights w_vc (converted from
    |X| to |X|^2 if they look like amplitudes).
    """
    with open(path, "r") as f:
        data = json.load(f)
    pc = data.get("input", {}).get("westpp_control", {})
    states = {}
    any_negative = False
    for key, entry in data.get("output", {}).items():
        m = re.fullmatch(r"E(\d+)", key)
        if not m:
            continue
        blocks = {}
        for kkey, kval in entry.items():
            if not re.fullmatch(r"K\d+", kkey):
                continue
            p = kval["projection"]
            v0, v1 = (int(x) for x in p["from_bandrange"])
            c0, c1 = (int(x) for x in p["to_bandrange"])
            X = np.asarray(p["vals"], dtype=np.float64).reshape(v1 - v0 + 1, c1 - c0 + 1)
            any_negative = any_negative or bool((X < 0).any())
            blocks[(int(p["from_spin"]), int(p["to_spin"]))] = {
                "occ": (v0, v1),
                "vir": (c0, c1),
                "X": X,
            }
        states[int(m.group(1))] = {
            "energy": float(entry.get("excitation_energy", np.nan)),
            "blocks": blocks,
        }
    signed = any_negative or bool(pc.get("westpp_sign", False))
    if not signed:
        for st in states.values():
            tot1 = sum(b["X"].sum() for b in st["blocks"].values())
            tot2 = sum((b["X"] ** 2).sum() for b in st["blocks"].values())
            if abs(tot2 - 1.0) < abs(tot1 - 1.0):  # values look like |X|
                for b in st["blocks"].values():
                    b["X"] = b["X"] ** 2
    return {"signed": signed, "path": path, "states": states}


def dominant_transitions(state, n=5):
    """Largest components of one decomposed state: list of (spin_from, v, spin_to, c, X)."""
    items = []
    for (fs, ts), b in state["blocks"].items():
        X = b["X"]
        flat = np.argsort(-np.abs(X), axis=None)[:n]
        for k in flat:
            iv, ic = np.unravel_index(k, X.shape)
            items.append((fs, b["occ"][0] + iv, ts, b["vir"][0] + ic, float(X[iv, ic])))
    items.sort(key=lambda t: -abs(t[4]))
    return items[:n]


# ---------------------------------------------------------------------------
# band window
# ---------------------------------------------------------------------------


def _band_weights(decomp, spin, nbnd):
    """Weight carried by each band (1..nbnd) of one spin, summed over states.

    Every state is normalized to total weight 1 first, so each state counts equally.
    """
    w = np.zeros(nbnd + 1)
    for st in decomp["states"].values():
        sq = {k: (b["X"] ** 2 if decomp["signed"] else b["X"]) for k, b in st["blocks"].items()}
        tot = sum(a.sum() for a in sq.values())
        if tot <= 0:
            continue
        for (fs, ts), b in st["blocks"].items():
            a = sq[(fs, ts)] / tot
            v0, v1 = b["occ"]
            c0, c1 = b["vir"]
            if fs == spin:
                w[v0 : v1 + 1] += a.sum(axis=1)
            if ts == spin:
                w[c0 : c1 + 1] += a.sum(axis=0)
    return w


def auto_band_window(decomp, spin, nbnd, weight_frac=0.999):
    """Smallest contiguous band range (lo, hi) holding >= weight_frac of the
    decomposition weight of this spin. Returns None if the spin carries no weight."""
    w = _band_weights(decomp, spin, nbnd)
    total = w.sum()
    if total <= 0:
        return None
    target = weight_frac * total
    best = (1, nbnd)
    lo, acc = 1, 0.0
    for hi in range(1, nbnd + 1):
        acc += w[hi]
        while lo < hi and acc - w[lo] >= target:
            acc -= w[lo]
            lo += 1
        if acc >= target and (hi - lo) < (best[1] - best[0]):
            best = (lo, hi)
    return best


def _restrict(block, win_occ, win_vir):
    """Part of a decomposition block inside the band windows of its two spins."""
    v0, v1 = block["occ"]
    c0, c1 = block["vir"]
    if win_occ is None or win_vir is None:
        return None, [], []
    vb = list(range(max(v0, win_occ[0]), min(v1, win_occ[1]) + 1))
    cb = list(range(max(c0, win_vir[0]), min(c1, win_vir[1]) + 1))
    if not vb or not cb:
        return None, vb, cb
    X = block["X"][vb[0] - v0 : vb[-1] - v0 + 1, cb[0] - c0 : cb[-1] - c0 + 1]
    return X, vb, cb


def retained_weight(state, windows, signed):
    """Fraction of a state's weight that lies inside the band windows."""
    tot, kept = 0.0, 0.0
    for (fs, ts), b in state["blocks"].items():
        a = b["X"] ** 2 if signed else b["X"]
        tot += a.sum()
        X, _, _ = _restrict(b, windows.get(fs), windows.get(ts))
        if X is not None:
            kept += (X**2 if signed else X).sum()
    return kept / tot if tot > 0 else 0.0


# ---------------------------------------------------------------------------
# one geometry: decomposition + KS window + eigenvalues
# ---------------------------------------------------------------------------


def wfc_filename(spin, nspin):
    if nspin == 1:
        return "wfc1.dat"
    return f"wfc{SPIN_NAME[spin]}1.dat"


def read_eigenvalues_xml(xml_file):
    """KS eigenvalues (Ry) per spin from data-file-schema.xml (first k-point)."""
    root = ET.parse(xml_file).getroot()
    bs = root.find("output/band_structure")
    lsda = bs.find("lsda").text.strip().lower() == "true"
    ev = np.array(bs.find("ks_energies/eigenvalues").text.split(), dtype=np.float64)
    ev = ev * HARTREE_TO_RY
    if lsda:
        nup = int(bs.find("nbnd_up").text)
        return {1: ev[:nup], 2: ev[nup:]}
    return {1: ev}


class TrackingGeometry:
    """Everything the identification needs about one geometry."""

    def __init__(self, decomp, windows, ks, eig, nspin, scf_iter=None, source=None):
        self.decomp = decomp  # see read_westpp_decomposition
        self.windows = windows  # {spin: (lo, hi)}
        self.ks = ks  # {spin: read_wfc dict} (empty if orbitals are not tracked)
        self.eig = eig  # {spin: eigenvalues in Ry}
        self.nspin = nspin
        self.scf_iter = scf_iter
        self.source = source

    @property
    def states(self):
        return sorted(self.decomp["states"])

    def retained(self):
        return {
            n: retained_weight(st, self.windows, self.decomp["signed"])
            for n, st in self.decomp["states"].items()
        }


def resolve_windows(decomp, nbnd_by_spin, band_window=None, weight_frac=0.999):
    """band_window: None (auto), (lo, hi) for all spins, or {spin: (lo, hi)}."""
    windows = {}
    for spin, nbnd in nbnd_by_spin.items():
        if band_window is None:
            win = auto_band_window(decomp, spin, nbnd, weight_frac)
        elif isinstance(band_window, dict):
            win = band_window.get(spin)
        else:
            win = (int(band_window[0]), int(band_window[1]))
        if win is not None:
            windows[spin] = (max(1, int(win[0])), min(nbnd, int(win[1])))
    return windows


def load_geometry(
    step_dir,
    pw_prefix="pwscf",
    west_prefix="west",
    outdir="./",
    band_window=None,
    weight_frac=0.999,
    read_orbitals=True,
    scf_iter=None,
):
    """Load the decomposition, eigenvalues and KS band window of one step directory.

    The wavefunctions are taken from <prefix>.save as they are when wbse.x runs
    (bfgs_iter restores the SCF wavefunctions after nscf; only ACE comes from nscf).
    """
    base = os.path.join(step_dir, _norm_outdir(outdir))
    save_dir = base + pw_prefix + ".save/"
    decomp = read_westpp_decomposition(find_westpp_cm_json(base + west_prefix + ".westpp.save"))
    eig = read_eigenvalues_xml(save_dir + "data-file-schema.xml")
    nspin = 2 if len(eig) == 2 else 1
    windows = resolve_windows(
        decomp, {s: len(e) for s, e in eig.items()}, band_window, weight_frac
    )
    ks = {}
    if read_orbitals:
        for spin, (lo, hi) in windows.items():
            ks[spin] = read_wfc(save_dir + wfc_filename(spin, nspin), range(lo, hi + 1))
    return TrackingGeometry(decomp, windows, ks, eig, nspin, scf_iter, os.path.abspath(step_dir))


# ---------------------------------------------------------------------------
# stash (persistent copy of the previous geometry)
# ---------------------------------------------------------------------------


def save_stash(geo, stash_dir):
    """Write a geometry to stash_dir.

    ks_s<spin>.npz holds exactly the read_wfc keys (miller, evc, gamma_only,
    npol, bands), so load_stash gives back dicts that go straight into overlap().
    """
    os.makedirs(stash_dir, exist_ok=True)
    for spin, wfc in geo.ks.items():
        np.savez_compressed(
            os.path.join(stash_dir, f"ks_s{spin}.npz"),
            miller=wfc["miller"],
            evc=wfc["evc"],
            gamma_only=wfc["gamma_only"],
            npol=wfc["npol"],
            bands=np.asarray(wfc["bands"], dtype=np.int64),
        )
    arrays, blocks_meta = {}, {}
    for n, st in geo.decomp["states"].items():
        blocks_meta[str(n)] = {"energy": st["energy"], "blocks": []}
        for (fs, ts), b in st["blocks"].items():
            name = f"E{n}_K{fs}{ts}"
            arrays[name] = b["X"]
            blocks_meta[str(n)]["blocks"].append(
                {"name": name, "from_spin": fs, "to_spin": ts, "occ": list(b["occ"]), "vir": list(b["vir"])}
            )
    np.savez_compressed(os.path.join(stash_dir, "decomp.npz"), **arrays)
    np.savez_compressed(
        os.path.join(stash_dir, "eigenvalues.npz"), **{f"s{s}": e for s, e in geo.eig.items()}
    )
    meta = {
        "scf_iter": geo.scf_iter,
        "source": geo.source,
        "nspin": geo.nspin,
        "signed": geo.decomp["signed"],
        "westpp_json": geo.decomp["path"],
        "windows": {str(s): list(w) for s, w in geo.windows.items()},
        "states": blocks_meta,
    }
    with open(os.path.join(stash_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_ks_stash(path):
    """Load one ks_s<spin>.npz back into the dict returned by read_wfc."""
    with np.load(path) as d:
        return {
            "miller": d["miller"],
            "evc": d["evc"],
            "gamma_only": bool(d["gamma_only"]),
            "npol": int(d["npol"]),
            "bands": [int(b) for b in d["bands"]],
        }


def load_stash(stash_dir):
    with open(os.path.join(stash_dir, "meta.json"), "r") as f:
        meta = json.load(f)
    states = {}
    with np.load(os.path.join(stash_dir, "decomp.npz")) as arr:
        for n, sm in meta["states"].items():
            blocks = {}
            for b in sm["blocks"]:
                blocks[(b["from_spin"], b["to_spin"])] = {
                    "occ": tuple(b["occ"]),
                    "vir": tuple(b["vir"]),
                    "X": arr[b["name"]],
                }
            states[int(n)] = {"energy": sm["energy"], "blocks": blocks}
    decomp = {"signed": meta["signed"], "path": meta["westpp_json"], "states": states}
    with np.load(os.path.join(stash_dir, "eigenvalues.npz")) as e:
        eig = {int(k[1:]): e[k] for k in e.files}
    ks = {}
    for path in glob.glob(os.path.join(stash_dir, "ks_s*.npz")):
        spin = int(re.search(r"ks_s(\d+)\.npz$", path).group(1))
        ks[spin] = load_ks_stash(path)
    windows = {int(s): tuple(w) for s, w in meta["windows"].items()}
    return TrackingGeometry(decomp, windows, ks, eig, meta["nspin"], meta["scf_iter"], meta["source"])


# ---------------------------------------------------------------------------
# overlaps and identification
# ---------------------------------------------------------------------------


def _window_bands(geo, spin):
    lo, hi = geo.windows[spin]
    return list(range(lo, hi + 1))


def ks_overlaps(prev, curr, l_track_orbitals=True):
    """S[spin][a, b] between window band a of prev and window band b of curr.

    Returns (S, bands) with bands[spin] = (prev_bands, curr_bands). With
    l_track_orbitals=False, S is the identity on matching band indices.
    All orbital overlaps go through qe_wfc_overlap.overlap.
    """
    S, bands = {}, {}
    for spin in sorted(set(prev.windows) & set(curr.windows)):
        if l_track_orbitals:
            A, B = prev.ks[spin], curr.ks[spin]
            S[spin] = overlap(A, B)
            bands[spin] = (list(A["bands"]), list(B["bands"]))
        else:
            pb, cb = _window_bands(prev, spin), _window_bands(curr, spin)
            S[spin] = (np.array(pb)[:, None] == np.array(cb)[None, :]).astype(np.float64)
            bands[spin] = (pb, cb)
    return S, bands


def excitation_overlap(state_i, prev, state_j, curr, S, bands, signed=True):
    """Overlap between decomposed state i at prev and state j at curr.

    signed:  O_ij = |sum X_vc^i S_vv' S_cc' X_v'c'^j| / (|X^i| |X^j|)
    weights: O_ij = sum w_vc^i |S_vv'|^2 |S_cc'|^2 w_v'c'^j / (|w^i| |w^j|)
    Only blocks with the same (from_spin, to_spin) are combined. Norms use the
    parts of the states inside the band windows.
    """
    num, ni, nj = 0.0, 0.0, 0.0
    for key in set(state_i["blocks"]) | set(state_j["blocks"]):
        fs, ts = key
        bi, bj = state_i["blocks"].get(key), state_j["blocks"].get(key)
        Xi = Xj = None
        if bi is not None:
            Xi, vi, ci = _restrict(bi, prev.windows.get(fs), prev.windows.get(ts))
            if Xi is not None:
                ni += (Xi**2).sum()
        if bj is not None:
            Xj, vj, cj = _restrict(bj, curr.windows.get(fs), curr.windows.get(ts))
            if Xj is not None:
                nj += (Xj**2).sum()
        if Xi is None or Xj is None or fs not in S or ts not in S:
            continue
        p_occ = {b: k for k, b in enumerate(bands[fs][0])}
        c_occ = {b: k for k, b in enumerate(bands[fs][1])}
        p_vir = {b: k for k, b in enumerate(bands[ts][0])}
        c_vir = {b: k for k, b in enumerate(bands[ts][1])}
        So = S[fs][np.ix_([p_occ[b] for b in vi], [c_occ[b] for b in vj])]
        Sv = S[ts][np.ix_([p_vir[b] for b in ci], [c_vir[b] for b in cj])]
        if signed:
            num += np.sum((So.T @ Xi @ Sv) * Xj)
        else:
            num += np.sum((np.abs(So.T) ** 2 @ Xi @ np.abs(Sv) ** 2) * Xj)
    if ni <= 0 or nj <= 0:
        return 0.0
    return float(abs(num) / np.sqrt(ni * nj))


def compare_geometries(prev, curr, label, l_track_orbitals=True, S=None, bands=None):
    """Overlap row O_label,j for every decomposed state j of curr."""
    if S is None:
        S, bands = ks_overlaps(prev, curr, l_track_orbitals)
    signed = prev.decomp["signed"] and curr.decomp["signed"]
    si = prev.decomp["states"][label]
    row = {
        j: excitation_overlap(si, prev, sj, curr, S, bands, signed)
        for j, sj in sorted(curr.decomp["states"].items())
    }
    return row, S, bands


def identify(row, overlap_thr=0.5, ambiguity_thr=0.1):
    """Pick j* = argmax_j O_ij and flag suspicious cases."""
    order = sorted(row, key=lambda j: -row[j])
    j_star = order[0]
    o_max = row[j_star]
    o_second = row[order[1]] if len(order) > 1 else 0.0
    return {
        "j_star": j_star,
        "max_overlap": o_max,
        "second": order[1] if len(order) > 1 else None,
        "second_overlap": o_second,
        "low_overlap": o_max < overlap_thr,
        "ambiguous": (o_max - o_second) < ambiguity_thr,
        "top_state": j_star == max(row),
    }


def degenerate_blocks(eig, bands, thr):
    """Groups (>= 2 bands) of consecutive bands with eigenvalue gaps below thr (Ry)."""
    groups, cur = [], [bands[0]] if bands else []
    for a, b in zip(bands[:-1], bands[1:]):
        if abs(eig[b - 1] - eig[a - 1]) < thr:
            cur.append(b)
        else:
            if len(cur) > 1:
                groups.append(cur)
            cur = [b]
    if len(cur) > 1:
        groups.append(cur)
    return groups


def degenerate_diagnostics(prev, S, bands, thr=1.0e-3):
    """For each near-degenerate block D of prev: sum_{m,n in D} |S_mn|^2 / |D|.

    About 1 means the block's subspace is preserved (an arbitrary rotation
    inside the block is harmless); clearly below 1 means it mixed with other bands.
    """
    out = []
    for spin, Sm in S.items():
        pb, cb = bands[spin]
        pidx = {b: k for k, b in enumerate(pb)}
        cidx = {b: k for k, b in enumerate(cb)}
        for block in degenerate_blocks(prev.eig[spin], pb, thr):
            if not all(b in cidx for b in block):
                continue
            sub = Sm[np.ix_([pidx[b] for b in block], [cidx[b] for b in block])]
            out.append((spin, block[0], block[-1], float((np.abs(sub) ** 2).sum() / len(block))))
    return out


def self_compare(geo, l_track_orbitals=True):
    """Full matrix O[i, j] of a geometry against itself (should be the identity)."""
    S, bands = ks_overlaps(geo, geo, l_track_orbitals)
    states = geo.states
    O = np.zeros((len(states), len(states)))
    for a, i in enumerate(states):
        row, _, _ = compare_geometries(geo, geo, i, l_track_orbitals, S, bands)
        O[a] = [row[j] for j in states]
    return states, O


# ---------------------------------------------------------------------------
# trajectory
# ---------------------------------------------------------------------------


def rewrite_last_traj_energy(traj_file, energy):
    """Replace energy=... in the comment line of the last extxyz frame."""
    with open(traj_file, "r") as f:
        lines = f.readlines()
    i, last_comment = 0, None
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        nat = int(lines[i].split()[0])
        last_comment = i + 1
        i += nat + 2
    if last_comment is None:
        return False
    value = repr(float(energy))  # plain float: repr(np.float64) is "np.float64(...)"
    new, n = re.subn(r"(?<![\w])energy=\S+", f"energy={value}", lines[last_comment])
    if n == 0:
        new = lines[last_comment].rstrip("\n") + f" energy={value}\n"
    lines[last_comment] = new
    with open(traj_file, "w") as f:
        f.writelines(lines)
    return True


# ---------------------------------------------------------------------------
# the BFGS subclass
# ---------------------------------------------------------------------------


class bfgs_iter_tracked(bfgs_iter):
    """BFGS relaxation that can follow an excited state by its character.

    Args (in addition to those of bfgs_iter, which are passed through):
        l_track_state (bool): master switch. If False, identical to bfgs_iter.
        l_track_orbitals (bool): include KS orbital overlaps (else compare by band index)
        l_test_mode (bool): keep every step directory and stash under tracking/test/
        run_westpp (string): full command to run westpp.x (required if l_track_state)
        westpp_input (string): user's westpp input (copied as-is into every step)
        band_window: None (automatic), (lo, hi), or {spin: (lo, hi)}; 1-based bands
        band_window_weight (float): weight kept by the automatic band window
        overlap_thr (float): warn if the best overlap is below this value
        ambiguity_thr (float): warn if the two best overlaps differ by less
        stop_on_low_overlap (bool): stop instead of warning (low overlap or ambiguity)
        degeneracy_thr (float): eigenvalue gap (Ry) for the degenerate-block diagnostic
    """

    def __init__(
        self,
        *args,
        l_track_state: bool = False,
        l_track_orbitals: bool = True,
        l_test_mode: bool = False,
        run_westpp: str = None,
        westpp_input: str = "westpp.in",
        band_window=None,
        band_window_weight: float = 0.999,
        overlap_thr: float = 0.5,
        ambiguity_thr: float = 0.1,
        stop_on_low_overlap: bool = False,
        degeneracy_thr: float = 1.0e-3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.l_track_state = l_track_state
        if not l_track_state:
            # plain bfgs_iter: nothing is read or written
            return

        if not run_westpp:
            raise ValueError("run_westpp is required when l_track_state is True")
        self.l_track_orbitals = l_track_orbitals
        self.l_test_mode = l_test_mode
        self.run_westpp = run_westpp
        self.westpp_input = westpp_input
        self.band_window = band_window
        self.band_window_weight = band_window_weight
        self.overlap_thr = overlap_thr
        self.ambiguity_thr = ambiguity_thr
        self.stop_on_low_overlap = stop_on_low_overlap
        self.degeneracy_thr = degeneracy_thr

        root_dir = os.getcwd() + "/"
        self.tracking_dir = root_dir + "tracking/"
        self.test_dir = self.tracking_dir + "test/"
        self.tracking_file = root_dir + "bfgs_tracking.json"
        self.history_file = self.tracking_dir + "history.json"

        l_restart = kwargs.get("l_restart", False)
        self._log("Excited-state tracking: ON")

        # validate the user's inputs (report only, never fix)
        wbse_in = root_dir + self.wbse_input
        self.nspin = read_nspin_pw_input(root_dir + self.pw_input)
        for w in validate_inputs(
            _load_yaml(wbse_in), _load_yaml(root_dir + westpp_input), self.nspin
        ):
            self._log(f"WARNING (tracking): {w}")

        # keep the user's original wbse.in
        if not os.path.exists(wbse_in + ".orig"):
            shutil.copy2(wbse_in, wbse_in + ".orig")

        self.history = []
        self.prev_stash = None
        self.track_label = get_forces_state(wbse_in)
        if l_restart and os.path.exists(self.tracking_file):
            with open(self.tracking_file, "r") as f:
                tdata = json.load(f)
            self.track_label = int(tdata["label"])
            self.history = tdata.get("history", [])
            self.prev_stash = tdata.get("stash")
            self._log(f"tracking restarted: label = {self.track_label}")
        elif l_restart and self.start_iter > 0:
            self._log(
                "WARNING (tracking): restart without bfgs_tracking.json; "
                "tracking restarts from forces_state in wbse.in with no reference"
            )
        elif os.path.exists(self.tracking_dir):
            backup = self.tracking_dir.rstrip("/") + time.strftime(".bak-%Y%m%d-%H%M%S")
            shutil.move(self.tracking_dir, backup)
            self._log(f"existing tracking/ moved to {backup}")
        os.makedirs(self.tracking_dir, exist_ok=True)

        self._log(f"tracked state (initial forces_state) = {self.track_label}")
        self._log(
            f"l_track_orbitals = {l_track_orbitals}, band_window = "
            f"{'auto' if band_window is None else band_window}, "
            f"overlap_thr = {overlap_thr}, ambiguity_thr = {ambiguity_thr}"
        )
        if l_test_mode:
            os.makedirs(self.test_dir, exist_ok=True)
            self._log(
                "WARNING (tracking): l_test_mode keeps every step directory "
                "(including wavefunction files) under tracking/test/: this uses a lot of disk"
            )
        self._log("")

    # ----------------------------------------------------------- overrides

    def _run_calc(self):
        if not self.l_track_state:
            return super()._run_calc()

        root_dir = os.getcwd() + "/"
        work_dir = root_dir + self.folder_name + str(self.scf_iter) + "/"
        label = self.track_label

        # 1. the root wbse.in carries the tracked label (_run_calc copies it)
        set_forces_state(root_dir + self.wbse_input, label)

        # 2. normal step: pw (+ nscf, wbse_init), wbse with forces on state `label`
        super()._run_calc()
        e_first = self.energy - self.gs_e

        # 3. decompose the excitations
        self._run_westpp(work_dir)

        # 4.-5. identify the continuation of the tracked state
        curr = load_geometry(
            work_dir,
            self.pw_prefix,
            self.west_prefix,
            self.outdir,
            self.band_window,
            self.band_window_weight,
            self.l_track_orbitals,
            self.scf_iter,
        )
        retained = curr.retained()
        record = {
            "scf_iter": self.scf_iter,
            "label_before": label,
            "label_after": label,
            "swap": False,
            "overlap_row": None,
            "max_overlap": None,
            "warnings": [],
            "windows": {str(s): list(w) for s, w in curr.windows.items()},
            "retained_weight_min": float(min(retained.values())) if retained else None,
            "degenerate_blocks": [],
        }
        self._log(f"tracking: label before   = {label}")
        rw = record["retained_weight_min"]
        self._log(
            "tracking: band windows   = "
            + (", ".join(f"{SPIN_NAME.get(s, s)} {w[0]}-{w[1]}" for s, w in curr.windows.items()) or "none")
            + (f" (min retained weight {rw:.4f})" if rw is not None else "")
        )

        j_star = label
        prev = load_stash(self.prev_stash) if self.prev_stash else None
        if label not in curr.decomp["states"]:
            msg = f"tracked state {label} is not among the decomposed states {curr.states}"
            record["warnings"].append(msg)
            self._log(f"WARNING (tracking): {msg}")
        elif prev is None:
            self._log("tracking: first step, stored as reference")
        elif label not in prev.decomp["states"]:
            msg = f"tracked state {label} missing from the previous decomposition"
            record["warnings"].append(msg)
            self._log(f"WARNING (tracking): {msg}")
        else:
            row, S, bands = compare_geometries(prev, curr, label, self.l_track_orbitals)
            res = identify(row, self.overlap_thr, self.ambiguity_thr)
            j_star = res["j_star"]
            record["overlap_row"] = {str(j): o for j, o in row.items()}
            record["max_overlap"] = res["max_overlap"]
            record["compared_with_scf_iter"] = prev.scf_iter
            self._log(
                "tracking: overlap row    = "
                + "  ".join(f"O({label},{j})={o:.4f}" for j, o in row.items())
            )
            for spin, b0, b1, val in degenerate_diagnostics(prev, S, bands, self.degeneracy_thr):
                record["degenerate_blocks"].append([spin, b0, b1, val])
                self._log(
                    f"tracking: degenerate block {SPIN_NAME.get(spin, spin)} "
                    f"{b0}-{b1}: sum|S|^2/n = {val:.4f}"
                )
            if res["low_overlap"]:
                record["warnings"].append(
                    f"max overlap {res['max_overlap']:.4f} < overlap_thr {self.overlap_thr}"
                )
            if res["ambiguous"]:
                record["warnings"].append(
                    f"ambiguous: O({label},{j_star})={res['max_overlap']:.4f} vs "
                    f"O({label},{res['second']})={res['second_overlap']:.4f}"
                )
            if res["top_state"] and len(row) > 1:
                record["warnings"].append(
                    f"state {j_star} is the highest decomposed state: the tracked state "
                    f"may be leaving the computed manifold"
                )
            for w in record["warnings"]:
                self._log(f"WARNING (tracking): {w}")
            if self.stop_on_low_overlap and (res["low_overlap"] or res["ambiguous"]):
                self.history.append(record)
                self._write_tracking_state(self.prev_stash)
                self._log("stop_on_low_overlap is True: stopping")
                sys.exit(1)

        # 6.-7. rerun wbse.x only if the tracked state changed index
        if j_star != label:
            self._log(f"STATE SWAP: {label} -> {j_star}")
            record["swap"] = True
            record["energy_first_run"] = e_first
            self._rerun_wbse(work_dir, label, j_star)
            record["energy_rerun"] = self.energy - self.gs_e
            record["energy_westpp_j"] = curr.decomp["states"][j_star]["energy"]

        # 8. update the label and log
        record["label_after"] = j_star
        record["forces_state"] = j_star
        record["gs_energy"] = self.gs_e
        record["exc_energy"] = self.energy - self.gs_e
        record["energy"] = self.energy
        if j_star in curr.decomp["states"]:
            record["dominant_transitions"] = dominant_transitions(curr.decomp["states"][j_star])
        self.track_label = j_star
        self._log(f"tracking: label after    = {j_star}")
        self.history.append(record)
        self._curr_geo = curr
        self._write_tracking_state(self.prev_stash)

    def _write_new_pw_input(self):
        if not self.l_track_state:
            return super()._write_new_pw_input()

        root_dir = os.getcwd() + "/"
        work_dir = root_dir + self.folder_name + str(self.scf_iter) + "/"
        self.history[-1]["total_force"] = float(self.grad_t)

        # stash Q_curr before super() deletes the step directory. The stash is
        # always the most recently *computed* geometry, also when BFGS rejects
        # the step and goes back to pos_p: it is the closest one in configuration
        # space to the next geometry.
        stash_dir = self.tracking_dir + f"stash/step-{self.scf_iter}"
        save_stash(self._curr_geo, stash_dir)
        if not self.l_test_mode:
            for old in glob.glob(self.tracking_dir + "stash/step-*"):
                if os.path.abspath(old) != os.path.abspath(stash_dir):
                    shutil.rmtree(old)
        self.prev_stash = stash_dir
        self._curr_geo = None

        if self.l_test_mode:
            self._copy_step_for_test(work_dir)
        self._write_tracking_state(stash_dir)

        super()._write_new_pw_input()

    def _save_final_geo(self):
        if not self.l_track_state:
            return super()._save_final_geo()

        root_dir = os.getcwd() + "/"
        work_dir = root_dir + self.folder_name + str(self.scf_iter) + "/"
        if self.history:
            self.history[-1]["total_force"] = float(self.grad_t)
        if self.l_test_mode:
            self._copy_step_for_test(work_dir)
        self._write_tracking_state(self.prev_stash)

        super()._save_final_geo()

    def _terminate_bfgs(self):
        if self.l_track_state:
            swaps = [r for r in self.history if r.get("swap")]
            self._log("")
            self._log(f"tracking: {len(swaps)} state swap(s), final tracked state = {self.track_label}")
            for r in swaps:
                self._log(
                    f"tracking: scf cycle {r['scf_iter']:3d}: STATE SWAP "
                    f"{r['label_before']} -> {r['label_after']}"
                )
        super()._terminate_bfgs()

    # ------------------------------------------------------------- helpers

    def _run_westpp(self, work_dir):
        self._log("Running westpp.x ...")
        shutil.copy2(self.westpp_input, work_dir + self.westpp_input)
        command = f"{self.run_westpp} -i {self.westpp_input} > westpp.out"
        try:
            subprocess.run(command, shell=True, cwd=work_dir, check=True)
        except subprocess.CalledProcessError:
            self._log(f"westpp.x failed: {work_dir}")
            exit()

    def _rerun_wbse(self, work_dir, old_state, new_state):
        """Rerun only wbse.x with forces on new_state, in the same step directory."""
        root_dir = os.getcwd() + "/"
        # keep the first run's output
        if os.path.exists(work_dir + "wbse.out"):
            os.rename(work_dir + "wbse.out", work_dir + f"wbse_state-{old_state}.out")
        save = work_dir + self.outdir + self.west_prefix + ".wbse.save/"
        # wbse.x does not overwrite wbse.json (it would write wbse_1.json), so
        # move the first run's file away before rerunning
        if os.path.exists(save + "wbse.json"):
            os.rename(save + "wbse.json", save + f"wbse_state-{old_state}.json")

        set_forces_state(root_dir + self.wbse_input, new_state)
        shutil.copy2(self.wbse_input, work_dir + self.wbse_input)

        self._log(f"Running wbse.x again for state {new_state} ...")
        command = f"{self.run_wbse} -i {self.wbse_input} > wbse.out"
        try:
            subprocess.run(command, shell=True, cwd=work_dir, check=True)
        except subprocess.CalledProcessError:
            self._log(f"wbse.x failed: {work_dir}")
            exit()

        # energy and gradient of the new state (reads forces_state from wbse.json)
        self._read_excited_state_forces_energy()
        # _save_traj ran before the rerun: fix the energy of the last frame
        rewrite_last_traj_energy(root_dir + self.traj_file, self.energy)

    def _copy_step_for_test(self, work_dir):
        dest = self.test_dir + self.folder_name + str(self.scf_iter)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(work_dir, dest)
        with open(dest + "/tracking_summary.txt", "w") as f:
            f.write(format_step_summary(self.history[-1]))

    def _write_tracking_state(self, stash_dir):
        tdata = {
            "label": self.track_label,
            "scf_iter": getattr(self, "scf_iter", None),
            "stash": stash_dir,
            "history": self.history,
        }
        _dump_json(self.tracking_file, tdata)
        _dump_json(self.history_file, self.history)
        if self.l_test_mode:
            with open(self.test_dir + "swap_summary.txt", "w") as f:
                f.write(format_swap_summary(self.history))


def _dump_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=float)
    os.replace(tmp, path)


def format_step_summary(r):
    lines = [
        f"scf cycle            : {r['scf_iter']}",
        f"label before         : {r['label_before']}",
        f"label after          : {r['label_after']}",
        f"state swap           : {'yes' if r.get('swap') else 'no'}",
    ]
    if r.get("overlap_row"):
        lines.append(
            "overlap row O_i.     : "
            + "  ".join(f"{j}:{o:.4f}" for j, o in r["overlap_row"].items())
        )
        lines.append(f"max overlap          : {r['max_overlap']:.6f}")
    else:
        lines.append("overlap row O_i.     : (reference step, no comparison)")
    lines.append(f"band windows         : {r.get('windows')}")
    lines.append(f"min retained weight  : {r.get('retained_weight_min')}")
    for spin, b0, b1, val in r.get("degenerate_blocks", []):
        lines.append(f"degenerate block     : spin {spin} bands {b0}-{b1} sum|S|^2/n = {val:.4f}")
    if r.get("swap"):
        lines.append(f"exc. energy 1st run  : {r.get('energy_first_run'):.10f} Ry (state {r['label_before']})")
        lines.append(f"exc. energy rerun    : {r.get('energy_rerun'):.10f} Ry (state {r['label_after']})")
        lines.append(f"westpp energy of j*  : {r.get('energy_westpp_j'):.10f} Ry")
    lines.append("dominant transitions : spin_from band_from -> spin_to band_to : X")
    for fs, v, ts, c, x in r.get("dominant_transitions", []):
        lines.append(f"     {fs} {v:6d} -> {ts} {c:6d} : {x:+.6f}")
    lines.append(f"ground-state energy  : {r.get('gs_energy'):.10f} Ry")
    lines.append(f"excitation energy    : {r.get('exc_energy'):.10f} Ry")
    tf = r.get("total_force")
    lines.append(f"total force          : {tf:.10f} Ry/Bohr" if tf is not None else "total force          : n/a")
    for w in r.get("warnings", []):
        lines.append(f"WARNING              : {w}")
    return "\n".join(lines) + "\n"


def format_swap_summary(history):
    lines = ["# step  forces_state  swap  max_overlap"]
    for r in history:
        mo = r.get("max_overlap")
        mo = f"{mo:.4f}" if mo is not None else "   -  "
        lines.append(
            f"  {r['scf_iter']:4d}  {r.get('forces_state', r['label_after']):12d}  "
            f"{'yes' if r.get('swap') else 'no ':>4s}  {mo:>11s}"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# dry-run tools (no executables are launched)
# ---------------------------------------------------------------------------


def replay(step_dirs, label=None, l_track_orbitals=True, band_window=None,
           band_window_weight=0.999, overlap_thr=0.5, ambiguity_thr=0.1, **prefixes):
    """Replay the identification across existing step directories (e.g. tracking/test).

    The starting label is taken from the first directory's wbse.in unless given.
    Returns a list of per-step dicts.
    """
    step_dirs = sorted(step_dirs, key=lambda d: int(re.search(r"(\d+)/?$", d).group(1)))
    if label is None:
        label = get_forces_state(os.path.join(step_dirs[0], "wbse.in"))
    out, prev = [], None
    for d in step_dirs:
        curr = load_geometry(d, band_window=band_window, weight_frac=band_window_weight,
                             read_orbitals=l_track_orbitals, **prefixes)
        rec = {"step_dir": d, "label_before": label, "label_after": label, "row": None}
        if prev is not None and label in prev.decomp["states"]:
            row, _, _ = compare_geometries(prev, curr, label, l_track_orbitals)
            res = identify(row, overlap_thr, ambiguity_thr)
            rec.update(row=row, **res)
            label = res["j_star"]
            rec["label_after"] = label
        out.append(rec)
        prev = curr
    return out


def _main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m westpy.tracking_cli",
        description="Dry-run tools for excited-state tracking (no executables are run).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("selfcompare", "replay"):
        s = sub.add_parser(name)
        s.add_argument("path", help="run/step directory (selfcompare) or directory with step-* (replay)")
        s.add_argument("--band-window", nargs=2, type=int, default=None)
        s.add_argument("--no-orbitals", action="store_true", help="l_track_orbitals=False")
        s.add_argument("--qe-prefix", default="pwscf")
        s.add_argument("--west-prefix", default="west")
        s.add_argument("--outdir", default="./")
    sub.choices["replay"].add_argument("--label", type=int, default=None)
    a = p.parse_args(argv)
    prefixes = {"pw_prefix": a.qe_prefix, "west_prefix": a.west_prefix, "outdir": a.outdir}
    orbitals = not a.no_orbitals
    np.set_printoptions(precision=4, suppress=True, linewidth=200)

    if a.cmd == "selfcompare":
        geo = load_geometry(a.path, band_window=a.band_window, read_orbitals=orbitals, **prefixes)
        print("band windows:", geo.windows, " retained weight:",
              {n: round(float(w), 5) for n, w in geo.retained().items()})
        states, O = self_compare(geo, orbitals)
        print("O[i, j], rows/cols = states", states)
        print(O)
        mapping = {i: states[int(np.argmax(O[k]))] for k, i in enumerate(states)}
        print("mapping i -> argmax_j:", mapping)
        print("identity:", all(i == j for i, j in mapping.items()),
              " min O_ii:", float(np.min(np.diag(O))))
    else:
        dirs = sorted(glob.glob(os.path.join(a.path, "step-*")))
        for r in replay(dirs, a.label, orbitals, a.band_window, **prefixes):
            row = "  ".join(f"{j}:{o:.3f}" for j, o in r["row"].items()) if r["row"] else "(reference)"
            flag = "  STATE SWAP" if r["label_before"] != r["label_after"] else ""
            print(f"{os.path.basename(r['step_dir'].rstrip('/')):>10s}  "
                  f"{r['label_before']} -> {r['label_after']}  {row}{flag}")


if __name__ == "__main__":
    _main()
