#!/usr/bin/env python3
"""
Overlap matrix between Kohn-Sham orbitals from two Quantum ESPRESSO runs.

Reads the Fortran-unformatted wfc files in <prefix>.save/ (wfcup1.dat,
wfcdw1.dat, or wfc1.dat) and computes S_mn = <psi_m(A)|psi_n(B)>.

Assumes:
  - same cell and ecutwfc in both runs (true for a fixed-cell relaxation)
  - norm-conserving pseudopotentials (no S-matrix needed; ONCV is fine)
  - Gamma point (single k-point), with or without gamma_only

Usage:
  python qe_wfc_overlap.py prev/pwscf.save curr/pwscf.save --spin up --bands 120-130
"""
import argparse
import numpy as np
from scipy.io import FortranFile


def read_wfc(path, bands=None):
    """Read a QE wfc*.dat file.

    bands: iterable of 1-based band indices to keep (None = all).
    Returns dict with miller (igwx,3), evc (nbands_kept, npol*igwx), gamma_only, bands.
    """
    f = FortranFile(path, "r")
    head = f.read_record(np.dtype([("ik", "<i4"), ("xk", "<f8", 3), ("ispin", "<i4"),
                                   ("gamma_only", "<i4"), ("scalef", "<f8")]))[0]
    ngw, igwx, npol, nbnd = f.read_ints("<i4")
    f.read_reals("<f8")                                # reciprocal lattice b1,b2,b3
    miller = f.read_ints("<i4").reshape(igwx, 3)

    keep = set(range(1, nbnd + 1)) if bands is None else set(bands)
    evc, kept = [], []
    for ib in range(1, nbnd + 1):
        rec = f.read_record("<c16")
        if ib in keep:
            evc.append(rec)
            kept.append(ib)
    f.close()
    return {"miller": miller, "evc": np.array(evc), "gamma_only": bool(head["gamma_only"]),
            "npol": npol, "bands": kept}


def _match_gvectors(mA, mB):
    """Indices into A and B of the common G-vectors (matched by Miller index)."""
    shift = 1 << 10                                    # Miller indices fit well within +/-512
    key = lambda m: ((m[:, 0] + shift) * (2 * shift) + (m[:, 1] + shift)) * (2 * shift) + (m[:, 2] + shift)
    _, iA, iB = np.intersect1d(key(mA), key(mB), return_indices=True)
    return iA, iB


def overlap(A, B):
    """S_mn = <psi_m(A)|psi_n(B)> for the bands stored in A and B."""
    if A["gamma_only"] != B["gamma_only"]:
        raise ValueError("Both runs must use the same gamma_only setting.")
    if A["npol"] != 1:
        raise NotImplementedError("Noncollinear (npol=2) not handled.")

    iA, iB = _match_gvectors(A["miller"], B["miller"])
    if len(iA) != len(A["miller"]) or len(iB) != len(B["miller"]):
        print(f"Warning: G-vector sets differ ({len(A['miller'])} vs {len(B['miller'])}, "
              f"{len(iA)} common). Check cell/cutoff.")
    cA, cB = A["evc"][:, iA], B["evc"][:, iB]
    S = cA.conj() @ cB.T

    if A["gamma_only"]:
        # Only half the G-sphere is stored: psi(-G) = psi(G)*.
        # <m|n> = 2 Re sum_half c_m* c_n - c_m(0) c_n(0)
        g0 = np.where(~A["miller"][iA].any(axis=1))[0]
        S = 2.0 * S.real
        if g0.size:
            S -= np.outer(cA[:, g0[0]].real, cB[:, g0[0]].real)
    return S


def parse_bands(s):
    lo, hi = (int(x) for x in s.split("-")) if "-" in s else (int(s), int(s))
    return list(range(lo, hi + 1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("save_A", help="prefix.save dir of the previous geometry")
    p.add_argument("save_B", help="prefix.save dir of the current geometry")
    p.add_argument("--spin", choices=["up", "dw", "none"], default="up")
    p.add_argument("--bands", required=True, help="1-based range, e.g. 120-130")
    args = p.parse_args()

    fname = "wfc1.dat" if args.spin == "none" else f"wfc{args.spin}1.dat"
    bands = parse_bands(args.bands)
    A = read_wfc(f"{args.save_A}/{fname}", bands)
    B = read_wfc(f"{args.save_B}/{fname}", bands)
    S = np.abs(overlap(A, B))

    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    print(f"|<psi_m(A)|psi_n(B)>|, spin {args.spin}, bands {bands[0]}-{bands[-1]}")
    print("rows = A bands, cols = B bands\n")
    print("      " + " ".join(f"{b:6d}" for b in B["bands"]))
    for i, b in enumerate(A["bands"]):
        print(f"{b:5d} " + " ".join(f"{x:6.3f}" for x in S[i]))

    print("\nBest match A -> B:")
    for i, b in enumerate(A["bands"]):
        j = int(np.argmax(S[i]))
        flag = "" if B["bands"][j] == b else "   <-- reordered"
        print(f"  {b:5d} -> {B['bands'][j]:5d}   |S| = {S[i, j]:.3f}{flag}")


if __name__ == "__main__":
    main()
