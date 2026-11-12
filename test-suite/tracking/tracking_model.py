"""Synthetic model shared by the tier-1 and tier-2 tracking tests.

Orbitals and excitons are defined in a fixed *orbital* basis phi_o (orthonormal
plane-wave coefficient vectors). At a given geometry the code sees *bands*
psi_b = sum_o C[b, o] phi_o, where C contains the effects a real calculation
can produce between two geometries:

  * a permutation of the virtual orbitals (band swap),
  * sign flips of individual bands,
  * a rotation inside a degenerate occupied pair.

An exciton with amplitudes A[o_v, o_c] (orbital basis) has band-basis
coefficients X = C_occ A C_vir^T, which is what westpp would report.

Layout per spin: 6 bands, bands 1-4 occupied, 5-6 virtual. Occupied orbitals
3 and 4 form a degenerate pair (e-like). Spin 2 ("down") carries most weight.
"""

import numpy as np

NBND = 6
NOCC = 4
NPW = 40  # plane waves in the fake basis

# excitons in the orbital basis: {name: {(spin, o_v, o_c): amplitude}} (1-based)
DIABATIC = {
    "A": {(2, 4, 5): 0.95, (2, 3, 5): 0.25, (1, 4, 5): 0.10},
    "B": {(2, 4, 6): 0.95, (2, 2, 6): 0.25, (1, 4, 6): 0.10},
    "C": {(2, 3, 6): 0.90, (2, 1, 5): 0.40},
}


def orbital_basis(spin, seed=1234):
    """NPW x NBND matrix of orthonormal orbital coefficient vectors (complex)."""
    rng = np.random.default_rng(seed + spin)
    M = rng.normal(size=(NPW, NBND)) + 1j * rng.normal(size=(NPW, NBND))
    Q, _ = np.linalg.qr(M)
    return Q[:, :NBND]


def miller_indices(seed=99):
    rng = np.random.default_rng(seed)
    pts = set()
    while len(pts) < NPW:
        pts.add(tuple(int(x) for x in rng.integers(-4, 5, size=3)))
    return np.array(sorted(pts), dtype=np.int64)


def band_transform(swap_virtual=False, flip=(), theta=0.0):
    """C[b, o] (0-based) for one spin: band b = sum_o C[b, o] phi_o."""
    C = np.eye(NBND)
    # rotation inside the degenerate occupied pair (orbitals 3, 4)
    c, s = np.cos(theta), np.sin(theta)
    C[2:4, 2:4] = [[c, s], [-s, c]]
    if swap_virtual:
        C[[4, 5]] = C[[5, 4]]
    for b in flip:  # 1-based band indices whose sign flips
        C[b - 1] *= -1.0
    return C


def amplitudes(name, spin):
    A = np.zeros((NBND, NBND))
    for (s, ov, oc), a in DIABATIC[name].items():
        if s == spin:
            A[ov - 1, oc - 1] = a
    return A


def normalization(name):
    return np.sqrt(sum(a * a for a in DIABATIC[name].values()))


def band_coefficients(name, spin, C, sign=1.0):
    """Occupied x virtual block X (NOCC x NBND-NOCC) of exciton `name` in the band basis."""
    A = amplitudes(name, spin) / normalization(name)
    Xfull = C @ A @ C.T
    return sign * Xfull[:NOCC, NOCC:]


def geometry_state(order, C_by_spin, exc_sign=None):
    """Decomposition dict (read_westpp_decomposition layout) for the excitons in `order`.

    order: list of exciton names, index 0 -> state 1 (lowest energy).
    """
    exc_sign = exc_sign or {}
    states = {}
    for n, name in enumerate(order, start=1):
        blocks = {}
        for spin in (1, 2):
            X = band_coefficients(name, spin, C_by_spin[spin], exc_sign.get(name, 1.0))
            blocks[(spin, spin)] = {"occ": (1, NOCC), "vir": (NOCC + 1, NBND), "X": X}
        states[n] = {"energy": 0.3 + 0.01 * n, "blocks": blocks}
    return {"signed": True, "path": "synthetic", "states": states}


def wavefunctions(C, spin):
    """read_wfc-style dict for one spin (gamma_only=False, all bands)."""
    U = orbital_basis(spin)
    evc = (C @ U.T)  # rows = bands, columns = plane waves
    return {
        "miller": miller_indices(),
        "evc": evc,
        "gamma_only": False,
        "npol": 1,
        "bands": list(range(1, NBND + 1)),
    }


def eigenvalues():
    """KS eigenvalues (Ry) per spin; bands 3/4 are degenerate."""
    return np.array([-1.0, -0.8, -0.5, -0.5, 0.2, 0.25])


def westpp_json(decomp, calc="CM", sign=True, weights=False, nliou=None):
    """A westpp JSON document with the real layout (occupied-major vals)."""
    states = decomp["states"]
    nliou = nliou or max(states)
    out = {}
    for n, st in states.items():
        entry = {"excitation_energy": st["energy"]}
        for k, ((fs, ts), b) in enumerate(sorted(st["blocks"].items()), start=1):
            X = b["X"]
            vals = (X**2 if weights else X).ravel(order="C")
            entry[f"K{k:06d}"] = {
                "projection": {
                    "from_spin": fs,
                    "to_spin": ts,
                    "from_bandrange": list(b["occ"]),
                    "to_bandrange": list(b["vir"]),
                    "vals": [float(v) for v in vals],
                }
            }
        entry["delta_spin_square"] = 0.0
        out[f"E{n:06d}"] = entry
    out["ground_state"] = {"spin_square": 2.0}
    return {
        "runjob": {},
        "input": {
            "westpp_control": {
                "westpp_calculation": calc,
                "westpp_range": [min(states), max(states)],
                "westpp_n_liouville_to_use": nliou,
                "westpp_sign": sign,
            }
        },
        "output": out,
    }
