"""Tier 1: startup validation of the westpp input against wbse.in (report only)."""

import copy

import pytest

from westpy.relaxation_tracked import validate_inputs

WBSE = {
    "input_west": {"qe_prefix": "pwscf", "west_prefix": "west", "outdir": "./"},
    "wbse_control": {"n_liouville_eigen": 4, "l_forces": True, "forces_state": 1, "l_spin_flip": False},
}
WESTPP = {
    "input_west": {"qe_prefix": "pwscf", "west_prefix": "west", "outdir": "./"},
    "westpp_control": {
        "westpp_calculation": "CM",
        "westpp_range": [1, 4],
        "westpp_n_liouville_to_use": 4,
        "westpp_format": "C",
        "westpp_sign": True,
        "westpp_l_spin_flip": False,
    },
}


def run(wbse_mod=None, pp_mod=None, nspin=2, forces_state=None):
    w, p = copy.deepcopy(WBSE), copy.deepcopy(WESTPP)
    for d, mod in ((w, wbse_mod), (p, pp_mod)):
        for path, val in (mod or {}).items():
            sec, key = path.split(".")
            d.setdefault(sec, {})[key] = val
    return validate_inputs(w, p, nspin, forces_state)


def test_consistent_inputs_give_no_warnings():
    assert run() == []
    # outdir with/without trailing slash and default prefixes are equivalent
    assert run(pp_mod={"input_west.outdir": "."}) == []


@pytest.mark.parametrize(
    "wbse_mod, pp_mod, nspin, expected",
    [
        (None, {"westpp_control.westpp_range": [2, 4]}, 2, "does not cover the initial forces_state 1"),
        (None, {"westpp_control.westpp_n_liouville_to_use": 3}, 2, "exceeds westpp_n_liouville_to_use"),
        ({"wbse_control.n_liouville_eigen": 3}, None, 2, "exceed n_liouville_eigen 3"),
        (None, {"westpp_control.westpp_l_spin_flip": True}, 2, "but wbse_control.l_spin_flip = False"),
        ({"wbse_control.l_spin_flip": True}, None, 2, "westpp_l_spin_flip = False but wbse_control.l_spin_flip = True"),
        (None, {"westpp_control.westpp_l_spin_flip": True}, 1, "used only when nspin = 2"),
        (None, {"westpp_control.westpp_l_compute": False}, 2, "unknown westpp_control key(s) ['westpp_l_compute']"),
        (None, {"westpp_control.westpp_calculation": "W"}, 2, "does not contain 'C'"),
        ({"wbse_control.l_forces": False}, None, 2, "l_forces is not True"),
        (None, {"westpp_control.westpp_sign": False}, 2, "only weights are compared"),
        (None, {"input_west.qe_prefix": "nv"}, 2, "input_west.qe_prefix differs"),
        ({"input_west.west_prefix": "w2"}, None, 2, "input_west.west_prefix differs"),
        (None, {"input_west.outdir": "./tmp"}, 2, "input_west.outdir differs"),
    ],
)
def test_each_warning(wbse_mod, pp_mod, nspin, expected):
    warnings = run(wbse_mod, pp_mod, nspin)
    assert any(expected in w for w in warnings), warnings


def test_spin_excitation_is_ignored_for_nspin2():
    """For nspin = 2 only l_spin_flip matters (spin_excitation is nspin = 1 only)."""
    assert run({"wbse_control.spin_excitation": "T"}, None, 2) == []


def test_spec_example_input_against_spin_conserving_run():
    """The example westpp input in the spec: unknown key + spin-flip mismatch."""
    pp = {
        "westpp_control.westpp_range": [1, 4],
        "westpp_control.westpp_l_spin_flip": True,
        "westpp_control.westpp_l_compute": False,
        "westpp_control.westpp_l_dipole_realspace": False,
    }
    warnings = run(None, pp, 2)
    assert len(warnings) == 2
    assert any("westpp_l_compute" in w for w in warnings)
    assert any("l_spin_flip" in w for w in warnings)


def test_forces_state_argument_overrides_wbse():
    assert any("forces_state 5" in w for w in run(forces_state=5))
