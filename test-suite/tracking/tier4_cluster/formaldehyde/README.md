# Tier 4a: formaldehyde smoke test (Phase 6, not prepared yet)

From WEST tutorial 12 (https://west-code.org/doc/training/formaldehyde/es/), with
`n_liouville_eigen: 4` and westpp over states 1-4. Two runs on the debug queue:
1. `l_track_state=False` (must match plain `bfgs_iter`);
2. `l_track_state=True, l_test_mode=True, maxiter=3`.
Not spin-polarized (wfc1.dat), so it exercises the nspin = 1 path.
