# Tier 4b: NV- in diamond, 2x2x2 (63 atoms), crossing test (Phase 7, not prepared yet)

Source (read-only): /global/cfs/cdirs/m4382/sstaylor/projects/NV-center-diamond/2x2x2/tddft
(pw.x + wbse.x, PBE, nspin = 2, 152 bands). The original wbse.in has
n_liouville_eigen: 1; the crossing test needs >= 4 states and westpp over 1-4.
Planned:
* a single-point fixture job (pw + wbse + westpp) -> $SCRATCH/tracking_fixture_diamond,
  used by tier3 (test_diamond_self_compare_identity) and to confirm the westpp format;
* a tracked relaxation with l_test_mode=True; check swap_summary.txt with
  `python -m westpy.tracking_cli replay tracking/test`.
