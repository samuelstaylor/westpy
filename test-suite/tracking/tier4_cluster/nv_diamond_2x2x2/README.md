# Tier 4b: NV- in diamond, 2x2x2 (63 atoms), crossing test

Source (read-only): /global/cfs/cdirs/m4382/sstaylor/projects/NV-center-diamond/2x2x2/tddft
(pw.x + wbse.x, PBE, nspin = 2, 152 bands, ecutwfc 85). Copied to
`BASE/source`, BASE = /global/cfs/cdirs/m4382/sstaylor/tracking_tests/nv_diamond_2x2x2
(CFS, because Slurm scratch licenses were held on 2026-11-12).

Start: the **initial** pw.in (C3v ground-state geometry, where the 3E pair is
degenerate), forces_state 1. wbse.in: n_liouville_eigen 4 (the source had 1).
westpp.in: CM, states 1-4, signed, spin-conserving.

    bash prepare.sh                 # builds BASE/fixture and BASE/relax

| Run | Job | What |
|---|---|---|
| `fixture/` | `job_fixture.sh` | single point: pw -> wbse -> westpp, then `tracking_cli selfcompare`. Becomes the tier-3 diamond fixture (`pytest -m fixture`). |
| `relax/` | `job_relax.sh` | `bfgs_iter_tracked`, tracking + test mode, maxiter 20; resubmit to continue (auto restart). |

CPU build `QEdir_cpu`, debug queue, 1 node x 32 MPI x 8 CPUs, 30 min.
`ESPRESSO_PSEUDO=./` because pw.in has `pseudo_dir` commented out.

Check afterwards:

    conda run -n westpy-dev python -m westpy.tracking_cli replay BASE/relax/tracking/test
    cat BASE/relax/tracking/test/swap_summary.txt
    cd ~/testing/westpy/test-suite/tracking && conda run -n westpy-dev python -m pytest -m fixture -v tier3_fixture
