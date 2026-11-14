"""Shared configuration for the excited-state tracking tests.

Tiers (see README.md):
  tier1_unit      synthetic data, pure functions                 (always run)
  tier2_mockwest  full BFGS loop with fake pw/wbse/westpp scripts (always run)
  tier3_fixture   real WEST output on disk                         (-m fixture, skipped if missing)
  tier4_cluster   job templates; nothing runs under pytest
"""

import os
import sys

import pytest

# one BLAS thread: the fake executables are tiny, and login nodes are shared
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # tracking_model, fake_west

# real-data fixtures (tier 3). Override with environment variables.
MGO_VEE_FIXTURE = os.environ.get(
    "TRACKING_FIXTURE", os.path.expandvars("$SCRATCH/tracking_fixture")
)
DIAMOND_FIXTURE = os.environ.get(
    "TRACKING_FIXTURE_DIAMOND", "/global/cfs/cdirs/m4382/sstaylor/tracking_tests/nv_diamond_2x2x2/fixture"
)


def pytest_configure(config):
    config.addinivalue_line("markers", "fixture: needs real WEST output on disk (tier 3, slow)")
    config.addinivalue_line("markers", "mockwest: runs the BFGS loop with fake executables (tier 2)")


def pytest_collection_modifyitems(config, items):
    # tier-3 tests are opt-in: run them with `-m fixture`
    if "fixture" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="real-data fixture test; run with -m fixture")
    for item in items:
        if "fixture" in item.keywords:
            item.add_marker(skip)
