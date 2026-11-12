"""Command-line dry-run tools for excited-state tracking (no executables are run).

    python -m westpy.tracking_cli selfcompare <run_or_step_dir>
    python -m westpy.tracking_cli replay <dir_with_step-*>   (e.g. tracking/test)

See westpy/relaxation_tracked.py and STATE_TRACKING.md.
"""

from westpy.relaxation_tracked import _main

if __name__ == "__main__":
    _main()
