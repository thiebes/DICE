"""
Facts about the running program that change what DICE can do.
"""

import sys


def parallel_processing_supported() -> bool:
    """
    Report whether simulations can run in parallel worker processes.

    The simulation runs in parallel through joblib's default (loky) backend,
    which cannot start workers from a standalone build: each worker re-runs
    the program's entry script instead of the simulation. PyInstaller marks a
    standalone build by setting sys.frozen.

    Returns
    -------
    bool
        False inside a standalone (frozen) build, True otherwise.
    """
    return not getattr(sys, 'frozen', False)
