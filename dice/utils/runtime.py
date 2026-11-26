"""
Facts about the running program that change what DICE can do.
"""

import sys
from pathlib import Path
from typing import Optional

# File name of the license notice that dice-gui.spec bundles (scripts/collect_licenses.py).
LICENSE_NOTICE_FILE_NAME = 'THIRD-PARTY-LICENSES.txt'


def is_standalone_build() -> bool:
    """
    Report whether DICE runs as a standalone (PyInstaller) application.

    PyInstaller marks a standalone build by setting sys.frozen.

    Returns
    -------
    bool
        True inside a standalone (frozen) build, False otherwise.
    """
    return bool(getattr(sys, 'frozen', False))


def joblib_workers_supported() -> bool:
    """
    Report whether joblib can start parallel worker processes.

    joblib's default (loky) backend cannot start workers from a standalone
    build: each worker re-runs the program's entry script instead of the
    simulation. A standalone build runs its parallel simulations in the
    standard library's process pool instead.

    Returns
    -------
    bool
        False inside a standalone (frozen) build, True otherwise.
    """
    return not is_standalone_build()


def examples_directory() -> Path:
    """
    Directory that holds the example parameter files, data/examples.

    A standalone build unpacks the folder that its PyInstaller spec bundles
    into sys._MEIPASS; a source checkout has it at the repository root. A
    wheel installation contains no data folder, so the returned directory
    does not exist there.

    Returns
    -------
    Path
        The data/examples directory for this program; it may not exist.
    """
    if is_standalone_build():
        return Path(sys._MEIPASS) / 'data' / 'examples'
    return Path(__file__).resolve().parents[2] / 'data' / 'examples'


def bundled_license_notice() -> Optional[Path]:
    """
    The license notice that a standalone build carries.

    The notice gives the license of DICE and of every package the
    application bundles. Only a standalone build has it.

    Returns
    -------
    Path or None
        The notice file, or None outside a standalone build or when the
        build did not bundle it.
    """
    bundle_folder = getattr(sys, '_MEIPASS', None)
    if not is_standalone_build() or bundle_folder is None:
        return None
    notice = Path(bundle_folder) / LICENSE_NOTICE_FILE_NAME
    return notice if notice.is_file() else None
