"""
Entry point for DICE GUI with splash screen.

Usage: python -m dice_gui
"""

import sys


def show_main_window(window, splash) -> None:
    """Show the main window, then close the splash screen.

    The order matters. Qt ends the application when its last visible window
    closes, so a splash screen that closes while the main window is not yet
    shown can end the application at startup.
    """
    window.show()
    splash.close()


def main():
    """Main entry point with splash screen."""
    # Import Qt first (lightweight)
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QTimer

    app = QApplication(sys.argv)

    # Show splash screen immediately
    from dice_gui.splash import DiceSplashScreen
    splash = DiceSplashScreen()
    splash.show()
    app.processEvents()

    # Import heavy modules with progress updates
    import random

    loading_messages = [
        "Exciting carriers...",
        "Diffusing excitons...",
        "Fitting Gaussians...",
        "Counting photons...",
        "Spreading the noise...",
        "Calibrating CNR...",
        "Warming up Monte Carlo...",
        "Aligning optics...",
        "Charging semiconductors...",
        "Propagating wavefunctions...",
        "Measuring lifetimes...",
        "Integrating profiles...",
        "Sampling distributions...",
        "Convolving signals...",
        "Normalizing amplitudes...",
    ]
    random.shuffle(loading_messages)
    msg_iter = iter(loading_messages)

    splash.set_progress(10, next(msg_iter))
    import numpy  # noqa: F401

    splash.set_progress(20, next(msg_iter))
    import pandas  # noqa: F401

    splash.set_progress(30, next(msg_iter))
    import scipy.signal  # noqa: F401

    splash.set_progress(45, next(msg_iter))
    import scipy.optimize  # noqa: F401

    splash.set_progress(50, next(msg_iter))
    import statsmodels.api  # noqa: F401

    splash.set_progress(55, next(msg_iter))
    import matplotlib.pyplot  # noqa: F401

    splash.set_progress(55, next(msg_iter))
    import seaborn  # noqa: F401

    splash.set_progress(60, next(msg_iter))
    import os
    from joblib import Parallel, delayed

    def _warmup_worker():
        """Pre-import simulation dependencies in worker processes."""
        from dice.analysis.simulation import run_single_simulation  # noqa: F401

    # A standalone build cannot start joblib workers; its simulations use a
    # process pool that starts with each simulation, so there is nothing to warm.
    from dice.utils.runtime import joblib_workers_supported

    if joblib_workers_supported():
        Parallel(n_jobs=-1)(
            delayed(_warmup_worker)() for _ in range(os.cpu_count() or 1)
        )

    splash.set_progress(80, next(msg_iter))
    from dice_gui.dice_gui import DiceGUI

    splash.set_progress(85, next(msg_iter))
    from dice_gui.styles import apply_theme

    splash.set_progress(90, next(msg_iter))
    apply_theme(app)

    splash.set_progress(95, next(msg_iter))
    window = DiceGUI()

    splash.set_progress(100, "Ready to simulate!")

    # Small delay to show 100% before the main window replaces the splash
    QTimer.singleShot(200, lambda: show_main_window(window, splash))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
