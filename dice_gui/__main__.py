"""
Entry point for DICE GUI with splash screen.

Usage: python -m dice_gui
"""

import sys


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

    splash.set_progress(25, next(msg_iter))
    import pandas  # noqa: F401

    splash.set_progress(40, next(msg_iter))
    from dice import __version__  # noqa: F401

    splash.set_progress(55, next(msg_iter))
    from dice_gui.dice_gui import DiceGUI

    splash.set_progress(70, next(msg_iter))
    from dice_gui.styles import apply_theme

    splash.set_progress(85, next(msg_iter))
    apply_theme(app)

    splash.set_progress(95, next(msg_iter))
    window = DiceGUI()

    splash.set_progress(100, "Ready to simulate!")

    # Small delay to show 100% before closing
    QTimer.singleShot(200, splash.close)
    QTimer.singleShot(200, window.show)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
