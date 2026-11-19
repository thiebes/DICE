"""
Entry point of the standalone (PyInstaller) DICE application.

dice-gui.spec builds the application from this script, which decides what
the process is for before it imports Qt or the GUI:

- A worker process of a parallel simulation starts the application again;
  multiprocessing.freeze_support() runs the worker and exits.
- Otherwise it starts the GUI.
"""

import multiprocessing


def main() -> None:
    """Run a worker or the GUI."""
    multiprocessing.freeze_support()

    from dice_gui.dice_gui import main as start_gui
    start_gui()


if __name__ == '__main__':
    main()
