"""
Simulation Thread Module

Background thread for running DICE simulations without blocking the GUI.
"""

from PyQt6.QtCore import QThread, pyqtSignal

from dice_gui.dice_interface import DiceInterface


class SimulationThread(QThread):
    """Thread for running simulations without blocking the GUI."""

    progress = pyqtSignal(str)
    iteration_progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, interface: DiceInterface, parameters: dict):
        super().__init__()
        self.interface = interface
        self.parameters = parameters
        self._stop_requested = False

    def run(self):
        """Run the simulation in a separate thread."""
        try:
            num_runs = self.parameters.get('number of runs', 1000)
            is_parallel = self.parameters.get('multiprocessing', True)

            if is_parallel:
                import os
                cpu_count = os.cpu_count() or 1
                self.progress.emit(f"Running {num_runs:,} iterations ({cpu_count} CPU cores)...")
            else:
                self.progress.emit("Starting simulation...")

            def progress_callback(current: int, total: int):
                if self._stop_requested:
                    raise InterruptedError("Simulation cancelled by user")
                self.iteration_progress.emit(current, total)

            result = self.interface.run_simulation(
                self.parameters,
                progress_callback=progress_callback
            )
            self.finished.emit(result)
        except InterruptedError:
            self.error.emit("Simulation cancelled by user")
        except Exception as e:
            self.error.emit(str(e))

    def request_stop(self):
        """Request the simulation to stop."""
        self._stop_requested = True
