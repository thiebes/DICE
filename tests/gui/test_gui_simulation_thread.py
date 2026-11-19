"""
SimulationThread: real serial runs of the quick_test example, observed only
through the thread's signals.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest
from PyQt6.QtCore import Qt

from dice.models.results import MonteCarloOutput
from dice_gui.dice_interface import DiceInterface
from dice_gui.simulation_thread import SimulationThread
from gui_support import OLS_SLOPE_COLUMN, WLS_SLOPE_COLUMN, quick_test_engine_parameters

pytestmark = pytest.mark.gui

SIGNAL_TIMEOUT_MS = 120_000


class _SignalLog:
    """Collects everything a SimulationThread emits."""

    def __init__(self, thread: SimulationThread):
        self.messages: list[str] = []
        self.iterations: list[tuple[int, int]] = []
        self.errors: list[str] = []
        self.results: list[object] = []
        thread.progress.connect(self.messages.append)
        thread.iteration_progress.connect(lambda current, total: self.iterations.append((current, total)))
        thread.error.connect(self.errors.append)
        thread.finished.connect(self.results.append)


def _run_to_signal(qtbot, parameters: dict, signal_name: str) -> tuple[SimulationThread, _SignalLog]:
    thread = SimulationThread(DiceInterface(), parameters)
    log = _SignalLog(thread)
    with qtbot.waitSignal(getattr(thread, signal_name), timeout=SIGNAL_TIMEOUT_MS):
        thread.start()
    thread.wait()
    return thread, log


class TestSuccessfulRun:

    @pytest.fixture
    def finished_run(self, qtbot) -> _SignalLog:
        _, log = _run_to_signal(qtbot, quick_test_engine_parameters(number_of_runs=5), "finished")
        return log

    def test_finished_carries_results_and_processed_parameters(self, finished_run):
        assert finished_run.errors == []
        [(result, processed)] = finished_run.results
        assert isinstance(result, MonteCarloOutput)
        assert result.num_runs == len(result.run_results) == 5
        assert processed["filename slug"] == "quick_test"
        assert processed["nominal diffusion length"] == pytest.approx(0.1)
        assert len(processed["time series"]) == 5

    def test_progress_reports_every_serial_iteration(self, finished_run):
        assert finished_run.messages == ["Starting simulation..."]
        assert finished_run.iterations == [(i, 5) for i in range(1, 6)]

    def test_outputs_are_written_under_the_slug(self, finished_run, work_dir):
        output_dir = work_dir / "output" / "quick_test"
        assert sorted(path.name for path in output_dir.iterdir()) == [
            "quick_test.csv", "quick_test_accuracy_histogram.png", "quick_test_summary.txt",
        ]
        frame = pd.read_csv(output_dir / "quick_test.csv")
        for column in (WLS_SLOPE_COLUMN, OLS_SLOPE_COLUMN):
            assert frame[column].notna().sum() == 5, column


class TestFailedRun:

    def test_unreadable_noise_file_emits_error(self, qtbot, work_dir):
        parameters = quick_test_engine_parameters(number_of_runs=5)
        del parameters["noise value"]
        parameters["estimate noise from data"] = str(work_dir / "absent.csv")
        parameters["multiprocessing"] = True

        _, log = _run_to_signal(qtbot, parameters, "error")

        assert log.errors == ["Simulation failed: Error reading CSV for noise estimation."]
        assert log.results == []
        assert log.messages == [f"Running 5 iterations ({os.cpu_count()} CPU cores)..."]

    def test_missing_required_parameter_is_named(self, qtbot):
        parameters = quick_test_engine_parameters(number_of_runs=5)
        del parameters["pixel width"]

        _, log = _run_to_signal(qtbot, parameters, "error")

        assert len(log.errors) == 1
        assert "'pixel width'" in log.errors[0]
        assert log.results == []


class TestCancellation:
    """A stop request ends the run as a cancellation, not as a failure."""

    def test_stop_request_emits_cancelled_and_no_error(self, qtbot):
        thread = SimulationThread(DiceInterface(), quick_test_engine_parameters(number_of_runs=300))
        log = _SignalLog(thread)
        thread.iteration_progress.connect(lambda current, total: thread.request_stop())

        with qtbot.waitSignal(thread.cancelled, timeout=SIGNAL_TIMEOUT_MS):
            thread.start()
        thread.wait()

        assert log.errors == []
        assert log.results == []
        assert 1 <= len(log.iterations) < 300

    def test_stop_button_reports_a_stop_without_an_error_dialog(self, qtbot, window, dialogs):
        window.num_runs_spin.setValue(300)
        window.multiprocessing_check.setChecked(False)
        qtbot.mouseClick(window.run_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: window.progress_bar.value() >= 1, timeout=SIGNAL_TIMEOUT_MS)

        qtbot.mouseClick(window.stop_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: window.simulation_thread is None, timeout=SIGNAL_TIMEOUT_MS)

        assert dialogs.titles("critical") == []
        assert window.status_label.text() == "Simulation stopped by user"
        assert window.run_button.isEnabled()
        assert not window.stop_button.isEnabled()
        assert window.progress_bar.isHidden()
        assert window.interface.last_result is None
