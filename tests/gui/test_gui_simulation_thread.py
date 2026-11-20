"""
SimulationThread: real runs of the quick_test example, serial or in the
standalone build's process pool, observed only through the thread's signals.
"""

from __future__ import annotations

import multiprocessing
import os
from typing import Callable, NamedTuple

import pandas as pd
import pytest
from joblib import effective_n_jobs
from PyQt6.QtCore import Qt

from dice.analysis.simulation import split_into_batches
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


class _ProcessPool(NamedTuple):
    new_children: Callable[[], set]


@pytest.fixture
def process_pool(monkeypatch) -> _ProcessPool:
    """Run parallel simulations in the standard library's process pool, as a standalone build does."""
    import dice.analysis.simulation as simulation

    monkeypatch.setattr(simulation, "joblib_workers_supported", lambda: False)
    # joblib's reusable workers from earlier tests are not the pool's.
    earlier = set(multiprocessing.active_children())
    return _ProcessPool(new_children=lambda: set(multiprocessing.active_children()) - earlier)


class TestProcessPoolRun:
    """A parallel run in the standalone build's process pool, through the GUI thread."""

    def test_finished_with_progress_after_each_batch(self, qtbot, process_pool):
        parameters = {**quick_test_engine_parameters(number_of_runs=20), "multiprocessing": True}

        _, log = _run_to_signal(qtbot, parameters, "finished")

        assert log.errors == []
        [(result, _)] = log.results
        assert [run.run_id for run in result.run_results] == list(range(20))
        assert log.iterations == [(batch.stop, 20) for batch in split_into_batches(20, effective_n_jobs(-1))]
        assert process_pool.new_children() == set()


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

    def test_stop_request_ends_the_process_pool_workers(self, qtbot, process_pool):
        parameters = {**quick_test_engine_parameters(number_of_runs=300), "multiprocessing": True}
        thread = SimulationThread(DiceInterface(), parameters)
        log = _SignalLog(thread)
        thread.iteration_progress.connect(lambda current, total: thread.request_stop())

        with qtbot.waitSignal(thread.cancelled, timeout=SIGNAL_TIMEOUT_MS):
            thread.start()
        thread.wait()

        assert log.errors == []
        assert log.results == []
        assert 1 <= len(log.iterations) < len(split_into_batches(300, effective_n_jobs(-1)))
        assert process_pool.new_children() == set()

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

    def test_stop_button_lets_the_process_pool_end_its_workers(self, qtbot, window, dialogs, process_pool):
        # Stop must not terminate the thread: a terminated parallel run leaves its workers running.
        window.num_runs_spin.setValue(300)
        qtbot.mouseClick(window.run_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: "/" in window.status_label.text(), timeout=SIGNAL_TIMEOUT_MS)

        qtbot.mouseClick(window.stop_button, Qt.MouseButton.LeftButton)
        assert window.status_label.text() == "Stopping..."
        assert not window.stop_button.isEnabled()
        assert not window.run_button.isEnabled()
        qtbot.waitUntil(lambda: window.simulation_thread is None, timeout=SIGNAL_TIMEOUT_MS)

        assert dialogs.titles("critical") == []
        assert window.status_label.text() == "Simulation stopped by user"
        assert window.run_button.isEnabled()
        assert window.interface.last_result is None
        assert process_pool.new_children() == set()


class TestThreadRelease:
    """The window releases a simulation thread only after its run() has returned.

    The thread emits finished, error or cancelled at the end of run(), and the
    window's slot runs while the thread may still be finishing. Deleting a
    running QThread aborts the whole process, which crashed CI runs on Linux,
    macOS and Windows. These tests stretch the moments after the signal and
    check that the thread had finished when the window released it.
    """

    @pytest.fixture
    def released(self, monkeypatch):
        import time

        run = SimulationThread.run

        def run_then_linger(thread):
            run(thread)
            time.sleep(0.5)  # the moments between the signal and the end of run(), stretched

        threads = []

        def record_release(thread):
            # Keeping the thread alive here, instead of deleting it, lets a
            # failing run report the problem instead of aborting the tests.
            threads.append((thread, thread.isFinished()))

        monkeypatch.setattr(SimulationThread, 'run', run_then_linger)
        monkeypatch.setattr(SimulationThread, 'deleteLater', record_release)
        yield threads
        for thread, _ in threads:
            thread.wait()

    def test_finished_run_is_released_after_it_returns(self, qtbot, window, dialogs, released):
        window.num_runs_spin.setValue(5)
        window.multiprocessing_check.setChecked(False)
        qtbot.mouseClick(window.run_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: window.simulation_thread is None, timeout=SIGNAL_TIMEOUT_MS)

        assert dialogs.titles("information") == ["Simulation Complete"]
        assert [finished for _, finished in released] == [True]

    def test_stopped_run_is_released_after_it_returns(self, qtbot, window, dialogs, released):
        window.num_runs_spin.setValue(300)
        window.multiprocessing_check.setChecked(False)
        qtbot.mouseClick(window.run_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: window.progress_bar.value() >= 1, timeout=SIGNAL_TIMEOUT_MS)
        qtbot.mouseClick(window.stop_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: window.simulation_thread is None, timeout=SIGNAL_TIMEOUT_MS)

        assert window.status_label.text() == "Simulation stopped by user"
        assert [finished for _, finished in released] == [True]

    def test_failed_run_is_released_after_it_returns(self, qtbot, window, dialogs, released, monkeypatch):
        def fail(*args, **kwargs):
            raise ValueError("a run that fails")

        monkeypatch.setattr(window.interface, 'run_simulation', fail)
        window.num_runs_spin.setValue(5)
        qtbot.mouseClick(window.run_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: window.simulation_thread is None, timeout=SIGNAL_TIMEOUT_MS)

        assert dialogs.titles("critical") == ["Simulation Error"]
        assert [finished for _, finished in released] == [True]
