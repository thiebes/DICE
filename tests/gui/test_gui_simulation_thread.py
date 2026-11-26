"""
SimulationThread: real runs of the quick_test example, serial or in the
standalone build's process pool, observed only through the thread's signals.
"""

from __future__ import annotations

import multiprocessing
import time
from typing import Callable, NamedTuple

import joblib
import pandas as pd
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox

from dice.models.results import MonteCarloOutput
from dice_gui.dice_interface import DiceInterface
from dice_gui.simulation_thread import SimulationThread
from gui_support import EXAMPLES_DIR, OLS_SLOPE_COLUMN, WLS_SLOPE_COLUMN, quick_test_engine_parameters

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


@pytest.fixture
def slow_serial_runs(monkeypatch) -> None:
    """Make each run of a serial simulation take at least 20 ms.

    A test that stops or closes during a simulation must find it still
    running. On a fast machine, 300 runs of the quick_test example end before
    the test has reacted to the first progress signal. The pause sits where
    the engine reports its progress, between two runs, so a stop request is
    still noticed at once.
    """
    run_simulation = DiceInterface.run_simulation

    def run_slowly(interface, parameters, progress_callback=None):
        def report_then_pause(current: int, total: int) -> None:
            progress_callback(current, total)
            time.sleep(0.02)

        return run_simulation(interface, parameters,
                              progress_callback=report_then_pause if progress_callback else None)

    monkeypatch.setattr(DiceInterface, "run_simulation", run_slowly)


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


class TestProgressBar:
    """The bar of a parallel run shows activity until the first report, and fills from then on."""

    def test_first_report_of_a_parallel_run_gives_the_bar_its_range(self, window):
        window.progress_bar.setRange(0, 0)  # as start_simulation leaves it for a parallel run
        window.elapsed_timer.start()

        window._update_iteration_progress(64, 20_000)

        assert (window.progress_bar.minimum(), window.progress_bar.maximum()) == (0, 20_000)
        assert window.progress_bar.value() == 64
        assert window.status_label.text() == "Running 64/20,000"

        window._update_iteration_progress(576, 20_000)

        assert window.progress_bar.maximum() == 20_000
        assert window.progress_bar.value() == 576


class TestProcessPoolRun:
    """A parallel run in the standalone build's process pool, through the GUI thread."""

    def test_finished_with_progress_after_each_batch(self, qtbot, process_pool):
        parameters = {**quick_test_engine_parameters(number_of_runs=20), "multiprocessing": True}

        _, log = _run_to_signal(qtbot, parameters, "finished")

        assert log.errors == []
        [(result, _)] = log.results
        assert [run.run_id for run in result.run_results] == list(range(20))
        # One report after each batch; the sizes of the batches follow the clock.
        assert log.iterations[-1] == (20, 20)
        assert [done for done, _ in log.iterations] == sorted({done for done, _ in log.iterations})
        assert {total for _, total in log.iterations} == {20}
        assert process_pool.new_children() == set()


class TestFailedRun:

    def test_unreadable_noise_file_emits_error(self, qtbot, work_dir, monkeypatch):
        parameters = quick_test_engine_parameters(number_of_runs=5)
        del parameters["noise value"]
        parameters["estimate noise from data"] = str(work_dir / "absent.csv")
        parameters["multiprocessing"] = True
        monkeypatch.setattr(joblib, "effective_n_jobs", lambda n_jobs: 4)

        _, log = _run_to_signal(qtbot, parameters, "error")

        [error] = log.errors
        assert error.startswith("Error reading CSV for noise estimation: ")
        assert "absent.csv" in error and "could not be read" in error
        assert log.results == []
        assert log.messages == ["Running 5 runs (4 at a time)..."]

    @pytest.mark.parametrize("number_of_runs, workers_of_joblib, status_text", [
        (5, 8, "Running 5 runs (5 at a time)..."),
        (1, 8, "Running 1 run (1 at a time)..."),
        (12000, 1, "Running 12,000 runs (1 at a time)..."),
        (12000, 8, "Running 12,000 runs (8 at a time)..."),
    ])
    def test_status_text_counts_the_runs_and_how_many_run_at_a_time(
            self, qtbot, work_dir, monkeypatch, number_of_runs, workers_of_joblib, status_text):
        """No more at a time than there are runs, and no more than joblib uses for all processors."""
        parameters = quick_test_engine_parameters(number_of_runs=number_of_runs)
        del parameters["noise value"]
        parameters["estimate noise from data"] = str(work_dir / "absent.csv")
        parameters["multiprocessing"] = True
        monkeypatch.setattr(joblib, "effective_n_jobs", lambda n_jobs: workers_of_joblib)

        _, log = _run_to_signal(qtbot, parameters, "error")

        assert log.messages == [status_text]

    def test_missing_required_parameter_is_named(self, qtbot):
        parameters = quick_test_engine_parameters(number_of_runs=5)
        del parameters["pixel width"]

        _, log = _run_to_signal(qtbot, parameters, "error")

        assert len(log.errors) == 1
        assert "'pixel width'" in log.errors[0]
        assert log.results == []


class TestCancellation:
    """A stop request ends the run as a cancellation, not as a failure."""

    def test_stop_request_emits_cancelled_and_no_error(self, qtbot, slow_serial_runs):
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
        # The stop takes effect at the report after the one that asked for it.
        assert log.iterations != []
        assert log.iterations[-1][0] < 300
        assert process_pool.new_children() == set()

    def test_stop_button_reports_a_stop_without_an_error_dialog(self, qtbot, window, dialogs, slow_serial_runs):
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

    def test_stopped_run_is_released_after_it_returns(self, qtbot, window, dialogs, released, slow_serial_runs):
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


def _start_long_serial_run(qtbot, window) -> SimulationThread:
    window.num_runs_spin.setValue(300)
    window.multiprocessing_check.setChecked(False)
    qtbot.mouseClick(window.run_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.progress_bar.value() >= 1, timeout=SIGNAL_TIMEOUT_MS)
    return window.simulation_thread


def _stop_and_wait(qtbot, window) -> None:
    qtbot.mouseClick(window.stop_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.simulation_thread is None, timeout=SIGNAL_TIMEOUT_MS)


@pytest.mark.usefixtures("slow_serial_runs")
class TestOneSimulationAtATime:
    """While a simulation runs, nothing in the form can start another one."""

    def test_run_stays_disabled_when_the_form_turns_valid_again(self, qtbot, window, dialogs):
        thread = _start_long_serial_run(qtbot, window)

        window.filename_slug_input.clear()
        window.validation_manager.validate_all()
        assert not window.validation_manager.is_valid()
        window.filename_slug_input.setText("valid_again")
        window.validation_manager.validate_all()
        assert window.validation_manager.is_valid()

        assert not window.run_button.isEnabled()
        assert window.status_label.text() != "Ready to run"
        # The Ctrl+R shortcut reaches run_simulation whatever the state of the button.
        window.run_simulation()
        assert window.simulation_thread is thread
        assert thread.isRunning() and not thread.stop_requested()

        _stop_and_wait(qtbot, window)
        assert window.status_label.text() == "Simulation stopped by user"
        assert window.run_button.isEnabled()
        assert dialogs.titles("critical") == []

    def test_run_stays_disabled_after_a_run_that_left_the_form_invalid(self, qtbot, window, dialogs):
        _start_long_serial_run(qtbot, window)
        window.filename_slug_input.clear()
        window.validation_manager.validate_all()

        _stop_and_wait(qtbot, window)

        assert not window.run_button.isEnabled()
        window.filename_slug_input.setText("valid_again")
        window.validation_manager.validate_all()
        assert window.run_button.isEnabled()
        assert window.status_label.text() == "Ready to run"


@pytest.mark.usefixtures("slow_serial_runs")
class TestClosingDuringARun:
    """Closing the window asks first, then waits for the run; it never terminates the thread."""

    @pytest.fixture
    def thread_calls(self, monkeypatch) -> dict:
        calls = {"terminate": 0, "wait": []}
        wait = SimulationThread.wait

        def record_wait(thread, *args):
            calls["wait"].append(args)
            return wait(thread, *args)

        def record_terminate(thread):
            calls["terminate"] += 1

        monkeypatch.setattr(SimulationThread, "wait", record_wait)
        monkeypatch.setattr(SimulationThread, "terminate", record_terminate)
        return calls

    def test_close_keeps_the_window_until_the_run_has_stopped(self, qtbot, window, dialogs, thread_calls):
        window.show()
        thread = _start_long_serial_run(qtbot, window)
        window.parameters_modified = False  # no unsaved-changes prompt before the one under test
        dialogs.question_answers.append(QMessageBox.StandardButton.Yes)

        assert window.close() is False  # the window stays while the run stops

        assert dialogs.titles("question") == ["Simulation Running"]
        assert thread.stop_requested()
        assert window.isVisible()
        assert not window.tabs.isEnabled() and not window.menuBar().isEnabled()
        assert not window.stop_button.isEnabled() and not window.run_button.isEnabled()
        # The message stays readable (an enabled label) and is not replaced by queued progress.
        assert window.status_label.isEnabled()
        assert window.status_label.text().startswith("Stopping. DICE exits when")

        # A second close while DICE waits asks nothing and changes nothing.
        assert window.close() is False
        assert dialogs.titles("question") == ["Simulation Running"]
        window.update_progress("Running 5 runs...")
        assert window.status_label.text().startswith("Stopping. DICE exits when")

        qtbot.waitUntil(lambda: not window.isVisible(), timeout=SIGNAL_TIMEOUT_MS)

        assert window.simulation_thread is None  # released after its run returned
        assert thread_calls["terminate"] == 0
        # A wait with a time limit would cut off a run that has not stopped yet.
        assert thread_calls["wait"] and all(arguments == () for arguments in thread_calls["wait"])
        # The questions were answered before the wait and are not asked again.
        assert dialogs.titles("question") == ["Simulation Running"]

    def test_run_that_ends_while_the_question_is_open_lets_the_window_close(
            self, qtbot, window, dialogs, monkeypatch):
        """The question runs an event loop; the run can end and release its thread before the answer."""
        window.show()
        _start_long_serial_run(qtbot, window)
        window.parameters_modified = False
        asked = []

        def end_the_run_then_answer_yes(parent, title, text, *args, **kwargs):
            asked.append(title)
            window.stop_simulation()
            qtbot.waitUntil(lambda: window.simulation_thread is None, timeout=SIGNAL_TIMEOUT_MS)
            return QMessageBox.StandardButton.Yes

        monkeypatch.setattr(QMessageBox, "question", staticmethod(end_the_run_then_answer_yes))

        assert window.close() is True

        assert asked == ["Simulation Running"]
        assert not window.isVisible()

    def test_close_with_unsaved_changes_asks_once(self, qtbot, window, dialogs, thread_calls):
        window.show()
        _start_long_serial_run(qtbot, window)
        window.parameters_modified = True
        dialogs.question_answers.extend([QMessageBox.StandardButton.Discard, QMessageBox.StandardButton.Yes])

        assert window.close() is False
        qtbot.waitUntil(lambda: not window.isVisible(), timeout=SIGNAL_TIMEOUT_MS)

        assert dialogs.titles("question") == ["Unsaved Changes", "Simulation Running"]

    def test_declining_keeps_the_window_and_the_run(self, qtbot, window, dialogs, thread_calls):
        thread = _start_long_serial_run(qtbot, window)
        window.parameters_modified = False  # no unsaved-changes prompt before the one under test
        dialogs.question_answers.append(QMessageBox.StandardButton.No)

        assert window.close() is False

        assert dialogs.titles("question") == ["Simulation Running"]
        assert thread.isRunning() and not thread.stop_requested()
        assert thread_calls == {"terminate": 0, "wait": []}
        _stop_and_wait(qtbot, window)


class TestOutputFilesThatCannotBeWritten:
    """A finished simulation is kept when its output files cannot be written."""

    def _block_the_results_file(self, work_dir) -> None:
        # A folder where the results file belongs makes the write fail, as a
        # file that another program holds open does on Windows.
        (work_dir / "output" / "quick_test" / "quick_test.csv").mkdir(parents=True)

    def test_interface_returns_the_result_and_writes_the_files_on_a_retry(self, work_dir):
        self._block_the_results_file(work_dir)
        interface = DiceInterface()

        result, processed = interface.run_simulation(quick_test_engine_parameters(number_of_runs=5))

        assert result.num_runs == 5
        assert interface.last_output_error.endswith("quick_test.csv: " + interface.last_output_error.split(": ")[-1])
        assert (interface.last_result, interface.last_parameters) == (result, processed)
        assert interface.retry_write_outputs() == interface.last_output_error != None  # noqa: E711

        (work_dir / "output" / "quick_test" / "quick_test.csv").rmdir()
        assert interface.retry_write_outputs() is None
        assert interface.last_output_error is None
        assert sorted(path.name for path in (work_dir / "output" / "quick_test").iterdir()) == [
            "quick_test.csv", "quick_test_accuracy_histogram.png", "quick_test_summary.txt",
        ]

    def _run_quick_test(self, qtbot, window, dialogs) -> None:
        window.load_parameters_from_file(str(EXAMPLES_DIR / "quick_test.txt"))
        window.num_runs_spin.setValue(5)
        window.multiprocessing_check.setChecked(False)
        dialogs.shown.clear()
        qtbot.mouseClick(window.run_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: window.simulation_thread is None, timeout=SIGNAL_TIMEOUT_MS)

    def test_window_keeps_the_results_and_says_why(self, qtbot, window, dialogs, work_dir):
        self._block_the_results_file(work_dir)

        self._run_quick_test(qtbot, window, dialogs)

        assert dialogs.titles() == ["Output Files Not Written"]
        assert "quick_test.csv" in dialogs.texts("warning")[0]
        assert window.interface.last_result.num_runs == 5
        assert window.results_status_label.text() == "Simulation results: quick_test (5 runs)"
        assert window.status_label.text() == "Simulation completed; output files not written"
        assert window.run_button.isEnabled()
        # The message names the file and the reason, not Python's text of the error.
        assert "[Errno" not in dialogs.texts("warning")[0] and "\\\\" not in dialogs.texts("warning")[0]
        assert "File > Write Output Files" in dialogs.texts("warning")[0]
        assert window.write_outputs_action.isEnabled()

    def test_files_can_be_written_later_from_the_file_menu(self, qtbot, window, dialogs, work_dir):
        self._block_the_results_file(work_dir)
        self._run_quick_test(qtbot, window, dialogs)  # the scripted warning dialog answers Close
        assert window.write_outputs_action.isEnabled()

        (work_dir / "output" / "quick_test" / "quick_test.csv").rmdir()
        dialogs.shown.clear()
        window.write_outputs_action.trigger()

        # Written at once, so there is nothing to ask: one message says where the files are.
        assert dialogs.titles() == ["Output Files Written"]
        assert dialogs.titles("information") == ["Output Files Written"]
        assert str(work_dir / "output" / "quick_test") in dialogs.texts("information")[0]
        assert sorted(path.name for path in (work_dir / "output" / "quick_test").iterdir()) == [
            "quick_test.csv", "quick_test_accuracy_histogram.png", "quick_test_summary.txt",
        ]
        assert not window.write_outputs_action.isEnabled()
        assert window.status_label.text() == "Output files written"

    def test_loading_results_asks_before_it_replaces_unwritten_results(
            self, qtbot, window, dialogs, work_dir, completed_run):
        self._block_the_results_file(work_dir)
        self._run_quick_test(qtbot, window, dialogs)
        unwritten = window.interface.last_result
        dialogs.shown.clear()

        dialogs.question_answers.append(QMessageBox.StandardButton.No)
        window.load_results_button.click()

        assert dialogs.titles("question") == ["Output Files Not Written"]
        assert dialogs.texts("question")[0].startswith(
            "Loading a results file discards the results of the last simulation, whose output files "
            "have not been written. Discard them?")
        assert "choose No, then File > Write Output Files" in dialogs.texts("question")[0]
        assert window.interface.last_result is unwritten
        assert window.write_outputs_action.isEnabled()

        dialogs.shown.clear()
        dialogs.question_answers.append(QMessageBox.StandardButton.Yes)
        dialogs.open_paths.append(str(completed_run.csv_path))
        window.load_results_button.click()

        assert window.interface.last_result is not unwritten
        assert window.interface.last_output_error is None
        assert not window.write_outputs_action.isEnabled()
        # Nothing waits for its files now: the action writes nothing and fails on nothing.
        assert window.interface.retry_write_outputs() is None
        assert window.status_label.text() == "Results loaded"
        window.write_output_files()
        assert window.status_label.text() == "Results loaded"  # not "Output files written"
        window.parameters_modified = False
        dialogs.shown.clear()
        assert window.close() is True
        assert dialogs.titles("question") == []

    def test_new_run_asks_before_it_replaces_unwritten_results(self, qtbot, window, dialogs, work_dir):
        self._block_the_results_file(work_dir)
        self._run_quick_test(qtbot, window, dialogs)
        dialogs.shown.clear()
        dialogs.question_answers.append(QMessageBox.StandardButton.No)

        qtbot.mouseClick(window.run_button, Qt.MouseButton.LeftButton)

        assert dialogs.titles("question") == ["Output Files Not Written"]
        assert dialogs.texts("question")[0].startswith("A new simulation discards the results of the last simulation")
        assert window.simulation_thread is None
        assert window.write_outputs_action.isEnabled()

    def test_exit_with_unwritten_files_asks_first(self, qtbot, window, dialogs, work_dir):
        self._block_the_results_file(work_dir)
        self._run_quick_test(qtbot, window, dialogs)
        window.parameters_modified = False
        dialogs.shown.clear()
        dialogs.question_answers.append(QMessageBox.StandardButton.No)

        assert window.close() is False

        assert dialogs.titles("question") == ["Output Files Not Written"]
        assert "are lost when DICE exits" in dialogs.texts("question")[0]

    def test_plot_that_cannot_be_saved_is_an_output_file_that_was_not_written(self, qtbot, window, dialogs, work_dir):
        """The results file and the summary are written; the plot is not, and Retry covers it."""
        (work_dir / "output" / "quick_test" / "quick_test_accuracy_histogram.png").mkdir(parents=True)

        self._run_quick_test(qtbot, window, dialogs)

        assert dialogs.titles() == ["Output Files Not Written"]
        assert "quick_test_accuracy_histogram.png" in dialogs.texts("warning")[0]
        assert (work_dir / "output" / "quick_test" / "quick_test.csv").is_file()
        assert window.write_outputs_action.isEnabled()

    def test_retry_writes_the_files_once_the_file_is_free(self, qtbot, window, dialogs, work_dir, monkeypatch):
        self._block_the_results_file(work_dir)
        blocked = work_dir / "output" / "quick_test" / "quick_test.csv"
        warnings_shown = []

        def free_the_file_and_retry(parent, title, text, *args, **kwargs):
            warnings_shown.append(title)
            blocked.rmdir()
            return QMessageBox.StandardButton.Retry

        monkeypatch.setattr(QMessageBox, "warning", staticmethod(free_the_file_and_retry))

        self._run_quick_test(qtbot, window, dialogs)

        assert warnings_shown == ["Output Files Not Written"]
        assert dialogs.titles() == ["Simulation Complete"]
        assert str(work_dir / "output" / "quick_test") in dialogs.texts("information")[0]
        assert blocked.is_file()
        assert window.status_label.text() == "Simulation completed successfully."
        assert not window.write_outputs_action.isEnabled()


@pytest.mark.usefixtures("slow_serial_runs")
class TestFocusDuringARun:
    """Run is disabled while a simulation runs, so the focus moves to Stop and returns afterwards."""

    def test_focus_goes_to_stop_and_back_to_run(self, qtbot, window, dialogs):
        window.run_button.setFocus()
        _start_long_serial_run(qtbot, window)

        assert window.focusWidget() is window.stop_button

        _stop_and_wait(qtbot, window)

        assert window.run_button.isEnabled()
        assert window.focusWidget() is window.run_button
