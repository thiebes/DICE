"""
Tests for dice_gui/standalone.py, the entry script of the standalone build,
and its --self-test option.
"""

import subprocess
import sys
from pathlib import Path

import joblib
import pytest

import dice.analysis.simulation as simulation
from dice_gui import standalone

REPO_ROOT = Path(__file__).resolve().parent.parent
ENTRY_SCRIPT = REPO_ROOT / 'dice_gui' / 'standalone.py'


def serial_pool(run_parameters, worker_count, progress_callback=None, drop_seeds=False):
    """Stands in for run_in_process_pool: the same runs in-process, optionally unseeded."""
    results = []
    for arguments in run_parameters:
        if drop_seeds:
            arguments = arguments[:-1] + (None,)
        results.append(simulation.run_single_simulation(*arguments))
    if progress_callback:
        progress_callback(len(run_parameters), len(run_parameters))
    return results


@pytest.fixture
def ten_runs(monkeypatch):
    """The self-test's example with 10 runs instead of 100, for the in-process tests."""
    import dice.io.parameters

    open_parameters = dice.io.parameters.open_parameters

    def open_with_ten_runs(filename):
        return {**open_parameters(filename), 'number of runs': 10}

    monkeypatch.setattr(dice.io.parameters, 'open_parameters', open_with_ten_runs)


class TestSelfTestRun:

    def test_self_test_passes_in_a_fresh_process_without_qt(self):
        # The run the release workflow makes, from source: real worker
        # processes, which costs a few seconds. -X importtime reaches the
        # workers too, so no PyQt6 line means no process imported Qt.
        code = (
            "import runpy, sys; sys.argv = ['DICE', '--self-test']; "
            f"runpy.run_path({str(ENTRY_SCRIPT)!r}, run_name='__main__')"
        )
        completed = subprocess.run(
            [sys.executable, '-X', 'importtime', '-c', code],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
        )

        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert 'Results agree to within rounding (relative 1e-12): yes' in completed.stdout
        assert 'Worker processes still running: 0' in completed.stdout
        assert completed.stdout.rstrip().endswith('Self-test passed')
        assert 'PyQt6' not in completed.stderr

    def test_differing_parallel_results_fail(self, monkeypatch, capsys, ten_runs):
        monkeypatch.setattr(
            simulation, 'run_in_process_pool',
            lambda *args, **kwargs: serial_pool(*args, **kwargs, drop_seeds=True),
        )

        assert standalone.run_self_test(sys.stdout) is False
        report = capsys.readouterr().out
        assert 'Results agree to within rounding (relative 1e-12): no' in report
        assert report.rstrip().endswith('Self-test failed')

    def test_identical_results_pass(self, monkeypatch, capsys, ten_runs):
        monkeypatch.setattr(simulation, 'run_in_process_pool', serial_pool)

        assert standalone.run_self_test(sys.stdout) is True
        report = capsys.readouterr().out
        assert 'Results agree to within rounding (relative 1e-12): yes' in report
        assert 'Progress reached 10 of 10 runs: yes' in report

    @pytest.mark.parametrize("workers_of_joblib, line", [
        (4, "Process pool of 4 worker processes: 10 runs in "),
        (64, "Process pool of 10 worker processes: 10 runs in "),
        (1, "Process pool of 1 worker process: 10 runs in "),
    ])
    def test_report_counts_the_worker_processes_of_the_pool(
            self, monkeypatch, capsys, ten_runs, workers_of_joblib, line):
        """No more workers than runs, and no more than joblib uses for all processors."""
        monkeypatch.setattr(simulation, 'run_in_process_pool', serial_pool)
        monkeypatch.setattr(joblib, 'effective_n_jobs', lambda n_jobs: workers_of_joblib)

        assert standalone.run_self_test(sys.stdout) is True

        assert line in capsys.readouterr().out

    def test_standalone_build_that_would_use_joblib_fails(self, monkeypatch, capsys, ten_runs):
        # The GUI would then start loky workers, which a frozen build cannot.
        import dice.utils.runtime as runtime

        examples = runtime.examples_directory()
        monkeypatch.setattr(simulation, 'run_in_process_pool', serial_pool)
        monkeypatch.setattr(runtime, 'examples_directory', lambda: examples)
        monkeypatch.setattr(runtime, 'is_standalone_build', lambda: True)
        monkeypatch.setattr(simulation, 'default_parallel_backend', lambda: simulation.JOBLIB_BACKEND)

        assert standalone.run_self_test(sys.stdout) is False
        assert 'use: joblib' in capsys.readouterr().out

    @pytest.mark.parametrize('bundles_notice', [True, False])
    def test_standalone_build_must_carry_its_license_notice(self, monkeypatch, capsys, tmp_path, ten_runs,
                                                             bundles_notice):
        import dice.utils.runtime as runtime

        examples = runtime.examples_directory()
        if bundles_notice:
            (tmp_path / runtime.LICENSE_NOTICE_FILE_NAME).write_text('notice', encoding='utf-8')
        monkeypatch.setattr(simulation, 'run_in_process_pool', serial_pool)
        monkeypatch.setattr(runtime, 'examples_directory', lambda: examples)
        monkeypatch.setattr(runtime, 'is_standalone_build', lambda: True)
        monkeypatch.setattr(sys, '_MEIPASS', str(tmp_path), raising=False)
        monkeypatch.setattr(simulation, 'default_parallel_backend', lambda: simulation.PROCESS_POOL_BACKEND)

        assert standalone.run_self_test(sys.stdout) is bundles_notice
        assert f"License notice bundled: {'yes' if bundles_notice else 'no'}" in capsys.readouterr().out

    def test_missing_example_fails(self, monkeypatch, tmp_path):
        import dice.utils.runtime as runtime

        monkeypatch.setattr(runtime, 'examples_directory', lambda: tmp_path)

        with pytest.raises(FileNotFoundError, match='quick_test.txt was not found'):
            standalone.run_self_test(None)


class TestEntryPoint:

    @pytest.fixture
    def calls(self, monkeypatch) -> list:
        """Record freeze_support and the GUI start instead of running them."""
        import multiprocessing
        import dice_gui.dice_gui

        calls = []
        monkeypatch.setattr(multiprocessing, 'freeze_support', lambda: calls.append('freeze_support'))
        monkeypatch.setattr(dice_gui.dice_gui, 'main', lambda: calls.append('gui'))
        return calls

    def test_without_the_option_the_gui_starts(self, monkeypatch, calls):
        monkeypatch.setattr(sys, 'argv', ['DICE'])
        standalone.main()
        assert calls == ['freeze_support', 'gui']

    def test_a_worker_process_ends_in_freeze_support(self, monkeypatch, calls):
        import multiprocessing

        def run_worker():
            calls.append('freeze_support')
            raise SystemExit(0)

        monkeypatch.setattr(multiprocessing, 'freeze_support', run_worker)
        monkeypatch.setattr(sys, 'argv', ['DICE', '--multiprocessing-fork', 'parent_pid=1', 'pipe_handle=2'])

        with pytest.raises(SystemExit):
            standalone.main()
        assert calls == ['freeze_support']

    @pytest.mark.parametrize(('passed', 'status'), [(True, 0), (False, 1)])
    def test_self_test_exit_status(self, monkeypatch, calls, passed, status):
        monkeypatch.setattr(sys, 'argv', ['DICE', '--self-test'])
        monkeypatch.setattr(standalone, 'run_self_test', lambda output: passed)

        with pytest.raises(SystemExit) as stop:
            standalone.main()

        assert stop.value.code == status
        assert calls == ['freeze_support']

    def test_self_test_error_is_printed_not_raised(self, monkeypatch, capsys, calls):
        def broken(output):
            raise RuntimeError("worker could not start")

        monkeypatch.setattr(sys, 'argv', ['DICE', '--self-test'])
        monkeypatch.setattr(standalone, 'run_self_test', broken)

        with pytest.raises(SystemExit) as stop:
            standalone.main()

        assert stop.value.code == 1
        assert 'RuntimeError: worker could not start' in capsys.readouterr().err
