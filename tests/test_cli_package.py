"""
Tests for the dice.cli package surface: running as a module, exported names,
and the console entry point.

Each check runs in a fresh interpreter, because the behavior under test
depends on what has already been imported.
"""

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import dice

RUNPY_WARNING = "found in sys.modules after import of package"


def run_python(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, '-W', 'default', *args],
        capture_output=True, text=True, timeout=120,
    )


def run_snippet(code: str) -> subprocess.CompletedProcess:
    return run_python('-c', textwrap.dedent(code))


class TestRunAsModule:
    """`python -m dice.cli.main` runs the CLI without the runpy warning."""

    def test_no_runpy_warning(self):
        completed = run_python('-m', 'dice.cli.main', '--version')
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == f"dice {dice.__version__}"
        assert RUNPY_WARNING not in completed.stderr
        assert "RuntimeWarning" not in completed.stderr

    def test_importing_package_leaves_main_unimported(self):
        completed = run_snippet("""
            import sys
            import dice.cli
            assert 'dice.cli.main' not in sys.modules
        """)
        assert completed.returncode == 0, completed.stderr


class TestModuleExitCode:
    """A failed run exits with a non-zero code under every way of starting the CLI."""

    @pytest.mark.parametrize('module', ['dice', 'dice.cli.main'])
    def test_failed_run_exits_with_one(self, tmp_path, module):
        parameters_file = tmp_path / 'missing_width.txt'
        parameters_file.write_text(
            "{'number of runs': 2, 'pixel width': 50, 'mean_0': 0, 'amplitude_0': 1}", encoding='utf-8')

        completed = run_python('-m', module, str(parameters_file), '--output-dir', str(tmp_path / 'out'))

        assert completed.returncode == 1, completed.stdout + completed.stderr
        assert "spatial width" in completed.stdout + completed.stderr

    @pytest.mark.parametrize('module', ['dice', 'dice.cli.main'])
    def test_version_exits_with_zero(self, module):
        completed = run_python('-m', module, '--version')
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == f"dice {dice.__version__}"


class TestPackageExports:
    """Every name in dice.cli.__all__ resolves, and `main` is always the entry function."""

    def test_all_exports_resolve(self):
        completed = run_snippet("""
            import inspect
            import dice.cli
            for name in dice.cli.__all__:
                assert inspect.isfunction(getattr(dice.cli, name)), name
            from dice.cli import main
            assert main.__module__ == 'dice.cli.main'
        """)
        assert completed.returncode == 0, completed.stderr

    def test_main_is_function_when_submodule_imported_first(self):
        completed = run_snippet("""
            import inspect
            import dice.cli.main
            from dice.cli import main
            assert inspect.isfunction(main), type(main)
        """)
        assert completed.returncode == 0, completed.stderr

    def test_main_is_stable_when_function_imported_first(self):
        completed = run_snippet("""
            import importlib
            import inspect
            from dice.cli import main
            submodule = importlib.import_module('dice.cli.main')
            from dice.cli import main as again
            assert inspect.isfunction(again) and again is main is submodule.main
        """)
        assert completed.returncode == 0, completed.stderr

    def test_unknown_name_raises_attribute_error(self):
        import dice.cli
        with pytest.raises(AttributeError, match="no attribute 'not_a_name'"):
            dice.cli.not_a_name


class TestConsoleEntryPoint:
    """The `dice` console script still targets dice.cli.main:main."""

    def test_entry_point_loads_main(self):
        completed = run_snippet("""
            from importlib.metadata import entry_points
            [entry_point] = entry_points(group='console_scripts', name='dice')
            assert entry_point.value == 'dice.cli.main:main', entry_point.value
            from dice.cli import main
            assert entry_point.load() is main
        """)
        assert completed.returncode == 0, completed.stderr

    def test_console_script_runs(self):
        script = shutil.which('dice', path=str(Path(sys.executable).parent))
        if script is None:
            pytest.skip("dice console script is not installed beside this interpreter")
        completed = subprocess.run([script, '--version'], capture_output=True, text=True, timeout=120)
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == f"dice {dice.__version__}"
