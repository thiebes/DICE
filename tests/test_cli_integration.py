"""
End-to-end tests of the DICE command-line interface, run in-process.

Each test runs dice.cli.main.main in a temporary working directory, because
the CLI writes to output/<slug>/ relative to it, and serially
(--multiprocessing 0), to avoid the start-up cost of the joblib workers.
Parameter files are the real examples in data/examples, or copies of
quick_test.txt with named values replaced.
"""

import ast
import contextlib
import importlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from dice.cli.main import main

EXAMPLES_DIR = Path(__file__).parent.parent / 'data' / 'examples'
QUICK_TEST = EXAMPLES_DIR / 'quick_test.txt'
SERIAL_QUIET = ['--quiet', '--multiprocessing', '0']
D_COLUMN = 'nominal diffusion coeff [cm²/s]'
WLS_D_COLUMN = 'weighted fit diffusion coeff [cm²/s]'
OLS_D_COLUMN = 'unweighted fit diffusion coeff [cm²/s]'


@dataclass
class CliRun:
    """Exit code and output locations of one CLI invocation."""
    exit_code: int
    output_dir: Path
    slug: str

    @property
    def csv_file(self) -> Path:
        return self.output_dir / f'{self.slug}.csv'

    @property
    def summary_file(self) -> Path:
        return self.output_dir / f'{self.slug}_summary.txt'

    @property
    def histogram_file(self) -> Path:
        return self.output_dir / f'{self.slug}_accuracy_histogram.png'

    def read_csv(self) -> pd.DataFrame:
        return pd.read_csv(self.csv_file)

    def read_summary(self) -> str:
        return self.summary_file.read_text(encoding='utf-8')


def run_cli(workdir: Path, params_file: Path, slug: str, *extra_args: str) -> CliRun:
    with contextlib.chdir(workdir):
        exit_code = main([str(params_file), *SERIAL_QUIET, *extra_args])
    return CliRun(exit_code, workdir / 'output' / slug, slug)


def write_quick_test_variant(
    directory: Path, slug: str, overrides: dict[str, Any], removed: tuple[str, ...] = ()
) -> Path:
    """Write quick_test.txt with values replaced or removed and a new filename slug."""
    params = ast.literal_eval(QUICK_TEST.read_text())
    for key in removed:
        del params[key]
    params.update(overrides)
    params['filename slug'] = slug
    params_file = directory / f'{slug}.txt'
    params_file.write_text(str(params))
    return params_file


def nanometer_picosecond_overrides() -> dict[str, Any]:
    """quick_test.txt restated in nanometers and picoseconds.

    Lengths scale by 1000 and the time values keep their numbers, so with the
    lifetime of 1 time unit that a diffusion-length file implies, the run is
    the same dimensionless problem: D = 100 nm x 100 nm / 1 ps = 1e4 nm^2/ps,
    which is 100 cm^2/s.
    """
    params = ast.literal_eval(QUICK_TEST.read_text())
    overrides: dict[str, Any] = {'length unit': 'nanometer', 'time unit': 'picosecond'}
    for key in ('FWHM_0', 'spatial width', 'nominal diffusion length'):
        overrides[key] = round(params[key] * 1000, 9)
    return overrides


class TestGlobalUnitsReachOutputs:
    """The CSV and summary convert from, and label with, the file's global units."""

    UNIT_RUNS = 10

    @pytest.fixture(scope='class')
    def micrometer_run(self, tmp_path_factory) -> CliRun:
        workdir = tmp_path_factory.mktemp('micrometer')
        params_file = write_quick_test_variant(
            workdir, 'units_um_ns', {'number of runs': self.UNIT_RUNS})
        return run_cli(workdir, params_file, 'units_um_ns', '--no-plots')

    @pytest.fixture(scope='class')
    def nanometer_run(self, tmp_path_factory) -> CliRun:
        workdir = tmp_path_factory.mktemp('nanometer')
        overrides = nanometer_picosecond_overrides() | {'number of runs': self.UNIT_RUNS}
        params_file = write_quick_test_variant(workdir, 'units_nm_ps', overrides)
        return run_cli(workdir, params_file, 'units_nm_ps', '--no-plots')

    def test_nanometer_picosecond_csv_in_cm2_per_s(self, nanometer_run):
        assert nanometer_run.exit_code == 0
        df = nanometer_run.read_csv()
        assert df[D_COLUMN].to_numpy() == pytest.approx(100.0, rel=1e-12)
        # 1 nm^2/ps = 1e-14 cm^2 / 1e-12 s = 1e-2 cm^2/s; D is half the MSD slope.
        assert df[WLS_D_COLUMN].to_numpy() == pytest.approx(
            df['weighted fit diffusion slope'].to_numpy() / 2 * 1e-2, rel=1e-12)
        assert df[OLS_D_COLUMN].to_numpy() == pytest.approx(
            df['unweighted fit diffusion slope'].to_numpy() / 2 * 1e-2, rel=1e-12)

    def test_nanometer_picosecond_summary_labels(self, nanometer_run):
        summary = nanometer_run.read_summary()
        assert "Spatial width:\n    5000 nanometer\n" in summary
        assert "Time range:\n    0.000 to 1.000 picosecond\n" in summary
        assert "Nominal diffusion length:\n    100.000 nanometer\n" in summary
        assert "Nominal diffusion coefficient:\n    10000.00000 nanometer^2/picosecond\n" in summary
        assert "Nominal lifetime:\n    1 picosecond\n" in summary
        assert "micrometer" not in summary
        assert "nanosecond" not in summary

    def test_micrometer_nanosecond_unchanged(self, micrometer_run):
        assert micrometer_run.exit_code == 0
        df = micrometer_run.read_csv()
        # D = 0.1 um x 0.1 um / 1 ns = 0.01 um^2/ns; 1 um^2/ns = 1e-8 cm^2 / 1e-9 s = 10 cm^2/s.
        assert df[D_COLUMN].to_numpy() == pytest.approx(0.1, rel=1e-12)
        assert df[WLS_D_COLUMN].to_numpy() == pytest.approx(
            df['weighted fit diffusion slope'].to_numpy() / 2 * 10, rel=1e-12)
        summary = micrometer_run.read_summary()
        assert "Spatial width:\n    5 micrometer\n" in summary
        assert "Nominal diffusion coefficient:\n    0.01000 micrometer^2/nanosecond\n" in summary

    def test_unit_systems_describe_the_same_problem(self, micrometer_run, nanometer_run):
        micrometer_df, nanometer_df = micrometer_run.read_csv(), nanometer_run.read_csv()
        assert list(micrometer_df.columns) == list(nanometer_df.columns)
        assert nanometer_df['nominal diffusion coeff'].to_numpy() == pytest.approx(
            micrometer_df['nominal diffusion coeff'].to_numpy() * 1e6, rel=1e-12)

    def test_unknown_unit_rejected_before_simulation(self, tmp_path, capsys):
        params_file = write_quick_test_variant(tmp_path, 'bad_unit', {'length unit': 'um'})
        run = run_cli(tmp_path, params_file, 'bad_unit')
        assert run.exit_code == 1
        assert "Unknown length unit 'um'" in capsys.readouterr().out
        assert not (tmp_path / 'output').exists()


class TestSeedReproducibility:
    """--seed N reproduces the CSV exactly, serially or in parallel; no seed does not.

    Values are compared between runs, never against stored numbers, which
    depend on the numpy, scipy, and fft-cnr versions and on the platform.
    """

    SEED_RUNS = 6
    NOISE_RANGE = {'noise range, reciprocal': [0.01, 0.05]}

    def _csv(self, workdir: Path, overrides: dict[str, Any], *cli_args: str) -> pd.DataFrame:
        removed = ('noise value',) if 'noise range, reciprocal' in overrides else ()
        params_file = write_quick_test_variant(
            workdir, 'seeded', {'number of runs': self.SEED_RUNS} | overrides, removed)
        run = run_cli(workdir, params_file, 'seeded', '--no-plots', *cli_args)
        assert run.exit_code == 0
        return run.read_csv()

    def test_same_seed_identical(self, tmp_path_factory):
        first = self._csv(tmp_path_factory.mktemp('a'), {}, '--seed', '3')
        second = self._csv(tmp_path_factory.mktemp('b'), {}, '--seed', '3')
        pd.testing.assert_frame_equal(first, second, check_exact=True)

    def test_different_seeds_differ(self, tmp_path_factory):
        first = self._csv(tmp_path_factory.mktemp('a'), {}, '--seed', '3')
        second = self._csv(tmp_path_factory.mktemp('b'), {}, '--seed', '4')
        assert not first[WLS_D_COLUMN].equals(second[WLS_D_COLUMN])

    def test_unseeded_runs_differ(self, tmp_path_factory):
        first = self._csv(tmp_path_factory.mktemp('a'), {})
        second = self._csv(tmp_path_factory.mktemp('b'), {})
        assert not first[WLS_D_COLUMN].equals(second[WLS_D_COLUMN])

    def test_serial_equals_parallel(self, tmp_path_factory):
        # The parallel run starts joblib workers, which costs tens of seconds.
        serial = self._csv(tmp_path_factory.mktemp('serial'), {}, '--seed', '3')
        parallel = self._csv(tmp_path_factory.mktemp('parallel'), {}, '--seed', '3',
                             '--multiprocessing', '-1')
        pd.testing.assert_frame_equal(serial, parallel, check_exact=True)

    def test_noise_range_draw_reproducible(self, tmp_path_factory):
        first = self._csv(tmp_path_factory.mktemp('a'), self.NOISE_RANGE, '--seed', '3')
        second = self._csv(tmp_path_factory.mktemp('b'), self.NOISE_RANGE, '--seed', '3')
        other = self._csv(tmp_path_factory.mktemp('c'), self.NOISE_RANGE, '--seed', '4')
        pd.testing.assert_frame_equal(first, second, check_exact=True)
        assert first['nominal CNR'].iloc[0] != other['nominal CNR'].iloc[0]


class TestMultiprocessingFlag:
    """--multiprocessing N reaches the simulation as its worker count."""

    @pytest.mark.parametrize('cli_args, expected', [
        ((), {'multiprocessing': True, 'n_jobs': -1}),
        (('--multiprocessing', '0'), {'multiprocessing': False}),
        (('--multiprocessing', '2'), {'multiprocessing': True, 'n_jobs': 2}),
        (('--multiprocessing', '-1'), {'multiprocessing': True, 'n_jobs': -1}),
    ])
    def test_flag_sets_worker_count(self, tmp_path, monkeypatch, cli_args, expected):
        # The package attribute dice.cli.main is the entry function, not the module.
        cli_main = importlib.import_module('dice.cli.main')
        simulate = cli_main.run_monte_carlo_simulation
        recorded_calls = []

        def record_then_run_serially(**kwargs):
            recorded_calls.append(kwargs)
            return simulate(**{**kwargs, 'multiprocessing': False})

        monkeypatch.setattr(cli_main, 'run_monte_carlo_simulation', record_then_run_serially)
        params_file = write_quick_test_variant(tmp_path, 'workers', {'number of runs': 2})
        with contextlib.chdir(tmp_path):
            assert main([str(params_file), '--quiet', '--no-plots', *cli_args]) == 0
        [call] = recorded_calls
        assert {key: call[key] for key in expected} == expected


class TestRetiredProximityLevel:
    """A file that still sets 'proximity level' warns once and runs exactly as without it."""

    def test_warns_and_has_no_effect(self, tmp_path_factory):
        seeded = ('--seed', '5', '--no-plots')
        plain_dir, retired_dir = tmp_path_factory.mktemp('plain'), tmp_path_factory.mktemp('retired')
        plain_file = write_quick_test_variant(plain_dir, 'proximity', {'number of runs': 6})
        retired_file = write_quick_test_variant(
            retired_dir, 'proximity', {'number of runs': 6, 'proximity level': 0.1})

        plain = run_cli(plain_dir, plain_file, 'proximity', *seeded)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            retired = run_cli(retired_dir, retired_file, 'proximity', *seeded)

        assert plain.exit_code == retired.exit_code == 0
        proximity_warnings = [
            w for w in caught if "'proximity level' no longer affects" in str(w.message)]
        assert [w.category for w in proximity_warnings] == [UserWarning]
        assert plain.csv_file.read_bytes() == retired.csv_file.read_bytes()
        assert plain.read_summary() == retired.read_summary()


class TestDryRun:
    """--dry-run loads and validates exactly as a real run does, then stops before any output."""

    INVALID_VALUES = [
        ({'noise value': 'loud'}, "Invalid noise value loud"),
        ({'number of runs': 0}, "Number of runs must be positive"),
        ({'length unit': 'um'}, "Unknown length unit 'um'"),
    ]

    def test_valid_example_exits_zero_without_output(self, tmp_path, capsys):
        with contextlib.chdir(tmp_path):
            assert main([str(QUICK_TEST), '--dry-run']) == 0
        out = capsys.readouterr().out
        assert "loaded and validated" in out
        out.encode('ascii')
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize('overrides, message', INVALID_VALUES)
    def test_invalid_value_fails_like_a_real_run(self, tmp_path, capsys, overrides, message):
        params_file = write_quick_test_variant(tmp_path, 'invalid', overrides)
        with contextlib.chdir(tmp_path):
            dry_exit = main([str(params_file), '--dry-run'])
            dry_error = capsys.readouterr().out.splitlines()[-1]
            real_exit = main([str(params_file), '--quiet'])
            real_error = capsys.readouterr().out.splitlines()[-1]
        assert dry_exit == real_exit == 1
        assert dry_error == real_error
        assert dry_error.startswith("Error: ") and message in dry_error
        assert not (tmp_path / 'output').exists()

    def test_missing_file_fails_like_a_real_run(self, tmp_path, capsys):
        missing = str(tmp_path / 'missing.txt')
        with pytest.raises(SystemExit) as dry_exit:
            main([missing, '--dry-run'])
        dry_error = capsys.readouterr().err
        with pytest.raises(SystemExit) as real_exit:
            main([missing])
        assert dry_exit.value.code == real_exit.value.code == 2
        assert "Parameters file not found" in dry_error
        assert dry_error == capsys.readouterr().err

    def test_output_is_ascii_when_redirected(self, tmp_path):
        import os
        import subprocess
        import sys
        completed = subprocess.run(
            [sys.executable, '-m', 'dice.cli.main', str(QUICK_TEST), '--dry-run'],
            cwd=tmp_path, capture_output=True, timeout=120,
            env={**os.environ, 'PYTHONIOENCODING': 'ascii:strict'},
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        completed.stdout.decode('ascii')
        assert list(tmp_path.iterdir()) == []


class TestQuietMode:
    """--quiet prints nothing on a successful run, including with --seed."""

    def test_quiet_seeded_run_prints_nothing(self, tmp_path, capsys):
        params_file = write_quick_test_variant(tmp_path, 'quiet', {'number of runs': 2})
        run = run_cli(tmp_path, params_file, 'quiet', '--seed', '1', '--no-plots')
        assert run.exit_code == 0
        assert capsys.readouterr().out == ''
