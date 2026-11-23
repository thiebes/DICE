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
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from dice.io.results import results_column

import dice
from dice.cli.main import main
from seeded_results import assert_seeded_frames_agree

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
            df[results_column(df.columns, 'weighted fit diffusion slope')].to_numpy() / 2 * 1e-2, rel=1e-12)
        assert df[OLS_D_COLUMN].to_numpy() == pytest.approx(
            df[results_column(df.columns, 'unweighted fit diffusion slope')].to_numpy() / 2 * 1e-2, rel=1e-12)

    def test_nanometer_picosecond_summary_labels(self, nanometer_run):
        summary = nanometer_run.read_summary()
        assert "Spatial width:\n    5000 nanometer\n" in summary
        assert "Time range:\n    0.000 to 1.000 picosecond\n" in summary
        assert "Nominal diffusion length:\n    100.000 nanometer\n" in summary
        assert "Nominal diffusion coefficient:\n    10000.00000 nanometer^2/picosecond\n" in summary
        assert ("Nominal lifetime:\n    1 picosecond (assumed: the file gives only the diffusion length, "
                "so its times are in units of the lifetime)\n") in summary
        assert "micrometer" not in summary
        assert "nanosecond" not in summary

    def test_micrometer_nanosecond_unchanged(self, micrometer_run):
        assert micrometer_run.exit_code == 0
        df = micrometer_run.read_csv()
        # D = 0.1 μm x 0.1 μm / 1 ns = 0.01 μm^2/ns; 1 μm^2/ns = 1e-8 cm^2 / 1e-9 s = 10 cm^2/s.
        assert df[D_COLUMN].to_numpy() == pytest.approx(0.1, rel=1e-12)
        assert df[WLS_D_COLUMN].to_numpy() == pytest.approx(
            df[results_column(df.columns, 'weighted fit diffusion slope')].to_numpy() / 2 * 10, rel=1e-12)
        summary = micrometer_run.read_summary()
        assert "Spatial width:\n    5 micrometer\n" in summary
        assert "Nominal diffusion coefficient:\n    0.01000 micrometer^2/nanosecond\n" in summary

    def test_unit_systems_describe_the_same_problem(self, micrometer_run, nanometer_run):
        micrometer_df, nanometer_df = micrometer_run.read_csv(), nanometer_run.read_csv()
        def without_simulation_units(frame):
            return [column.replace('[μm²/ns]', '[]').replace('[nm²/ps]', '[]').replace('[μm²]', '[]')
                    .replace('[nm²]', '[]').replace('[μm]', '[]').replace('[nm]', '[]')
                    .replace('[ns]', '[]').replace('[ps]', '[]') for column in frame.columns]

        assert without_simulation_units(micrometer_df) == without_simulation_units(nanometer_df)
        assert 'nominal lifetime [ns]' in micrometer_df.columns and 'nominal lifetime [ps]' in nanometer_df.columns
        assert nanometer_df[results_column(nanometer_df.columns, 'nominal diffusion coeff')].to_numpy() == pytest.approx(
            micrometer_df[results_column(micrometer_df.columns, 'nominal diffusion coeff')].to_numpy() * 1e6, rel=1e-12)

    def test_unknown_unit_rejected_before_simulation(self, tmp_path, capsys):
        params_file = write_quick_test_variant(tmp_path, 'bad_unit', {'length unit': 'um'})
        run = run_cli(tmp_path, params_file, 'bad_unit')
        assert run.exit_code == 1
        assert "Unknown length unit 'um'" in capsys.readouterr().out
        assert not (tmp_path / 'output').exists()


class TestOutputDiffusionUnits:
    """'output_diffusion_length_unit' and 'output_diffusion_time_unit' set the D unit of the CSV and summary."""

    @pytest.fixture(scope='class')
    def nm2_per_ps_run(self, tmp_path_factory) -> CliRun:
        workdir = tmp_path_factory.mktemp('output_units')
        params_file = write_quick_test_variant(workdir, 'out_nm_ps', {
            'number of runs': 6, 'output_diffusion_length_unit': 'nanometer',
            'output_diffusion_time_unit': 'picosecond'})
        return run_cli(workdir, params_file, 'out_nm_ps', '--no-plots', '--seed', '2')

    def test_csv_columns_in_the_output_unit(self, nm2_per_ps_run):
        assert nm2_per_ps_run.exit_code == 0
        df = nm2_per_ps_run.read_csv()
        assert list(df.columns) == [c.replace('cm²/s', 'nm²/ps') for c in CSV_COLUMNS]
        # 0.01 micrometer^2/ns = 0.01 x 1e6 nm^2 / 1e3 ps = 10 nm^2/ps
        assert df['nominal diffusion coeff [nm²/ps]'].to_numpy() == pytest.approx(10.0, rel=1e-12)
        assert df[results_column(df.columns, 'nominal diffusion coeff')].to_numpy() == pytest.approx(0.01)

    def test_summary_diffusion_coefficients_agree_with_the_csv(self, nm2_per_ps_run):
        df = nm2_per_ps_run.read_csv()
        values = summary_label_to_values(nm2_per_ps_run.read_summary())
        unit = 'nanometer^2/picosecond'
        assert values['Nominal diffusion coefficient'] == [f"10.00000 {unit}"]
        estimates = {'wls': df['weighted fit diffusion coeff [nm²/ps]'].to_numpy(),
                     'ols': df['unweighted fit diffusion coeff [nm²/ps]'].to_numpy()}
        assert values['Mean estimate'] == [f"{estimates[m].mean():.5f} {unit}" for m in ('wls', 'ols')]
        assert values['Standard deviation'] == [f"{estimates[m].std(ddof=1):.5f} {unit}" for m in ('wls', 'ols')]
        # Lengths, times, and the unitless ratios stay as they were.
        assert values['Nominal diffusion length'] == ['0.100 micrometer']
        assert values['Relative standard deviation'] == [
            f"{estimates[m].std(ddof=1) / 10.0:.3f}" for m in ('wls', 'ols')]

    @pytest.mark.parametrize('overrides, message', [
        ({'output_time_unit': 'picosecond'}, "'output_time_unit' is not supported"),
        ({'output_diffusion_time_unit': 'second'},
         "Give both 'output_diffusion_length_unit' and 'output_diffusion_time_unit', or neither."),
        ({'output_diffusion_length_unit': 'centimeter', 'output_diffusion_time_unit': 'sec'},
         "Unknown time unit 'sec' for 'output_diffusion_time_unit'"),
    ])
    def test_rejected_before_simulation(self, tmp_path, capsys, overrides, message):
        params_file = write_quick_test_variant(tmp_path, 'bad_output_unit', overrides)
        run = run_cli(tmp_path, params_file, 'bad_output_unit')
        assert run.exit_code == 1
        assert message in capsys.readouterr().out
        assert not (tmp_path / 'output').exists()


class TestSeedReproducibility:
    """--seed N reproduces the CSV to within floating-point rounding, serially or in parallel; no seed does not.

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
        assert_seeded_frames_agree(first, second)

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
        assert_seeded_frames_agree(serial, parallel)

    def test_noise_range_draw_reproducible(self, tmp_path_factory):
        first = self._csv(tmp_path_factory.mktemp('a'), self.NOISE_RANGE, '--seed', '3')
        second = self._csv(tmp_path_factory.mktemp('b'), self.NOISE_RANGE, '--seed', '3')
        other = self._csv(tmp_path_factory.mktemp('c'), self.NOISE_RANGE, '--seed', '4')
        assert_seeded_frames_agree(first, second)
        assert first['nominal CNR'].iloc[0] != other['nominal CNR'].iloc[0]
        # Each run draws its own noise level.
        assert first['noise standard deviation'].nunique() == self.SEED_RUNS

    def test_noise_range_identical_serially_and_in_both_worker_pools(self, tmp_path_factory, monkeypatch):
        # Starts joblib workers and a process pool of two workers each, which costs some seconds.
        import dice.analysis.simulation as simulation
        serial = self._csv(tmp_path_factory.mktemp('serial'), self.NOISE_RANGE, '--seed', '3')
        joblib_pool = self._csv(tmp_path_factory.mktemp('joblib'), self.NOISE_RANGE, '--seed', '3',
                                '--multiprocessing', '2')
        # A standalone build's default: the standard library's process pool.
        pool_calls = []
        run_in_process_pool = simulation.run_in_process_pool
        monkeypatch.setattr(simulation, 'joblib_workers_supported', lambda: False)
        monkeypatch.setattr(simulation, 'run_in_process_pool',
                            lambda *args: pool_calls.append(args[1]) or run_in_process_pool(*args))
        process_pool = self._csv(tmp_path_factory.mktemp('pool'), self.NOISE_RANGE, '--seed', '3',
                                 '--multiprocessing', '2')
        assert pool_calls == [2]
        assert_seeded_frames_agree(serial, joblib_pool)
        assert_seeded_frames_agree(serial, process_pool)


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
        assert_seeded_frames_agree(retired.read_csv(), plain.read_csv())
        assert plain.read_summary() == retired.read_summary()


class TestDryRun:
    """--dry-run loads and validates exactly as a real run does, then stops before any output."""

    INVALID_VALUES = [
        ({'noise value': 'loud'}, "Invalid noise value loud"),
        ({'number of runs': 0}, "'number of runs' must be >= 1, got 0"),
        # These two used to pass the checks, run the whole simulation, and fail at the plot.
        ({'nominal diffusion length': 0}, "The nominal diffusion coefficient must be greater than 0"),
        ({'image numbins': 0}, "'image numbins' must be >= 1, got 0"),
        ({'image numbins': 35.5}, "'image numbins' must be an integer"),
        ({'number of runs': 3.5}, "'number of runs' must be an integer"),
        ({'time range': [0, 1, 1]}, "The number of steps in 'time range' must be >= 2"),
        ({'noise value': -0.02}, "'noise value' must be 0 or greater"),
        ({'FWHM_0': 0}, "'FWHM_0' must be greater than 0"),
        ({'pixel width': 1}, "'pixel width' (the number of pixels) is 1"),
        ({'mean_0': 1.3}, "outside the central 40% of the spatial window"),
        ({'length unit': 'um'}, "Unknown length unit 'um'"),
        ({'image type': 'bmp'}, "Unknown image type 'bmp'"),
        ({'output_length_unit': 'nanometer'}, "'output_length_unit' is not supported"),
        ({'output_diffusion_length_unit': 'furlong', 'output_diffusion_time_unit': 'second'},
         "Unknown length unit 'furlong' for 'output_diffusion_length_unit'"),
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

    @pytest.mark.parametrize('series, message', [
        ([], "'time series'"),
        ([0.5], "'time series' holds 1 time point"),
        ([0, 0.5, 0.5], "Time points must be unique"),
    ])
    def test_invalid_time_series_fails_like_a_real_run(self, tmp_path, capsys, series, message):
        params_file = write_quick_test_variant(
            tmp_path, 'invalid_series', {'time series': series}, removed=('time range',))
        with contextlib.chdir(tmp_path):
            dry_exit = main([str(params_file), '--dry-run'])
            dry_error = capsys.readouterr().out.splitlines()[-1]
            real_exit = main([str(params_file), '--quiet'])
            real_error = capsys.readouterr().out.splitlines()[-1]
        assert dry_exit == real_exit == 1
        assert dry_error == real_error
        assert dry_error.startswith("Error: ") and message in dry_error
        assert not (tmp_path / 'output').exists()

    def test_unsorted_time_series_runs_in_ascending_order(self, tmp_path):
        params_file = write_quick_test_variant(
            tmp_path, 'unsorted', {'time series': [1, 0, 0.5], 'number of runs': 3, 'multiprocessing': False},
            removed=('time range',))
        run = run_cli(tmp_path, params_file, 'unsorted')
        assert run.exit_code == 0
        assert "Time range:\n    0.000 to 1.000 nanosecond" in run.read_summary()

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


class TestZeroNoiseRun:
    """'noise value': 0 runs to completion and writes a complete summary."""

    def test_noise_free_run_completes(self, tmp_path):
        params_file = write_quick_test_variant(tmp_path, 'noise_free', {'noise value': 0, 'number of runs': 3})
        run = run_cli(tmp_path, params_file, 'noise_free', '--no-plots')
        assert run.exit_code == 0
        summary = run.read_summary()
        assert "Initial CNR (Contrast-to-Noise Ratio):\n    inf\n" in summary
        assert summary.endswith("Simulation completed successfully.\n")
        assert (run.read_csv()['nominal CNR'] == float('inf')).all()


class TestZeroLifetimeRun:
    """'nominal lifetime (tau)': 0 means no decay: every output reports an infinite diffusion length."""

    def test_infinite_diffusion_length_in_every_output(self, tmp_path):
        params_file = write_quick_test_variant(
            tmp_path, 'no_decay',
            {'nominal diffusion coefficient': 0.01, 'nominal lifetime (tau)': 0, 'number of runs': 4},
            removed=('nominal diffusion length',))
        run = run_cli(tmp_path, params_file, 'no_decay', '--export-format', 'both')
        assert run.exit_code == 0

        values = summary_label_to_values(run.read_summary())
        assert values['Nominal diffusion length'] == ['inf micrometer']
        assert values['Nominal lifetime'] == ['0 nanosecond']  # given in the file, so not marked as assumed
        df = run.read_csv()
        assert (df[results_column(df.columns, 'nominal diffusion length')] == np.inf).all()
        assert (df[results_column(df.columns, 'nominal lifetime')] == 0).all()
        assert df[results_column(df.columns, 'nominal diffusion coeff')].to_numpy() == pytest.approx(0.01)
        exported = json.loads((run.output_dir / 'no_decay.json').read_text())
        assert exported['parameters']['nominal diffusion length'] == np.inf
        assert exported['analysis']['summary']['nominal_diffusion_length'] == np.inf
        # Nothing downstream divides by the diffusion length: the histogram is drawn.
        assert run.histogram_file.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'


CSV_COLUMNS = [
    'run number',
    'nominal diffusion coeff [μm²/ns]',
    'nominal lifetime [ns]',
    'nominal diffusion length [μm]',
    'nominal CNR',
    'estimated CNR',
    'nominal sigma^2_0 [μm²]',
    'estimated sigma^2_0 [μm²]',
    'unweighted fit diffusion slope (2D) [μm²/ns]',
    'unweighted fit diffusion slope stderr [μm²/ns]',
    'unweighted fit intercept [μm²]',
    'unweighted fit intercept stderr [μm²]',
    'unweighted fit diffusion coeff [cm²/s]',
    'unweighted fit diffusion stderr [cm²/s]',
    'weighted fit diffusion slope (2D) [μm²/ns]',
    'weighted fit diffusion slope stderr [μm²/ns]',
    'weighted fit intercept [μm²]',
    'weighted fit intercept stderr [μm²]',
    'weighted fit diffusion coeff [cm²/s]',
    'weighted fit diffusion stderr [cm²/s]',
    'nominal diffusion coeff [cm²/s]',
]

# The summary for quick_test.txt with every number on a value line replaced by '#'.
FRACTION_LINE = "    WLS: #% (# of #; #% to #%)    OLS: #% (# of #; #% to #%)"

SUMMARY_SKELETON = [
    "DICE Simulation Summary",
    "=" * 50,
    "",
    "DICE version:", "    <version>",
    "Seed:", "    #",
    "",
    "Simulation Parameters:",
    "-" * 30,
    "Number of runs:", "    #",
    "Spatial width:", "    # micrometer",
    "Pixel width:", "    # pixels",
    "Time frames:", "    # frames",
    "Time range:", "    # to # nanosecond",
    "Noise standard deviation values:", "    [#]",
    "Initial CNR (Contrast-to-Noise Ratio):", "    #",
    "Initial profile squared width (sigma^2):", "    # micrometer^2",
    "Nominal diffusion length:", "    # micrometer",
    "Nominal diffusion coefficient:", "    # micrometer^2/nanosecond",
    "Nominal lifetime:",
    "    # nanosecond (assumed: the file gives only the diffusion length, so its times are in units of the lifetime)",
    "",
    "Analysis Results:",
    "-" * 30,
    "Diffusion Coefficient Estimate Statistics:",
    "  Weighted Least Squares:",
    "    Estimates:", "        # of # runs (# without an estimate)",
    "    Standard deviation:", "        # micrometer^2/nanosecond",
    "    Relative standard deviation:", "        #",
    "    Mean estimate:", "        # micrometer^2/nanosecond",
    "    Standard error of the mean and of the bias:", "        # micrometer^2/nanosecond",
    "    Bias (mean - nominal):", "        # micrometer^2/nanosecond",
    "    Mean estimate / nominal:", "        #",
    "    Root Mean Square Error:", "        # micrometer^2/nanosecond",
    "  Ordinary (unweighted) Least Squares, for comparison:",
    "    Estimates:", "        # of # runs (# without an estimate)",
    "    Standard deviation:", "        # micrometer^2/nanosecond",
    "    Relative standard deviation:", "        #",
    "    Mean estimate:", "        # micrometer^2/nanosecond",
    "    Standard error of the mean and of the bias:", "        # micrometer^2/nanosecond",
    "    Bias (mean - nominal):", "        # micrometer^2/nanosecond",
    "    Mean estimate / nominal:", "        #",
    "    Root Mean Square Error:", "        # micrometer^2/nanosecond",
    "",
    "Precision at standard thresholds:",
    "  (each fraction with its count of estimates and its 95% interval)",
    "  Fraction within ±5% of nominal:", FRACTION_LINE,
    "  Fraction within ±10% of nominal:", FRACTION_LINE,
    "  Fraction within ±20% of nominal:", FRACTION_LINE,
    "  Fraction within ±50% of nominal:", FRACTION_LINE,
    "",
    "Simulation completed successfully.",
]

SUMMARY_NUMBER = re.compile(r"(?<![\^\w.])-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|\binf\b|\bnan\b")


def summary_skeleton(summary: str) -> list[str]:
    """Summary lines with the numbers on value lines (indented lines after a label) masked."""
    skeleton, previous = [], ""
    for line in summary.splitlines():
        is_value_line = previous.endswith(":") and line.startswith("    ")
        if previous == "DICE version:":
            line = line.replace(dice.__version__, "<version>")
        elif is_value_line:
            line = SUMMARY_NUMBER.sub("#", line)
        skeleton.append(line)
        previous = line
    return skeleton


def summary_label_to_values(summary: str) -> dict[str, list[str]]:
    """Each summary label mapped to the value lines under it, in file order."""
    label_to_values: dict[str, list[str]] = {}
    previous = ""
    for line in summary.splitlines():
        if previous.endswith(":") and line.startswith("    "):
            label_to_values.setdefault(previous.strip().rstrip(":"), []).append(line.strip())
        previous = line
    return label_to_values


class TestSmallValuesInSummary:
    """In meters and seconds the summary shows small values in scientific notation, not as zero."""

    @pytest.fixture(scope='class')
    def meter_second_run(self, tmp_path_factory) -> CliRun:
        # quick_test.txt restated in meters and seconds: lengths scale by 1e-6, and the
        # time values keep their numbers with a lifetime of 1 second, so D = L^2 = 1e-14 m^2/s.
        params = ast.literal_eval(QUICK_TEST.read_text())
        overrides: dict[str, Any] = {'length unit': 'meter', 'time unit': 'second', 'number of runs': 10}
        for key in ('FWHM_0', 'spatial width', 'nominal diffusion length'):
            overrides[key] = params[key] * 1e-6
        workdir = tmp_path_factory.mktemp('meter')
        return run_cli(workdir, write_quick_test_variant(workdir, 'units_m_s', overrides), 'units_m_s', '--no-plots')

    def test_no_value_prints_as_zero(self, meter_second_run):
        assert meter_second_run.exit_code == 0
        values = summary_label_to_values(meter_second_run.read_summary())
        nominal_d, unit = values['Nominal diffusion coefficient'][0].split(' ', 1)
        assert unit == 'meter^2/second'
        assert float(nominal_d) == pytest.approx(1e-14, rel=1e-5)
        assert 'e-' in nominal_d
        for label in ('Nominal diffusion length', 'Initial profile squared width (sigma^2)'):
            number = values[label][0].split(' ', 1)[0]
            assert float(number) > 0 and 'e-' in number, (label, number)

    def test_estimate_lines_share_the_nominal_format(self, meter_second_run):
        summary = meter_second_run.read_summary()
        values = summary_label_to_values(summary)
        csv = meter_second_run.read_csv()
        for label in ('Standard deviation', 'Mean estimate', 'Root Mean Square Error'):
            for line in values[label]:
                number, unit = line.split(' ', 1)
                assert unit == 'meter^2/second' and 'e' in number, (label, line)
        wls_mean = float(values['Mean estimate'][0].split(' ', 1)[0])
        assert wls_mean == pytest.approx(csv[WLS_D_COLUMN].mean() * 1e-4, rel=1e-5)


class TestQuickTestOutputSchema:
    """Pins the output files, CSV columns, and summary structure for data/examples/quick_test.txt.

    Values are noisy, so the tests pin names and structure, check the inputs
    that reach the outputs unchanged, and check that the summary statistics
    agree with the CSV they summarize.
    """

    @pytest.fixture(scope='class')
    def quick_test_run(self, tmp_path_factory) -> CliRun:
        workdir = tmp_path_factory.mktemp('quick_test')
        run = run_cli(workdir, QUICK_TEST, 'quick_test', '--seed', '1')
        assert run.exit_code == 0
        return run

    def test_output_files(self, quick_test_run):
        assert sorted(p.name for p in quick_test_run.output_dir.iterdir()) == [
            'quick_test.csv', 'quick_test_accuracy_histogram.png', 'quick_test_summary.txt']
        assert quick_test_run.histogram_file.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'

    def test_csv_columns(self, quick_test_run):
        assert list(quick_test_run.read_csv().columns) == CSV_COLUMNS

    def test_csv_rows_carry_inputs(self, quick_test_run):
        from dice.utils.converters import fwhm_to_sigma2
        df = quick_test_run.read_csv()
        assert df['run number'].tolist() == list(range(100))
        assert df[results_column(df.columns, 'nominal diffusion coeff')].to_numpy() == pytest.approx(0.01)
        assert df[results_column(df.columns, 'nominal lifetime')].to_numpy() == pytest.approx(1.0)
        assert df[results_column(df.columns, 'nominal diffusion length')].to_numpy() == pytest.approx(0.1)
        assert df['nominal CNR'].to_numpy() == pytest.approx(50.0)
        assert df[results_column(df.columns, 'nominal sigma^2_0')].to_numpy() == pytest.approx(fwhm_to_sigma2(1.0))
        assert df[D_COLUMN].to_numpy() == pytest.approx(0.1, rel=1e-12)
        for prefix in ('weighted fit', 'unweighted fit'):
            # 1 μm^2/ns is 10 cm^2/s, and D is half the MSD slope.
            assert df[f'{prefix} diffusion coeff [cm²/s]'].to_numpy() == pytest.approx(
                df[results_column(df.columns, f'{prefix} diffusion slope')].to_numpy() * 5, rel=1e-12)
            assert df[f'{prefix} diffusion stderr [cm²/s]'].to_numpy() == pytest.approx(
                df[results_column(df.columns, f'{prefix} diffusion slope stderr')].to_numpy() * 5, rel=1e-12)
        assert not df.isna().any().any()

    def test_summary_skeleton(self, quick_test_run):
        assert summary_skeleton(quick_test_run.read_summary()) == SUMMARY_SKELETON

    def test_summary_agrees_with_csv(self, quick_test_run):
        df = quick_test_run.read_csv()
        values = summary_label_to_values(quick_test_run.read_summary())
        assert values['Number of runs'] == ['100']
        assert values['Initial CNR (Contrast-to-Noise Ratio)'] == ['50.000']
        assert values['Noise standard deviation values'] == ['[0.02]']
        d_estimates = {
            'wls': df[results_column(df.columns, 'weighted fit diffusion slope')].to_numpy() / 2,
            'ols': df[results_column(df.columns, 'unweighted fit diffusion slope')].to_numpy() / 2,
        }
        # WLS is listed before OLS under each statistic.
        d_unit = 'micrometer^2/nanosecond'
        assert values['Mean estimate'] == [
            f"{d_estimates[m].mean():.5f} {d_unit}" for m in ('wls', 'ols')]
        assert values['Standard deviation'] == [
            f"{d_estimates[m].std(ddof=1):.5f} {d_unit}" for m in ('wls', 'ols')]
        assert values['Bias (mean - nominal)'] == [
            f"{d_estimates[m].mean() - 0.01:.5f} {d_unit}" for m in ('wls', 'ols')]
        assert values['Root Mean Square Error'] == [
            f"{np.sqrt(np.mean((d_estimates[m] - 0.01) ** 2)):.5f} {d_unit}" for m in ('wls', 'ols')]
        assert values['Relative standard deviation'] == [
            f"{d_estimates[m].std(ddof=1) / 0.01:.3f}" for m in ('wls', 'ols')]
        assert values['Standard error of the mean and of the bias'] == [
            f"{d_estimates[m].std(ddof=1) / np.sqrt(100):.5f} {d_unit}" for m in ('wls', 'ols')]
        # The ratio is the bias in the paper's form: 1 + bias / nominal.
        assert values['Mean estimate / nominal'] == [f"{d_estimates[m].mean() / 0.01:.3f}" for m in ('wls', 'ols')]
        assert values['Estimates'] == ["100 of 100 runs (0 without an estimate)"] * 2
        assert values['DICE version'] == [dice.__version__]
        assert values['Seed'] == ['1']

        # The count and the 95% interval of the fraction, recomputed with statsmodels.
        from statsmodels.stats.proportion import proportion_confint

        def fraction(method: str) -> str:
            count = int(np.sum(np.abs(d_estimates[method] / 0.01 - 1) <= 0.1))
            lower, upper = proportion_confint(count, 100, alpha=0.05, method='wilson')
            return f"{count / 100:.1%} ({count} of 100; {lower:.1%} to {upper:.1%})"

        assert values['Fraction within ±10% of nominal'] == [
            f"WLS: {fraction('wls')}    OLS: {fraction('ols')}"]


class TestOutputOptions:
    """Export format, output directory, plot format, and their prerequisites."""

    @pytest.mark.parametrize('overrides, removed, plot_name', [
        ({}, (), 'no_plot_accuracy_histogram.png'),
        ({'noise range, reciprocal': [0.01, 0.05]}, ('noise value',), 'no_plot_accuracy_vs_cnr.png'),
    ])
    def test_plot_that_cannot_be_saved_fails_the_run_after_the_results_are_saved(
            self, tmp_path, capsys, overrides, removed, plot_name):
        """A folder stands where the plot file would go, which blocks the save on every system."""
        blocked = tmp_path / 'output' / 'no_plot' / plot_name
        blocked.mkdir(parents=True)
        params_file = write_quick_test_variant(
            tmp_path, 'no_plot', {'number of runs': 4} | overrides, removed=removed)

        with contextlib.chdir(tmp_path):
            assert main([str(params_file), '--multiprocessing', '0']) == 1

        console = capsys.readouterr().out
        assert 'saved to: ' + str(blocked) not in console
        last = console.splitlines()[-1]
        assert last.startswith("Error: The results were saved, but the plot could not be saved: ")
        assert plot_name in last
        # The file and the reason, not Python's text of the error.
        assert "[Errno" not in last and "\\\\" not in last
        assert (blocked.parent / 'no_plot.csv').exists()
        assert (blocked.parent / 'no_plot_summary.txt').read_text(encoding='utf-8').endswith(
            "Simulation completed successfully.\n")

    OPTION_RUNS = 4

    def _run(self, workdir: Path, slug: str, *cli_args: str, overrides=None) -> CliRun:
        params_file = write_quick_test_variant(
            workdir, slug, {'number of runs': self.OPTION_RUNS} | (overrides or {}))
        return run_cli(workdir, params_file, slug, *cli_args)

    def test_seed_of_the_command_line_is_recorded(self, tmp_path):
        """--seed N reproduces the run, although the runs draw from a child of N."""
        run = self._run(tmp_path, 'seeded', '--seed', '7', '--export-format', 'both', '--no-plots')
        assert run.exit_code == 0
        assert "Seed:\n    7\n" in run.read_summary()
        assert json.loads((run.output_dir / 'seeded.json').read_text())['summary']['seed'] == 7

    def test_console_ends_with_the_completion_line_of_the_summary(self, tmp_path, capsys):
        params_file = write_quick_test_variant(tmp_path, 'spoken', {'number of runs': self.OPTION_RUNS})
        with contextlib.chdir(tmp_path):
            assert main([str(params_file), '--multiprocessing', '1', '--no-plots']) == 0
        console = capsys.readouterr().out.splitlines()
        summary = (tmp_path / 'output' / 'spoken' / 'spoken_summary.txt').read_text(encoding='utf-8').splitlines()
        assert console[-1] == summary[-1] == "Simulation completed successfully."

    def test_json_and_csv_export(self, tmp_path):
        run = self._run(tmp_path, 'both', '--export-format', 'both', '--no-plots')
        assert run.exit_code == 0
        assert sorted(p.name for p in run.output_dir.iterdir()) == [
            'both.csv', 'both.json', 'both_summary.txt']
        exported = json.loads((run.output_dir / 'both.json').read_text())
        assert set(exported) == {'analysis', 'parameters', 'summary'}
        assert exported['summary'] == {
            'dice_version': dice.__version__, 'seed': None,
            'total_runs': self.OPTION_RUNS, 'noise_values': [0.02],
            'units': {'length': 'micrometer', 'time': 'nanosecond', 'diffusion_coefficient': 'μm²/ns'},
        }
        assert set(exported['analysis']) == {
            'summary', 'precision', 'precision_intervals', 'accuracy', 'noise_analysis', 'statistics'}
        interval = exported['analysis']['precision_intervals']['10%']['wls']
        assert set(interval) == {'count', 'total', 'lower', 'upper'}
        assert interval['total'] == self.OPTION_RUNS
        assert interval['count'] == round(exported['analysis']['precision']['10%']['wls'] * self.OPTION_RUNS)
        assert set(exported['analysis']['precision']) == {'5%', '10%', '20%', '50%'}
        assert set(exported['analysis']['accuracy']) == {'ols', 'wls'}
        assert exported['parameters']['filename slug'] == 'both'

    def test_json_only_export_writes_no_csv(self, tmp_path):
        run = self._run(tmp_path, 'json_only', '--export-format', 'json', '--no-plots')
        assert run.exit_code == 0
        assert sorted(p.name for p in run.output_dir.iterdir()) == [
            'json_only.json', 'json_only_summary.txt']

    def test_output_dir_created(self, tmp_path):
        target = tmp_path / 'nested' / 'results'
        params_file = write_quick_test_variant(tmp_path, 'elsewhere', {'number of runs': self.OPTION_RUNS})
        with contextlib.chdir(tmp_path):
            assert main([str(params_file), *SERIAL_QUIET, '--output-dir', str(target), '--no-plots']) == 0
        assert sorted(p.name for p in target.iterdir()) == ['elsewhere.csv', 'elsewhere_summary.txt']
        assert not (tmp_path / 'output').exists()

    def test_output_dir_that_is_a_file_rejected(self, tmp_path, capsys):
        blocker = tmp_path / 'not_a_directory'
        blocker.write_text('')
        params_file = write_quick_test_variant(tmp_path, 'blocked', {'number of runs': self.OPTION_RUNS})
        with pytest.raises(SystemExit) as exit_info:
            main([str(params_file), *SERIAL_QUIET, '--output-dir', str(blocker)])
        assert exit_info.value.code == 2
        assert "exists but is not a directory" in capsys.readouterr().err

    def test_image_type_sets_histogram_format(self, tmp_path):
        run = self._run(tmp_path, 'vector', overrides={'image type': 'svg'})
        assert run.exit_code == 0
        histogram = run.output_dir / 'vector_accuracy_histogram.svg'
        assert histogram.read_text(encoding='utf-8').lstrip().startswith('<?xml')
        assert not run.histogram_file.exists()

    def test_image_keys_reach_the_histogram(self, tmp_path, monkeypatch):
        """Every image key in the parameter file sets the matching figure property."""
        cli_main = importlib.import_module('dice.cli.main')
        figures = []
        real_plot = cli_main.plot_accuracy_histogram

        def recording_plot(**kwargs):
            figures.append((kwargs, real_plot(**kwargs)))
            return figures[-1][1]

        monkeypatch.setattr(cli_main, 'plot_accuracy_histogram', recording_plot)
        image_keys = {
            'image width': 4, 'image height': 3, 'image dpi': 50, 'image font size': 7,
            'image tick length': 3, 'image tick width': 1.5, 'image numbins': 11,
            'image x_lim': [0.5, 1.5],
        }

        run = self._run(tmp_path, 'styled', overrides=image_keys)

        assert run.exit_code == 0
        (kwargs, fig), = figures
        assert {key: kwargs[key] for key in (
            'width', 'height', 'dpi', 'font_size', 'tick_length', 'tick_width', 'num_bins', 'x_lim',
        )} == {
            'width': 4, 'height': 3, 'dpi': 50, 'font_size': 7, 'tick_length': 3,
            'tick_width': 1.5, 'num_bins': 11, 'x_lim': [0.5, 1.5],
        }
        ax = fig.axes[0]
        assert ax.get_xlim() == (0.5, 1.5)
        assert len(ax.patches) == 11
        assert run.histogram_file.exists()

    def test_unknown_image_type_rejected_before_simulation(self, tmp_path, capsys):
        run = self._run(tmp_path, 'bitmap', overrides={'image type': 'bmp'})
        assert run.exit_code == 1
        out = capsys.readouterr().out
        assert "Error: Unknown image type 'bmp'. Valid types: " in out
        assert "png" in out.splitlines()[-1] and "svg" in out.splitlines()[-1]
        assert "Starting DICE simulation" not in out
        assert not (tmp_path / 'output').exists()

    def test_malformed_x_lim_rejected_before_simulation(self, tmp_path, capsys):
        run = self._run(tmp_path, 'bad_limits', overrides={'image x_lim': [1.5, 0.5]})
        assert run.exit_code == 1
        assert "'image x_lim' must be two increasing finite numbers" in capsys.readouterr().out
        assert not (tmp_path / 'output').exists()

    @pytest.mark.parametrize(('key_value', 'cli_args', 'expected'), [
        (None, (), False),
        (1, (), True),
        (0, ('--retain-profiles',), True),
        (0, (), False),
    ])
    def test_retain_profile_data_key_or_flag_turns_retention_on(
            self, tmp_path, monkeypatch, key_value, cli_args, expected):
        cli_main = importlib.import_module('dice.cli.main')
        retained = []
        real_run = cli_main.run_monte_carlo_simulation

        def recording_run(*args, **kwargs):
            retained.append(kwargs['retain_profile_data'])
            return real_run(*args, **kwargs)

        monkeypatch.setattr(cli_main, 'run_monte_carlo_simulation', recording_run)
        overrides = {} if key_value is None else {'retain profile data': key_value}

        run = self._run(tmp_path, 'retention', *cli_args, '--no-plots', overrides=overrides)

        assert run.exit_code == 0
        assert retained == [expected]

    def test_estimated_noise_reaches_summary(self, tmp_path):
        from fft_cnr import fft_cnr
        profile_file = EXAMPLES_DIR / 'example_profile.csv'
        file_cnr = fft_cnr(np.loadtxt(profile_file, delimiter=',')).cnr
        params_file = write_quick_test_variant(
            tmp_path, 'estimated',
            {'number of runs': self.OPTION_RUNS, 'estimate noise from data': str(profile_file)},
            removed=('noise value',))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            run = run_cli(tmp_path, params_file, 'estimated', '--no-plots')
        assert run.exit_code == 0
        # The repository profile's peak is too narrow for fft-cnr, which the run reports.
        assert [w.category for w in caught if 'pixels wide at half maximum' in str(w.message)] == [
            UserWarning]
        values = summary_label_to_values(run.read_summary())
        assert values['Initial CNR (Contrast-to-Noise Ratio)'] == [f"{file_cnr:.3f}"]
        assert run.read_csv()['nominal CNR'].to_numpy() == pytest.approx(file_cnr, rel=1e-12)


class TestRelativeNoiseDataPath:
    """A parameters file names its measured profile relative to its own folder, not the working directory."""

    def test_cli_from_another_directory(self, tmp_path):
        import shutil
        from fft_cnr import fft_cnr
        params_dir, workdir = tmp_path / 'project', tmp_path / 'elsewhere'
        (params_dir / 'data').mkdir(parents=True)
        workdir.mkdir()
        shutil.copy(EXAMPLES_DIR / 'example_profile.csv', params_dir / 'data' / 'measured.csv')
        params_file = write_quick_test_variant(
            params_dir, 'relative_noise',
            {'number of runs': 4, 'estimate noise from data': 'data/measured.csv'},
            removed=('noise value',))

        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            run = run_cli(workdir, params_file, 'relative_noise', '--no-plots')

        assert run.exit_code == 0
        assert run.output_dir.parent.parent == workdir
        file_cnr = fft_cnr(np.loadtxt(EXAMPLES_DIR / 'example_profile.csv', delimiter=',')).cnr
        values = summary_label_to_values(run.read_summary())
        assert values['Initial CNR (Contrast-to-Noise Ratio)'] == [f"{file_cnr:.3f}"]


class TestNarrowPeakWarning:
    """The CLI shows the warning for a measured profile too narrow for fft-cnr, on stderr."""

    @staticmethod
    def _dry_run(directory: Path, profile_file: Path) -> 'subprocess.CompletedProcess[str]':
        import subprocess
        import sys
        params_file = write_quick_test_variant(
            directory, 'estimated', {'estimate noise from data': str(profile_file)},
            removed=('noise value',))
        return subprocess.run(
            [sys.executable, '-m', 'dice.cli.main', str(params_file), '--dry-run'],
            cwd=directory, capture_output=True, text=True, encoding='utf-8', timeout=120,
        )

    def test_repository_profile_warns(self, tmp_path):
        completed = self._dry_run(tmp_path, EXAMPLES_DIR / 'example_profile.csv')
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert "Warning: The peak of the measured profile" in completed.stderr
        assert "is 4.6 pixels wide at half maximum" in completed.stderr
        assert "loaded and validated" in completed.stdout

    def test_wide_profile_does_not_warn(self, tmp_path):
        from dice.core.profiles import gaussian
        from dice.utils.converters import fwhm_to_sigma2
        x = np.arange(101) - 50
        rng = np.random.default_rng(5)
        wide = 1000 * gaussian(x, 0.3, fwhm_to_sigma2(15.0), 1.0) + rng.normal(0, 50, x.size)
        profile_file = tmp_path / 'wide_profile.csv'
        np.savetxt(profile_file, wide.reshape(1, -1), delimiter=',', fmt='%.17g')
        completed = self._dry_run(tmp_path, profile_file)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert "Warning" not in completed.stderr


class TestNominalCnrUsesAmplitude:
    """Nominal CNR is amplitude_0 / noise sigma in the CSV and the summary.

    quick_test.txt is run as is (amplitude 1, noise 0.02) and with amplitude 2
    and noise 0.04, which keeps the initial CNR at 50, the regime in which the
    fft-cnr estimate was characterized for this 50-pixel profile. With the same
    seed the second run's noisy profiles are exactly twice the first run's.
    """

    @pytest.fixture(scope='class')
    def unit_amplitude_run(self, tmp_path_factory) -> CliRun:
        workdir = tmp_path_factory.mktemp('amplitude_1')
        return run_cli(workdir, QUICK_TEST, 'quick_test', '--seed', '4', '--no-plots')

    @pytest.fixture(scope='class')
    def double_amplitude_run(self, tmp_path_factory) -> CliRun:
        workdir = tmp_path_factory.mktemp('amplitude_2')
        params_file = write_quick_test_variant(
            workdir, 'amplitude_2', {'amplitude_0': 2, 'noise value': 0.04})
        return run_cli(workdir, params_file, 'amplitude_2', '--seed', '4', '--no-plots')

    def test_nominal_cnr_is_amplitude_over_noise(self, double_amplitude_run):
        assert double_amplitude_run.exit_code == 0
        assert double_amplitude_run.read_csv()['nominal CNR'].to_numpy() == pytest.approx(2 / 0.04)
        values = summary_label_to_values(double_amplitude_run.read_summary())
        assert values['Initial CNR (Contrast-to-Noise Ratio)'] == ['50.000']

    def test_mean_estimated_cnr_agrees_with_nominal(self, double_amplitude_run):
        # Tolerance: four standard errors of the mean of the run's own estimates,
        # plus 1% of the nominal value for the estimator's bias on this profile.
        # On 4000 simulated quick_test profiles at CNR 50, fft-cnr 0.4.1 read
        # +0.47% high (standard error 0.21%), with a per-profile SD of 6.5, so
        # 100 runs give a standard error near 0.65. Ignoring amplitude_0 would
        # put the nominal value at 25, about 38 standard errors away.
        df = double_amplitude_run.read_csv()
        estimates, nominal = df['estimated CNR'].to_numpy(), df['nominal CNR'].iloc[0]
        standard_error = estimates.std(ddof=1) / np.sqrt(estimates.size)
        assert abs(estimates.mean() - nominal) <= 4 * standard_error + 0.01 * nominal

    def test_unit_amplitude_unchanged(self, unit_amplitude_run, double_amplitude_run):
        assert unit_amplitude_run.exit_code == 0
        unit_df, double_df = unit_amplitude_run.read_csv(), double_amplitude_run.read_csv()
        assert unit_df['nominal CNR'].to_numpy() == pytest.approx(1 / 0.02)
        values = summary_label_to_values(unit_amplitude_run.read_summary())
        assert values['Initial CNR (Contrast-to-Noise Ratio)'] == ['50.000']
        # Scaling signal and noise together leaves every CNR, and every slope, unchanged.
        assert double_df['estimated CNR'].to_numpy() == pytest.approx(
            unit_df['estimated CNR'].to_numpy(), rel=1e-9)
        assert double_df[results_column(double_df.columns, 'weighted fit diffusion slope')].to_numpy() == pytest.approx(
            unit_df[results_column(unit_df.columns, 'weighted fit diffusion slope')].to_numpy(), rel=1e-6)


class TestNoiseRangeExample:
    """data/examples/noise_range.txt draws a noise level for each run and reports accuracy against CNR.

    The CSV, the JSON export, and the summary are checked against each other:
    the JSON records the noise of each CSV row, and the summary and the JSON
    report the same CNR bins, whose statistics are recomputed from the CSV.
    """

    RUNS = 100
    LOW, HIGH = 0.005, 0.1

    @pytest.fixture(scope='class')
    def example_run(self, tmp_path_factory) -> CliRun:
        workdir = tmp_path_factory.mktemp('noise_range')
        run = run_cli(workdir, EXAMPLES_DIR / 'noise_range.txt', 'noise_range',
                      '--seed', '2', '--export-format', 'both')
        assert run.exit_code == 0
        return run

    @staticmethod
    def _json(run: CliRun) -> dict:
        return json.loads((run.output_dir / 'noise_range.json').read_text(encoding='utf-8'))

    def test_output_files(self, example_run):
        assert sorted(p.name for p in example_run.output_dir.iterdir()) == [
            'noise_range.csv', 'noise_range.json', 'noise_range_accuracy_vs_cnr.png',
            'noise_range_summary.txt']
        assert (example_run.output_dir / 'noise_range_accuracy_vs_cnr.png').read_bytes()[:8] == (
            b'\x89PNG\r\n\x1a\n')

    def test_csv_records_the_noise_of_each_run(self, example_run):
        df = example_run.read_csv()
        expected_columns = list(CSV_COLUMNS)
        expected_columns.insert(expected_columns.index('nominal CNR'), 'noise standard deviation')
        assert list(df.columns) == expected_columns
        noise = df['noise standard deviation'].to_numpy()
        assert len(noise) == len(np.unique(noise)) == self.RUNS
        assert np.all((noise >= self.LOW) & (noise <= self.HIGH))
        # The CSV keeps about 16 significant digits of each value.
        assert df['nominal CNR'].to_numpy() == pytest.approx(1 / noise, rel=1e-12)

    def test_json_records_each_run_and_each_bin(self, example_run):
        exported = self._json(example_run)
        df = example_run.read_csv()
        assert exported['summary']['total_runs'] == self.RUNS
        assert exported['summary']['noise_values'] == pytest.approx(
            df['noise standard deviation'].tolist(), rel=1e-12)
        assert exported['summary']['noise_range'] == {'low': self.LOW, 'high': self.HIGH, 'logarithmic': True}
        bins = exported['analysis']['cnr_bins']
        assert [entry['num_runs'] for entry in bins] == [20] * 5
        assert exported['analysis']['noise_analysis'] == {}
        for entry in bins:
            # The CSV keeps about 16 significant digits, so the bin limits get a relative margin.
            in_bin = df[df['nominal CNR'].between(
                entry['cnr_min'] * (1 - 1e-12), entry['cnr_max'] * (1 + 1e-12))]
            assert len(in_bin) == entry['num_runs']
            wls = in_bin[results_column(in_bin.columns, 'weighted fit diffusion slope')].dropna().to_numpy() / 2
            assert entry['accuracy']['wls']['mean'] == pytest.approx(wls.mean(), rel=1e-12)
            assert entry['accuracy']['wls']['std'] == pytest.approx(wls.std(ddof=1), rel=1e-12)
            assert entry['statistics']['wls']['num_successful'] == len(wls)
            assert entry['precision']['10%']['wls'] == pytest.approx(np.mean(np.abs(wls / 0.01 - 1) <= 0.1))

    def test_summary_agrees_with_csv_and_json(self, example_run):
        df = example_run.read_csv()
        summary = example_run.read_summary()
        values = summary_label_to_values(summary)
        noise, cnr = df['noise standard deviation'], df['nominal CNR']
        assert values['Number of runs'] == ['100']
        assert values['Noise standard deviation, drawn for each run'] == [
            f"100 values from {noise.min():.4f} to {noise.max():.4f} (range 0.0050 to 0.1000)"]
        assert values['Initial CNR (Contrast-to-Noise Ratio), drawn for each run'] == [
            f"{cnr.min():.3f} to {cnr.max():.3f} (range 10.000 to 200.000, uniform in log10 of the CNR)"]
        assert 'Noise standard deviation values' not in values
        bin_headers = [line for line in summary.splitlines() if re.match(r'Initial CNR [0-9.]+ to ', line)]
        assert bin_headers == [
            f"Initial CNR {entry['cnr_min']:.3f} to {entry['cnr_max']:.3f} ({entry['num_runs']} runs):"
            for entry in self._json(example_run)['analysis']['cnr_bins']]
        d_unit = 'micrometer^2/nanosecond'
        assert values['Mean estimate'] == [
            f"{entry['accuracy'][key]['mean']:.5f} {d_unit}"
            for entry in self._json(example_run)['analysis']['cnr_bins'] for key in ('wls', 'ols')]

    @pytest.mark.parametrize('noise_range, message', [
        ([0.05, 0.01], "Invalid noise range [0.05, 0.01]. The first value (low) must be less than the second (high)."),
        ([0.0, 0.05], "Invalid noise range [0.0, 0.05]. Both values must be positive and finite."),
        ([0.01], "Invalid noise range [0.01]. Must be two numeric values."),
    ])
    def test_malformed_range_rejected_before_simulation(self, tmp_path, capsys, noise_range, message):
        params_file = write_quick_test_variant(
            tmp_path, 'bad_range', {'noise range, reciprocal log': noise_range}, removed=('noise value',))
        run = run_cli(tmp_path, params_file, 'bad_range')
        assert run.exit_code == 1
        assert message in capsys.readouterr().out
        assert not (tmp_path / 'output').exists()

    def test_dice_runner_rejects_what_the_command_line_rejects(self, tmp_path):
        """The Python entry point validates the file, so an off-center profile does not run."""
        import dice
        params_file = write_quick_test_variant(tmp_path, 'runner_off_center', {'mean_0': 1.3, 'number of runs': 3})
        with pytest.raises(ValueError, match="'mean_0' is 1.3, outside the central 40% of the spatial window"):
            dice.dice_runner(str(params_file))

    def test_dice_runner_draws_a_noise_level_for_each_run(self, tmp_path):
        import dice
        from dice.models.parameters import NoiseRange
        params_file = write_quick_test_variant(
            tmp_path, 'runner_range',
            {'noise range, reciprocal': [0.01, 0.05], 'number of runs': 6, 'multiprocessing': False},
            removed=('noise value',))
        result = dice.dice_runner(str(params_file))
        assert result.noise_range == NoiseRange(0.01, 0.05, logarithmic=False)
        assert result.num_runs == len(result.run_results) == len(result.noise_values) == 6
        assert [run.noise_sigma for run in result.run_results] == result.noise_values


class TestNoPlotOfAnEarlierRunStaysBesideNewResults:
    """A run replaces the results of an earlier run with the same slug, and removes its plots."""

    RUNS = {'number of runs': 4}

    def _plots(self, run: CliRun) -> list:
        return sorted(path.name for path in run.output_dir.iterdir() if 'accuracy' in path.name)

    def test_another_image_type_replaces_the_earlier_plot(self, tmp_path):
        first = run_cli(tmp_path, write_quick_test_variant(tmp_path, 'rerun', self.RUNS), 'rerun')
        assert self._plots(first) == ['rerun_accuracy_histogram.png']

        second = run_cli(tmp_path, write_quick_test_variant(tmp_path, 'rerun', self.RUNS | {'image type': 'svg'}),
                         'rerun')

        assert second.exit_code == 0
        assert self._plots(second) == ['rerun_accuracy_histogram.svg']

    def test_noise_range_after_a_fixed_noise_value_and_back(self, tmp_path):
        run_cli(tmp_path, write_quick_test_variant(tmp_path, 'rerun', self.RUNS), 'rerun')

        ranged = run_cli(tmp_path, write_quick_test_variant(
            tmp_path, 'rerun', self.RUNS | {'noise range, reciprocal': [0.01, 0.05]}, removed=('noise value',)),
            'rerun')
        assert self._plots(ranged) == ['rerun_accuracy_vs_cnr.png']

        fixed = run_cli(tmp_path, write_quick_test_variant(tmp_path, 'rerun', self.RUNS), 'rerun')
        assert self._plots(fixed) == ['rerun_accuracy_histogram.png']

    def test_run_without_plots_removes_the_earlier_plot(self, tmp_path):
        params_file = write_quick_test_variant(tmp_path, 'rerun', self.RUNS)
        first = run_cli(tmp_path, params_file, 'rerun')
        earlier_results = first.read_csv()

        second = run_cli(tmp_path, params_file, 'rerun', '--no-plots')

        assert second.exit_code == 0
        assert self._plots(second) == []
        # The results are those of the second run, which the earlier plot did not show.
        assert not second.read_csv().equals(earlier_results)

    def test_files_of_another_slug_are_kept(self, tmp_path):
        other = run_cli(tmp_path, write_quick_test_variant(tmp_path, 'other', self.RUNS), 'other')
        run_cli(tmp_path, write_quick_test_variant(tmp_path, 'rerun', self.RUNS), 'rerun', '--output-dir',
                str(other.output_dir))
        assert 'other_accuracy_histogram.png' in self._plots(other)


class TestProgressMessages:
    """With --verbose, a message each time a multiple of the progress interval is passed, and at the end."""

    def _messages(self, capsys, counts: list[int], total: int, interval: int) -> list[str]:
        from dice.cli.main import create_progress_callback
        callback = create_progress_callback(verbose=True, quiet=False, interval=interval)
        for count in counts:
            callback(count, total)
        return [line.split(" runs completed")[0] for line in capsys.readouterr().out.splitlines()]

    def test_serial_run_reports_at_each_multiple_of_the_interval(self, capsys):
        messages = self._messages(capsys, list(range(1, 251)), total=250, interval=100)
        assert messages == ["Progress: 100/250", "Progress: 200/250", "Progress: 250/250"]

    def test_parallel_run_reports_when_a_multiple_is_passed(self, capsys):
        """A parallel simulation reports at counts that follow the clock, which seldom are multiples."""
        messages = self._messages(capsys, [37, 95, 130, 180, 260, 297, 300], total=300, interval=100)
        assert messages == ["Progress: 130/300", "Progress: 260/300", "Progress: 300/300"]

    def test_no_callback_without_verbose_or_with_quiet(self):
        from dice.cli.main import create_progress_callback
        assert create_progress_callback(verbose=False, quiet=False) is None
        assert create_progress_callback(verbose=True, quiet=True) is None
