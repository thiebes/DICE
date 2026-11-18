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


def write_quick_test_variant(directory: Path, slug: str, overrides: dict[str, Any]) -> Path:
    """Write quick_test.txt with the given values replaced and a new filename slug."""
    params = ast.literal_eval(QUICK_TEST.read_text())
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
