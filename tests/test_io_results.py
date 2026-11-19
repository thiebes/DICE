"""
Integration tests for dice.io.results, fed by a real DICE simulation.

The fixtures parse data/examples/quick_test.txt and run a short serial
simulation, so the writers receive the same objects the CLI passes them.
"""

from pathlib import Path

import numpy as np
import pytest

from dice.analysis.simulation import run_monte_carlo_simulation
from dice.analysis.statistics import analyze_simulation_results
from dice.io.parameters import open_parameters
from dice.io.results import (
    export_collated_results,
    load_multiple_results,
    load_results_csv,
    print_and_append,
    write_summary_file,
)
from dice.models.parameters import OutputUnitPreferences
from dice.models.results import MonteCarloOutput
from dice.utils.legacy_compatibility import create_parameters_from_legacy

EXAMPLES_DIR = Path(__file__).parent.parent / 'data' / 'examples'
PLUS_MINUS = '±'
FIXTURE_RUNS = 8


@pytest.fixture(scope='module')
def quick_test_parameters() -> dict:
    return open_parameters(EXAMPLES_DIR / 'quick_test.txt')


@pytest.fixture(scope='module')
def quick_test_simulation(quick_test_parameters) -> MonteCarloOutput:
    params = quick_test_parameters
    sim_params = create_parameters_from_legacy(
        parameters_dict=params,
        diffusion_coefficient=params['nominal diffusion coefficient'],
        lifetime=params['nominal lifetime (tau)'],
        diffusion_length=params['nominal diffusion length'],
    )
    return run_monte_carlo_simulation(
        parameters=sim_params,
        x_axis=params['x array'],
        time_axis=params['time series'],
        noise_values=params['noise series'],
        num_runs=FIXTURE_RUNS,
        multiprocessing=False,
    )


@pytest.fixture(scope='module')
def quick_test_analysis(quick_test_simulation) -> dict:
    return analyze_simulation_results(quick_test_simulation)


class TestTextOutputEncoding:
    """Text outputs are UTF-8 on every platform, whatever the locale encoding."""

    def test_summary_file_is_utf8(self, tmp_path, quick_test_simulation,
                                  quick_test_parameters, quick_test_analysis):
        summary_file = tmp_path / 'summary.txt'
        write_summary_file(quick_test_simulation, summary_file,
                           quick_test_parameters, quick_test_analysis)
        text = summary_file.read_bytes().decode('utf-8')
        assert f"Fraction within {PLUS_MINUS}5% of nominal:" in text

    def test_print_and_append_is_utf8(self, tmp_path):
        log_file = tmp_path / 'log.txt'
        message = f"precision within {PLUS_MINUS}10% at 5 μm"
        print_and_append(log_file, message, print_to_console=False)
        print_and_append(log_file, message, print_to_console=False)
        assert log_file.read_text(encoding='utf-8') == f"{message}\n{message}\n"


class TestZeroNoise:
    """A noise-free run has an infinite initial CNR, which the summary reports as 'inf'."""

    def test_summary_reports_infinite_cnr(self, tmp_path, quick_test_simulation,
                                          quick_test_parameters, quick_test_analysis):
        import dataclasses
        noise_free = dataclasses.replace(quick_test_simulation, noise_values=[0.0])
        summary_file = tmp_path / 'summary.txt'
        write_summary_file(noise_free, summary_file, quick_test_parameters, quick_test_analysis)
        text = summary_file.read_text(encoding='utf-8')
        assert "Initial CNR (Contrast-to-Noise Ratio):\n    inf\n" in text
        assert text.endswith("Simulation completed successfully.\n")


def simulate_quick_test(params: dict, time_axis, num_runs: int, **options) -> MonteCarloOutput:
    sim_params = create_parameters_from_legacy(
        params, params['nominal diffusion coefficient'],
        params['nominal lifetime (tau)'], params['nominal diffusion length'])
    return run_monte_carlo_simulation(
        sim_params, params['x array'], time_axis, params['noise series'],
        num_runs=num_runs, multiprocessing=False, seed=3, **options)


class TestAnalysisToSummarySchema:
    """The analysis dictionary carries every field write_summary_file reads."""

    def test_analysis_fields(self, quick_test_analysis):
        assert {'summary', 'precision', 'accuracy'} <= set(quick_test_analysis)
        assert 'nominal_diffusion_coefficient' in quick_test_analysis['summary']
        for method in ('wls', 'ols'):
            assert {'std', 'mean', 'bias', 'rmse'} <= set(quick_test_analysis['accuracy'][method])
        assert list(quick_test_analysis['precision']) == ['5%', '10%', '20%', '50%']
        for fractions in quick_test_analysis['precision'].values():
            assert set(fractions) == {'wls', 'ols'}

    def test_summary_without_analysis(self, tmp_path, quick_test_simulation, quick_test_parameters):
        summary_file = tmp_path / 'summary.txt'
        write_summary_file(quick_test_simulation, summary_file, quick_test_parameters)
        text = summary_file.read_text(encoding='utf-8')
        assert "Analysis Results:" not in text
        assert text.endswith("Simulation completed successfully.\n")

    def test_summary_without_precision(self, tmp_path, quick_test_simulation,
                                       quick_test_parameters, quick_test_analysis):
        analysis = {key: value for key, value in quick_test_analysis.items() if key != 'precision'}
        summary_file = tmp_path / 'summary.txt'
        write_summary_file(quick_test_simulation, summary_file, quick_test_parameters, analysis)
        text = summary_file.read_text(encoding='utf-8')
        assert "Weighted Least Squares:" in text
        assert "Precision at standard thresholds:" not in text


class TestExportCollatedResults:
    """CSV export: returned frame, unit labels, and the columns present per run type."""

    def test_returned_frame_matches_file(self, tmp_path, quick_test_simulation):
        csv_file = tmp_path / 'results.csv'
        exported = export_collated_results(quick_test_simulation, csv_file)
        reloaded = load_results_csv(csv_file)
        assert list(reloaded.columns) == list(exported.columns)
        np.testing.assert_allclose(reloaded.to_numpy(dtype=float), exported.to_numpy(dtype=float), rtol=1e-12)
        assert len(exported) == FIXTURE_RUNS

    def test_output_units_relabel_and_convert(self, tmp_path, quick_test_simulation):
        micrometer_per_ns = OutputUnitPreferences(diffusion_length='micrometer', diffusion_time='nanosecond')
        exported = export_collated_results(quick_test_simulation, tmp_path / 'results.csv',
                                           output_units=micrometer_per_ns)
        label = 'weighted fit diffusion coeff [μm²/ns]'
        assert label in exported.columns
        assert not any('cm²/s' in column for column in exported.columns)
        assert exported[label].to_numpy() == pytest.approx(
            exported['weighted fit diffusion slope'].to_numpy() / 2, rel=1e-12)
        assert exported['nominal diffusion coeff [μm²/ns]'].to_numpy() == pytest.approx(0.01)

    def test_single_frame_runs_have_no_fit_columns(self, tmp_path, quick_test_parameters):
        single_frame = simulate_quick_test(
            quick_test_parameters, quick_test_parameters['time series'][:1], num_runs=2)
        exported = export_collated_results(single_frame, tmp_path / 'results.csv')
        assert list(exported.columns) == [
            'run number', 'nominal diffusion coeff', 'nominal lifetime', 'nominal diffusion length',
            'nominal CNR', 'estimated CNR', 'nominal sigma^2_0', 'estimated sigma^2_0',
            'nominal diffusion coeff [cm²/s]',
        ]


class TestLoadMultipleResults:
    """Result CSVs in one directory concatenate, with unreadable files skipped."""

    def test_concatenates_and_skips_unreadable(self, tmp_path, quick_test_simulation, capsys):
        export_collated_results(quick_test_simulation, tmp_path / 'a.csv')
        export_collated_results(quick_test_simulation, tmp_path / 'b.csv')
        (tmp_path / 'c_empty.csv').write_text('')
        combined = load_multiple_results(tmp_path)
        assert len(combined) == 2 * FIXTURE_RUNS
        assert sorted(combined['source_file'].unique()) == ['a.csv', 'b.csv']
        assert "c_empty.csv" in capsys.readouterr().out

    def test_no_files_and_no_valid_files(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No files matching"):
            load_multiple_results(tmp_path)
        (tmp_path / 'empty.csv').write_text('')
        with pytest.raises(ValueError, match="No valid CSV files"):
            load_multiple_results(tmp_path)


class TestFormatQuantity:
    """Fixed decimals unless they would show fewer than two significant digits."""

    @pytest.mark.parametrize('value, decimals, reference, expected', [
        (0.01, 5, None, '0.01000'),
        (12.5, 3, None, '12.500'),
        (0.0, 3, None, '0.000'),
        (float('inf'), 3, None, 'inf'),
        (1e-14, 5, None, '1.00000e-14'),
        (0.0001, 5, None, '0.00010'),
        (0.00009, 5, None, '9.00000e-05'),
        (0.0004, 5, 1e-6, '4.00000e-04'),
        (-3e-7, 5, 0.01, '-0.00000'),
    ])
    def test_format(self, value, decimals, reference, expected):
        from dice.io.results import format_quantity
        assert format_quantity(value, decimals, reference) == expected
