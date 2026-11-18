"""
Integration tests for dice.io.results, fed by a real DICE simulation.

The fixtures parse data/examples/quick_test.txt and run a short serial
simulation, so the writers receive the same objects the CLI passes them.
"""

from pathlib import Path

import pytest

from dice.analysis.simulation import run_monte_carlo_simulation
from dice.analysis.statistics import analyze_simulation_results
from dice.io.parameters import open_parameters
from dice.io.results import export_legacy_format, print_and_append, write_summary_file
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

    def test_legacy_summary_file_is_utf8(self, tmp_path, quick_test_simulation,
                                         quick_test_parameters):
        summary_file = tmp_path / 'legacy_summary.txt'
        legacy_analysis = {'% fits within proximity': {
            '10%': {'unweighted fit': 50.0, 'weighted fit': 60.0},
        }}
        export_legacy_format(
            quick_test_simulation,
            {'parameters': quick_test_parameters, 'analysis': legacy_analysis},
            tmp_path / 'legacy.csv',
            summary_file,
        )
        text = summary_file.read_bytes().decode('utf-8')
        assert f"D_estimate / D_nominal = 1 {PLUS_MINUS} 10%:" in text

    def test_print_and_append_is_utf8(self, tmp_path):
        log_file = tmp_path / 'log.txt'
        message = f"precision within {PLUS_MINUS}10% at 5 μm"
        print_and_append(log_file, message, print_to_console=False)
        print_and_append(log_file, message, print_to_console=False)
        assert log_file.read_text(encoding='utf-8') == f"{message}\n{message}\n"
