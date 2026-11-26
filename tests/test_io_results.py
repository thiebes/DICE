"""
Integration tests for dice.io.results, fed by a real DICE simulation.

The fixtures parse data/examples/quick_test.txt and run a short serial
simulation, so the writers receive the same objects the CLI passes them.
"""

from pathlib import Path

import numpy as np
import pytest

from dice.io.results import column_unit, results_column

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
from dice.models.parameters import simulation_parameters_from_dict

EXAMPLES_DIR = Path(__file__).parent.parent / 'data' / 'examples'
PLUS_MINUS = '±'
FIXTURE_RUNS = 8


@pytest.fixture(scope='module')
def quick_test_parameters() -> dict:
    return open_parameters(EXAMPLES_DIR / 'quick_test.txt')


@pytest.fixture(scope='module')
def quick_test_simulation(quick_test_parameters) -> MonteCarloOutput:
    params = quick_test_parameters
    sim_params = simulation_parameters_from_dict(
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
    sim_params = simulation_parameters_from_dict(
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
        exported = export_collated_results(quick_test_simulation, csv_file, length_unit='micrometer', time_unit='nanosecond')
        reloaded = load_results_csv(csv_file)
        assert list(reloaded.columns) == list(exported.columns)
        np.testing.assert_allclose(reloaded.to_numpy(dtype=float), exported.to_numpy(dtype=float), rtol=1e-12)
        assert len(exported) == FIXTURE_RUNS

    def test_output_units_relabel_and_convert(self, tmp_path, quick_test_simulation):
        micrometer_per_ns = OutputUnitPreferences(diffusion_length='micrometer', diffusion_time='nanosecond')
        exported = export_collated_results(quick_test_simulation, tmp_path / 'results.csv', length_unit='micrometer', time_unit='nanosecond',
                                           output_units=micrometer_per_ns)
        label = 'weighted fit diffusion coeff [μm²/ns]'
        assert label in exported.columns
        assert not any('cm²/s' in column for column in exported.columns)
        assert exported[label].to_numpy() == pytest.approx(
            exported[results_column(exported.columns, 'weighted fit diffusion slope')].to_numpy() / 2, rel=1e-12)
        assert exported['nominal diffusion coeff [μm²/ns]'].to_numpy() == pytest.approx(0.01)

    def test_single_frame_runs_have_no_fit_columns(self, tmp_path, quick_test_parameters):
        single_frame = simulate_quick_test(
            quick_test_parameters, quick_test_parameters['time series'][:1], num_runs=2)
        exported = export_collated_results(single_frame, tmp_path / 'results.csv', length_unit='micrometer', time_unit='nanosecond')
        assert list(exported.columns) == [
            'run number', 'nominal diffusion coeff [μm²/ns]', 'nominal lifetime [ns]',
            'nominal diffusion length [μm]', 'nominal CNR', 'estimated CNR',
            'nominal sigma^2_0 [μm²]', 'estimated sigma^2_0 [μm²]',
            'nominal diffusion coeff [cm²/s]',
        ]


class TestLoadMultipleResults:
    """Result CSVs in one directory concatenate, with unreadable files skipped."""

    def test_concatenates_and_skips_unreadable(self, tmp_path, quick_test_simulation, capsys):
        export_collated_results(quick_test_simulation, tmp_path / 'a.csv', length_unit='micrometer', time_unit='nanosecond')
        export_collated_results(quick_test_simulation, tmp_path / 'b.csv', length_unit='micrometer', time_unit='nanosecond')
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


RANGE_RUNS = 45  # two CNR bins, of 23 and 22 runs


@pytest.fixture(scope='module')
def range_simulation(quick_test_parameters) -> MonteCarloOutput:
    """quick_test.txt with an amplitude of 2 and a noise level drawn for each run."""
    from dice.core.noise import make_noise_distribution
    from dice.models.parameters import NoiseRange
    params = quick_test_parameters
    noise_range = NoiseRange(0.01, 0.05, logarithmic=False)
    sim_params = simulation_parameters_from_dict(
        params | {'amplitude_0': 2.0}, params['nominal diffusion coefficient'],
        params['nominal lifetime (tau)'], params['nominal diffusion length'])
    return run_monte_carlo_simulation(
        sim_params, params['x array'], params['time series'],
        make_noise_distribution(0.01, 0.05, RANGE_RUNS, seed=8), num_runs=RANGE_RUNS,
        multiprocessing=False, seed=9, noise_range=noise_range)


class TestNoiseRangeOutputs:
    """A simulation with a noise range: the CSV records each run's noise; the summary reports CNR bins."""

    def test_csv_gains_the_noise_of_each_run(self, tmp_path, range_simulation, quick_test_simulation):
        ranged = export_collated_results(range_simulation, tmp_path / 'range.csv', length_unit='micrometer', time_unit='nanosecond')
        fixed = export_collated_results(quick_test_simulation, tmp_path / 'fixed.csv', length_unit='micrometer', time_unit='nanosecond')

        expected_columns = list(fixed.columns)
        expected_columns.insert(expected_columns.index('nominal CNR'), 'noise standard deviation')
        assert list(ranged.columns) == expected_columns
        assert 'noise standard deviation' not in fixed.columns
        reloaded = load_results_csv(tmp_path / 'range.csv')
        sigmas = [run.noise_sigma for run in range_simulation.run_results]
        assert reloaded['noise standard deviation'].tolist() == pytest.approx(sigmas, rel=1e-15)
        assert reloaded['nominal CNR'].to_numpy() == pytest.approx(2.0 / np.array(sigmas), rel=1e-15)
        assert reloaded['run number'].tolist() == list(range(RANGE_RUNS))

    def _summary(self, tmp_path, simulation, parameters, analysis) -> str:
        summary_file = tmp_path / 'summary.txt'
        write_summary_file(simulation, summary_file, parameters, analysis)
        return summary_file.read_text(encoding='utf-8')

    def test_summary_reports_the_drawn_range(self, tmp_path, range_simulation, quick_test_parameters):
        text = self._summary(tmp_path, range_simulation, quick_test_parameters,
                             analyze_simulation_results(range_simulation))
        sigmas = np.array(range_simulation.noise_values)

        assert "Noise standard deviation values:" not in text
        assert str(range_simulation.noise_values[0]) not in text  # no list of every value
        assert (f"Noise standard deviation, drawn for each run:\n"
                f"    {RANGE_RUNS} values from {sigmas.min():.4f} to {sigmas.max():.4f}"
                f" (range 0.0100 to 0.0500)\n") in text
        assert (f"Initial CNR (Contrast-to-Noise Ratio), drawn for each run:\n"
                f"    {2 / sigmas.max():.3f} to {2 / sigmas.min():.3f}"
                f" (range 40.000 to 200.000, uniform in CNR)\n") in text
        assert "Number of runs:\n    45\n" in text

    def test_summary_reports_each_cnr_bin(self, tmp_path, range_simulation, quick_test_parameters):
        analysis = analyze_simulation_results(range_simulation)
        text = self._summary(tmp_path, range_simulation, quick_test_parameters, analysis)

        # Recompute the bins from the runs: sorted by CNR, the first 23 runs, then 22.
        runs = sorted(range_simulation.run_results, key=lambda run: -run.noise_sigma)
        bins = [runs[:23], runs[23:]]
        assert "Statistics in bins of initial CNR with equal numbers of runs:\n" in text
        for runs_in_bin in bins:
            cnr = [2.0 / run.noise_sigma for run in runs_in_bin]
            header = f"\nInitial CNR {min(cnr):.3f} to {max(cnr):.3f} ({len(runs_in_bin)} runs):\n"
            assert header in text
            section = text.split(header)[1].split("\nInitial CNR ")[0]
            wls = np.array([run.wls_slope / 2 for run in runs_in_bin])
            assert f"      Mean estimate:\n          {wls.mean():.5f} micrometer^2/nanosecond\n" in section
            assert f"      Standard deviation:\n          {wls.std(ddof=1):.5f} micrometer^2/nanosecond\n" in section
            count = int(np.sum(np.abs(wls / 0.01 - 1) <= 0.1))
            assert (f"    Fraction within ±10% of nominal:\n"
                    f"      WLS: {count / len(wls):.1%} ({count} of {len(wls)}; ") in section
            assert f"      Estimates:\n          {len(wls)} of {len(runs_in_bin)} runs (0 without an estimate)\n" in section
            assert (f"      Standard error of the mean and of the bias:\n"
                    f"          {wls.std(ddof=1) / np.sqrt(len(wls)):.5f} micrometer^2/nanosecond\n") in section
        # The pooled statistics of all runs are not reported.
        assert "\nDiffusion Coefficient Estimate Statistics:" not in text
        assert text.count("Diffusion Coefficient Estimate Statistics:") == 2
        assert text.endswith("\n\nSimulation completed successfully.\n")

    def test_summary_names_the_logarithmic_distribution(self, tmp_path, range_simulation,
                                                        quick_test_parameters):
        import dataclasses
        from dice.models.parameters import NoiseRange
        logarithmic = dataclasses.replace(range_simulation, noise_range=NoiseRange(0.01, 0.05, True))
        text = self._summary(tmp_path, logarithmic, quick_test_parameters,
                             analyze_simulation_results(logarithmic))
        assert "(range 40.000 to 200.000, uniform in log10 of the CNR)\n" in text

    def test_analysis_without_bins_is_rejected(self, tmp_path, range_simulation, quick_test_parameters):
        analysis = analyze_simulation_results(range_simulation)
        del analysis['cnr_bins']
        with pytest.raises(KeyError, match='cnr_bins'):
            self._summary(tmp_path, range_simulation, quick_test_parameters, analysis)


class TestWritersTakeTheUnitsOfTheRun:
    """The writers do not assume micrometer and nanosecond for results in other units."""

    def test_csv_export_requires_the_units(self, tmp_path, quick_test_simulation):
        with pytest.raises(TypeError, match="length_unit"):
            export_collated_results(quick_test_simulation, tmp_path / 'results.csv')

    def test_summary_takes_the_units_from_the_parameters(self, tmp_path, quick_test_simulation,
                                                          quick_test_parameters, quick_test_analysis):
        parameters = quick_test_parameters | {'length unit': 'nanometer', 'time unit': 'picosecond',
                                              'length_unit': 'nanometer', 'time_unit': 'picosecond'}
        derived, explicit = tmp_path / 'derived.txt', tmp_path / 'explicit.txt'

        write_summary_file(quick_test_simulation, derived, parameters, quick_test_analysis)
        write_summary_file(quick_test_simulation, explicit, parameters, quick_test_analysis,
                           length_unit='nanometer', time_unit='picosecond')

        text = derived.read_text(encoding='utf-8')
        assert text == explicit.read_text(encoding='utf-8')
        assert 'nanometer' in text and 'picosecond' in text
        assert 'micrometer' not in text and 'nanosecond' not in text


class TestSummaryRecordsWhatReproducesIt:
    """The summary names the DICE version and the seed of the simulation."""

    def test_version_and_seed_of_a_seeded_simulation(self, tmp_path, quick_test_parameters):
        import dice

        seeded = simulate_quick_test(quick_test_parameters, quick_test_parameters['time series'], num_runs=3)
        summary_file = tmp_path / 'summary.txt'
        write_summary_file(seeded, summary_file, quick_test_parameters, analyze_simulation_results(seeded))
        text = summary_file.read_text(encoding='utf-8')

        assert seeded.seed == 3
        assert f"DICE version:\n    {dice.__version__}\nSeed:\n    3\n" in text

    def test_unseeded_simulation_says_so(self, tmp_path, quick_test_simulation, quick_test_parameters,
                                         quick_test_analysis):
        summary_file = tmp_path / 'summary.txt'
        write_summary_file(quick_test_simulation, summary_file, quick_test_parameters, quick_test_analysis)

        assert quick_test_simulation.seed is None
        assert "Seed:\n    none (another simulation" in summary_file.read_text(encoding='utf-8')

    @staticmethod
    def _simulate(params, seed):
        from dice.models.parameters import simulation_parameters_from_dict as adapt
        return run_monte_carlo_simulation(
            adapt(params, params['nominal diffusion coefficient'], params['nominal lifetime (tau)'],
                  params['nominal diffusion length']),
            params['x array'], params['time series'], params['noise series'],
            num_runs=2, multiprocessing=False, seed=seed)

    def test_seed_sequence_of_one_integer_records_the_integer(self, quick_test_parameters):
        assert self._simulate(quick_test_parameters, np.random.SeedSequence(41)).seed == 41

    def test_child_of_a_seed_sequence_records_what_reproduces_it(self, quick_test_parameters):
        """The entropy alone names the parent, whose noise is different."""
        first, second = np.random.SeedSequence(41).spawn(2)
        of_second = self._simulate(quick_test_parameters, second)

        assert of_second.seed == "SeedSequence(entropy=41, spawn_key=(1,))"
        assert self._simulate(quick_test_parameters, first).seed == "SeedSequence(entropy=41, spawn_key=(0,))"

        again = self._simulate(quick_test_parameters, np.random.SeedSequence(entropy=41, spawn_key=(1,)))
        np.testing.assert_array_equal(again.d_estimates('wls'), of_second.d_estimates('wls'))
        parent = self._simulate(quick_test_parameters, 41)
        assert not np.array_equal(parent.d_estimates('wls'), of_second.d_estimates('wls'))

    def test_seed_sequence_of_several_integers_is_not_recorded_as_unseeded(self, quick_test_parameters, tmp_path):
        result = self._simulate(quick_test_parameters, np.random.SeedSequence([1, 2]))
        assert result.seed == "SeedSequence(entropy=[1, 2], spawn_key=())"

        summary_file = tmp_path / 'summary.txt'
        write_summary_file(result, summary_file, quick_test_parameters, analyze_simulation_results(result))
        assert "Seed:\n    SeedSequence(entropy=[1, 2], spawn_key=())\n" in summary_file.read_text(encoding='utf-8')


class TestColumnsNameTheirUnits:
    """Every column with a unit names it, and results_column finds a column without the unit."""

    @pytest.fixture
    def nanometer_frame(self, tmp_path, quick_test_simulation):
        return export_collated_results(quick_test_simulation, tmp_path / 'results.csv',
                                       length_unit='nanometer', time_unit='picosecond')

    def test_units_follow_the_units_of_the_simulation(self, nanometer_frame):
        columns = list(nanometer_frame.columns)
        for expected in ['nominal diffusion coeff [nm²/ps]', 'nominal lifetime [ps]',
                         'nominal diffusion length [nm]', 'nominal sigma^2_0 [nm²]',
                         'estimated sigma^2_0 [nm²]', 'weighted fit diffusion slope (2D) [nm²/ps]',
                         'weighted fit diffusion slope stderr [nm²/ps]', 'weighted fit intercept [nm²]',
                         'weighted fit intercept stderr [nm²]', 'weighted fit diffusion coeff [cm²/s]',
                         'nominal diffusion coeff [cm²/s]']:
            assert expected in columns
        without_unit = [column for column in columns if column_unit(column) is None]
        assert without_unit == ['run number', 'nominal CNR', 'estimated CNR']

    def test_results_column_finds_the_column_from_its_name(self, nanometer_frame):
        columns = nanometer_frame.columns
        assert results_column(columns, 'weighted fit diffusion slope') == 'weighted fit diffusion slope (2D) [nm²/ps]'
        assert results_column(columns, 'weighted fit diffusion slope stderr') == (
            'weighted fit diffusion slope stderr [nm²/ps]')
        assert results_column(columns, 'weighted fit intercept') == 'weighted fit intercept [nm²]'
        assert results_column(columns, 'nominal CNR') == 'nominal CNR'
        # Two columns hold the nominal diffusion coefficient; the first is in the units of the slopes.
        assert results_column(columns, 'nominal diffusion coeff') == 'nominal diffusion coeff [nm²/ps]'
        with pytest.raises(KeyError, match="no column 'weighted fit diffusion'"):
            results_column(columns, 'weighted fit diffusion')

    def test_slope_is_twice_the_diffusion_coefficient_in_the_named_units(self, nanometer_frame):
        """The name says 2D; the values agree with the column in cm²/s through the units both names give."""
        slope = nanometer_frame[results_column(nanometer_frame.columns, 'weighted fit diffusion slope')]
        in_cm2_per_s = nanometer_frame['weighted fit diffusion coeff [cm²/s]']
        nm2_per_ps_in_cm2_per_s = (1e-7) ** 2 / 1e-12
        assert (slope / 2 * nm2_per_ps_in_cm2_per_s).to_numpy() == pytest.approx(in_cm2_per_s.to_numpy(), rel=1e-12)

    def test_output_units_equal_to_the_simulation_units_give_one_nominal_column(self, tmp_path, quick_test_simulation):
        from dice.models.parameters import OutputUnitPreferences
        frame = export_collated_results(
            quick_test_simulation, tmp_path / 'results.csv', length_unit='micrometer', time_unit='nanosecond',
            output_units=OutputUnitPreferences(diffusion_length='micrometer', diffusion_time='nanosecond'))
        assert [c for c in frame.columns if c.startswith('nominal diffusion coeff')] == [
            'nominal diffusion coeff [μm²/ns]']

    def test_results_file_of_dice_1_2_is_read_with_the_same_call(self):
        """The columns of 1.2.0 have no units; a script written with results_column reads both."""
        columns = ['run number', 'nominal diffusion coeff', 'nominal lifetime', 'weighted fit diffusion slope',
                   'weighted fit diffusion slope stderr', 'nominal diffusion coeff [cm²/s]']
        assert results_column(columns, 'weighted fit diffusion slope') == 'weighted fit diffusion slope'
        assert results_column(columns, 'nominal diffusion coeff') == 'nominal diffusion coeff'
        assert column_unit('weighted fit diffusion slope') is None


class TestResultsFileEncoding:
    """The results file is UTF-8 with a byte-order mark, which spreadsheet programs need for 'μ' and '²'."""

    def test_file_starts_with_the_mark_and_reads_back_without_it(self, tmp_path, quick_test_simulation):
        import csv

        import pandas as pd

        results_file = tmp_path / 'results.csv'
        exported = export_collated_results(quick_test_simulation, results_file,
                                           length_unit='micrometer', time_unit='nanosecond')

        assert results_file.read_bytes().startswith(b'\xef\xbb\xbf')
        assert list(pd.read_csv(results_file).columns) == list(exported.columns)
        assert pd.read_csv(results_file).columns[0] == 'run number'
        with open(results_file, newline='', encoding='utf-8-sig') as handle:
            assert next(csv.reader(handle)) == list(exported.columns)
        assert 'nominal diffusion coeff [μm²/ns]' in exported.columns
