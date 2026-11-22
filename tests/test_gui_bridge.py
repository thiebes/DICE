"""
The GUI bridge (dice_gui.dice_interface.DiceInterface) without Qt.

Pins which diffusion estimates each fit method reads, from results in memory
and from an exported CSV, and the messages for a missing estimator. The
histogram writer is replaced by a function that records the ratios it receives.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dice.io.results import export_collated_results
from dice.models.results import MonteCarloOutput, RunResult
from dice.utils.legacy_compatibility import create_parameters_from_legacy
from dice_gui.dice_interface import DiceInterface

NOMINAL_D = 0.01
# Per-run MSD slopes (2 D); None where that estimator produced no fit.
WLS_SLOPES = [0.02, 0.0216, None, 0.03]     # D / D_nominal: 1.0, 1.08, -, 1.5
OLS_SLOPES = [0.019, None, 0.017, 0.026]    # D / D_nominal: 0.95, -, 0.85, 1.3
# Keyword arguments that select each fit method.
METHOD_ARGUMENTS = {'wls': {'estimator': 'wls'}, 'ols': {'estimator': 'ols'}}
METHOD_SLOPES = {'wls': WLS_SLOPES, 'ols': OLS_SLOPES}
IMAGE_SETTINGS = {'image_type': 'png'}


def _output(wls_slopes, ols_slopes) -> MonteCarloOutput:
    parameters = create_parameters_from_legacy(
        {'sigma^2_0': 0.18, 'amplitude_0': 1.0, 'mu_0': 0.0},
        diffusion_coefficient=NOMINAL_D, lifetime=1.0, diffusion_length=0.1)
    runs = [
        RunResult(run_id=i, nominal_diffusion_coefficient=NOMINAL_D, nominal_lifetime=1.0,
                  nominal_diffusion_length=0.1, nominal_sigma2_0=0.18, noise_sigma=0.02,
                  cnr_0_estimate=50.0, estimated_sigma2_0=0.18,
                  wls_slope=wls, wls_slope_stderr=None if wls is None else 0.001,
                  ols_slope=ols, ols_slope_stderr=None if ols is None else 0.001)
        for i, (wls, ols) in enumerate(zip(wls_slopes, ols_slopes))
    ]
    return MonteCarloOutput(parameters=parameters, run_results=runs, num_runs=len(runs),
                            noise_values=[0.02])


def _interface_with(output: MonteCarloOutput) -> DiceInterface:
    interface = DiceInterface()
    interface.last_result = output
    interface.last_parameters = {'nominal diffusion coefficient': NOMINAL_D,
                                 'filename slug': 'bridge'}
    return interface


def _ratios(slopes):
    return (np.array([s / 2 for s in slopes if s is not None]) / NOMINAL_D).tolist()


@pytest.fixture
def plotted(monkeypatch):
    """Records the D / D_nominal ratios each histogram call receives."""
    import dice.visualization.histograms as histograms
    calls = []

    def record(simulation_result, filename, **options):
        calls.append(simulation_result['collated results']['d_est_over_d_nom'])

    monkeypatch.setattr(histograms, 'plot_accuracy_histogram', record)
    return calls


class TestProximityOnDemand:

    @pytest.mark.parametrize('method, level, expected', [
        ('wls', 0.1, 2 / 3), ('wls', 0.2, 2 / 3), ('wls', 0.6, 1.0),
        ('ols', 0.1, 1 / 3), ('ols', 0.2, 2 / 3), ('ols', 0.6, 1.0),
    ])
    def test_fraction_within_level_uses_the_selected_estimates(self, method, level, expected):
        interface = _interface_with(_output(WLS_SLOPES, OLS_SLOPES))
        fraction = interface.calculate_proximity_on_demand(level, **METHOD_ARGUMENTS[method])
        assert fraction == expected

    @pytest.mark.parametrize('method', list(METHOD_ARGUMENTS))
    def test_no_estimates_gives_none(self, method):
        interface = _interface_with(_output([None, None], [None, None]))
        assert interface.calculate_proximity_on_demand(0.1, **METHOD_ARGUMENTS[method]) is None

    def test_no_results_gives_none(self):
        assert DiceInterface().calculate_proximity_on_demand(0.1) is None


class TestRegeneratePlotFromMemory:

    @pytest.mark.parametrize('method', list(METHOD_ARGUMENTS))
    def test_histogram_receives_the_selected_estimates(self, method, plotted, tmp_path):
        interface = _interface_with(_output(WLS_SLOPES, OLS_SLOPES))
        interface.regenerate_plot_from_memory(
            str(tmp_path / 'plot.png'), IMAGE_SETTINGS, **METHOD_ARGUMENTS[method])
        assert plotted == [_ratios(METHOD_SLOPES[method])]

    @pytest.mark.parametrize('method, name', [('wls', 'WLS'), ('ols', 'OLS')])
    def test_missing_estimates_are_named(self, method, name, plotted, tmp_path):
        slopes = [0.02, 0.022]
        output = _output([None, None], slopes) if method == 'wls' else _output(slopes, [None, None])
        with pytest.raises(RuntimeError) as error:
            _interface_with(output).regenerate_plot_from_memory(
                str(tmp_path / 'plot.png'), IMAGE_SETTINGS, **METHOD_ARGUMENTS[method])
        assert str(error.value) == (
            f"Failed to regenerate plot: No {name} diffusion estimates found in results")
        assert plotted == []


class TestDiffusionCoefficientUnit:
    """The unit the GUI reports the standard deviation and bias of the estimates in."""

    @pytest.mark.parametrize('parameters, expected', [
        ({'length unit': 'nanometer', 'time unit': 'picosecond'}, 'nm²/ps'),
        ({'length_unit': 'micrometer', 'time_unit': 'nanosecond'}, 'μm²/ns'),
        ({}, 'μm²/ns'),
    ])
    def test_run_results_use_the_global_units(self, parameters, expected):
        interface = _interface_with(_output(WLS_SLOPES, OLS_SLOPES))
        interface.last_parameters |= parameters
        assert interface.diffusion_coefficient_unit() == (1.0, expected)

    def test_run_results_follow_the_summary_output_unit(self):
        interface = _interface_with(_output(WLS_SLOPES, OLS_SLOPES))
        interface.last_parameters |= {
            'length unit': 'micrometer', 'time unit': 'nanosecond',
            'output_diffusion_length_unit': 'centimeter', 'output_diffusion_time_unit': 'second'}
        scale, unit = interface.diffusion_coefficient_unit()
        assert unit == 'cm\u00b2/s'
        assert scale == pytest.approx(10, rel=1e-12)


class TestLoadAndPlotFromCsv:

    @pytest.fixture
    def exported_csv(self, tmp_path) -> Path:
        csv_file = tmp_path / 'exported.csv'
        export_collated_results(_output(WLS_SLOPES, OLS_SLOPES), csv_file, length_unit='micrometer', time_unit='nanosecond')
        return csv_file

    @pytest.mark.parametrize('method', list(METHOD_ARGUMENTS))
    def test_histogram_receives_the_selected_column(self, method, exported_csv, plotted, tmp_path):
        interface = DiceInterface()
        interface.load_and_plot_from_csv(
            str(exported_csv), str(tmp_path / 'plot.png'), IMAGE_SETTINGS,
            **METHOD_ARGUMENTS[method])
        assert plotted == [_ratios(METHOD_SLOPES[method])]

    @pytest.mark.parametrize('method', list(METHOD_ARGUMENTS))
    def test_loaded_results_keep_each_row_as_one_run(
            self, method, exported_csv, plotted, tmp_path):
        interface = DiceInterface()
        interface.load_and_plot_from_csv(
            str(exported_csv), str(tmp_path / 'plot.png'), IMAGE_SETTINGS,
            **METHOD_ARGUMENTS[method])

        runs = interface.last_result.run_results
        assert interface.last_result.num_runs == len(runs) == len(WLS_SLOPES)
        assert [r.wls_slope for r in runs] == WLS_SLOPES
        assert [r.ols_slope for r in runs] == OLS_SLOPES
        assert [r.nominal_diffusion_coefficient for r in runs] == [NOMINAL_D] * len(WLS_SLOPES)
        # 0.01 micrometer^2/ns is 0.1 cm^2/s, the unit of the labeled nominal column.
        assert interface.last_parameters == {
            'nominal diffusion coefficient': NOMINAL_D, 'filename slug': 'exported',
            'diffusion coefficient unit': 'cm²/s',
            'diffusion coefficient scale': pytest.approx(10, rel=1e-12),
        }
        assert interface.last_result.estimator_keys == ('ols', 'wls')
        # The other fit method works from the stored slopes without reloading.
        assert interface.calculate_proximity_on_demand(0.1, **METHOD_ARGUMENTS['wls']) == 2 / 3
        assert interface.calculate_proximity_on_demand(0.1, **METHOD_ARGUMENTS['ols']) == 1 / 3

    def test_gap_in_one_estimator_does_not_drop_the_others_runs(self, plotted, tmp_path):
        # WLS has one gap and OLS none, so the two columns have different counts.
        wls = [0.02, None, 0.03, 0.021]
        ols = [0.019, 0.0216, 0.017, 0.026]   # D / D_nominal: 0.95, 1.08, 0.85, 1.3
        csv_file = tmp_path / 'one_gap.csv'
        export_collated_results(_output(wls, ols), csv_file, length_unit='micrometer', time_unit='nanosecond')

        interface = DiceInterface()
        interface.load_and_plot_from_csv(
            str(csv_file), str(tmp_path / 'plot.png'), IMAGE_SETTINGS, **METHOD_ARGUMENTS['wls'])

        assert [r.ols_slope for r in interface.last_result.run_results] == ols
        # Two of the four OLS estimates lie within 10% of the nominal value.
        assert interface.calculate_proximity_on_demand(0.1, **METHOD_ARGUMENTS['ols']) == 2 / 4

    def test_missing_column_is_named(self, exported_csv, plotted, tmp_path):
        frame = pd.read_csv(exported_csv)
        ols_only = tmp_path / 'ols_only.csv'
        frame[[c for c in frame.columns if not c.startswith('weighted fit')]].to_csv(
            ols_only, index=False)

        interface = DiceInterface()
        with pytest.raises(ValueError) as error:
            interface.load_and_plot_from_csv(
                str(ols_only), str(tmp_path / 'plot.png'), IMAGE_SETTINGS,
                **METHOD_ARGUMENTS['wls'])
        assert str(error.value).startswith(
            "CSV missing WLS slope column. Available columns: ['run number',")
        assert plotted == []
        assert interface.last_result is None

        interface.load_and_plot_from_csv(
            str(ols_only), str(tmp_path / 'plot.png'), IMAGE_SETTINGS, **METHOD_ARGUMENTS['ols'])
        assert plotted == [_ratios(OLS_SLOPES)]
        assert [r.wls_slope for r in interface.last_result.run_results] == [None] * len(OLS_SLOPES)
        assert interface.last_result.estimator_keys == ('ols',)
        assert interface.calculate_proximity_on_demand(0.1, **METHOD_ARGUMENTS['wls']) is None

    def test_statistics_unit_of_loaded_results_is_the_labeled_column_unit(
            self, plotted, tmp_path):
        from dice.models.parameters import OutputUnitPreferences
        csv_file = tmp_path / 'nm_per_ps.csv'
        export_collated_results(
            _output(WLS_SLOPES, OLS_SLOPES), csv_file, length_unit='micrometer', time_unit='nanosecond',
            output_units=OutputUnitPreferences(diffusion_length='nanometer', diffusion_time='picosecond'))
        interface = DiceInterface()
        interface.load_and_plot_from_csv(
            str(csv_file), str(tmp_path / 'plot.png'), IMAGE_SETTINGS, **METHOD_ARGUMENTS['wls'])
        # 1 micrometer^2/ns = 1e6 nm^2 / 1e3 ps = 1000 nm^2/ps
        scale, unit = interface.diffusion_coefficient_unit()
        assert unit == 'nm²/ps'
        assert scale == pytest.approx(1000, rel=1e-12)

    def test_statistics_have_no_unit_when_the_csv_has_no_labeled_column(
            self, exported_csv, plotted, tmp_path):
        frame = pd.read_csv(exported_csv)
        unlabeled = tmp_path / 'unlabeled.csv'
        frame[[c for c in frame.columns if '[' not in c]].to_csv(unlabeled, index=False)
        interface = DiceInterface()
        interface.load_and_plot_from_csv(
            str(unlabeled), str(tmp_path / 'plot.png'), IMAGE_SETTINGS, **METHOD_ARGUMENTS['wls'])
        assert interface.diffusion_coefficient_unit() == (1.0, None)

    def test_unknown_estimator_is_rejected(self, exported_csv, plotted, tmp_path):
        interface = DiceInterface()
        with pytest.raises(RuntimeError, match="unknown estimator 'nope'"):
            interface.load_and_plot_from_csv(
                str(exported_csv), str(tmp_path / 'plot.png'), IMAGE_SETTINGS, estimator='nope')
        assert plotted == []
        assert interface.last_result is None


def _quick_test_parameters(**overrides) -> dict:
    import ast
    params = ast.literal_eval(
        (Path(__file__).parent.parent / 'data' / 'examples' / 'quick_test.txt').read_text())
    params.update({'number of runs': 2, 'multiprocessing': False}, **overrides)
    return params


class TestRunValidation:

    @pytest.mark.parametrize('overrides, message', [
        ({'image type': 'bmp'}, "Unknown image type 'bmp'"),
        ({'output_length_unit': 'nanometer'}, "'output_length_unit' is not supported"),
    ])
    def test_invalid_output_setting_fails_before_the_simulation(
            self, tmp_path, monkeypatch, overrides, message):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(RuntimeError, match=f"Simulation failed: {message}"):
            DiceInterface().run_simulation(_quick_test_parameters(**overrides))
        assert list(tmp_path.iterdir()) == []

    def test_output_diffusion_units_reach_the_csv_and_summary(self, tmp_path, monkeypatch):
        """The GUI export path converts like the CLI, though the GUI form has no control for the keys."""
        monkeypatch.chdir(tmp_path)
        params = _quick_test_parameters(**{
            'output_diffusion_length_unit': 'nanometer', 'output_diffusion_time_unit': 'picosecond'})
        DiceInterface().run_simulation(params)
        output_dir = tmp_path / 'output' / 'quick_test'
        frame = pd.read_csv(output_dir / 'quick_test.csv')
        assert frame['nominal diffusion coeff [nm\u00b2/ps]'].to_numpy() == pytest.approx(10.0, rel=1e-12)
        summary = (output_dir / 'quick_test_summary.txt').read_text(encoding='utf-8')
        assert "Nominal diffusion coefficient:\n    10.00000 nanometer^2/picosecond\n" in summary



class TestFiguresAreClosed:
    """Each histogram the bridge draws is closed after saving, so a long GUI session holds no figures."""

    def test_csv_load_and_regeneration_leave_no_open_figure(self, tmp_path):
        import matplotlib.pyplot as plt

        csv_file = tmp_path / 'exported.csv'
        export_collated_results(_output(WLS_SLOPES, OLS_SLOPES), csv_file, length_unit='micrometer', time_unit='nanosecond')
        plt.close('all')
        interface = DiceInterface()

        interface.load_and_plot_from_csv(
            str(csv_file), str(tmp_path / 'loaded.png'), IMAGE_SETTINGS, **METHOD_ARGUMENTS['wls'])
        for method in ('wls', 'ols'):
            interface.regenerate_plot_from_memory(
                str(tmp_path / f'again_{method}.png'), IMAGE_SETTINGS, **METHOD_ARGUMENTS[method])

        assert plt.get_fignums() == []
        assert all((tmp_path / name).stat().st_size > 0
                   for name in ('loaded.png', 'again_wls.png', 'again_ols.png'))
