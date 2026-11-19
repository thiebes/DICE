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


class TestLoadAndPlotFromCsv:

    @pytest.fixture
    def exported_csv(self, tmp_path) -> Path:
        csv_file = tmp_path / 'exported.csv'
        export_collated_results(_output(WLS_SLOPES, OLS_SLOPES), csv_file)
        return csv_file

    @pytest.mark.parametrize('method', list(METHOD_ARGUMENTS))
    def test_histogram_receives_the_selected_column(self, method, exported_csv, plotted, tmp_path):
        interface = DiceInterface()
        interface.load_and_plot_from_csv(
            str(exported_csv), str(tmp_path / 'plot.png'), IMAGE_SETTINGS,
            **METHOD_ARGUMENTS[method])
        assert plotted == [_ratios(METHOD_SLOPES[method])]

    @pytest.mark.parametrize('method', list(METHOD_ARGUMENTS))
    def test_loaded_results_hold_each_columns_values_in_file_order(
            self, method, exported_csv, plotted, tmp_path):
        interface = DiceInterface()
        interface.load_and_plot_from_csv(
            str(exported_csv), str(tmp_path / 'plot.png'), IMAGE_SETTINGS,
            **METHOD_ARGUMENTS[method])

        runs = interface.last_result.run_results
        stored_wls = [s for s in WLS_SLOPES if s is not None]
        stored_ols = [s for s in OLS_SLOPES if s is not None]
        assert interface.last_result.num_runs == len(runs) == 3
        assert [r.wls_slope for r in runs] == stored_wls
        assert [r.ols_slope for r in runs] == stored_ols
        assert [r.nominal_diffusion_coefficient for r in runs] == [NOMINAL_D] * 3
        assert interface.last_parameters == {'nominal diffusion coefficient': NOMINAL_D,
                                             'filename slug': 'exported'}
        assert interface.last_result.estimator_keys == ('ols', 'wls')
        # The other fit method works from the stored slopes without reloading.
        assert interface.calculate_proximity_on_demand(0.1, **METHOD_ARGUMENTS['wls']) == 2 / 3
        assert interface.calculate_proximity_on_demand(0.1, **METHOD_ARGUMENTS['ols']) == 1 / 3

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
        assert [r.wls_slope for r in interface.last_result.run_results] == [None] * 3
        assert interface.last_result.estimator_keys == ('ols',)
        assert interface.calculate_proximity_on_demand(0.1, **METHOD_ARGUMENTS['wls']) is None

    def test_unknown_estimator_is_rejected(self, exported_csv, plotted, tmp_path):
        interface = DiceInterface()
        with pytest.raises(RuntimeError, match="unknown estimator 'nope'"):
            interface.load_and_plot_from_csv(
                str(exported_csv), str(tmp_path / 'plot.png'), IMAGE_SETTINGS, estimator='nope')
        assert plotted == []
        assert interface.last_result is None
