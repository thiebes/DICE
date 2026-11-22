"""
Tests for dice.analysis.statistics module.
"""

import pytest
import numpy as np
from dice.analysis.statistics import (
    calculate_precision,
    calculate_accuracy_metrics,
    analyze_simulation_results,
    calculate_confidence_intervals,
)
from dice.models.results import RunResult, MonteCarloOutput


def get_legacy_simulation_params():
    """Simulation parameters with fixed values, in the structure the engine reads."""
    from dice.utils.legacy_compatibility import MockSimulationParameters
    return MockSimulationParameters()


class TestCalculatePrecision:
    """Test precision calculation."""
    
    def test_perfect_precision(self):
        """Test with perfect estimates."""
        estimates = np.array([1.0, 1.0, 1.0, 1.0])
        nominal = 1.0
        
        precision = calculate_precision(estimates, nominal, proximity_level=0.1)
        assert precision == 1.0  # All within ±10%
    
    def test_no_precision(self):
        """Test with no estimates within proximity."""
        estimates = np.array([2.0, 3.0, 0.5, 0.3])
        nominal = 1.0
        
        precision = calculate_precision(estimates, nominal, proximity_level=0.1)
        assert precision == 0.0  # None within ±10%
    
    def test_partial_precision(self):
        """Test with partial precision."""
        estimates = np.array([0.95, 1.05, 1.15, 0.85])
        nominal = 1.0
        
        precision = calculate_precision(estimates, nominal, proximity_level=0.1)
        assert precision == 0.5  # 2 out of 4 within ±10%
    
    def test_different_proximity_levels(self):
        """Test with different proximity levels."""
        estimates = np.array([0.9, 1.1, 0.8, 1.2])
        nominal = 1.0
        
        precision_10 = calculate_precision(estimates, nominal, 0.1)
        precision_20 = calculate_precision(estimates, nominal, 0.2)
        precision_30 = calculate_precision(estimates, nominal, 0.3)
        
        assert precision_10 == 0.5  # 2/4 within ±10%
        assert precision_20 == 1.0  # 4/4 within ±20%
        assert precision_30 == 1.0  # 4/4 within ±30%
    
    def test_empty_estimates(self):
        """Test with empty estimates."""
        estimates = np.array([])
        nominal = 1.0
        
        precision = calculate_precision(estimates, nominal, 0.1)
        assert precision == 0.0
    
    def test_zero_nominal(self):
        """Test with zero nominal value."""
        estimates = np.array([0.1, 0.2, 0.3])
        nominal = 0.0
        
        precision = calculate_precision(estimates, nominal, 0.1)
        assert precision == 0.0


class TestCalculateAccuracyMetrics:
    """Test accuracy metrics calculation."""
    
    def test_perfect_accuracy(self):
        """Test with perfect estimates."""
        estimates = np.array([1.0, 1.0, 1.0])
        nominal = 1.0
        
        metrics = calculate_accuracy_metrics(estimates, nominal)
        
        assert metrics['mean'] == 1.0
        assert metrics['median'] == 1.0
        assert metrics['std'] == 0.0
        assert metrics['bias'] == 0.0
        assert metrics['relative_bias'] == 0.0
        assert metrics['rmse'] == 0.0
        assert metrics['relative_rmse'] == 0.0
    
    def test_biased_estimates(self):
        """Test with biased estimates."""
        estimates = np.array([1.1, 1.2, 1.3])
        nominal = 1.0
        
        metrics = calculate_accuracy_metrics(estimates, nominal)
        
        assert metrics['mean'] == pytest.approx(1.2)
        assert metrics['median'] == 1.2
        assert metrics['bias'] == pytest.approx(0.2)
        assert metrics['relative_bias'] == pytest.approx(0.2)
        assert metrics['rmse'] > 0
    
    def test_variable_estimates(self):
        """Test with variable estimates."""
        estimates = np.array([0.8, 1.0, 1.2])
        nominal = 1.0
        
        metrics = calculate_accuracy_metrics(estimates, nominal)
        
        assert metrics['mean'] == pytest.approx(1.0)
        assert metrics['median'] == 1.0
        assert metrics['bias'] == pytest.approx(0.0)
        # Standard deviation of [0.8, 1.0, 1.2] is ~0.163, not 0.2
        assert metrics['std'] == pytest.approx(0.16329931618554516)
        assert metrics['rmse'] == pytest.approx(np.sqrt(0.08/3))
    
    def test_empty_estimates(self):
        """Test with empty estimates."""
        estimates = np.array([])
        nominal = 1.0
        
        metrics = calculate_accuracy_metrics(estimates, nominal)
        
        assert np.isnan(metrics['mean'])
        assert np.isnan(metrics['median'])
        assert np.isnan(metrics['std'])


class TestAnalyzeMonteCarloOutputs:
    """Test comprehensive simulation analysis."""
    
    def test_basic_analysis(self):
        """Test basic analysis of simulation results."""
        # Create mock simulation results
        params = get_legacy_simulation_params()
        params.physics.diffusion_coefficient = 1.0
        params.physics.lifetime = 10.0
        params.physics.diffusion_length = np.sqrt(10.0)
        
        runs = []
        for i in range(10):
            run = RunResult(
                run_id=i,
                nominal_diffusion_coefficient=1.0,
                nominal_lifetime=10.0,
                nominal_diffusion_length=np.sqrt(10.0),
                noise_sigma=0.05,
                cnr_0_estimate=20.0,
                nominal_sigma2_0=1.0,
                estimated_sigma2_0=1.05,
                ols_slope=2.0 + np.random.normal(0, 0.1),  # 2*D with noise
                ols_slope_stderr=0.05,
                ols_intercept=0.0,
                ols_intercept_stderr=0.01,
                wls_slope=2.0 + np.random.normal(0, 0.08),  # Slightly better
                wls_slope_stderr=0.04,
                wls_intercept=0.0,
                wls_intercept_stderr=0.008
            )
            runs.append(run)
        
        result = MonteCarloOutput(
            parameters=params,
            run_results=runs,
            num_runs=10,
            noise_values=[0.05]
        )
        
        analysis = analyze_simulation_results(result, proximity_levels=[0.05, 0.1, 0.2])
        
        assert 'summary' in analysis
        assert analysis['summary']['total_runs'] == 10
        assert analysis['summary']['nominal_diffusion_coefficient'] == 1.0
        
        assert 'precision' in analysis
        assert '5%' in analysis['precision']
        assert '10%' in analysis['precision']
        assert '20%' in analysis['precision']
        
        assert 'accuracy' in analysis
        assert 'ols' in analysis['accuracy']
        assert 'wls' in analysis['accuracy']
        
        assert 'statistics' in analysis
        assert analysis['statistics']['ols']['num_successful'] == 10
        assert analysis['statistics']['ols']['success_rate'] == 1.0
    
    def test_multiple_noise_levels(self):
        """Test analysis with multiple noise levels."""
        params = get_legacy_simulation_params()
        params.physics.diffusion_coefficient = 1.0
        
        runs = []
        for noise in [0.01, 0.05, 0.1]:
            for i in range(3):
                run = RunResult(
                    run_id=len(runs),
                    nominal_diffusion_coefficient=1.0,
                    nominal_lifetime=10.0,
                    nominal_diffusion_length=np.sqrt(10.0),
                    noise_sigma=noise,
                    cnr_0_estimate=1.0/noise,
                    nominal_sigma2_0=1.0,
                    estimated_sigma2_0=1.0,
                    ols_slope=2.0 + np.random.normal(0, noise),
                    ols_slope_stderr=noise/2,
                    ols_intercept=0.0,
                    ols_intercept_stderr=noise/10,
                    wls_slope=2.0 + np.random.normal(0, noise*0.8),
                    wls_slope_stderr=noise*0.4,
                    wls_intercept=0.0,
                    wls_intercept_stderr=noise/12
                )
                runs.append(run)
        
        result = MonteCarloOutput(
            parameters=params,
            run_results=runs,
            num_runs=9,
            noise_values=[0.01, 0.05, 0.1]
        )
        
        analysis = analyze_simulation_results(result)
        
        assert 'noise_analysis' in analysis
        assert 'noise_0.01' in analysis['noise_analysis']
        assert 'noise_0.05' in analysis['noise_analysis']
        assert 'noise_0.1' in analysis['noise_analysis']
        
        # Check CNR values
        assert analysis['noise_analysis']['noise_0.01']['cnr'] == pytest.approx(100.0)
        assert analysis['noise_analysis']['noise_0.05']['cnr'] == pytest.approx(20.0)
        assert analysis['noise_analysis']['noise_0.1']['cnr'] == pytest.approx(10.0)


class TestCalculateConfidenceIntervals:
    """Test confidence interval calculation."""
    
    def test_95_percent_ci(self):
        """Test 95% confidence interval."""
        np.random.seed(42)
        estimates = np.random.normal(1.0, 0.1, 1000)
        
        lower, upper = calculate_confidence_intervals(estimates, 0.95)
        
        # Should contain approximately 95% of the data
        within_ci = np.sum((estimates >= lower) & (estimates <= upper))
        assert within_ci / len(estimates) > 0.94
        assert within_ci / len(estimates) < 0.96
    
    def test_90_percent_ci(self):
        """Test 90% confidence interval."""
        estimates = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
        
        lower, upper = calculate_confidence_intervals(estimates, 0.90)
        
        assert lower == pytest.approx(0.82, abs=0.01)
        assert upper == pytest.approx(1.18, abs=0.01)
    
    def test_empty_estimates(self):
        """Test with empty estimates."""
        estimates = np.array([])
        
        lower, upper = calculate_confidence_intervals(estimates, 0.95)
        
        assert np.isnan(lower)
        assert np.isnan(upper)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

class TestNoiseAnalysisCnr:
    """Per-noise CNR in the analysis is amplitude_0 / noise sigma, from a real simulation."""

    def test_cnr_scales_with_amplitude(self):
        from pathlib import Path
        from dice.analysis.simulation import run_monte_carlo_simulation
        from dice.io.parameters import open_parameters
        from dice.utils.legacy_compatibility import create_parameters_from_legacy
        params = open_parameters(Path(__file__).parent.parent / 'data' / 'examples' / 'quick_test.txt')
        amplitude_0 = 2.0
        sim_params = create_parameters_from_legacy(
            params | {'amplitude_0': amplitude_0}, params['nominal diffusion coefficient'],
            params['nominal lifetime (tau)'], params['nominal diffusion length'])
        noise_values = [0.04, 0.08]
        result = run_monte_carlo_simulation(
            sim_params, params['x array'], params['time series'], noise_values,
            num_runs=2, multiprocessing=False, seed=6)

        noise_analysis = analyze_simulation_results(result)['noise_analysis']

        for noise_sigma in noise_values:
            assert noise_analysis[f'noise_{noise_sigma}']['cnr'] == pytest.approx(amplitude_0 / noise_sigma)


class TestCustomProximityLevels:
    """Each requested proximity level gets its own key, named by its exact percentage."""

    def test_levels_are_keyed_by_their_exact_percentage(self):
        from pathlib import Path
        from dice.analysis.simulation import run_monte_carlo_simulation
        from dice.io.parameters import open_parameters
        from dice.utils.legacy_compatibility import create_parameters_from_legacy
        params = open_parameters(Path(__file__).parent.parent / 'data' / 'examples' / 'quick_test.txt')
        sim_params = create_parameters_from_legacy(
            params, params['nominal diffusion coefficient'],
            params['nominal lifetime (tau)'], params['nominal diffusion length'])
        result = run_monte_carlo_simulation(
            sim_params, params['x array'], params['time series'], params['noise series'],
            num_runs=20, multiprocessing=False, seed=11)
        # 0.29 * 100 and 0.565 * 100 fall just below 29 and 56.5 in binary floating point.
        levels = {0.29: '29%', 0.565: '56.5%', 0.57: '57%', 0.025: '2.5%'}

        precision = analyze_simulation_results(result, proximity_levels=list(levels))['precision']

        assert sorted(precision) == sorted(levels.values())
        nominal_d = params['nominal diffusion coefficient']
        for level, label in levels.items():
            for key in ('ols', 'wls'):
                assert precision[label][key] == calculate_precision(
                    result.d_estimates(key), nominal_d, level)


class TestCnrBins:
    """A simulation with a noise range is analyzed in bins of nominal CNR with equal numbers of runs."""

    BIN_KEYS = ['cnr_min', 'cnr_max', 'num_runs', 'precision', 'precision_intervals', 'accuracy', 'statistics']

    @staticmethod
    def _range_output(noise_sigmas, amplitude_0=1.0, failed_wls_runs=()):
        """Runs whose D estimates deviate from the nominal 0.01 in proportion to their noise sigma."""
        from dice.models.parameters import NoiseRange
        from dice.utils.legacy_compatibility import create_parameters_from_legacy
        parameters = create_parameters_from_legacy(
            {'sigma^2_0': 0.18, 'amplitude_0': amplitude_0, 'mu_0': 0.0},
            diffusion_coefficient=0.01, lifetime=1.0, diffusion_length=0.1)
        runs = []
        for run_id, sigma in enumerate(noise_sigmas):
            sign = 1 if run_id % 2 else -1
            fit = {'slope': 0.02 * (1 + sign * 4 * sigma), 'slope_stderr': 0.001,
                   'intercept': 0.0, 'intercept_stderr': 0.001}
            fields = {f'ols_{name}': value for name, value in fit.items()}
            if run_id not in failed_wls_runs:
                fields |= {f'wls_{name}': value * (1.01 if name == 'slope' else 1) for name, value in fit.items()}
            runs.append(RunResult(
                run_id=run_id, nominal_diffusion_coefficient=0.01, nominal_lifetime=1.0,
                nominal_diffusion_length=0.1, nominal_sigma2_0=0.18, noise_sigma=sigma,
                cnr_0_estimate=amplitude_0 / sigma, **fields))
        return MonteCarloOutput(parameters=parameters, run_results=runs, num_runs=len(runs),
                                noise_values=list(noise_sigmas), noise_range=NoiseRange(0.01, 0.05))

    @pytest.mark.parametrize('run_count, bins', [
        (0, 1), (1, 1), (39, 1), (40, 2), (59, 2), (60, 3), (99, 4), (100, 5), (101, 5), (10000, 5)])
    def test_bin_count_rule(self, run_count, bins):
        from dice.analysis.statistics import MAX_CNR_BINS, MIN_RUNS_PER_CNR_BIN, cnr_bin_count
        assert (MAX_CNR_BINS, MIN_RUNS_PER_CNR_BIN) == (5, 20)
        assert cnr_bin_count(run_count) == bins

    def test_bins_hold_every_run_once_in_order_of_cnr(self):
        from dice.analysis.statistics import runs_by_cnr_bin
        rng = np.random.default_rng(3)
        sigmas = list(rng.uniform(0.01, 0.05, size=63))
        sigmas[10] = sigmas[40]  # a tie keeps run order
        bins = runs_by_cnr_bin(self._range_output(sigmas, amplitude_0=2.0))

        assert [len(runs) for runs in bins] == [21, 21, 21]
        ordered = [run for runs in bins for run in runs]
        assert sorted(run.run_id for run in ordered) == list(range(63))
        cnr = [2.0 / run.noise_sigma for run in ordered]
        assert cnr == sorted(cnr)
        tied = [run.run_id for run in ordered if run.noise_sigma == sigmas[40]]
        assert tied == [10, 40]

    def test_bin_sizes_differ_by_at_most_one(self):
        from dice.analysis.statistics import runs_by_cnr_bin
        bins = runs_by_cnr_bin(self._range_output(list(np.linspace(0.05, 0.01, 103))))
        assert [len(runs) for runs in bins] == [21, 21, 21, 20, 20]

    def test_each_bin_reports_the_statistics_of_its_runs(self):
        from dice.analysis.statistics import calculate_accuracy_metrics
        sigmas = list(np.random.default_rng(5).uniform(0.01, 0.05, size=45))
        output = self._range_output(sigmas, amplitude_0=2.0, failed_wls_runs=(0, 1, 2))
        analysis = analyze_simulation_results(output, proximity_levels=[0.05, 0.2])

        bins = analysis['cnr_bins']
        assert [list(entry) for entry in bins] == [self.BIN_KEYS] * 2
        assert [entry['num_runs'] for entry in bins] == [23, 22]
        assert bins[0]['cnr_max'] <= bins[1]['cnr_min']

        # Recompute each bin independently from the runs whose CNR falls in its interval.
        run_cnr = np.array([2.0 / s for s in sigmas])
        for entry in bins:
            in_bin = [run for run, cnr in zip(output.run_results, run_cnr)
                      if entry['cnr_min'] <= cnr <= entry['cnr_max']]
            assert len(in_bin) == entry['num_runs']
            assert entry['cnr_min'] == pytest.approx(min(2.0 / run.noise_sigma for run in in_bin))
            assert entry['cnr_max'] == pytest.approx(max(2.0 / run.noise_sigma for run in in_bin))
            for key in ('wls', 'ols'):
                d = np.array([getattr(run, f'{key}_slope') / 2 for run in in_bin
                              if getattr(run, f'{key}_slope') is not None])
                assert entry['accuracy'][key] == pytest.approx(calculate_accuracy_metrics(d, 0.01))
                for label, level in (('5%', 0.05), ('20%', 0.2)):
                    assert entry['precision'][label][key] == pytest.approx(
                        np.mean(np.abs(d / 0.01 - 1) <= level))
                assert entry['statistics'][key] == {
                    'num_successful': len(d), 'num_failed': len(in_bin) - len(d),
                    'success_rate': len(d) / len(in_bin)}
        assert sum(entry['statistics']['wls']['num_failed'] for entry in bins) == 3

    def test_noise_range_has_no_per_level_analysis(self):
        analysis = analyze_simulation_results(self._range_output(list(np.linspace(0.01, 0.05, 30))))
        assert analysis['noise_analysis'] == {}
        assert analysis['summary']['total_runs'] == 30
        assert analysis['statistics']['wls']['num_successful'] == 30

    def test_fixed_noise_has_no_cnr_bins(self):
        import dataclasses
        fixed = dataclasses.replace(self._range_output([0.02] * 30), noise_values=[0.02], noise_range=None)
        assert 'cnr_bins' not in analyze_simulation_results(fixed)

    def test_no_runs_no_bins(self):
        analysis = analyze_simulation_results(self._range_output([]))
        assert analysis['cnr_bins'] == []


class TestUncertaintyOfTheStatistics:
    """A statistic of a finite number of runs comes with its uncertainty."""

    @pytest.mark.parametrize('count, total', [(0, 20), (1, 20), (19, 100), (265, 1500), (20, 20), (7, 7)])
    def test_interval_agrees_with_statsmodels(self, count, total):
        from statsmodels.stats.proportion import proportion_confint
        from dice.analysis.statistics import proportion_interval

        lower, upper = proportion_interval(count, total)
        expected = proportion_confint(count, total, alpha=0.05, method='wilson')

        assert (lower, upper) == pytest.approx(expected, abs=1e-12)
        assert 0.0 <= lower <= count / total <= upper <= 1.0
        assert upper - lower > 0  # also for 0 and for all of the estimates

    def test_interval_narrows_with_more_runs(self):
        from dice.analysis.statistics import proportion_interval

        widths = [upper - lower for lower, upper in
                  (proportion_interval(total // 5, total) for total in (20, 100, 500, 2500))]
        assert widths == sorted(widths, reverse=True)

    def test_no_estimates_give_no_interval(self):
        from dice.analysis.statistics import proportion_interval

        assert all(np.isnan(limit) for limit in proportion_interval(0, 0))

    def test_standard_error_is_that_of_the_mean(self):
        estimates = np.array([0.009, 0.011, 0.0105, 0.0098, 0.0102, 0.0089])
        metrics = calculate_accuracy_metrics(estimates, 0.01)

        assert metrics['count'] == 6
        assert metrics['standard_error'] == pytest.approx(estimates.std(ddof=1) / np.sqrt(6))
        assert np.isnan(calculate_accuracy_metrics(estimates[:1], 0.01)['standard_error'])
        empty = calculate_accuracy_metrics(np.array([]), 0.01)
        assert empty['count'] == 0 and np.isnan(empty['standard_error'])
