"""
Tests for dice.analysis.statistics module.
"""

import pytest
import warnings
import numpy as np
from dice.analysis.statistics import (
    calculate_precision,
    calculate_accuracy_metrics,
    analyze_simulation_results,
    calculate_confidence_intervals,
)
from dice.models.results import RunResult, MonteCarloOutput


def get_legacy_simulation_params():
    """
    Get legacy simulation parameters for testing.

    Uses MockSimulationParameters with deprecation warning suppressed.
    For new tests, prefer using the default_simulation_params fixture.
    """
    from dice.utils.legacy_compatibility import MockSimulationParameters
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
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
