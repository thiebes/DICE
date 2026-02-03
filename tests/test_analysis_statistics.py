"""
Tests for dice.analysis.statistics module.
"""

import pytest
import numpy as np
import pandas as pd
from dice.analysis.statistics import (
    calculate_precision,
    calculate_accuracy_metrics,
    estimates_precision,
    analyze_simulation_results,
    precision_counts,
    calculate_confidence_intervals,
    analyze_cnr_dependence,
)
from dice.models.results import RunResult, MonteCarloOutput

# Import mock classes for testing
from dice.utils.legacy_compatibility import MockSimulationParameters as SimulationParameters


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


class TestEstimatesPrecision:
    """Test legacy precision function."""
    
    def test_dataframe_precision(self):
        """Test precision calculation from DataFrame."""
        df = pd.DataFrame({
            # Using values clearly outside bounds: 0.89 < 0.9 and 1.11 > 1.1
            'weighted fit diffusion coeff [cm^2/s]': [0.95, 1.05, 0.89, 1.11],
            'unweighted fit diffusion coeff [cm^2/s]': [0.92, 1.08, 0.88, 1.12],
            'nominal diffusion coeff [cm^2/s]': [1.0, 1.0, 1.0, 1.0]
        })
        
        result = estimates_precision(df, proximity_level=0.1)
        
        assert '% fits within proximity' in result
        # Now truly 2/4 values (0.95, 1.05) are within [0.9, 1.1]
        assert result['% fits within proximity']['weighted fit'] == 50.0  # 2/4
        # 2/4 values (0.92, 1.08) are within [0.9, 1.1]  
        assert result['% fits within proximity']['unweighted fit'] == 50.0  # 2/4
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="DataFrame is empty"):
            estimates_precision(df, 0.1)
    
    def test_missing_columns(self):
        """Test with missing columns."""
        df = pd.DataFrame({
            'nominal diffusion coeff [cm^2/s]': [1.0, 1.0]
        })
        
        result = estimates_precision(df, 0.1)
        
        assert result['% fits within proximity']['weighted fit'] == 0.0
        assert result['% fits within proximity']['unweighted fit'] == 0.0


class TestAnalyzeMonteCarloOutputs:
    """Test comprehensive simulation analysis."""
    
    def test_basic_analysis(self):
        """Test basic analysis of simulation results."""
        # Create mock simulation results
        params = SimulationParameters()
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
        params = SimulationParameters()
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


class TestAnalyzeCNRDependence:
    """Test CNR dependence analysis."""
    
    def test_cnr_analysis(self):
        """Test CNR dependence analysis."""
        # Create DataFrame with different CNR values
        data = []
        for cnr in [10, 20, 50, 100]:
            for _ in range(10):
                noise = 1.0 / cnr
                data.append({
                    'nominal CNR': cnr,
                    'weighted fit diffusion coeff [cm^2/s]': 1.0 + np.random.normal(0, noise),
                    'nominal diffusion coeff [cm^2/s]': 1.0
                })
        
        df = pd.DataFrame(data)
        
        analysis = analyze_cnr_dependence(df)
        
        assert 10.0 in analysis
        assert 20.0 in analysis
        assert 50.0 in analysis
        assert 100.0 in analysis
        
        # Higher CNR should have better precision
        assert analysis[100.0]['precision_10%'] > analysis[10.0]['precision_10%']
        
        # Check structure
        for cnr_val in [10.0, 20.0, 50.0, 100.0]:
            assert 'num_runs' in analysis[cnr_val]
            assert 'mean_ratio' in analysis[cnr_val]
            assert 'std_ratio' in analysis[cnr_val]
            assert 'precision_10%' in analysis[cnr_val]
            assert 'ci_95%_lower' in analysis[cnr_val]
            assert 'ci_95%_upper' in analysis[cnr_val]


class TestPrecisionCounts:
    """Test precision counting by CNR bins."""
    
    def test_binned_precision(self):
        """Test precision counting in CNR bins."""
        # Create DataFrame with various CNR values
        data = []
        for cnr in np.linspace(5, 50, 20):
            data.append({
                'nominal CNR': cnr,
                'weighted fit diffusion coeff [cm^2/s]': 1.0 + np.random.normal(0, 0.1/cnr),
                'unweighted fit diffusion coeff [cm^2/s]': 1.0 + np.random.normal(0, 0.15/cnr),
                'nominal diffusion coeff [cm^2/s]': 1.0
            })
        
        df = pd.DataFrame(data)
        cnr_bins = np.array([0, 10, 20, 30, 40, 50])
        proximity_levels = [0.1, 0.2]
        
        results = precision_counts(df, cnr_bins, proximity_levels)
        
        assert 'proximity_10%' in results
        assert 'proximity_20%' in results
        
        # Check that bins are present
        for prox_key in results:
            bins_found = list(results[prox_key].keys())
            assert len(bins_found) > 0
            
            for bin_key in bins_found:
                assert 'count' in results[prox_key][bin_key]
                assert 'ols_precision' in results[prox_key][bin_key]
                assert 'wls_precision' in results[prox_key][bin_key]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])