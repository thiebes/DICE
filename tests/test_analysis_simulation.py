"""
Tests for dice.analysis.simulation module.
"""

import pytest
import warnings
import numpy as np
from dice.analysis.simulation import (
    run_single_simulation,
    calculate_wls_weights,
    run_monte_carlo_simulation,
    scan_runner_compatibility,
)


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


class TestCalculateWLSWeights:
    """Test weight calculation for WLS fitting."""
    
    def test_basic_weights(self):
        """Test basic weight calculation."""
        sigma2_values = np.array([1.0, 2.0, 3.0, 4.0])
        # Different relative errors: 0.05, 0.1, 0.15, 0.2
        sigma2_errors = np.array([0.05, 0.2, 0.45, 0.8])
        
        weights = calculate_wls_weights(sigma2_values, sigma2_errors)
        
        assert len(weights) == 4
        # Weights should be normalized to sum to number of points
        assert np.sum(weights) == pytest.approx(4.0, rel=0.01)
        # Smaller relative errors should have larger weights
        assert weights[0] > weights[1] > weights[2] > weights[3]
    
    def test_zero_error_handling(self):
        """Test handling of zero errors."""
        sigma2_values = np.array([1.0, 2.0, 3.0])
        sigma2_errors = np.array([0.1, 0.0, 0.3])  # Zero error
        
        weights = calculate_wls_weights(sigma2_values, sigma2_errors)
        
        assert weights[1] == 0  # Zero weight for zero error
        assert weights[0] > 0
        assert weights[2] > 0
    
    def test_zero_value_handling(self):
        """Test handling of zero values."""
        sigma2_values = np.array([0.0, 2.0, 3.0])  # Zero value
        sigma2_errors = np.array([0.1, 0.2, 0.3])
        
        weights = calculate_wls_weights(sigma2_values, sigma2_errors)
        
        assert weights[0] == 0  # Zero weight for zero value
        assert weights[1] > 0
        assert weights[2] > 0
    
    def test_nan_handling(self):
        """Test handling of NaN errors."""
        sigma2_values = np.array([1.0, 2.0, 3.0])
        sigma2_errors = np.array([0.1, np.nan, 0.3])  # NaN error
        
        weights = calculate_wls_weights(sigma2_values, sigma2_errors)
        
        assert weights[1] == 0  # Zero weight for NaN
        assert weights[0] > 0
        assert weights[2] > 0
    
    def test_all_zero_fallback(self):
        """Test fallback when all weights would be zero."""
        sigma2_values = np.array([0.0, 0.0, 0.0])
        sigma2_errors = np.array([0.1, 0.2, 0.3])
        
        weights = calculate_wls_weights(sigma2_values, sigma2_errors)
        
        # Should fall back to uniform weights
        assert np.all(weights == 1.0)


class TestRunSingleSimulation:
    """Test single simulation runs."""
    
    def test_basic_simulation(self):
        """Test basic simulation run."""
        # Set up parameters
        params = get_legacy_simulation_params()
        params.profile.sigma2_0 = 1.0
        params.profile.amplitude_0 = 1.0
        params.profile.mu_0 = 0.0
        params.physics.diffusion_coefficient = 0.5
        params.physics.lifetime = 10.0
        params.physics.diffusion_length = np.sqrt(0.5 * 10.0)
        
        x_axis = np.linspace(-10, 10, 201)
        time_axis = np.array([0, 1, 2, 3, 4])
        
        result = run_single_simulation(
            run_id=0,
            x_axis=x_axis,
            time_axis=time_axis,
            parameters=params,
            noise_sigma=0.01,
            retain_profile_data=False
        )
        
        assert result.run_id == 0
        assert result.nominal_diffusion_coefficient == 0.5
        assert result.nominal_lifetime == 10.0
        assert result.noise_sigma == 0.01
        assert result.cnr_0_estimate > 0
        assert result.estimated_sigma2_0 > 0
        
        # Should have OLS and WLS results
        assert result.ols_slope is not None
        assert result.wls_slope is not None
    
    def test_single_timepoint(self):
        """Test with single time point (no diffusion fit)."""
        params = get_legacy_simulation_params()
        params.profile.sigma2_0 = 1.0
        params.profile.amplitude_0 = 1.0
        params.profile.mu_0 = 0.0
        params.physics.diffusion_coefficient = 0.5
        params.physics.lifetime = 10.0
        params.physics.diffusion_length = np.sqrt(5.0)
        
        x_axis = np.linspace(-10, 10, 201)
        time_axis = np.array([0])  # Single time point
        
        result = run_single_simulation(
            run_id=1,
            x_axis=x_axis,
            time_axis=time_axis,
            parameters=params,
            noise_sigma=0.05,
            retain_profile_data=False
        )
        
        assert result.run_id == 1
        # No diffusion fit possible with single time point
        assert result.ols_slope is None
        assert result.wls_slope is None
        assert result.cnr_0_estimate > 0
    
    def test_retain_profile_data(self):
        """Test retaining profile data."""
        params = get_legacy_simulation_params()
        params.profile.sigma2_0 = 1.0
        params.profile.amplitude_0 = 1.0
        params.profile.mu_0 = 0.0
        params.physics.diffusion_coefficient = 0.5
        params.physics.lifetime = 10.0
        params.physics.diffusion_length = np.sqrt(5.0)
        
        x_axis = np.linspace(-10, 10, 101)
        time_axis = np.array([0, 1, 2])
        
        result = run_single_simulation(
            run_id=2,
            x_axis=x_axis,
            time_axis=time_axis,
            parameters=params,
            noise_sigma=0.02,
            retain_profile_data=True
        )
        
        # Profile data should be retained
        assert hasattr(result, 'nominal_profiles')
        assert hasattr(result, 'noisy_profiles')
        assert hasattr(result, 'fitted_sigma2_t')
        assert hasattr(result, 'fitted_sigma2_stderrs')
        assert hasattr(result, 'weights')
        
        # Check shapes
        assert result.nominal_profiles.shape == (3, 101)
        assert result.noisy_profiles.shape == (3, 101)
        assert len(result.fitted_sigma2_t) == 3
    
    def test_high_noise(self):
        """Test with high noise level."""
        params = get_legacy_simulation_params()
        params.profile.sigma2_0 = 1.0
        params.profile.amplitude_0 = 1.0
        params.profile.mu_0 = 0.0
        params.physics.diffusion_coefficient = 0.5
        params.physics.lifetime = 10.0
        params.physics.diffusion_length = np.sqrt(5.0)
        
        x_axis = np.linspace(-10, 10, 201)
        time_axis = np.array([0, 1, 2])
        
        result = run_single_simulation(
            run_id=3,
            x_axis=x_axis,
            time_axis=time_axis,
            parameters=params,
            noise_sigma=0.5,  # High noise
            retain_profile_data=False
        )
        
        # Should still complete, but CNR will be low
        assert result.cnr_0_estimate < 10  # Low CNR due to high noise
        assert result.ols_slope is not None  # Should still attempt fitting


class TestRunMonteCarloSimulation:
    """Test Monte Carlo simulation runs."""
    
    def test_basic_monte_carlo(self):
        """Test basic Monte Carlo simulation."""
        params = get_legacy_simulation_params()
        params.profile.sigma2_0 = 1.0
        params.profile.amplitude_0 = 1.0
        params.profile.mu_0 = 0.0
        params.physics.diffusion_coefficient = 0.5
        params.physics.lifetime = 10.0
        params.physics.diffusion_length = np.sqrt(5.0)
        
        x_axis = np.linspace(-10, 10, 101)
        time_axis = np.array([0, 1, 2])
        noise_values = [0.01, 0.02]
        num_runs = 5
        
        result = run_monte_carlo_simulation(
            parameters=params,
            x_axis=x_axis,
            time_axis=time_axis,
            noise_values=noise_values,
            num_runs=num_runs,
            multiprocessing=False,  # Easier to test
            retain_profile_data=False
        )
        
        # Should have correct number of runs
        assert result.num_runs == 10  # 5 runs × 2 noise values
        assert len(result.run_results) == 10
        assert result.noise_values == noise_values
        
        # Check run IDs are sequential
        run_ids = [r.run_id for r in result.run_results]
        assert run_ids == list(range(10))
        
        # Check noise values are distributed correctly
        noise_sigmas = [r.noise_sigma for r in result.run_results]
        assert noise_sigmas.count(0.01) == 5
        assert noise_sigmas.count(0.02) == 5
    
    def test_single_noise_value(self):
        """Test with single noise value."""
        params = get_legacy_simulation_params()
        params.profile.sigma2_0 = 1.0
        params.profile.amplitude_0 = 1.0
        params.profile.mu_0 = 0.0
        params.physics.diffusion_coefficient = 0.5
        params.physics.lifetime = 10.0
        params.physics.diffusion_length = np.sqrt(5.0)
        
        x_axis = np.linspace(-10, 10, 101)
        time_axis = np.array([0, 1])
        
        result = run_monte_carlo_simulation(
            parameters=params,
            x_axis=x_axis,
            time_axis=time_axis,
            noise_values=[0.05],
            num_runs=3,
            multiprocessing=False
        )
        
        assert result.num_runs == 3
        assert len(result.run_results) == 3
        assert all(r.noise_sigma == 0.05 for r in result.run_results)
    
    def test_progress_callback(self):
        """Test progress callback functionality."""
        params = get_legacy_simulation_params()
        params.profile.sigma2_0 = 1.0
        params.profile.amplitude_0 = 1.0
        params.profile.mu_0 = 0.0
        params.physics.diffusion_coefficient = 0.5
        params.physics.lifetime = 10.0
        params.physics.diffusion_length = np.sqrt(5.0)
        
        x_axis = np.linspace(-10, 10, 51)
        time_axis = np.array([0, 1])
        
        progress_calls = []
        
        def progress_callback(current, total):
            progress_calls.append((current, total))
        
        result = run_monte_carlo_simulation(
            parameters=params,
            x_axis=x_axis,
            time_axis=time_axis,
            noise_values=[0.05],
            num_runs=3,
            multiprocessing=False,
            progress_callback=progress_callback
        )
        
        # Should have been called 3 times
        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3)


class TestScanRunnerCompatibility:
    """Test backward compatibility wrapper."""
    
    def test_legacy_format(self):
        """Test legacy format compatibility."""
        indices = {
            'x axis': np.linspace(-10, 10, 101),
            'time axis': np.array([0, 1, 2])
        }
        
        parameters = {
            'sigma^2_0': 1.0,
            'amplitude_0': 1.0,
            'mu_0': 0.0
        }
        
        result = scan_runner_compatibility(
            indices=indices,
            parameters=parameters,
            ld=2.0,
            this_diff=0.5,
            this_tau=8.0,
            this_noise=0.05,
            this_run=42,
            retain_profile_data=False
        )
        
        # Check legacy format structure
        assert 'run_42' in result
        run_data = result['run_42']
        
        assert run_data['run'] == 42
        assert run_data['run parameters']['nominal diffusion coefficient'] == 0.5
        assert run_data['run parameters']['nominal lifetime'] == 8.0
        assert run_data['run parameters']['nominal diffusion length'] == 2.0
        assert run_data['run parameters']['noise stdev'] == 0.05
        
        assert 'cnr_0 estimate' in run_data
        assert 'diffusion' in run_data
        assert 'unweighted fit' in run_data['diffusion']
        assert 'weighted fit' in run_data['diffusion']
    
    def test_legacy_with_profile_data(self):
        """Test legacy format with profile data retention."""
        indices = {
            'x axis': np.linspace(-5, 5, 51),
            'time axis': np.array([0, 1])
        }
        
        parameters = {
            'sigma^2_0': 0.5,
            'amplitude_0': 2.0,
            'mu_0': 0.5
        }
        
        result = scan_runner_compatibility(
            indices=indices,
            parameters=parameters,
            ld=1.0,
            this_diff=1.0,
            this_tau=1.0,
            this_noise=0.01,
            this_run=0,
            retain_profile_data=True
        )
        
        run_data = result['run_0']
        
        # Should have profile data
        assert 'nominal profiles' in run_data
        assert 'y_values_t' in run_data['nominal profiles']
        assert 'noisy profiles' in run_data
        assert 'y_values_t' in run_data['noisy profiles']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])