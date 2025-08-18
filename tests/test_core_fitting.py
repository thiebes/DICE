"""
Tests for dice.core.fitting module.
"""

import pytest
import numpy as np
from dice.core.fitting import (
    gauss_fitting,
    fit_gaussian_profile,
    diffusion_ols_fit,
    diffusion_wls_fit,
    calculate_fit_weights,
)
from dice.core.profiles import gaussian


class TestGaussFitting:
    """Test Gaussian fitting to noisy profiles."""
    
    def test_single_clean_profile(self):
        """Test fitting a single clean Gaussian profile."""
        x = np.linspace(-10, 10, 201)
        y = gaussian(x, mu=0, sig2=2.0, amp=1.0)
        
        result = gauss_fitting(x, y)
        
        assert 'sigma^2_t estimates' in result
        assert 'sigma^2_t standard errors' in result
        assert len(result['sigma^2_t estimates']) == 1
        
        # Should recover the true variance
        assert np.isclose(result['sigma^2_t estimates'][0], 2.0, rtol=0.01)
    
    def test_multiple_profiles(self):
        """Test fitting multiple profiles."""
        x = np.linspace(-10, 10, 201)
        profiles = []
        true_variances = [1.0, 2.0, 3.0]
        
        for sig2 in true_variances:
            y = gaussian(x, mu=0, sig2=sig2, amp=1.0)
            # Add small noise
            np.random.seed(42)
            y_noisy = y + np.random.normal(0, 0.01, len(y))
            profiles.append(y_noisy)
        
        result = gauss_fitting(x, np.array(profiles))
        
        assert len(result['sigma^2_t estimates']) == 3
        assert len(result['sigma^2_t standard errors']) == 3
        
        # Check recovered variances
        for i, true_sig2 in enumerate(true_variances):
            assert np.isclose(result['sigma^2_t estimates'][i], true_sig2, rtol=0.1)
    
    def test_noisy_profile(self):
        """Test fitting with significant noise."""
        x = np.linspace(-10, 10, 201)
        true_sig2 = 2.0
        y = gaussian(x, mu=0, sig2=true_sig2, amp=1.0)
        
        # Add significant noise
        np.random.seed(42)
        y_noisy = y + np.random.normal(0, 0.1, len(y))
        
        result = gauss_fitting(x, y_noisy)
        
        # Should still be reasonably close despite noise
        assert np.isclose(result['sigma^2_t estimates'][0], true_sig2, rtol=0.2)
        
        # Standard error should be non-zero
        assert result['sigma^2_t standard errors'][0] > 0
    
    def test_offset_gaussian(self):
        """Test fitting Gaussian with non-zero mean."""
        x = np.linspace(-10, 10, 201)
        y = gaussian(x, mu=3.0, sig2=1.5, amp=0.8)
        
        result = gauss_fitting(x, y)
        
        # Should recover the variance regardless of mean
        assert np.isclose(result['sigma^2_t estimates'][0], 1.5, rtol=0.01)
    
    def test_mismatched_lengths_error(self):
        """Test error when profile and x_axis have different lengths."""
        x = np.linspace(-10, 10, 100)
        y = np.ones(50)  # Different length
        
        with pytest.raises(ValueError, match="same length"):
            gauss_fitting(x, y)
    
    def test_zero_profile(self):
        """Test handling of zero profile."""
        x = np.linspace(-10, 10, 201)
        y = np.zeros(201)
        
        result = gauss_fitting(x, y)
        
        # Should return some result even for zero profile
        assert len(result['sigma^2_t estimates']) == 1
        assert len(result['sigma^2_t standard errors']) == 1
    
    def test_array_vs_list_input(self):
        """Test that both array and list inputs work."""
        x = np.linspace(-10, 10, 201)
        y = gaussian(x, mu=0, sig2=1.0, amp=1.0)
        
        # Test with array
        result_array = gauss_fitting(x, np.array([y]))
        
        # Test with list
        result_list = gauss_fitting(x, [y])
        
        # Results should be the same
        assert np.isclose(
            result_array['sigma^2_t estimates'][0],
            result_list['sigma^2_t estimates'][0]
        )


class TestFitGaussianProfile:
    """Test single Gaussian profile fitting."""
    
    def test_basic_fit(self):
        """Test basic Gaussian fitting."""
        x = np.linspace(-5, 5, 101)
        y = gaussian(x, mu=0, sig2=1.0, amp=2.0)
        
        result = fit_gaussian_profile(x, y)
        
        assert result['success'] is True
        assert np.isclose(result['mu'], 0, atol=0.01)
        assert np.isclose(result['sigma2'], 1.0, rtol=0.01)
        assert np.isclose(result['amplitude'], 2.0, rtol=0.01)
    
    def test_with_initial_guess(self):
        """Test fitting with initial guess."""
        x = np.linspace(-5, 5, 101)
        y = gaussian(x, mu=1.0, sig2=0.5, amp=1.5)
        
        # Provide good initial guess
        result = fit_gaussian_profile(x, y, initial_guess=(1.0, 0.5, 1.5))
        
        assert result['success'] is True
        assert np.isclose(result['mu'], 1.0, atol=0.01)
        assert np.isclose(result['sigma2'], 0.5, rtol=0.01)
    
    def test_with_noise(self):
        """Test fitting noisy data."""
        x = np.linspace(-5, 5, 101)
        y = gaussian(x, mu=0, sig2=1.0, amp=1.0)
        
        # Add noise
        np.random.seed(42)
        y_noisy = y + np.random.normal(0, 0.05, len(y))
        
        result = fit_gaussian_profile(x, y_noisy)
        
        assert result['success'] is True
        # Parameters should be close despite noise
        assert np.isclose(result['mu'], 0, atol=0.1)
        assert np.isclose(result['sigma2'], 1.0, rtol=0.1)
    
    def test_mismatched_lengths(self):
        """Test error for mismatched x and y lengths."""
        x = np.linspace(-5, 5, 100)
        y = np.ones(50)
        
        with pytest.raises(ValueError, match="same length"):
            fit_gaussian_profile(x, y)
    
    def test_failed_fit(self):
        """Test handling of fit failure."""
        x = np.linspace(-5, 5, 10)
        # Create pathological data that's hard to fit
        y = np.random.random(10) * 1e-10
        
        result = fit_gaussian_profile(x, y, initial_guess=(0, 1, 1))
        
        # Even if fit fails, should return result
        assert 'success' in result
        assert 'mu' in result
        assert 'sigma2' in result


class TestDiffusionOLSFit:
    """Test OLS fitting for diffusion coefficient estimation."""
    
    def test_perfect_linear_msd(self):
        """Test with perfect linear MSD data."""
        # Create perfect MSD = 2*D*t data
        time_axis = np.array([0, 1, 2, 3, 4])
        D_true = 0.5
        sigma2_0 = 1.0
        sigma2_t = sigma2_0 + 2 * D_true * time_axis  # Perfect diffusion
        
        result = diffusion_ols_fit(time_axis, sigma2_t)
        
        assert 'MSD_t slope estimate' in result
        assert 'r_squared' in result
        
        # Should recover exact slope (2*D)
        assert np.isclose(result['MSD_t slope estimate'], 2 * D_true)
        assert np.isclose(result['r_squared'], 1.0)  # Perfect fit
        assert np.isclose(result['intercept estimate'], 0)  # MSD starts at 0
    
    def test_noisy_msd(self):
        """Test with noisy MSD data."""
        np.random.seed(42)
        time_axis = np.linspace(0, 10, 20)
        D_true = 0.3
        sigma2_0 = 1.0
        sigma2_t = sigma2_0 + 2 * D_true * time_axis
        
        # Add noise
        sigma2_t += np.random.normal(0, 0.05, len(time_axis))
        
        result = diffusion_ols_fit(time_axis, sigma2_t)
        
        # Should be close to true value
        assert np.isclose(result['MSD_t slope estimate'], 2 * D_true, rtol=0.1)
        assert result['r_squared'] > 0.9  # Good fit despite noise
    
    def test_minimum_points(self):
        """Test with minimum number of points."""
        time_axis = np.array([0, 1])
        sigma2_t = np.array([1.0, 2.0])
        
        result = diffusion_ols_fit(time_axis, sigma2_t)
        
        # With 2 points, should fit perfectly
        assert np.isclose(result['MSD_t slope estimate'], 1.0)  # MSD slope
        assert np.isclose(result['r_squared'], 1.0)
    
    def test_insufficient_points_error(self):
        """Test error with insufficient points."""
        with pytest.raises(ValueError, match="at least 2 points"):
            diffusion_ols_fit(np.array([0]), np.array([1]))
    
    def test_mismatched_arrays_error(self):
        """Test error with mismatched array lengths."""
        with pytest.raises(ValueError, match="same length"):
            diffusion_ols_fit(np.array([0, 1, 2]), np.array([1, 2]))
    
    def test_standard_errors(self):
        """Test that standard errors are calculated."""
        time_axis = np.array([0, 1, 2, 3, 4])
        sigma2_t = np.array([1.0, 1.5, 2.1, 2.4, 3.0])
        
        result = diffusion_ols_fit(time_axis, sigma2_t)
        
        assert 'MSD_t slope std error' in result
        assert 'intercept standard error' in result
        assert result['MSD_t slope std error'] > 0
        assert not np.isnan(result['MSD_t slope std error'])


class TestDiffusionWLSFit:
    """Test WLS fitting for diffusion coefficient estimation."""
    
    def test_perfect_linear_msd(self):
        """Test with perfect linear MSD data."""
        time_axis = np.array([0, 1, 2, 3, 4])
        D_true = 0.5
        sigma2_0 = 1.0
        sigma2_t = sigma2_0 + 2 * D_true * time_axis
        
        result = diffusion_wls_fit(time_axis, sigma2_t)
        
        # Should recover exact slope with or without weights
        assert np.isclose(result['MSD_t slope estimate'], 2 * D_true)
        assert np.isclose(result['r_squared'], 1.0)
    
    def test_with_custom_weights(self):
        """Test with custom weights."""
        time_axis = np.array([0, 1, 2, 3])
        sigma2_t = np.array([1.0, 2.0, 3.0, 4.0])
        
        # Give more weight to later points
        weights = np.array([1.0, 1.0, 2.0, 2.0])
        
        result = diffusion_wls_fit(time_axis, sigma2_t, weights=weights)
        
        assert 'MSD_t slope estimate' in result
        assert result['MSD_t slope estimate'] > 0
    
    def test_automatic_weights(self):
        """Test automatic weight generation."""
        time_axis = np.array([0, 1, 2, 3])
        sigma2_t = np.array([1.0, 2.0, 4.0, 8.0])
        
        # Without explicit weights, should use inverse variance
        result = diffusion_wls_fit(time_axis, sigma2_t)
        
        assert 'MSD_t slope estimate' in result
        # Early points (smaller variance) should have more influence
    
    def test_heteroscedastic_data(self):
        """Test with heteroscedastic noise."""
        np.random.seed(42)
        time_axis = np.linspace(0, 10, 20)
        D_true = 0.3
        sigma2_0 = 1.0
        sigma2_t = sigma2_0 + 2 * D_true * time_axis
        
        # Add heteroscedastic noise (increases with time)
        noise_scale = 0.01 + 0.02 * time_axis
        sigma2_t += np.random.normal(0, noise_scale)
        
        result_ols = diffusion_ols_fit(time_axis, sigma2_t)
        result_wls = diffusion_wls_fit(time_axis, sigma2_t)
        
        # Both should give reasonable results
        assert np.isclose(result_ols['MSD_t slope estimate'], 2 * D_true, rtol=0.2)
        assert np.isclose(result_wls['MSD_t slope estimate'], 2 * D_true, rtol=0.2)
    
    def test_weight_length_error(self):
        """Test error when weights have wrong length."""
        time_axis = np.array([0, 1, 2])
        sigma2_t = np.array([1, 2, 3])
        weights = np.array([1, 2])  # Wrong length
        
        with pytest.raises(ValueError, match="same length"):
            diffusion_wls_fit(time_axis, sigma2_t, weights=weights)
    
    def test_minimum_points(self):
        """Test with minimum number of points."""
        time_axis = np.array([0, 1])
        sigma2_t = np.array([1.0, 2.0])
        
        result = diffusion_wls_fit(time_axis, sigma2_t)
        
        assert result['MSD_t slope estimate'] == 1.0
        assert result['r_squared'] == 1.0


class TestCalculateFitWeights:
    """Test weight calculation for WLS fitting."""
    
    def test_inverse_variance_weights(self):
        """Test inverse variance weighting."""
        errors = np.array([0.1, 0.2, 0.5, 1.0])
        
        weights = calculate_fit_weights(errors, method='inverse_variance')
        
        assert len(weights) == 4
        # Smaller errors should have larger weights
        assert weights[0] > weights[1] > weights[2] > weights[3]
        
        # Check normalization
        assert np.isclose(np.mean(weights), 1.0)
    
    def test_uniform_weights(self):
        """Test uniform weighting."""
        errors = np.array([0.1, 0.2, 0.5, 1.0])
        
        weights = calculate_fit_weights(errors, method='uniform')
        
        assert len(weights) == 4
        assert np.all(weights == 1.0)
    
    def test_zero_error_handling(self):
        """Test handling of zero errors."""
        errors = np.array([0.0, 0.1, 0.2])
        
        # Should not crash with zero error
        weights = calculate_fit_weights(errors, method='inverse_variance')
        
        assert len(weights) == 3
        assert not np.any(np.isnan(weights))
        assert not np.any(np.isinf(weights))
    
    def test_invalid_method_error(self):
        """Test error for invalid method."""
        errors = np.array([0.1, 0.2])
        
        with pytest.raises(ValueError, match="Unknown weighting method"):
            calculate_fit_weights(errors, method='invalid')
    
    def test_single_value(self):
        """Test with single error value."""
        errors = np.array([0.5])
        
        weights = calculate_fit_weights(errors, method='inverse_variance')
        
        assert len(weights) == 1
        assert weights[0] == 1.0  # Normalized to 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])