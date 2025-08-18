"""
Tests for dice.core.profiles module.
"""

import pytest
import numpy as np
from dice.core.profiles import (
    gaussian,
    integrated_intensity,
    kinetic_decay_intensities,
    diffusion_sigma2_t,
    make_diffusion_decay,
)


class TestGaussian:
    """Test Gaussian profile generation."""
    
    def test_gaussian_basic(self):
        """Test basic Gaussian generation."""
        x = np.linspace(-5, 5, 101)
        y = gaussian(x, mu=0, sig2=1, amp=1)
        
        # Check peak is at mu
        assert np.argmax(y) == 50  # Center index
        
        # Check peak value
        assert np.isclose(y[50], 1.0)
        
        # Check symmetry
        assert np.allclose(y[:50], y[51:][::-1])
    
    def test_gaussian_different_parameters(self):
        """Test Gaussian with different parameters."""
        x = np.linspace(-10, 10, 201)
        
        # Different mean
        y = gaussian(x, mu=3, sig2=1, amp=1)
        max_idx = np.argmax(y)
        assert np.isclose(x[max_idx], 3, atol=0.1)
        
        # Different variance (wider)
        y_wide = gaussian(x, mu=0, sig2=4, amp=1)
        y_narrow = gaussian(x, mu=0, sig2=0.25, amp=1)
        
        # Both should have same peak value (amplitude = 1)
        assert np.isclose(np.max(y_wide), 1.0)
        assert np.isclose(np.max(y_narrow), 1.0)
        
        # Wide gaussian should decay more slowly
        # At x=2, wide gaussian should be higher than narrow
        x2_idx = np.argmin(np.abs(x - 2))
        assert y_wide[x2_idx] > y_narrow[x2_idx]
        
        # Different amplitude
        y2 = gaussian(x, mu=0, sig2=1, amp=2)
        assert np.isclose(np.max(y2), 2.0)
    
    def test_gaussian_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        x = np.linspace(-5, 5, 101)
        
        # Negative variance
        with pytest.raises(ValueError):
            gaussian(x, mu=0, sig2=-1, amp=1)
        
        # Zero variance
        with pytest.raises(ValueError, match="must be positive"):
            gaussian(x, mu=0, sig2=0, amp=1)
        
        # Negative amplitude is allowed (for fitting)
        # But should raise error based on our validation
        with pytest.raises(ValueError):
            gaussian(x, mu=0, sig2=1, amp=-1)


class TestIntegratedIntensity:
    """Test integrated intensity calculation."""
    
    def test_integrated_intensity_basic(self):
        """Test basic integrated intensity calculation."""
        # For unit Gaussian (amp=1, sig2=1)
        # Integrated intensity should be sqrt(2*pi)
        result = integrated_intensity(sig2=1, amp=1)
        expected = np.sqrt(2 * np.pi)
        assert np.isclose(result, expected)
    
    def test_integrated_intensity_scaling(self):
        """Test scaling properties."""
        # Double amplitude -> double intensity
        i1 = integrated_intensity(sig2=1, amp=1)
        i2 = integrated_intensity(sig2=1, amp=2)
        assert np.isclose(i2, 2 * i1)
        
        # Four times variance -> double intensity (sqrt scaling)
        i3 = integrated_intensity(sig2=4, amp=1)
        assert np.isclose(i3, 2 * i1)
    
    def test_integrated_intensity_invalid(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            integrated_intensity(sig2=0, amp=1)
        
        with pytest.raises(ValueError):
            integrated_intensity(sig2=1, amp=0)


class TestKineticDecay:
    """Test kinetic decay functions."""
    
    def test_no_decay(self):
        """Test with tau=0 (no decay)."""
        t = np.array([0, 1, 2, 3, 4])
        i0 = 100
        result = kinetic_decay_intensities(i0, tau=0, t_values=t)
        
        # Should be constant
        assert np.all(result == i0)
    
    def test_exponential_decay(self):
        """Test exponential decay."""
        t = np.array([0, 1, 2, 3])
        i0 = 100
        tau = 2.0
        
        result = kinetic_decay_intensities(i0, tau=tau, t_values=t)
        
        # Check initial value
        assert result[0] == i0
        
        # Check decay
        expected = i0 * np.exp(-t / tau)
        assert np.allclose(result, expected)
        
        # At t=tau, intensity should be i0/e
        t_tau = np.array([tau])
        result_tau = kinetic_decay_intensities(i0, tau=tau, t_values=t_tau)
        assert np.isclose(result_tau[0], i0 / np.e)
    
    def test_invalid_inputs(self):
        """Test error handling."""
        t = np.array([0, 1, 2])
        
        # Negative initial intensity
        with pytest.raises(ValueError):
            kinetic_decay_intensities(-100, tau=1, t_values=t)
        
        # Negative tau
        with pytest.raises(ValueError):
            kinetic_decay_intensities(100, tau=-1, t_values=t)


class TestDiffusionSigma2:
    """Test variance evolution due to diffusion."""
    
    def test_no_diffusion(self):
        """Test with D=0 (no diffusion)."""
        t = np.array([0, 1, 2, 3])
        sig2_0 = 1.0
        result = diffusion_sigma2_t(0, sig2_0, t)
        
        # Should be constant
        assert np.all(result == sig2_0)
    
    def test_linear_growth(self):
        """Test linear growth of variance."""
        t = np.array([0, 1, 2, 3, 4])
        sig2_0 = 1.0
        D = 0.5
        
        result = diffusion_sigma2_t(D, sig2_0, t)
        
        # Check initial value
        assert result[0] == sig2_0
        
        # Check linear growth: sig2(t) = sig2_0 + 2*D*t
        expected = sig2_0 + 2 * D * t
        assert np.allclose(result, expected)
        
        # Check MSD = 2*D*t
        msd = result - result[0]
        assert np.allclose(msd, 2 * D * t)
    
    def test_negative_time(self):
        """Test error with negative time."""
        t = np.array([-1, 0, 1])
        
        with pytest.raises(ValueError, match="non-negative"):
            diffusion_sigma2_t(1.0, 1.0, t)
    
    def test_invalid_variance(self):
        """Test error with invalid initial variance."""
        t = np.array([0, 1, 2])
        
        with pytest.raises(ValueError):
            diffusion_sigma2_t(1.0, 0, t)


class TestMakeDiffusionDecay:
    """Test complete diffusion and decay profile generation."""
    
    def test_basic_profile_evolution(self):
        """Test basic profile evolution."""
        x_axis = np.linspace(-10, 10, 101)
        time_axis = np.array([0, 1, 2])
        
        params = {
            'x axis': x_axis,
            'time axis': time_axis,
            'sigma^2_0': 1.0,
            'amplitude_0': 1.0,
            'mu_0': 0.0,
            'nominal diffusion coefficient': 0.5,
            'nominal lifetime': 10.0,
        }
        
        result = make_diffusion_decay(params)
        
        # Check structure
        assert 'parameters_t' in result
        assert 'y_values_t' in result
        
        # Check time points
        assert len(result['y_values_t']) == 3
        assert len(result['parameters_t']['sigma^2_t']) == 3
        
        # Check diffusion (variance should increase)
        sig2_t = result['parameters_t']['sigma^2_t']
        assert sig2_t[2] > sig2_t[1] > sig2_t[0]
        
        # Check decay (amplitude should decrease)
        amp_t = result['parameters_t']['amplitude_t']
        assert amp_t[2] < amp_t[1] < amp_t[0]
    
    def test_no_diffusion_no_decay(self):
        """Test with no diffusion and no decay."""
        x_axis = np.linspace(-10, 10, 101)
        time_axis = np.array([0, 1, 2])
        
        params = {
            'x axis': x_axis,
            'time axis': time_axis,
            'sigma^2_0': 1.0,
            'amplitude_0': 1.0,
            'mu_0': 0.0,
            'nominal diffusion coefficient': 0,
            'nominal lifetime': 0,  # No decay (tau=0 means no decay)
        }
        
        result = make_diffusion_decay(params)
        
        # Variance should be constant
        sig2_t = result['parameters_t']['sigma^2_t']
        assert np.allclose(sig2_t, 1.0)
        
        # Amplitude should be constant (no decay)
        amp_t = result['parameters_t']['amplitude_t']
        assert np.allclose(amp_t, 1.0)
    
    def test_missing_parameters(self):
        """Test error handling for missing parameters."""
        params = {
            'x axis': np.linspace(-10, 10, 101),
            'time axis': np.array([0, 1, 2]),
            # Missing other required parameters
        }
        
        with pytest.raises(KeyError):
            make_diffusion_decay(params)
    
    def test_conservation_of_intensity(self):
        """Test that integrated intensity follows decay correctly."""
        x_axis = np.linspace(-20, 20, 201)  # Wide enough to capture full profile
        time_axis = np.array([0, 1, 2])
        tau = 2.0
        
        params = {
            'x axis': x_axis,
            'time axis': time_axis,
            'sigma^2_0': 1.0,
            'amplitude_0': 1.0,
            'mu_0': 0.0,
            'nominal diffusion coefficient': 0.1,
            'nominal lifetime': tau,
        }
        
        result = make_diffusion_decay(params)
        
        # Check integrated intensity follows exponential decay
        ii_t = result['parameters_t']['integrated intensity_t']
        i0 = ii_t[0]
        
        for i, t in enumerate(time_axis):
            expected = i0 * np.exp(-t / tau)
            assert np.isclose(ii_t[i], expected, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])