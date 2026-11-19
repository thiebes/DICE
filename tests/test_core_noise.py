"""
Tests for dice.core.noise module.
"""

from pathlib import Path

import pytest
import numpy as np
from fft_cnr import fft_cnr
from dice.core.noise import (
    MIN_CNR_PEAK_FWHM,
    add_noise,
    make_noise_distribution,
    estimate_noise_from_profile,
    peak_fwhm_samples,
)
from dice.core.profiles import gaussian
from dice.utils.converters import fwhm_to_sigma2

EXAMPLE_PROFILE = Path(__file__).parent.parent / 'data' / 'examples' / 'example_profile.csv'


class TestAddNoise:
    """Test noise addition to profiles."""
    
    def test_add_noise_single_profile(self):
        """Test adding noise to a single profile."""
        profile = np.ones(100)
        result = add_noise(profile, noise_sigma=0.1, seed=42)
        
        assert 'y_values_t' in result
        assert result['y_values_t'].shape == (100,)
        
        # Should not be exactly ones anymore
        assert not np.allclose(result['y_values_t'], 1.0)
        
        # Mean should still be close to 1
        assert np.abs(np.mean(result['y_values_t']) - 1.0) < 0.05
    
    def test_add_noise_multiple_profiles(self):
        """Test adding noise to multiple profiles."""
        profiles = np.ones((5, 100))
        result = add_noise(profiles, noise_sigma=0.1, seed=42)
        
        assert result['y_values_t'].shape == (5, 100)
        
        # Each profile should be different
        for i in range(5):
            for j in range(i+1, 5):
                assert not np.allclose(result['y_values_t'][i], result['y_values_t'][j])
    
    def test_zero_noise(self):
        """Test with zero noise (no change)."""
        profile = np.ones(100)
        result = add_noise(profile, noise_sigma=0.0, seed=42)
        
        # Should be unchanged
        assert np.allclose(result['y_values_t'], 1.0)
    
    def test_reproducibility_with_seed(self):
        """Test that seed makes results reproducible."""
        profile = np.ones(100)
        result1 = add_noise(profile, noise_sigma=0.1, seed=42)
        result2 = add_noise(profile, noise_sigma=0.1, seed=42)
        result3 = add_noise(profile, noise_sigma=0.1, seed=123)
        
        # Same seed should give same result
        assert np.allclose(result1['y_values_t'], result2['y_values_t'])
        
        # Different seed should give different result
        assert not np.allclose(result1['y_values_t'], result3['y_values_t'])
    
    def test_negative_noise_error(self):
        """Test error for negative noise sigma."""
        with pytest.raises(ValueError):
            add_noise(np.ones(100), noise_sigma=-0.1)


class TestMakeNoiseDistribution:
    """Test noise distribution generation."""
    
    def test_linear_distribution(self):
        """Test uniform distribution in reciprocal (CNR) space."""
        noise_values = make_noise_distribution(0.1, 1.0, 100, logarithmic=False, seed=42)
        
        assert len(noise_values) == 100
        assert all(0.1 <= n <= 1.0 for n in noise_values)
        
        # Convert to CNR values
        cnr_values = [1/n for n in noise_values]
        
        # CNR should be between 1 and 10
        assert all(1.0 <= c <= 10.0 for c in cnr_values)
    
    def test_logarithmic_distribution(self):
        """Test logarithmic distribution in reciprocal space."""
        noise_values = make_noise_distribution(0.01, 1.0, 100, logarithmic=True, seed=42)
        
        assert len(noise_values) == 100
        assert all(0.01 <= n <= 1.0 for n in noise_values)
    
    def test_single_value(self):
        """Test generating single noise value."""
        noise_values = make_noise_distribution(0.1, 1.0, 1, seed=42)
        
        assert len(noise_values) == 1
        assert 0.1 <= noise_values[0] <= 1.0
    
    def test_invalid_bounds(self):
        """Test error for invalid bounds."""
        # Zero bound
        with pytest.raises(ValueError, match="must be positive"):
            make_noise_distribution(0, 1.0, 10)
        
        # Negative bound
        with pytest.raises(ValueError):
            make_noise_distribution(-0.1, 1.0, 10)
        
        # Lower >= upper
        with pytest.raises(ValueError, match="less than"):
            make_noise_distribution(1.0, 0.1, 10)
    
    def test_reproducibility(self):
        """Test reproducibility with seed."""
        result1 = make_noise_distribution(0.1, 1.0, 10, seed=42)
        result2 = make_noise_distribution(0.1, 1.0, 10, seed=42)
        result3 = make_noise_distribution(0.1, 1.0, 10, seed=123)
        
        assert result1 == result2
        assert result1 != result3


class TestFFTCNR:
    """Test FFT-based CNR estimation via the fft-cnr package."""

    def test_clean_gaussian(self):
        """Clean signal should produce high CNR."""
        from dice.core.profiles import gaussian

        x = np.linspace(-10, 10, 201)
        y = gaussian(x, mu=0, sig2=1, amp=1)

        np.random.seed(42)
        y_noisy = y + np.random.normal(0, 0.01, len(y))

        result = fft_cnr(y_noisy)
        assert result.cnr > 10

    def test_noisy_profile(self):
        """Noisy signal should produce lower CNR than clean signal."""
        from dice.core.profiles import gaussian

        x = np.linspace(-10, 10, 201)
        y = gaussian(x, mu=0, sig2=1, amp=1)

        np.random.seed(42)
        y_noisy = y + np.random.normal(0, 0.1, len(y))

        result = fft_cnr(y_noisy)
        assert result.cnr > 0
        assert result.cnr < 50

    def test_result_has_expected_fields(self):
        """CNREstimate should expose cnr, noise_rms, and amplitude."""
        from dice.core.profiles import gaussian

        x = np.linspace(-10, 10, 201)
        y = gaussian(x, mu=0, sig2=1, amp=1)

        np.random.seed(42)
        y_noisy = y + np.random.normal(0, 0.05, len(y))

        result = fft_cnr(y_noisy)
        assert hasattr(result, 'cnr')
        assert hasattr(result, 'noise_rms')
        assert hasattr(result, 'amplitude')
        assert result.noise_rms > 0

    def test_noisier_signal_has_lower_cnr(self):
        """More noise should produce lower CNR."""
        from dice.core.profiles import gaussian

        x = np.linspace(-10, 10, 201)
        y = gaussian(x, mu=0, sig2=1, amp=1)

        np.random.seed(42)
        y_low_noise = y + np.random.normal(0, 0.01, len(y))
        np.random.seed(42)
        y_high_noise = y + np.random.normal(0, 0.2, len(y))

        cnr_low = fft_cnr(y_low_noise).cnr
        cnr_high = fft_cnr(y_high_noise).cnr

        assert cnr_low > cnr_high


class TestEstimateNoiseFromProfile:
    """Test noise estimation from profile."""
    
    def test_fft_method(self):
        """Test FFT-based noise estimation."""
        from dice.core.profiles import gaussian
        
        x = np.linspace(-10, 10, 201)
        y = gaussian(x, mu=0, sig2=1, amp=1)
        
        # Add known noise
        np.random.seed(42)
        noise_true = 0.05
        y_noisy = y + np.random.normal(0, noise_true, len(y))
        
        noise_est = estimate_noise_from_profile(y_noisy, method='fft')
        
        # Should be close to true noise level
        assert np.abs(noise_est - noise_true) < 0.02
    
    def test_high_freq_method(self):
        """Test high-frequency based noise estimation."""
        from dice.core.profiles import gaussian
        
        x = np.linspace(-10, 10, 201)
        y = gaussian(x, mu=0, sig2=1, amp=1)
        
        # Add known noise
        np.random.seed(42)
        noise_true = 0.05
        y_noisy = y + np.random.normal(0, noise_true, len(y))
        
        noise_est = estimate_noise_from_profile(y_noisy, method='high_freq')
        
        # High-freq method is less accurate but should be in ballpark
        assert 0.01 < noise_est < 0.1
    
    def test_invalid_method(self):
        """Test error for invalid method."""
        profile = np.ones(100)
        
        with pytest.raises(ValueError, match="Unknown method"):
            estimate_noise_from_profile(profile, method='invalid')
    
    def test_short_profile_high_freq(self):
        """Test error for too short profile with high-freq method."""
        with pytest.raises(ValueError, match="too short"):
            estimate_noise_from_profile(np.array([1]), method='high_freq')
    
    def test_zero_amplitude_profile(self):
        """Test handling of zero amplitude profile."""
        profile = np.zeros(100)
        profile[50] = 1.0  # Single spike
        
        # Should still work
        noise_est = estimate_noise_from_profile(profile, method='fft')
        assert isinstance(noise_est, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

class TestNominalCnr:
    """Nominal CNR is amplitude over noise sigma, infinite without noise."""

    def test_amplitude_over_noise(self):
        from dice.core.noise import nominal_cnr
        assert nominal_cnr(2.0, 0.04) == pytest.approx(50.0)
        assert nominal_cnr(1.0, 0.02) == pytest.approx(50.0)

    def test_zero_noise_is_infinite(self):
        from dice.core.noise import nominal_cnr
        assert nominal_cnr(1.0, 0.0) == float('inf')


class TestPeakFwhmSamples:
    """Width of a measured peak in samples, for the narrow-peak warning on 'estimate noise from data'."""

    @staticmethod
    def _gaussian_samples(fwhm: float, n: int = 101) -> np.ndarray:
        """DICE's own Gaussian, sampled once per pixel, slightly off the pixel grid."""
        x = np.arange(n) - n // 2
        return gaussian(x, 0.3, fwhm_to_sigma2(fwhm), 1.0)

    def test_repository_profile_is_narrow(self):
        measured = np.loadtxt(EXAMPLE_PROFILE, delimiter=',')
        width = peak_fwhm_samples(measured)
        assert width == pytest.approx(4.6, abs=0.1)
        assert width < MIN_CNR_PEAK_FWHM

    @pytest.mark.parametrize('fwhm', [4.6, 9.0, 14.0])
    def test_noise_free_gaussian_width_is_exact(self, fwhm):
        assert peak_fwhm_samples(self._gaussian_samples(fwhm)) == pytest.approx(fwhm, rel=1e-6)

    def test_wide_noisy_peak_is_not_narrow(self):
        """At CNR 5 a 15-pixel peak still reads wider than the threshold."""
        rng = np.random.default_rng(7)
        noisy = self._gaussian_samples(15.0) + rng.normal(0, 0.2, 101)
        assert peak_fwhm_samples(noisy) == pytest.approx(15.0, rel=0.15)

    def test_sign_and_baseline_do_not_change_the_width(self):
        measured = np.loadtxt(EXAMPLE_PROFILE, delimiter=',')
        width = peak_fwhm_samples(measured)
        assert peak_fwhm_samples(-measured) == pytest.approx(width, rel=1e-9)
        assert peak_fwhm_samples(measured + 500.0) == pytest.approx(width, rel=1e-9)

    def test_profile_without_a_peak_has_no_width(self):
        assert np.isnan(peak_fwhm_samples(np.full(41, 3.0)))

    def test_too_few_samples_to_fit_falls_back_to_the_crossings(self):
        # Four fit parameters, three samples: the half-maximum crossings are at 0.5 and 1.5.
        assert peak_fwhm_samples([0.0, 1.0, 0.0]) == pytest.approx(1.0)

    def test_failed_fit_falls_back_to_the_crossings(self, monkeypatch):
        import scipy.optimize

        def failing_fit(*args, **kwargs):
            raise RuntimeError("Optimal parameters not found")

        monkeypatch.setattr(scipy.optimize, 'curve_fit', failing_fit)
        # Linear interpolation across the half-maximum points of a sampled
        # Gaussian lands slightly outside them, so the crossing width is a little wide.
        assert peak_fwhm_samples(self._gaussian_samples(9.0)) == pytest.approx(9.0, rel=0.02)

    def test_two_dimensional_input_rejected(self):
        with pytest.raises(ValueError):
            peak_fwhm_samples(np.ones((2, 41)))
