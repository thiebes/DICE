"""
Unit tests for dice.utils module.
"""
import pytest
import numpy as np
from dice.utils import (
    gaussian,
    make_x_axis,
    make_time_axis,
    sigma_to_fwhm,
    sigma2_to_fwhm,
    fwhm_to_sigma,
    fwhm_to_sigma2,
    add_noise,
    make_noise_distribution,
    gauss_fitting,
)


@pytest.mark.unit
class TestGaussian:
    """Tests for gaussian() function."""

    def test_gaussian_basic(self, sample_x_array):
        """Test basic Gaussian generation."""
        result = gaussian(sample_x_array, mu=0.0, sig2=1.0, amp=1.0)
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_x_array.shape
        # Maximum should be close to amplitude (within 1%)
        assert result.max() == pytest.approx(1.0, rel=0.01)

    def test_gaussian_invalid_variance(self, sample_x_array):
        """Test that negative variance raises ValueError."""
        with pytest.raises(ValueError, match="Variance.*must be a positive"):
            gaussian(sample_x_array, mu=0.0, sig2=-1.0, amp=1.0)

    def test_gaussian_zero_variance(self, sample_x_array):
        """Test that zero variance raises ValueError."""
        with pytest.raises(ValueError, match="Variance.*must be a positive"):
            gaussian(sample_x_array, mu=0.0, sig2=0.0, amp=1.0)

    def test_gaussian_invalid_amplitude(self, sample_x_array):
        """Test that negative amplitude raises ValueError."""
        with pytest.raises(ValueError, match="Amplitude.*must be non-negative"):
            gaussian(sample_x_array, mu=0.0, sig2=1.0, amp=-1.0)


@pytest.mark.unit
class TestMakeXAxis:
    """Tests for make_x_axis() function."""

    def test_make_x_axis_basic(self):
        """Test basic x-axis creation."""
        result = make_x_axis(scan_width=10.0, scan_width_pixels=100, mu=0.0)
        assert isinstance(result, np.ndarray)
        assert len(result) == 100
        assert result.min() == pytest.approx(-5.0, rel=1e-5)
        assert result.max() == pytest.approx(5.0, rel=1e-5)

    def test_make_x_axis_invalid_pixels(self):
        """Test that non-positive pixel count raises ValueError."""
        with pytest.raises(ValueError, match="scan_width_pixels must be a positive"):
            make_x_axis(scan_width=10.0, scan_width_pixels=0, mu=0.0)

    def test_make_x_axis_negative_pixels(self):
        """Test that negative pixel count raises ValueError."""
        with pytest.raises(ValueError, match="scan_width_pixels must be a positive"):
            make_x_axis(scan_width=10.0, scan_width_pixels=-10, mu=0.0)

    def test_make_x_axis_invalid_width(self):
        """Test that non-positive scan width raises ValueError."""
        with pytest.raises(ValueError, match="scan_width must be a positive"):
            make_x_axis(scan_width=0.0, scan_width_pixels=100, mu=0.0)


@pytest.mark.unit
class TestMakeTimeAxis:
    """Tests for make_time_axis() function."""

    def test_make_time_axis_basic(self):
        """Test basic time axis creation."""
        result = make_time_axis(t_start=0.0, t_end=5.0, t_frames=6)
        assert isinstance(result, np.ndarray)
        assert len(result) == 6
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(5.0)

    def test_make_time_axis_invalid_order(self):
        """Test that start >= end raises ValueError."""
        with pytest.raises(ValueError, match="Start timestamp must be less than"):
            make_time_axis(t_start=5.0, t_end=0.0, t_frames=6)

    def test_make_time_axis_equal_bounds(self):
        """Test that start == end raises ValueError."""
        with pytest.raises(ValueError, match="Start timestamp must be less than"):
            make_time_axis(t_start=5.0, t_end=5.0, t_frames=6)

    def test_make_time_axis_invalid_frames(self):
        """Test that non-positive frames raises ValueError."""
        with pytest.raises(ValueError, match="Number of time frames must be a positive"):
            make_time_axis(t_start=0.0, t_end=5.0, t_frames=0)


@pytest.mark.unit
class TestConversions:
    """Tests for sigma/FWHM conversion functions."""

    def test_sigma_to_fwhm(self):
        """Test sigma to FWHM conversion."""
        sigma = 1.0
        fwhm = sigma_to_fwhm(sigma)
        expected = sigma * 2 * np.sqrt(2 * np.log(2))
        assert fwhm == pytest.approx(expected)

    def test_sigma2_to_fwhm(self):
        """Test sigma^2 to FWHM conversion."""
        sigma2 = 1.0
        fwhm = sigma2_to_fwhm(sigma2)
        expected = np.sqrt(sigma2) * 2 * np.sqrt(2 * np.log(2))
        assert fwhm == pytest.approx(expected)

    def test_fwhm_to_sigma(self):
        """Test FWHM to sigma conversion."""
        fwhm = 2.355
        sigma = fwhm_to_sigma(fwhm)
        expected = fwhm / (2 * np.sqrt(2 * np.log(2)))
        assert sigma == pytest.approx(expected)

    def test_fwhm_to_sigma2(self):
        """Test FWHM to sigma^2 conversion."""
        fwhm = 2.355
        sigma2 = fwhm_to_sigma2(fwhm)
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        expected = sigma ** 2
        assert sigma2 == pytest.approx(expected)

    def test_roundtrip_conversion(self):
        """Test that converting sigma -> FWHM -> sigma returns original."""
        original_sigma = 1.5
        fwhm = sigma_to_fwhm(original_sigma)
        sigma = fwhm_to_sigma(fwhm)
        assert sigma == pytest.approx(original_sigma)


@pytest.mark.unit
class TestAddNoise:
    """Tests for add_noise() function."""

    def test_add_noise_shape(self):
        """Test that add_noise preserves array shape."""
        nominal = np.ones((5, 100))
        result = add_noise(noise_sigma=0.01, nominal_profiles=nominal)
        assert 'y_values_t' in result
        assert result['y_values_t'].shape == nominal.shape

    def test_add_noise_modifies_data(self):
        """Test that add_noise actually modifies the data."""
        np.random.seed(42)
        nominal = np.ones((5, 100))
        result = add_noise(noise_sigma=0.1, nominal_profiles=nominal)
        # With noise, data should not be exactly 1.0 everywhere
        assert not np.allclose(result['y_values_t'], nominal)


@pytest.mark.unit
class TestMakeNoiseDistribution:
    """Tests for make_noise_distribution() function."""

    def test_make_noise_distribution_linear(self):
        """Test linear noise distribution generation."""
        result = make_noise_distribution(0.01, 0.1, 10, logarithmic=False)
        assert isinstance(result, list)
        assert len(result) == 10
        assert all(0.01 <= val <= 0.1 for val in result)

    def test_make_noise_distribution_invalid_bounds(self):
        """Test that negative bounds raise ValueError."""
        with pytest.raises(ValueError, match="Noise bounds must be positive"):
            make_noise_distribution(-0.01, 0.1, 10)

    def test_make_noise_distribution_invalid_order(self):
        """Test that lower >= upper raises ValueError."""
        with pytest.raises(ValueError, match="lower bound must be less than"):
            make_noise_distribution(0.1, 0.01, 10)


@pytest.mark.unit
class TestGaussFitting:
    """Tests for gauss_fitting() function."""

    def test_gauss_fitting_basic(self, sample_x_array):
        """Test basic Gaussian fitting."""
        # Create a perfect Gaussian
        true_mu = 0.0
        true_sig2 = 1.0
        true_amp = 1.0
        y = gaussian(sample_x_array, true_mu, true_sig2, true_amp)

        # Add small noise
        np.random.seed(42)
        y_noisy = y + np.random.normal(0, 0.001, y.shape)

        # Fit should recover approximate parameters
        result = gauss_fitting(sample_x_array, [y_noisy])

        assert 'sigma^2_t estimates' in result
        assert 'sigma^2_t standard errors' in result
        assert len(result['sigma^2_t estimates']) == 1

        # Check that fitted sigma^2 is close to true value
        assert result['sigma^2_t estimates'][0] == pytest.approx(true_sig2, rel=0.1)
