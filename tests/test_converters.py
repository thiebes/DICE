"""
Tests for dice.utils.converters module.
"""

import pytest
import numpy as np
from dice.utils.converters import (
    sigma_to_fwhm,
    sigma2_to_fwhm,
    fwhm_to_sigma,
    fwhm_to_sigma2,
    calculate_pixel_size,
    slope_to_diffusion_constant,
)


class TestFWHMConversions:
    """Test FWHM <-> sigma conversions."""
    
    def test_sigma_to_fwhm(self):
        """Test sigma to FWHM conversion."""
        # Known value: FWHM = sigma * 2 * sqrt(2 * ln(2))
        sigma = 1.0
        expected = 2.354820045  # 2 * sqrt(2 * ln(2))
        result = sigma_to_fwhm(sigma)
        assert np.isclose(result, expected, rtol=1e-6)
        
        # Test with different value
        sigma = 5.0
        result = sigma_to_fwhm(sigma)
        assert np.isclose(result, sigma * 2.354820045, rtol=1e-6)
    
    def test_sigma_to_fwhm_invalid(self):
        """Test sigma to FWHM with invalid input."""
        with pytest.raises(ValueError, match="must be a positive"):
            sigma_to_fwhm(0)
        
        with pytest.raises(ValueError, match="must be a positive"):
            sigma_to_fwhm(-1)
    
    def test_fwhm_to_sigma(self):
        """Test FWHM to sigma conversion."""
        fwhm = 2.354820045
        expected = 1.0
        result = fwhm_to_sigma(fwhm)
        assert np.isclose(result, expected, rtol=1e-6)
    
    def test_round_trip_conversions(self):
        """Test that conversions are inverses of each other."""
        # Test sigma -> FWHM -> sigma
        sigma = 3.5
        fwhm = sigma_to_fwhm(sigma)
        sigma_back = fwhm_to_sigma(fwhm)
        assert np.isclose(sigma, sigma_back, rtol=1e-10)
        
        # Test FWHM -> sigma -> FWHM
        fwhm = 7.2
        sigma = fwhm_to_sigma(fwhm)
        fwhm_back = sigma_to_fwhm(sigma)
        assert np.isclose(fwhm, fwhm_back, rtol=1e-10)
    
    def test_sigma2_conversions(self):
        """Test sigma^2 conversions."""
        sigma2 = 4.0  # sigma = 2.0
        expected_fwhm = sigma_to_fwhm(2.0)
        result = sigma2_to_fwhm(sigma2)
        assert np.isclose(result, expected_fwhm, rtol=1e-6)
        
        # Test inverse
        fwhm = 5.0
        sigma2 = fwhm_to_sigma2(fwhm)
        assert np.isclose(sigma2, fwhm_to_sigma(fwhm)**2, rtol=1e-10)


class TestDiffusionConversion:
    """Test slope to diffusion coefficient conversion."""
    
    def test_basic_conversion(self):
        """Test basic unit conversion."""
        # MSD slope of 2 μm²/ns should give D = 1 μm²/ns
        slope = 2.0
        D = slope_to_diffusion_constant(slope, 'micrometer', 'nanosecond')
        # D in cm²/s = 1 μm²/ns * (10^-4 cm/μm)² / (10^-9 s/ns)
        # = 1 * 10^-8 / 10^-9 = 10 cm²/s
        expected = 10.0
        assert np.isclose(D, expected, rtol=1e-10)
    
    def test_different_units(self):
        """Test conversion with different units."""
        slope = 1.0
        
        # nm²/ps -> cm²/s
        D = slope_to_diffusion_constant(slope, 'nanometer', 'picosecond')
        # D = 0.5 nm²/ps * (10^-7)² / 10^-12 = 0.5 * 10^-14 / 10^-12 = 0.5 * 10^-2
        expected = 0.5e-2
        assert np.isclose(D, expected, rtol=1e-10)
        
        # cm²/s -> cm²/s (no conversion needed)
        slope = 2.0
        D = slope_to_diffusion_constant(slope, 'centimeter', 'second')
        expected = 1.0  # slope/2
        assert np.isclose(D, expected, rtol=1e-10)
    
    def test_invalid_units(self):
        """Test error handling for invalid units."""
        slope = 1.0
        
        with pytest.raises(ValueError, match="Unknown length unit"):
            slope_to_diffusion_constant(slope, 'invalid_unit', 'second')

        with pytest.raises(ValueError, match="Unknown time unit"):
            slope_to_diffusion_constant(slope, 'meter', 'invalid_unit')
    
    def test_zero_slope(self):
        """Test with zero slope (no diffusion)."""
        slope = 0.0
        D = slope_to_diffusion_constant(slope, 'micrometer', 'nanosecond')
        assert D == 0.0


class TestPixelSizeConversion:
    """Test pixel size calculation."""

    def test_basic_calculation(self):
        """Test basic pixel size calculation."""
        pixel_size = calculate_pixel_size(10.0, 100)
        assert np.isclose(pixel_size, 0.1, rtol=1e-10)

        pixel_size = calculate_pixel_size(50.0, 500)
        assert np.isclose(pixel_size, 0.1, rtol=1e-10)

    def test_different_values(self):
        """Test with different input values."""
        pixel_size = calculate_pixel_size(1.0, 10)
        assert np.isclose(pixel_size, 0.1, rtol=1e-10)

        pixel_size = calculate_pixel_size(100.0, 1)
        assert np.isclose(pixel_size, 100.0, rtol=1e-10)

    def test_invalid_spatial_width(self):
        """Test error handling for invalid spatial width."""
        with pytest.raises(ValueError, match="Spatial width must be positive"):
            calculate_pixel_size(0, 100)

        with pytest.raises(ValueError, match="Spatial width must be positive"):
            calculate_pixel_size(-10, 100)

    def test_invalid_pixel_count(self):
        """Test error handling for invalid pixel count."""
        with pytest.raises(ValueError, match="Pixel count must be positive"):
            calculate_pixel_size(10, 0)

        with pytest.raises(ValueError, match="Pixel count must be positive"):
            calculate_pixel_size(10, -1)


if __name__ == "__main__":
    pytest.main([__file__])