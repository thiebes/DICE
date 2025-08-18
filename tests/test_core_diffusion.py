"""
Tests for dice.core.diffusion module.
"""

import pytest
import numpy as np
from dice.core.diffusion import (
    calculate_diffusion_coefficient,
    calculate_diffusion_length,
    calculate_msd,
    estimate_diffusion_from_msd,
    calculate_peclet_number,
    einstein_relation,
)


class TestCalculateDiffusionCoefficient:
    """Test diffusion coefficient calculation from MSD slope."""
    
    def test_1d_diffusion(self):
        """Test 1D diffusion coefficient calculation."""
        # In 1D: MSD = 2*D*t, so D = slope/2
        slope = 10.0
        D = calculate_diffusion_coefficient(slope, dimensions=1)
        assert D == 5.0
    
    def test_2d_diffusion(self):
        """Test 2D diffusion coefficient calculation."""
        # In 2D: MSD = 4*D*t, so D = slope/4
        slope = 10.0
        D = calculate_diffusion_coefficient(slope, dimensions=2)
        assert D == 2.5
    
    def test_3d_diffusion(self):
        """Test 3D diffusion coefficient calculation."""
        # In 3D: MSD = 6*D*t, so D = slope/6
        slope = 12.0
        D = calculate_diffusion_coefficient(slope, dimensions=3)
        assert D == 2.0
    
    def test_zero_slope(self):
        """Test with zero slope (no diffusion)."""
        D = calculate_diffusion_coefficient(0, dimensions=1)
        assert D == 0
    
    def test_invalid_dimensions(self):
        """Test error for invalid dimensions."""
        with pytest.raises(ValueError, match="Dimensions must be 1, 2, or 3"):
            calculate_diffusion_coefficient(10, dimensions=4)
        
        with pytest.raises(ValueError, match="Dimensions must be 1, 2, or 3"):
            calculate_diffusion_coefficient(10, dimensions=0)
    
    def test_negative_slope(self):
        """Test error for negative slope."""
        with pytest.raises(ValueError):
            calculate_diffusion_coefficient(-10, dimensions=1)


class TestCalculateDiffusionLength:
    """Test diffusion length calculation."""
    
    def test_basic_calculation(self):
        """Test basic diffusion length calculation."""
        # L_D = sqrt(D * tau)
        D = 4.0
        tau = 9.0
        L_D = calculate_diffusion_length(D, tau)
        assert L_D == 6.0
    
    def test_zero_diffusion(self):
        """Test with zero diffusion coefficient."""
        L_D = calculate_diffusion_length(0, 10)
        assert L_D == 0
    
    def test_zero_lifetime(self):
        """Test with zero lifetime."""
        L_D = calculate_diffusion_length(10, 0)
        assert L_D == 0
    
    def test_negative_values(self):
        """Test error for negative values."""
        with pytest.raises(ValueError):
            calculate_diffusion_length(-1, 10)
        
        with pytest.raises(ValueError):
            calculate_diffusion_length(10, -1)


class TestCalculateMSD:
    """Test MSD calculation."""
    
    def test_basic_msd(self):
        """Test basic MSD calculation."""
        sigma2_t = np.array([1.0, 2.0, 3.0, 4.0])
        msd = calculate_msd(sigma2_t)
        
        # MSD = sigma2(t) - sigma2(0)
        expected = np.array([0.0, 1.0, 2.0, 3.0])
        assert np.allclose(msd, expected)
    
    def test_explicit_sigma2_0(self):
        """Test with explicit initial variance."""
        sigma2_t = np.array([2.0, 3.0, 4.0])
        sigma2_0 = 1.5
        msd = calculate_msd(sigma2_t, sigma2_0)
        
        expected = np.array([0.5, 1.5, 2.5])
        assert np.allclose(msd, expected)
    
    def test_negative_msd_error(self):
        """Test error for significantly negative MSD."""
        # Variance decreasing significantly should raise error
        sigma2_t = np.array([10.0, 5.0, 1.0])
        
        with pytest.raises(ValueError, match="negative values"):
            calculate_msd(sigma2_t)
    
    def test_numerical_error_cleanup(self):
        """Test that tiny negative values from numerical errors are cleaned."""
        # Tiny negative values should be set to zero
        sigma2_t = np.array([1.0, 1.0 - 1e-12, 1.0 + 1e-6])
        msd = calculate_msd(sigma2_t)
        
        assert msd[1] == 0  # Tiny negative set to 0
        assert msd[2] > 0  # Positive value preserved


class TestEstimateDiffusionFromMSD:
    """Test diffusion estimation from MSD data."""
    
    def test_perfect_linear_fit(self):
        """Test with perfect linear MSD data."""
        # Create perfect MSD = 2*D*t data with D=0.5
        t = np.array([0, 1, 2, 3, 4])
        D_true = 0.5
        msd = 2 * D_true * t  # Perfect 1D diffusion
        
        D_est, slope, r2 = estimate_diffusion_from_msd(t, msd, dimensions=1)
        
        assert np.isclose(D_est, D_true)
        assert np.isclose(slope, 2 * D_true)
        assert np.isclose(r2, 1.0)  # Perfect fit
    
    def test_2d_diffusion_estimation(self):
        """Test 2D diffusion estimation."""
        t = np.array([0, 1, 2, 3])
        D_true = 0.25
        msd = 4 * D_true * t  # 2D diffusion
        
        D_est, slope, r2 = estimate_diffusion_from_msd(t, msd, dimensions=2)
        
        assert np.isclose(D_est, D_true)
        assert np.isclose(slope, 4 * D_true)
    
    def test_noisy_data(self):
        """Test with noisy MSD data."""
        np.random.seed(42)
        t = np.linspace(0, 10, 20)
        D_true = 0.5
        msd = 2 * D_true * t + np.random.normal(0, 0.1, len(t))
        
        D_est, slope, r2 = estimate_diffusion_from_msd(t, msd, dimensions=1)
        
        # Should be close but not exact due to noise
        assert np.isclose(D_est, D_true, rtol=0.1)
        assert r2 > 0.9  # Good but not perfect fit
    
    def test_insufficient_data(self):
        """Test error with insufficient data points."""
        with pytest.raises(ValueError, match="at least 2 points"):
            estimate_diffusion_from_msd(np.array([0]), np.array([0]))
    
    def test_mismatched_arrays(self):
        """Test error with mismatched array lengths."""
        with pytest.raises(ValueError, match="same length"):
            estimate_diffusion_from_msd(np.array([0, 1]), np.array([0, 1, 2]))


class TestCalculatePecletNumber:
    """Test Péclet number calculation."""
    
    def test_basic_peclet(self):
        """Test basic Péclet number calculation."""
        # Pe = v * L / D
        velocity = 2.0
        length = 3.0
        D = 0.5
        
        Pe = calculate_peclet_number(velocity, length, D)
        assert Pe == 12.0
    
    def test_zero_velocity(self):
        """Test with zero velocity (pure diffusion)."""
        Pe = calculate_peclet_number(0, 10, 1)
        assert Pe == 0
    
    def test_zero_diffusion_error(self):
        """Test error for zero diffusion coefficient."""
        with pytest.raises(ValueError, match="cannot be zero"):
            calculate_peclet_number(1, 1, 0)
    
    def test_negative_values(self):
        """Test error for negative values."""
        with pytest.raises(ValueError):
            calculate_peclet_number(-1, 1, 1)
        
        with pytest.raises(ValueError):
            calculate_peclet_number(1, -1, 1)


class TestEinsteinRelation:
    """Test Einstein relation calculations."""
    
    def test_mobility_calculation(self):
        """Test mobility calculation from diffusion coefficient."""
        D = 1e-10  # m^2/s
        T = 300  # K
        k_B = 1.380649e-23
        
        mobility = einstein_relation(D, T)
        expected = D / (k_B * T)
        
        assert np.isclose(mobility, expected)
    
    def test_stokes_einstein(self):
        """Test Stokes-Einstein relation."""
        D = 1e-10
        T = 300
        viscosity = 1e-3  # Pa·s (water)
        radius = 1e-9  # 1 nm
        k_B = 1.380649e-23
        
        D_expected = einstein_relation(D, T, viscosity, radius)
        D_stokes = k_B * T / (6 * np.pi * viscosity * radius)
        
        assert np.isclose(D_expected, D_stokes)
    
    def test_zero_temperature_error(self):
        """Test error for zero temperature."""
        with pytest.raises(ValueError, match="cannot be zero"):
            einstein_relation(1e-10, 0)
    
    def test_zero_viscosity_error(self):
        """Test error for zero viscosity when calculating Stokes-Einstein."""
        with pytest.raises(ValueError, match="cannot be zero"):
            einstein_relation(1e-10, 300, viscosity=0, radius=1e-9)
    
    def test_zero_radius_error(self):
        """Test error for zero radius when calculating Stokes-Einstein."""
        with pytest.raises(ValueError, match="cannot be zero"):
            einstein_relation(1e-10, 300, viscosity=1e-3, radius=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])