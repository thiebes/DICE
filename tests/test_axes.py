"""
Tests for dice.utils.axes module.
"""

import pytest
import numpy as np
from dice.utils.axes import (
    make_x_axis,
    make_time_axis,
    make_time_series,
    make_spatial_grid,
)


class TestMakeXAxis:
    """Test spatial axis generation."""
    
    def test_basic_x_axis(self):
        """Test basic x-axis creation."""
        x = make_x_axis(10.0, 11, mu=0.0)
        
        # Check length
        assert len(x) == 11
        
        # Check range
        assert x[0] == -5.0
        assert x[-1] == 5.0
        assert x[5] == 0.0  # Center point
        
        # Check spacing
        spacing = np.diff(x)
        assert np.allclose(spacing, 1.0)
    
    def test_off_center_axis(self):
        """Test x-axis with non-zero center."""
        x = make_x_axis(10.0, 11, mu=5.0)
        
        assert x[0] == 0.0
        assert x[-1] == 10.0
        assert x[5] == 5.0  # Center point
    
    def test_different_resolution(self):
        """Test with different pixel counts."""
        x = make_x_axis(20.0, 101, mu=0.0)
        
        assert len(x) == 101
        assert x[0] == -10.0
        assert x[-1] == 10.0
        assert x[50] == 0.0
        
        # Check spacing
        spacing = np.diff(x)
        expected_spacing = 20.0 / 100
        assert np.allclose(spacing, expected_spacing)
    
    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        # Zero or negative pixels
        with pytest.raises(ValueError):
            make_x_axis(10.0, 0, mu=0.0)
        
        with pytest.raises(ValueError):
            make_x_axis(10.0, -5, mu=0.0)
        
        # Zero width
        with pytest.raises(ValueError, match="must be greater than 0"):
            make_x_axis(0.0, 10, mu=0.0)
        
        # Negative width (should be caught by validator)
        with pytest.raises(ValueError):
            make_x_axis(-10.0, 10, mu=0.0)


class TestMakeTimeAxis:
    """Test time axis generation."""
    
    def test_basic_time_axis(self):
        """Test basic time axis creation."""
        t = make_time_axis(0.0, 10.0, 11)
        
        assert len(t) == 11
        assert t[0] == 0.0
        assert t[-1] == 10.0
        assert t[5] == 5.0
        
        # Check uniform spacing
        spacing = np.diff(t)
        assert np.allclose(spacing, 1.0)
    
    def test_non_zero_start(self):
        """Test time axis with non-zero start."""
        t = make_time_axis(5.0, 15.0, 6)
        
        assert len(t) == 6
        assert t[0] == 5.0
        assert t[-1] == 15.0
        
        spacing = np.diff(t)
        assert np.allclose(spacing, 2.0)
    
    def test_single_frame(self):
        """Test with single time frame."""
        t = make_time_axis(1.0, 1.0, 1)
        assert len(t) == 1
        assert t[0] == 1.0
    
    def test_invalid_time_range(self):
        """Test error handling for invalid time range."""
        # Start >= end
        with pytest.raises(ValueError, match="must be less than"):
            make_time_axis(10.0, 5.0, 10)
        
        with pytest.raises(ValueError, match="must be less than"):
            make_time_axis(5.0, 5.0, 10)
        
        # Invalid frame count
        with pytest.raises(ValueError):
            make_time_axis(0.0, 10.0, 0)


class TestMakeTimeSeries:
    """Test explicit time series creation."""
    
    def test_sorted_series(self):
        """Test with already sorted time points."""
        times = [0.1, 0.3, 0.5, 0.7, 0.9]
        t = make_time_series(times)
        
        assert len(t) == 5
        assert np.array_equal(t, times)
    
    def test_unsorted_series(self):
        """Test automatic sorting of unsorted times."""
        times = [0.5, 0.1, 0.9, 0.3, 0.7]
        t = make_time_series(times)
        
        assert len(t) == 5
        assert np.array_equal(t, [0.1, 0.3, 0.5, 0.7, 0.9])
    
    def test_numpy_input(self):
        """Test with numpy array input."""
        times = np.array([1.0, 2.0, 3.0])
        t = make_time_series(times)
        
        assert isinstance(t, np.ndarray)
        assert np.array_equal(t, times)
    
    def test_duplicate_times(self):
        """Test error on duplicate time points."""
        times = [0.1, 0.3, 0.3, 0.5]
        
        with pytest.raises(ValueError, match="must be unique"):
            make_time_series(times)
    
    def test_non_finite_times(self):
        """Test error on non-finite values."""
        times = [0.1, np.inf, 0.3]
        
        with pytest.raises(ValueError, match="must be finite"):
            make_time_series(times)
        
        times = [0.1, np.nan, 0.3]
        
        with pytest.raises(ValueError, match="must be finite"):
            make_time_series(times)
    
    def test_empty_series(self):
        """Test error on empty time series."""
        with pytest.raises(ValueError, match="cannot be empty"):
            make_time_series([])


class TestMakeSpatialGrid:
    """Test 2D spatial grid generation."""
    
    def test_basic_grid(self):
        """Test basic 2D grid creation."""
        X, Y = make_spatial_grid(10.0, 8.0, 11, 9, center=(0.0, 0.0))
        
        # Check shapes
        assert X.shape == (9, 11)
        assert Y.shape == (9, 11)
        
        # Check X values (constant along columns)
        assert X[0, 0] == -5.0
        assert X[0, -1] == 5.0
        assert np.allclose(X[:, 5], 0.0)  # Center column
        
        # Check Y values (constant along rows)
        assert Y[0, 0] == -4.0
        assert Y[-1, 0] == 4.0
        assert np.allclose(Y[4, :], 0.0)  # Center row
    
    def test_off_center_grid(self):
        """Test grid with non-zero center."""
        X, Y = make_spatial_grid(10.0, 10.0, 5, 5, center=(5.0, -5.0))
        
        assert X.shape == (5, 5)
        assert Y.shape == (5, 5)
        
        # Check centering
        assert X[2, 2] == 5.0
        assert Y[2, 2] == -5.0


if __name__ == "__main__":
    pytest.main([__file__])