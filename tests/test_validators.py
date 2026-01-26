"""
Tests for dice.utils.validators module.
"""

import pytest
import numpy as np
import os
import tempfile
from dice.utils.validators import (
    validate_numeric,
    validate_array_like,
    validate_integer,
    validate_string_choice,
    validate_file_path,
)


class TestValidateNumeric:
    """Test numeric validation."""
    
    def test_valid_numeric(self):
        """Test with valid numeric inputs."""
        assert validate_numeric(5) == 5.0
        assert validate_numeric(3.14) == 3.14
        assert validate_numeric(np.int32(10)) == 10.0
        assert validate_numeric(np.float64(2.5)) == 2.5
    
    def test_numeric_bounds(self):
        """Test numeric bounds checking."""
        # Within bounds
        assert validate_numeric(5, min_val=0, max_val=10) == 5.0
        assert validate_numeric(0, min_val=0) == 0.0
        assert validate_numeric(10, max_val=10) == 10.0
        
        # Out of bounds
        with pytest.raises(ValueError, match="must be >= 0"):
            validate_numeric(-1, min_val=0)
        
        with pytest.raises(ValueError, match="must be <= 10"):
            validate_numeric(11, max_val=10)
    
    def test_infinite_values(self):
        """Test handling of infinite values."""
        # Allow infinity
        assert validate_numeric(np.inf, allow_inf=True) == np.inf
        assert validate_numeric(-np.inf, allow_inf=True) == -np.inf
        
        # Disallow infinity (default)
        with pytest.raises(ValueError, match="must be finite"):
            validate_numeric(np.inf)
        
        with pytest.raises(ValueError, match="must be finite"):
            validate_numeric(np.nan)
    
    def test_invalid_types(self):
        """Test with invalid types."""
        with pytest.raises(TypeError, match="must be numeric"):
            validate_numeric("5")
        
        with pytest.raises(TypeError, match="must be numeric"):
            validate_numeric([1, 2, 3])


class TestValidateArrayLike:
    """Test array validation."""
    
    def test_valid_arrays(self):
        """Test with valid array-like inputs."""
        # List
        arr = validate_array_like([1, 2, 3])
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 3
        
        # Tuple
        arr = validate_array_like((4, 5, 6))
        assert len(arr) == 3
        
        # Already numpy array
        original = np.array([7, 8, 9])
        arr = validate_array_like(original)
        assert np.array_equal(arr, original)
    
    def test_dimension_checking(self):
        """Test dimension validation."""
        # 1D array
        arr = validate_array_like([1, 2, 3], ndim=1)
        assert arr.ndim == 1
        
        # 2D array
        arr = validate_array_like([[1, 2], [3, 4]], ndim=2)
        assert arr.ndim == 2
        
        # Wrong dimensions
        with pytest.raises(ValueError, match="must have 1 dimensions"):
            validate_array_like([[1, 2]], ndim=1)
    
    def test_shape_checking(self):
        """Test exact shape validation."""
        arr = validate_array_like([1, 2, 3], shape=(3,))
        assert arr.shape == (3,)
        
        arr = validate_array_like([[1, 2], [3, 4]], shape=(2, 2))
        assert arr.shape == (2, 2)
        
        with pytest.raises(ValueError, match="must have shape"):
            validate_array_like([1, 2], shape=(3,))
    
    def test_empty_array(self):
        """Test empty array handling."""
        # Disallow empty (default)
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_array_like([])
        
        # Allow empty
        arr = validate_array_like([], non_empty=False)
        assert len(arr) == 0


class TestValidateInteger:
    """Test integer validation."""
    
    def test_valid_integers(self):
        """Test with valid integer inputs."""
        assert validate_integer(5) == 5
        assert validate_integer(np.int32(10)) == 10
        assert validate_integer(5.0) == 5  # Float with no decimal part
    
    def test_integer_bounds(self):
        """Test integer bounds checking."""
        assert validate_integer(5, min_val=0, max_val=10) == 5
        assert validate_integer(0, min_val=0) == 0
        
        with pytest.raises(ValueError, match="must be >= 1"):
            validate_integer(0, min_val=1)
    
    def test_invalid_integers(self):
        """Test with invalid inputs."""
        with pytest.raises(TypeError, match="must be an integer"):
            validate_integer(3.14)
        
        with pytest.raises(TypeError, match="must be an integer"):
            validate_integer("5")


class TestValidateStringChoice:
    """Test string choice validation."""
    
    def test_valid_choices(self):
        """Test with valid string choices."""
        choices = ['red', 'green', 'blue']
        assert validate_string_choice('red', choices) == 'red'
        assert validate_string_choice('blue', choices) == 'blue'
    
    def test_case_sensitivity(self):
        """Test case-sensitive and case-insensitive matching."""
        choices = ['Red', 'Green', 'Blue']
        
        # Case sensitive (default)
        with pytest.raises(ValueError, match="must be one of"):
            validate_string_choice('red', choices)
        
        # Case insensitive
        result = validate_string_choice('red', choices, case_sensitive=False)
        assert result == 'Red'  # Returns original case from choices
        
        result = validate_string_choice('GREEN', choices, case_sensitive=False)
        assert result == 'Green'
    
    def test_invalid_choices(self):
        """Test with invalid choices."""
        choices = ['a', 'b', 'c']
        
        with pytest.raises(ValueError, match="must be one of"):
            validate_string_choice('d', choices)
        
        with pytest.raises(TypeError, match="must be a string"):
            validate_string_choice(123, choices)


class TestValidateFilePath:
    """Test file path validation."""
    
    def test_valid_path(self):
        """Test with valid file path."""
        # Any string is valid if we don't check existence
        path = validate_file_path("/some/path.txt")
        assert path == "/some/path.txt"
    
    def test_file_existence(self):
        """Test file existence checking."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # File exists
            path = validate_file_path(tmp_path, must_exist=True)
            assert path == tmp_path
            
            # File doesn't exist
            os.unlink(tmp_path)
            with pytest.raises(ValueError, match="does not exist"):
                validate_file_path(tmp_path, must_exist=True)
        finally:
            # Clean up if file still exists
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_extension_checking(self):
        """Test file extension validation."""
        path = validate_file_path("data.txt", extension=".txt")
        assert path == "data.txt"
        
        with pytest.raises(ValueError, match="must have extension"):
            validate_file_path("data.csv", extension=".txt")
    
    def test_invalid_path_type(self):
        """Test with invalid path type."""
        with pytest.raises(TypeError, match="must be a string"):
            validate_file_path(123)


if __name__ == "__main__":
    pytest.main([__file__])