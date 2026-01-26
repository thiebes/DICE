"""
Tests for dice.io.parameters module.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from dice.io.parameters import (
    open_parameters,
    check_for_unique_key,
    handle_time_parameters,
    handle_noise_parameters,
    handle_profile_width_parameters,
    handle_diffusion_parameters,
    parameter_parser,
)


class TestCheckForUniqueKey:
    """Test unique key checking function."""
    
    def test_single_key_present(self):
        """Test with exactly one key present."""
        params = {'key1': 'value1', 'other': 'value'}
        result = check_for_unique_key(params, ['key1', 'key2', 'key3'])
        assert result == 'key1'
    
    def test_no_keys_present(self):
        """Test with no matching keys."""
        params = {'other': 'value'}
        with pytest.raises(ValueError, match="No parameter provided"):
            check_for_unique_key(params, ['key1', 'key2'])
    
    def test_multiple_keys_present(self):
        """Test with multiple matching keys."""
        params = {'key1': 'value1', 'key2': 'value2'}
        with pytest.raises(ValueError, match="More than one parameter"):
            check_for_unique_key(params, ['key1', 'key2'])
    
    def test_invalid_input_types(self):
        """Test with invalid input types."""
        with pytest.raises(TypeError):
            check_for_unique_key("not_a_dict", ['key1'])
        
        with pytest.raises(TypeError):
            check_for_unique_key({}, "not_a_list")


class TestHandleTimeParameters:
    """Test time parameter handling."""
    
    def test_time_range(self):
        """Test time range specification."""
        params = {'time range': [0, 10, 11]}
        handle_time_parameters(params)
        
        assert 'time series' in params
        assert len(params['time series']) == 11
        assert params['time series'][0] == 0
        assert params['time series'][-1] == 10
        assert 'time range' not in params
    
    def test_time_series_provided(self):
        """Test when time series is directly provided."""
        params = {'time series': np.array([0, 1, 2, 3])}
        handle_time_parameters(params)
        
        assert 'time series' in params
        assert len(params['time series']) == 4
    
    def test_invalid_time_range(self):
        """Test invalid time range values."""
        # Wrong number of values
        params = {'time range': [0, 10]}
        with pytest.raises(ValueError, match="three values"):
            handle_time_parameters(params)
        
        # Non-numeric values
        params = {'time range': [0, 'ten', 11]}
        with pytest.raises(ValueError, match="must be numeric"):
            handle_time_parameters(params)
    
    def test_missing_time_specification(self):
        """Test missing time specification."""
        params = {}
        with pytest.raises(ValueError, match="No parameter provided"):
            handle_time_parameters(params)


class TestHandleNoiseParameters:
    """Test noise parameter handling."""
    
    def test_single_noise_value(self):
        """Test single noise value specification."""
        params = {'noise value': 0.1}
        handle_noise_parameters(params)
        
        assert 'noise series' in params
        assert params['noise series'] == [0.1]
    
    def test_noise_range_linear(self):
        """Test linear noise range."""
        params = {'noise range, reciprocal': [0.1, 1.0]}
        handle_noise_parameters(params, num_vals=5)
        
        assert 'noise series' in params
        assert len(params['noise series']) == 5
        assert all(0.1 <= n <= 1.0 for n in params['noise series'])
    
    def test_noise_range_log(self):
        """Test logarithmic noise range."""
        params = {'noise range, reciprocal log': [0.01, 1.0]}
        handle_noise_parameters(params, num_vals=5)
        
        assert 'noise series' in params
        assert len(params['noise series']) == 5
        assert all(0.01 <= n <= 1.0 for n in params['noise series'])
    
    def test_invalid_noise_value(self):
        """Test invalid noise value."""
        params = {'noise value': 'not_a_number'}
        with pytest.raises(ValueError, match="Must be numeric"):
            handle_noise_parameters(params)
    
    def test_invalid_noise_range(self):
        """Test invalid noise range."""
        params = {'noise range, reciprocal': [0.1]}  # Only one value
        with pytest.raises(ValueError, match="two numeric values"):
            handle_noise_parameters(params)


class TestHandleProfileWidthParameters:
    """Test profile width parameter handling."""
    
    def test_fwhm_conversion(self):
        """Test FWHM to sigma^2 conversion."""
        params = {'FWHM_0': 2.0}
        sigma2 = handle_profile_width_parameters(params)
        
        # FWHM = 2*sqrt(2*ln(2))*sigma, so sigma^2 = (FWHM/(2*sqrt(2*ln(2))))^2
        expected = (2.0 / (2 * np.sqrt(2 * np.log(2)))) ** 2
        assert np.isclose(sigma2, expected)
    
    def test_sigma_conversion(self):
        """Test sigma to sigma^2 conversion."""
        params = {'sigma_0': 1.5}
        sigma2 = handle_profile_width_parameters(params)
        
        assert sigma2 == 1.5 ** 2
    
    def test_invalid_fwhm(self):
        """Test invalid FWHM value."""
        params = {'FWHM_0': 'not_a_number'}
        with pytest.raises(ValueError, match="Must be numeric"):
            handle_profile_width_parameters(params)
    
    def test_missing_width_parameter(self):
        """Test missing width parameter."""
        params = {}
        with pytest.raises(ValueError, match="No parameter provided"):
            handle_profile_width_parameters(params)


class TestHandleDiffusionParameters:
    """Test diffusion parameter handling."""
    
    def test_diffusion_and_lifetime_provided(self):
        """Test when D and tau are provided."""
        params = {
            'nominal diffusion coefficient': 0.5,
            'nominal lifetime (tau)': 10.0
        }
        handle_diffusion_parameters(params)
        
        assert 'nominal diffusion length' in params
        assert np.isclose(params['nominal diffusion length'], np.sqrt(5.0))
    
    def test_diffusion_length_provided(self):
        """Test when diffusion length is provided."""
        params = {'nominal diffusion length': 2.0}
        handle_diffusion_parameters(params)
        
        assert params['nominal diffusion coefficient'] == 4.0
        assert params['nominal lifetime (tau)'] == 1
    
    def test_all_parameters_provided(self):
        """Test error when all parameters are provided."""
        params = {
            'nominal diffusion coefficient': 0.5,
            'nominal lifetime (tau)': 10.0,
            'nominal diffusion length': 2.0
        }
        with pytest.raises(ValueError, match="not all three"):
            handle_diffusion_parameters(params)
    
    def test_no_parameters_provided(self):
        """Test error when no parameters are provided."""
        params = {}
        with pytest.raises(ValueError, match="Provide"):
            handle_diffusion_parameters(params)
    
    def test_invalid_values(self):
        """Test invalid parameter values."""
        params = {
            'nominal diffusion coefficient': 'not_a_number',
            'nominal lifetime (tau)': 10.0
        }
        with pytest.raises(ValueError, match="must be numeric"):
            handle_diffusion_parameters(params)


class TestParameterParser:
    """Test complete parameter parsing."""
    
    def test_minimal_valid_parameters(self):
        """Test parsing minimal valid parameters."""
        params = {
            'number of runs': 100,
            'spatial width': 20.0,
            'pixel width': 201,
            'mean_0': 0.0,
            'amplitude_0': 1.0,
            'time range': [0, 10, 11],
            'noise value': 0.05,
            'FWHM_0': 2.0,
            'nominal diffusion coefficient': 0.5,
            'nominal lifetime (tau)': 10.0
        }
        
        result = parameter_parser(params)
        
        assert 'x array' in result
        assert len(result['x array']) == 201
        assert 'time series' in result
        assert len(result['time series']) == 11
        assert 'noise series' in result
        assert result['noise series'] == [0.05]
        assert 'sigma^2_0' in result
        assert 'nominal diffusion length' in result
    
    def test_missing_required_parameter(self):
        """Test error with missing required parameter."""
        params = {
            'spatial width': 20.0,
            'pixel width': 201,
            'mean_0': 0.0,
            'amplitude_0': 1.0,
        }
        # Missing 'number of runs'
        
        with pytest.raises(KeyError, match="number of runs"):
            parameter_parser(params)
    
    def test_default_values(self):
        """Test that default values are set."""
        params = {
            'number of runs': 100,
            'spatial width': 20.0,
            'pixel width': 201,
            'mean_0': 0.0,
            'amplitude_0': 1.0,
            'time range': [0, 10, 11],
            'noise value': 0.05,
            'FWHM_0': 2.0,
            'nominal diffusion coefficient': 0.5,
            'nominal lifetime (tau)': 10.0
        }
        
        result = parameter_parser(params)

        # Check defaults (proximity level is NOT defaulted - handled by CLI)
        assert result['multiprocessing'] == 1
        assert result['filename slug'] == 'dice_simulation'
        assert result['retain profile data'] == 0
        assert result['image type'] == 'png'


class TestOpenParameters:
    """Test parameter file loading."""
    
    def test_load_valid_file(self):
        """Test loading a valid parameter file."""
        # Create a temporary parameter file
        params_dict = {
            'number of runs': 10,
            'spatial width': 10.0,
            'pixel width': 101,
            'mean_0': 0.0,
            'amplitude_0': 1.0,
            'time range': [0, 5, 6],
            'noise value': 0.1,
            'sigma_0': 1.0,
            'nominal diffusion length': 2.0
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(str(params_dict))
            temp_file = f.name
        
        try:
            result = open_parameters(temp_file)
            
            assert result['number of runs'] == 10
            assert 'x array' in result
            assert 'time series' in result
            assert 'noise series' in result
        finally:
            os.unlink(temp_file)
    
    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            open_parameters('nonexistent_file.txt')
    
    def test_invalid_file_content(self):
        """Test error with invalid file content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is not a valid Python dictionary")
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Error evaluating"):
                open_parameters(temp_file)
        finally:
            os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])