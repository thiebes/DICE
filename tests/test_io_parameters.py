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
    
    def test_zero_lifetime_gives_infinite_length(self):
        """A lifetime of 0 means no decay, so the diffusion length is unbounded."""
        params = {'nominal diffusion coefficient': 0.01, 'nominal lifetime (tau)': 0}
        handle_diffusion_parameters(params)
        assert params['nominal diffusion length'] == np.inf
        assert params['nominal diffusion coefficient'] == 0.01
        assert params['nominal lifetime (tau)'] == 0

    @pytest.mark.parametrize('diff, tau, name', [(-0.01, 1.0, 'diffusion_coeff'), (0.01, -1.0, 'lifetime')])
    def test_negative_coefficient_or_lifetime_rejected(self, diff, tau, name):
        params = {'nominal diffusion coefficient': diff, 'nominal lifetime (tau)': tau}
        with pytest.raises(ValueError, match=f"{name} must be >= 0"):
            handle_diffusion_parameters(params)

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


class TestMixedUnitParameters:
    """Test parameter parsing and resolution with per-parameter unit overrides."""

    def _base_params(self):
        """Return a minimal valid parameter set."""
        return {
            'number of runs': 10,
            'spatial width': 10.0,
            'pixel width': 101,
            'mean_0': 0.0,
            'amplitude_0': 1.0,
            'time range': [0, 1, 10],
            'noise value': 0.05,
            'FWHM_0': 1.0,
            'nominal diffusion coefficient': 0.5,
            'nominal lifetime (tau)': 2.0,
            'length unit': 'micrometer',
            'time unit': 'nanosecond',
        }

    def test_unit_keys_pass_through_parser(self):
        """Verify that _unit keys survive parameter_parser normalization."""
        params = self._base_params()
        params['fwhm_0_unit'] = 'nanometer'
        params['lifetime_unit'] = 'picosecond'
        result = parameter_parser(params)
        assert result.get('fwhm_0_unit') == 'nanometer'
        assert result.get('lifetime_unit') == 'picosecond'

    def test_resolve_before_parser_flow(self):
        """resolve_units must be called before parameter_parser because the
        parser converts FWHM to sigma^2 -- the FWHM value must already be
        in the global unit system when the parser processes it."""
        from dice.utils.units import resolve_units
        params = self._base_params()
        params['FWHM_0'] = 500.0
        params['fwhm_0_unit'] = 'nanometer'
        # Resolve first to convert 500 nm -> 0.5 um
        resolved = resolve_units(params)
        assert np.isclose(resolved['FWHM_0'], 0.5, rtol=1e-12)
        # Then parse (which converts FWHM to sigma^2)
        result = parameter_parser(resolved)
        from dice.utils.converters import fwhm_to_sigma2
        expected_sigma2 = fwhm_to_sigma2(0.5)
        assert np.isclose(result['sigma2_0'], expected_sigma2, rtol=1e-10)

    def test_mixed_units_equivalent_to_single_unit_system(self):
        """A mixed-unit parameter set should produce the same parsed result
        as an equivalent single-unit-system parameter set."""
        from dice.utils.units import resolve_units
        from dice.utils.converters import fwhm_to_sigma2

        # Single unit system: everything in um/ns
        params_uniform = self._base_params()
        params_uniform['FWHM_0'] = 0.5
        params_uniform['nominal lifetime (tau)'] = 2.0
        result_uniform = parameter_parser(params_uniform)

        # Mixed units: FWHM in nm, lifetime in ps
        params_mixed = self._base_params()
        params_mixed['FWHM_0'] = 500.0
        params_mixed['fwhm_0_unit'] = 'nanometer'
        params_mixed['nominal lifetime (tau)'] = 2000.0
        params_mixed['lifetime_unit'] = 'picosecond'
        resolved_mixed = resolve_units(params_mixed)
        result_mixed = parameter_parser(resolved_mixed)

        # Compare key derived values
        assert np.isclose(
            result_uniform['sigma2_0'],
            result_mixed['sigma2_0'],
            rtol=1e-10
        )
        assert np.isclose(
            result_uniform['nominal lifetime (tau)'],
            result_mixed['nominal lifetime (tau)'],
            rtol=1e-10
        )

    def test_file_with_unit_overrides(self):
        """Test loading a parameter file that contains _unit keys."""
        from dice.utils.units import resolve_units
        params_dict = {
            'number of runs': 10,
            'spatial width': 10.0,
            'pixel width': 101,
            'mean_0': 0.0,
            'amplitude_0': 1.0,
            'time range': [0, 1, 10],
            'noise value': 0.05,
            'FWHM_0': 500,
            'fwhm_0_unit': 'nanometer',
            'nominal diffusion length': 0.1,
            'length unit': 'micrometer',
            'time unit': 'nanosecond',
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(str(params_dict))
            temp_file = f.name

        try:
            result = open_parameters(temp_file)
            result = resolve_units(result)
            # fwhm_0_unit should be consumed
            assert 'fwhm_0_unit' not in result
        finally:
            os.unlink(temp_file)

    def test_no_unit_keys_backward_compatible(self):
        """Existing parameter files without _unit keys work unchanged."""
        from dice.utils.units import resolve_units
        params = self._base_params()
        resolved = resolve_units(params)
        result = parameter_parser(resolved)
        assert 'x array' in result
        assert 'time series' in result
        assert 'sigma2_0' in result


EXAMPLES_DIR = Path(__file__).parent.parent / 'data' / 'examples'
EXAMPLE_PROFILE = EXAMPLES_DIR / 'example_profile.csv'


class TestEstimateNoiseFromData:
    """'estimate noise from data' must set the noise so the simulated initial CNR
    equals the CNR fft-cnr estimates from the measured profile."""

    def _write_parameters(self, directory: Path, noise_file: Path, amplitude_0: float) -> Path:
        import ast
        params = ast.literal_eval((EXAMPLES_DIR / 'quick_test.txt').read_text())
        del params['noise value']
        params['estimate noise from data'] = str(noise_file)
        params['amplitude_0'] = amplitude_0
        params_file = directory / 'estimate_noise.txt'
        params_file.write_text(str(params))
        return params_file

    def _write_profile(self, directory: Path, profile: np.ndarray) -> Path:
        profile_file = directory / 'profile.csv'
        np.savetxt(profile_file, profile.reshape(1, -1), delimiter=',', fmt='%.17g')
        return profile_file

    @pytest.mark.parametrize('amplitude_0', [1.0, 2.5])
    def test_simulated_cnr_equals_file_cnr(self, tmp_path, amplitude_0):
        from fft_cnr import fft_cnr
        file_estimate = fft_cnr(np.loadtxt(EXAMPLE_PROFILE, delimiter=','))

        params = open_parameters(self._write_parameters(tmp_path, EXAMPLE_PROFILE, amplitude_0))

        assert len(params['noise series']) == 1
        noise_sigma = params['noise series'][0]
        assert amplitude_0 / noise_sigma == pytest.approx(file_estimate.cnr, rel=1e-12)
        # The file's noise_rms is in its own intensity units and must not leak through.
        assert noise_sigma != pytest.approx(file_estimate.noise_rms)

    def test_constant_profile_rejected(self, tmp_path):
        measured = np.loadtxt(EXAMPLE_PROFILE, delimiter=',')
        constant_file = self._write_profile(tmp_path, np.full_like(measured, measured[0]))
        params_file = self._write_parameters(tmp_path, constant_file, 1.0)
        with pytest.raises(ValueError, match="finite, positive CNR"):
            open_parameters(params_file)

    def test_short_profile_rejected(self, tmp_path):
        measured = np.loadtxt(EXAMPLE_PROFILE, delimiter=',')
        short_file = self._write_profile(tmp_path, measured[:10])
        params_file = self._write_parameters(tmp_path, short_file, 1.0)
        with pytest.raises(ValueError, match="too short"):
            open_parameters(params_file)

    def test_non_numeric_profile_rejected(self, tmp_path):
        text = EXAMPLE_PROFILE.read_text().replace('1530', 'peak', 1)
        bad_file = tmp_path / 'non_numeric.csv'
        bad_file.write_text(text)
        params_file = self._write_parameters(tmp_path, bad_file, 1.0)
        with pytest.raises(ValueError, match="Error reading CSV"):
            open_parameters(params_file)

    def test_missing_amplitude_rejected(self):
        params = {'estimate noise from data': str(EXAMPLE_PROFILE)}
        with pytest.raises(ValueError, match="amplitude_0"):
            handle_noise_parameters(params)


class TestRetiredProximityLevel:
    """'proximity level' no longer affects a run; a file that sets it gets one warning."""

    def _quick_test_params(self) -> dict:
        import ast
        return ast.literal_eval((EXAMPLES_DIR / 'quick_test.txt').read_text())

    @pytest.mark.parametrize('key', ['proximity level', 'proximity_level'])
    def test_warns_once_naming_key_and_replacement(self, key):
        params = self._quick_test_params() | {key: 0.1}
        with pytest.warns(UserWarning) as caught:
            parameter_parser(params)
        messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        assert len(messages) == 1
        assert f"'{key}' no longer affects the simulation" in messages[0]
        assert "analyze_simulation_results" in messages[0]
        assert "Explore Accuracy Threshold" in messages[0]

    def test_no_warning_without_key(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            parameter_parser(self._quick_test_params())

    def test_check_returns_messages_only_for_retired_keys(self):
        from dice.io.parameters import check_retired_parameters
        assert check_retired_parameters(self._quick_test_params()) == []
        assert len(check_retired_parameters({'proximity level': 0.1})) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])