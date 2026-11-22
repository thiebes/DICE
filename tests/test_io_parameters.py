"""
Tests for dice.io.parameters module.
"""

import re

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
    validate_parameters,
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

    def test_negative_diffusion_length_rejected(self):
        params = {'nominal diffusion length': -0.1}
        with pytest.raises(ValueError, match="'nominal diffusion length' must be >= 0, got -0.1"):
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
        # Resolve first to convert 500 nm -> 0.5 μm
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

        # Single unit system: everything in μm/ns
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

        with pytest.warns(UserWarning, match="pixels wide at half maximum"):
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

    def test_profile_with_a_byte_order_mark_reads_as_without(self, tmp_path):
        """A spreadsheet program saves "CSV UTF-8" with a byte-order mark."""
        marked_file = tmp_path / 'marked.csv'
        marked_file.write_bytes(b'\xef\xbb\xbf' + EXAMPLE_PROFILE.read_bytes())
        with pytest.warns(UserWarning, match="pixels wide at half maximum"):
            marked = open_parameters(self._write_parameters(tmp_path, marked_file, 1.0))
            plain = open_parameters(self._write_parameters(tmp_path, EXAMPLE_PROFILE, 1.0))
        assert marked['noise series'] == plain['noise series']

    def test_unreadable_profile_names_the_file_and_the_cause(self, tmp_path):
        absent = tmp_path / 'absent.csv'
        params_file = self._write_parameters(tmp_path, absent, 1.0)
        with pytest.raises(ValueError, match="Error reading CSV for noise estimation: .*absent.csv.* could not be read") as error:
            open_parameters(params_file)
        assert "absent.csv not found" in str(error.value)

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

    def test_profile_of_several_lines_rejected(self, tmp_path):
        """The measured profile is one line; a file of several profiles is named as such, not as a shape error."""
        measured = np.loadtxt(EXAMPLE_PROFILE, delimiter=',')
        two_lines = tmp_path / 'two_profiles.csv'
        np.savetxt(two_lines, np.vstack([measured, measured]), delimiter=',', fmt='%.17g')
        params_file = self._write_parameters(tmp_path, two_lines, 1.0)
        with pytest.raises(ValueError, match=re.escape(
                f"The measured profile {str(two_lines)!r} must be a single line of comma-separated "
                f"values; it has 2 lines of {measured.size} values.")):
            open_parameters(params_file)

    def test_missing_amplitude_rejected(self):
        params = {'estimate noise from data': str(EXAMPLE_PROFILE)}
        with pytest.raises(ValueError, match="amplitude_0"):
            handle_noise_parameters(params)

    def test_narrow_repository_profile_warns(self, tmp_path):
        """data/examples/example_profile.csv has a peak 4.6 pixels wide, too narrow for fft-cnr."""
        params_file = self._write_parameters(tmp_path, EXAMPLE_PROFILE, 1.0)
        with pytest.warns(UserWarning) as caught:
            open_parameters(params_file)
        messages = [str(w.message) for w in caught if 'half maximum' in str(w.message)]
        assert len(messages) == 1
        assert f"'{EXAMPLE_PROFILE}' is 4.6 pixels wide at half maximum" in messages[0]
        assert "narrower than about 9 pixels" in messages[0]
        assert "too low" in messages[0]

    def test_wide_profile_does_not_warn(self, tmp_path):
        """A measured-like profile 15 pixels wide at half maximum, from DICE's own Gaussian with noise."""
        import warnings
        from dice.core.profiles import gaussian
        from dice.utils.converters import fwhm_to_sigma2
        x = np.arange(101) - 50
        rng = np.random.default_rng(11)
        wide = 1000 * gaussian(x, 0.3, fwhm_to_sigma2(15.0), 1.0) + 40 + rng.normal(0, 50, x.size)
        params_file = self._write_parameters(tmp_path, self._write_profile(tmp_path, wide), 1.0)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            params = open_parameters(params_file)
        assert 0 < params['noise series'][0] < 1

    def test_warning_names_nothing_for_a_profile_without_a_peak(self):
        from dice.io.parameters import narrow_peak_warning
        assert narrow_peak_warning(np.zeros(41), 'flat.csv') is None
        measured = np.loadtxt(EXAMPLE_PROFILE, delimiter=',')
        assert narrow_peak_warning(measured, 'measured.csv').startswith(
            "The peak of the measured profile 'measured.csv' is 4.6 pixels wide")


class TestNoiseDataPath:
    """A relative 'estimate noise from data' path is relative to the parameters file's folder."""

    def test_relative_path_joins_the_base_folder(self, tmp_path):
        from dice.io.parameters import resolve_noise_data_path
        params = {'estimate noise from data': 'profiles/measured.csv', 'number of runs': 3}
        resolved = resolve_noise_data_path(params, tmp_path)
        assert resolved == {'estimate noise from data': str(tmp_path / 'profiles' / 'measured.csv'),
                            'number of runs': 3}
        assert params['estimate noise from data'] == 'profiles/measured.csv'

    def test_parent_reference_is_normalized(self, tmp_path):
        from dice.io.parameters import resolve_noise_data_path
        resolved = resolve_noise_data_path({'estimate_noise_from_data': '../measured.csv'}, tmp_path / 'params')
        assert resolved['estimate_noise_from_data'] == str(tmp_path / 'measured.csv')

    @pytest.mark.parametrize('value', [str(EXAMPLE_PROFILE), '', 0.02, None])
    def test_absolute_path_and_non_path_values_unchanged(self, tmp_path, value):
        from dice.io.parameters import resolve_noise_data_path
        params = {'estimate noise from data': value}
        assert resolve_noise_data_path(params, tmp_path) == params

    def test_parameter_without_a_noise_file_unchanged(self, tmp_path):
        from dice.io.parameters import resolve_noise_data_path
        params = {'noise value': 0.02}
        assert resolve_noise_data_path(params, tmp_path) == params

    def test_open_parameters_resolves_against_the_file_folder(self, tmp_path, monkeypatch):
        import ast
        import shutil
        params_dir, elsewhere = tmp_path / 'params', tmp_path / 'elsewhere'
        params_dir.mkdir()
        elsewhere.mkdir()
        shutil.copy(EXAMPLE_PROFILE, params_dir / 'measured.csv')
        params = ast.literal_eval((EXAMPLES_DIR / 'quick_test.txt').read_text())
        del params['noise value']
        params['estimate noise from data'] = 'measured.csv'
        params_file = params_dir / 'relative.txt'
        params_file.write_text(str(params))
        monkeypatch.chdir(elsewhere)

        with pytest.warns(UserWarning, match="pixels wide at half maximum"):
            parsed = open_parameters(params_file)

        assert parsed['estimate noise from data'] == str(params_dir / 'measured.csv')
        from fft_cnr import fft_cnr
        file_cnr = fft_cnr(np.loadtxt(EXAMPLE_PROFILE, delimiter=',')).cnr
        assert parsed['amplitude_0'] / parsed['noise series'][0] == pytest.approx(file_cnr, rel=1e-12)

    def test_path_relative_to_the_working_directory_is_not_used(self, tmp_path, monkeypatch):
        import ast
        import shutil
        params_dir = tmp_path / 'params'
        params_dir.mkdir()
        shutil.copy(EXAMPLE_PROFILE, tmp_path / 'measured.csv')
        params = ast.literal_eval((EXAMPLES_DIR / 'quick_test.txt').read_text())
        del params['noise value']
        params['estimate noise from data'] = 'measured.csv'
        params_file = params_dir / 'relative.txt'
        params_file.write_text(str(params))
        monkeypatch.chdir(tmp_path)

        with pytest.raises(ValueError, match="Error reading CSV for noise estimation"):
            open_parameters(params_file)


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


class TestNoiseRangeDraws:
    """A noise range draws one noise sigma for each run from the parser's seed, uniformly in 1/sigma or its log."""

    RUNS = 20000
    LOW, HIGH = 0.01, 0.05

    def _parsed(self, key='noise range, reciprocal', runs=None, value=None, seed=None) -> dict:
        import ast
        params = ast.literal_eval((EXAMPLES_DIR / 'quick_test.txt').read_text())
        del params['noise value']
        params[key] = [self.LOW, self.HIGH] if value is None else value
        params['number of runs'] = self.RUNS if runs is None else runs
        return parameter_parser(params, seed=seed)

    @pytest.mark.parametrize('key, logarithmic', [
        ('noise range, reciprocal', False), ('noise_range_reciprocal', False),
        ('noise range, reciprocal log', True), ('noise_range_reciprocal_log', True),
    ])
    def test_one_value_per_run_and_the_range_is_recognized(self, key, logarithmic):
        from dice.io.parameters import noise_range_from_parameters
        from dice.models.parameters import NoiseRange
        parsed = self._parsed(key, runs=7, seed=1)
        assert len(parsed['noise series']) == 7
        assert len(set(parsed['noise series'])) == 7
        assert noise_range_from_parameters(parsed) == NoiseRange(self.LOW, self.HIGH, logarithmic)

    @pytest.mark.parametrize('noise_key', ['noise value', 'estimate noise from data'])
    def test_fixed_noise_has_no_range(self, noise_key):
        import ast
        import warnings
        from dice.io.parameters import noise_range_from_parameters
        params = ast.literal_eval((EXAMPLES_DIR / 'quick_test.txt').read_text())
        if noise_key != 'noise value':
            del params['noise value']
            params[noise_key] = str(EXAMPLE_PROFILE)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # the example profile's narrow-peak warning
            parsed = parameter_parser(params)
        assert len(parsed['noise series']) == 1
        assert noise_range_from_parameters(parsed) is None

    @pytest.mark.parametrize('key, transform', [
        ('noise range, reciprocal', lambda sigma: 1 / sigma),
        ('noise range, reciprocal log', lambda sigma: np.log10(1 / sigma)),
    ])
    def test_draws_are_uniform_within_the_range(self, key, transform):
        """1/sigma (or its log10) of 20000 draws lies in the range and is uniform within sampling error."""
        from numpy.random import SeedSequence
        from scipy.stats import kstest
        values = transform(np.array(self._parsed(key, seed=SeedSequence(5))['noise series']))
        low, high = transform(self.HIGH), transform(self.LOW)

        # In range, up to rounding in the reciprocal and the logarithm.
        assert values.min() >= low * (1 - 1e-12) and values.max() <= high * (1 + 1e-12)
        # 20 equal-width bins: each count is binomial(n, 1/20), so it lies within 5
        # standard deviations of n/20 unless the draw is not uniform (the chance
        # that one of 20 bins of a uniform draw does not is about 1e-5).
        counts, _ = np.histogram(values, bins=20, range=(low, high))
        assert counts.sum() == self.RUNS
        assert np.all(np.abs(counts - self.RUNS / 20) <= 5 * np.sqrt(self.RUNS * (1 / 20) * (19 / 20)))
        # The mean of a uniform variable lies within 5 standard errors of the midpoint.
        standard_error = (high - low) / np.sqrt(12) / np.sqrt(self.RUNS)
        assert abs(values.mean() - (low + high) / 2) <= 5 * standard_error
        assert kstest(values, 'uniform', args=(low, high - low)).pvalue > 1e-4

    @pytest.mark.parametrize('key, other_transform', [
        ('noise range, reciprocal', lambda sigma: np.log10(1 / sigma)),
        ('noise range, reciprocal log', lambda sigma: 1 / sigma),
    ])
    def test_the_other_scale_is_not_uniform(self, key, other_transform):
        """The uniformity check tells the two keys apart: over a 5-fold range the other scale is far from uniform."""
        from numpy.random import SeedSequence
        from scipy.stats import kstest
        values = other_transform(np.array(self._parsed(key, seed=SeedSequence(5))['noise series']))
        low, high = other_transform(self.HIGH), other_transform(self.LOW)
        assert kstest(values, 'uniform', args=(low, high - low)).pvalue < 1e-10

    def test_seed_reproduces_the_draws(self):
        from numpy.random import SeedSequence
        first = self._parsed(runs=50, seed=SeedSequence(9))['noise series']
        assert self._parsed(runs=50, seed=SeedSequence(9))['noise series'] == first
        assert self._parsed(runs=50, seed=SeedSequence(10))['noise series'] != first

    @pytest.mark.parametrize('value, message', [
        ([0.1], "Invalid noise range [0.1]. Must be two numeric values."),
        ([0.01, 0.02, 0.03], "Invalid noise range [0.01, 0.02, 0.03]. Must be two numeric values."),
        ([0.01, 'high'], "Invalid noise range [0.01, 'high']. Must be two numeric values."),
        ([True, 0.05], "Invalid noise range [True, 0.05]. Must be two numeric values."),
        (0.05, "Invalid noise range 0.05. Must be two numeric values."),
        ([0, 0.05], "Invalid noise range [0, 0.05]. Both values must be positive and finite."),
        ([-0.01, 0.05], "Invalid noise range [-0.01, 0.05]. Both values must be positive and finite."),
        ([0.01, float('inf')], "Invalid noise range [0.01, inf]. Both values must be positive and finite."),
        ([0.05, 0.01],
         "Invalid noise range [0.05, 0.01]. The first value (low) must be less than the second (high)."),
        ([0.05, 0.05],
         "Invalid noise range [0.05, 0.05]. The first value (low) must be less than the second (high)."),
    ])
    @pytest.mark.parametrize('key', ['noise range, reciprocal', 'noise range, reciprocal log'])
    def test_malformed_range_rejected(self, key, value, message):
        import re
        with pytest.raises(ValueError, match=re.escape(message)):
            self._parsed(key, runs=5, value=value)

    @pytest.mark.parametrize('runs, error', [
        (0, ValueError), (-3, ValueError), (2.5, TypeError), ('ten', TypeError)])
    def test_number_of_runs_must_be_a_positive_integer(self, runs, error):
        with pytest.raises(error, match="'number of runs' must be"):
            self._parsed(runs=runs)

    def test_file_with_malformed_range_is_rejected_by_name(self, tmp_path):
        import ast
        params = ast.literal_eval((EXAMPLES_DIR / 'quick_test.txt').read_text())
        del params['noise value']
        params['noise range, reciprocal log'] = [0.05, 0.01]
        params_file = tmp_path / 'bad_range.txt'
        params_file.write_text(str(params))
        with pytest.raises(ValueError, match=r"Invalid noise range \[0\.05, 0\.01\]\. The first value"):
            open_parameters(params_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


QUICK_TEST_FILE = Path(__file__).parent.parent / 'data' / 'examples' / 'quick_test.txt'


def quick_test_raw() -> dict:
    """The dictionary in data/examples/quick_test.txt, before any parsing."""
    import ast
    return ast.literal_eval(QUICK_TEST_FILE.read_text(encoding='utf-8'))


class TestOptionalKeysInEitherSpelling:
    """A value given under the underscore spelling of an optional key is not replaced by the default."""

    CANONICAL_VALUES = {
        'filename_slug': 'my_run', 'image_type': 'svg', 'image_dpi': 300,
        'image_width': 20, 'image_height': 12, 'retain_profile_data': 1,
    }
    LEGACY_KEYS = {
        'filename_slug': 'filename slug', 'image_type': 'image type', 'image_dpi': 'image dpi',
        'image_width': 'image width', 'image_height': 'image height',
        'retain_profile_data': 'retain profile data',
    }

    def test_canonical_optional_keys_keep_their_values(self):
        raw = {key: value for key, value in quick_test_raw().items()
               if key not in self.LEGACY_KEYS.values()}
        parsed = parameter_parser(raw | self.CANONICAL_VALUES)
        for canonical_key, value in self.CANONICAL_VALUES.items():
            assert parsed[canonical_key] == value
            assert parsed[self.LEGACY_KEYS[canonical_key]] == value

    def test_defaults_fill_absent_optional_keys(self):
        raw = {key: value for key, value in quick_test_raw().items()
               if key not in self.LEGACY_KEYS.values()}
        parsed = parameter_parser(raw)
        assert parsed['filename slug'] == 'dice_simulation'
        assert (parsed['image type'], parsed['image width'], parsed['image height'], parsed['image dpi']) \
            == ('png', 10, 6, 100)
        assert parsed['retain profile data'] == 0


class TestFilenameSlugIsCheckedBeforeTheRun:
    """validate_parameters rejects a slug that would write outside the output folder."""

    @pytest.mark.parametrize('slug', ['../elsewhere/run', 'sub/run', 'C:/somewhere/run', '..', '.', 'run.', 'run ',
                                      'NUL', 'com1.v2', 'run\tone'])
    def test_slug_that_is_not_a_plain_name_is_rejected(self, slug):
        parsed = parameter_parser(quick_test_raw() | {'filename slug': slug})
        with pytest.raises(ValueError, match="'filename slug' cannot"):
            validate_parameters(parsed)

    def test_plain_slug_is_accepted(self):
        validate_parameters(parameter_parser(quick_test_raw() | {'filename slug': 'run_2.v3'}))


class TestProfileCenterInsideTheFitRange:
    """validate_parameters rejects a profile centered where the fit does not look for it."""

    @pytest.mark.parametrize('mean', [1.01, -1.3, 2.0])
    def test_center_outside_the_central_part_of_the_window_is_rejected(self, mean):
        parsed = parameter_parser(quick_test_raw() | {'mean_0': mean})
        with pytest.raises(ValueError, match="'mean_0' is .* outside the central 40% of the spatial window"):
            validate_parameters(parsed)

    @pytest.mark.parametrize('mean', [0.0, 0.9, -1.0, 1.0])
    def test_center_inside_is_accepted(self, mean):
        validate_parameters(parameter_parser(quick_test_raw() | {'mean_0': mean}))

    def test_limits_are_those_the_fit_uses(self):
        """An accepted center at the limit is fitted correctly; this ties the rule to the fit."""
        from dice.core.fitting import center_search_limits, gauss_fitting
        from dice.core.profiles import gaussian

        parsed = parameter_parser(quick_test_raw())
        x_axis = parsed['x array']
        low, high = center_search_limits(x_axis)
        assert (low, high) == pytest.approx((-1.0, 1.0))

        sigma2 = parsed['sigma^2_0']
        at_limit = gaussian(x_axis, high, sigma2, 1.0)
        beyond = gaussian(x_axis, high + 0.3, sigma2, 1.0)
        fitted_at_limit = gauss_fitting(x_axis, [at_limit])
        fitted_beyond = gauss_fitting(x_axis, [beyond])

        assert fitted_at_limit['sigma^2_t estimates'][0] == pytest.approx(sigma2, rel=1e-6)
        assert fitted_beyond['sigma^2_t estimates'][0] != pytest.approx(sigma2, rel=0.2)


class TestFileThatCannotRunIsRejected:
    """A value the run would fail on, or would read differently from the GUI, fails when the file is parsed."""

    @pytest.mark.parametrize('changes, removed, error, message', [
        ({'time range': [0, 1, 1]}, (), ValueError, r"The number of steps in 'time range' must be >= 2, got 1"),
        ({'time range': [0, 1, 5.5]}, (), TypeError, r"The number of steps in 'time range' must be an integer"),
        ({'time range': [1, 0, 5]}, (), ValueError, r"The start of 'time range' \(1\) must be less than its stop \(0\)"),
        ({'time range': [1, 1, 5]}, (), ValueError, r"The start of 'time range'"),
        ({'time series': []}, ('time range',), ValueError, r"'time series'"),
        ({'time series': [0.5]}, ('time range',), ValueError, r"'time series' holds 1 time point"),
        ({'time series': [0, 0.5, 0.5]}, ('time range',), ValueError,
         r"Invalid 'time series': Time points must be unique"),
        ({'time series': [0, float('nan')]}, ('time range',), ValueError, r"Invalid 'time series': .*finite"),
        ({'number of runs': 3.5}, (), TypeError, r"'number of runs' must be an integer"),
        ({'noise value': -0.02}, (), ValueError, r"'noise value' must be 0 or greater, got -0.02"),
        ({'FWHM_0': 0}, (), ValueError, r"'FWHM_0' must be greater than 0"),
        ({'sigma_0': 0}, ('FWHM_0',), ValueError, r"'sigma_0' must be greater than 0"),
        ({'sigma_0': -0.4}, ('FWHM_0',), ValueError, r"'sigma_0' must be greater than 0"),
        ({'pixel width': 1}, (), ValueError, r"'pixel width' \(the number of pixels\) is 1; a profile needs at least 16"),
        ({'pixel width': 15}, (), ValueError, r"'pixel width' \(the number of pixels\) is 15"),
        ({'pixel width': 50.5}, (), TypeError, r"'pixel width' must be an integer"),
        ({'spatial width': 0}, (), ValueError, r"'spatial width' must be a number greater than 0"),
    ])
    def test_value_is_rejected_with_the_key_of_the_file(self, changes, removed, error, message):
        raw = {key: value for key, value in quick_test_raw().items() if key not in removed}
        with pytest.raises(error, match=message):
            parameter_parser(raw | changes)

    def test_time_series_is_sorted_as_the_gui_sorts_it(self):
        raw = {key: value for key, value in quick_test_raw().items() if key != 'time range'}
        parsed = parameter_parser(raw | {'time series': [1, 0, 0.5]})
        assert parsed['time series'].tolist() == [0.0, 0.5, 1.0]

    def test_whole_number_given_as_a_float_becomes_an_int(self):
        parsed = parameter_parser(quick_test_raw() | {'number of runs': 100.0, 'pixel width': 50.0})
        assert parsed['number of runs'] == 100 and isinstance(parsed['number of runs'], int)
        assert len(parsed['x array']) == 50

    def test_smallest_profile_the_cnr_estimate_takes_is_accepted_and_runs(self):
        """The limit is the one of the CNR estimate: a run with it gives an estimate."""
        from dice.core.noise import MINIMUM_PROFILE_POINTS
        from dice.analysis.simulation import run_single_simulation
        from dice.utils.legacy_compatibility import create_parameters_from_legacy

        parsed = parameter_parser(quick_test_raw() | {'pixel width': MINIMUM_PROFILE_POINTS})
        parameters = create_parameters_from_legacy(
            parameters_dict=parsed,
            diffusion_coefficient=parsed['nominal diffusion coefficient'],
            lifetime=parsed['nominal lifetime (tau)'],
            diffusion_length=parsed['nominal diffusion length'],
        )
        run = run_single_simulation(0, parsed['x array'], parsed['time series'], parameters,
                                    parsed['noise series'][0], seed=1)
        assert np.isfinite(run.cnr_0_estimate)


class TestValuesThatOnlyThePlotWouldReject:
    """A value that the plot cannot use is rejected when the file is parsed, not after the simulation."""

    def test_number_of_bins_given_as_a_float_becomes_an_int(self):
        parsed = parameter_parser(quick_test_raw() | {'image numbins': 35.0})
        assert parsed['image numbins'] == 35 and isinstance(parsed['image numbins'], int)

    @pytest.mark.parametrize('bins, error', [(0, ValueError), (-5, ValueError), (35.5, TypeError), ('many', TypeError)])
    def test_number_of_bins_that_is_no_count_is_rejected(self, bins, error):
        with pytest.raises(error, match="'image numbins'"):
            parameter_parser(quick_test_raw() | {'image numbins': bins})

    def test_zero_diffusion_is_rejected(self):
        """The estimates are reported as ratios to the nominal diffusion coefficient."""
        with pytest.raises(ValueError, match="The nominal diffusion coefficient must be greater than 0"):
            parameter_parser(quick_test_raw() | {'nominal diffusion length': 0})
        raw = {key: value for key, value in quick_test_raw().items() if key != 'nominal diffusion length'}
        with pytest.raises(ValueError, match="The nominal diffusion coefficient must be greater than 0"):
            parameter_parser(raw | {'nominal diffusion coefficient': 0, 'nominal lifetime (tau)': 1})

    def test_pixel_count_given_as_a_float_is_stored_as_an_int(self):
        parsed = parameter_parser(quick_test_raw() | {'pixel width': 50.0})
        assert parsed['pixel width'] == 50 and isinstance(parsed['pixel width'], int)

    def test_number_of_runs_below_one_is_rejected_by_the_parser(self):
        with pytest.raises(ValueError, match="'number of runs' must be >= 1, got 0"):
            parameter_parser(quick_test_raw() | {'number of runs': 0})
