"""
Unit tests for dice.parameters module.
"""
import pytest
import numpy as np
from dice.parameters import (
    check_for_unique_key,
    handle_time_parameters,
    handle_profile_width_parameters,
    parameter_parser,
    open_parameters,
)


@pytest.mark.unit
class TestCheckForUniqueKey:
    """Tests for check_for_unique_key() function."""

    def test_unique_key_found(self):
        """Test that unique key is found and returned."""
        params = {'time range': (0, 5, 6)}
        result = check_for_unique_key(params, ['time range', 'time series'])
        assert result == 'time range'

    def test_multiple_keys_error(self):
        """Test that multiple keys raises ValueError."""
        params = {'time range': (0, 5, 6), 'time series': [1, 2, 3]}
        with pytest.raises(ValueError, match="More than one parameter provided"):
            check_for_unique_key(params, ['time range', 'time series'])

    def test_no_keys_error(self):
        """Test that no keys raises ValueError."""
        params = {'something else': 42}
        with pytest.raises(ValueError, match="No parameter provided"):
            check_for_unique_key(params, ['time range', 'time series'])

    def test_invalid_dict_type(self):
        """Test that non-dict raises TypeError."""
        with pytest.raises(TypeError, match="must be a dictionary"):
            check_for_unique_key("not a dict", ['time range'])

    def test_invalid_keys_type(self):
        """Test that non-list raises TypeError."""
        with pytest.raises(TypeError, match="must be a list"):
            check_for_unique_key({}, "not a list")


@pytest.mark.unit
class TestHandleTimeParameters:
    """Tests for handle_time_parameters() function."""

    def test_time_range_to_series(self):
        """Test conversion of time range to time series."""
        params = {'time range': (0.0, 5.0, 6)}
        handle_time_parameters(params)

        assert 'time series' in params
        assert 'time range' not in params
        assert isinstance(params['time series'], np.ndarray)
        assert len(params['time series']) == 6
        assert params['time series'][0] == pytest.approx(0.0)
        assert params['time series'][-1] == pytest.approx(5.0)

    def test_time_series_unchanged(self):
        """Test that existing time series is left unchanged."""
        original_series = np.array([1, 2, 3, 4, 5])
        params = {'time series': original_series}
        handle_time_parameters(params)

        assert 'time series' in params
        assert np.array_equal(params['time series'], original_series)

    def test_invalid_range_not_three_values(self):
        """Test that time range without 3 values raises ValueError."""
        params = {'time range': (0.0, 5.0)}
        with pytest.raises(ValueError, match="must contain three values"):
            handle_time_parameters(params)

    def test_invalid_range_non_numeric(self):
        """Test that non-numeric time range raises ValueError."""
        params = {'time range': ('a', 'b', 'c')}
        with pytest.raises(ValueError, match="must be numeric"):
            handle_time_parameters(params)

    def test_invalid_range_not_sequence(self):
        """Test that non-sequence time range raises ValueError."""
        params = {'time range': 42}
        with pytest.raises(ValueError, match="must be a sequence"):
            handle_time_parameters(params)


@pytest.mark.unit
class TestHandleProfileWidthParameters:
    """Tests for handle_profile_width_parameters() function."""

    def test_fwhm_conversion(self):
        """Test FWHM to sigma^2 conversion."""
        fwhm = 2.355
        params = {'FWHM_0': fwhm}
        result = handle_profile_width_parameters(params)

        # FWHM = 2.355 * sigma, so sigma = FWHM / 2.355
        # sigma^2 = (FWHM / 2.355)^2
        expected = (fwhm / (2 * np.sqrt(2 * np.log(2)))) ** 2
        assert result == pytest.approx(expected)

    def test_sigma_conversion(self):
        """Test sigma to sigma^2 conversion."""
        sigma = 1.5
        params = {'sigma_0': sigma}
        result = handle_profile_width_parameters(params)

        expected = sigma ** 2
        assert result == pytest.approx(expected)

    def test_both_parameters_error(self):
        """Test that providing both FWHM and sigma raises ValueError."""
        params = {'FWHM_0': 2.355, 'sigma_0': 1.0}
        with pytest.raises(ValueError, match="More than one parameter"):
            handle_profile_width_parameters(params)

    def test_no_parameters_error(self):
        """Test that providing neither FWHM nor sigma raises ValueError."""
        params = {}
        with pytest.raises(ValueError, match="No parameter provided"):
            handle_profile_width_parameters(params)


@pytest.mark.unit
class TestParameterParser:
    """Tests for parameter_parser() function."""

    def test_basic_parsing(self):
        """Test basic parameter parsing."""
        params = {
            'number of runs': 10,
            'spatial width': 10.0,
            'pixel width': 100,
            'mean_0': 0.0,
            'amplitude_0': 1.0,
            'sigma_0': 1.0,
            'time range': (0.0, 5.0, 6),
            'noise value': 0.01,
            'nominal diffusion length': 1.0,
        }

        result = parameter_parser(params)

        assert 'x array' in result
        assert 'time series' in result
        assert 'noise series' in result
        assert 't0 Gaussian sigma^2, amplitude, mean' in result
        assert isinstance(result['x array'], np.ndarray)
        assert isinstance(result['time series'], np.ndarray)

    def test_missing_required_key(self):
        """Test that missing required parameter raises KeyError."""
        params = {
            'number of runs': 10,
            'spatial width': 10.0,
            'pixel width': 100,
            # Missing 'mean_0' and 'amplitude_0'
        }

        with pytest.raises(KeyError, match="required parameter.*missing"):
            parameter_parser(params)

    def test_with_fwhm_instead_of_sigma(self):
        """Test parameter parsing with FWHM_0 instead of sigma_0."""
        params = {
            'number of runs': 10,
            'spatial width': 10.0,
            'pixel width': 100,
            'mean_0': 0.0,
            'amplitude_0': 1.0,
            'FWHM_0': 2.355,
            'time series': np.array([0, 1, 2, 3, 4]),
            'noise value': 0.01,
            'nominal diffusion length': 1.0,
        }

        result = parameter_parser(params)

        assert 't0 Gaussian sigma^2, amplitude, mean' in result
        # FWHM should have been converted to sigma^2
        assert result['t0 Gaussian sigma^2, amplitude, mean'][0] > 0


@pytest.mark.integration
class TestOpenParameters:
    """Integration tests for open_parameters() function."""

    def test_open_parameters_basic(self, mock_parameters_file):
        """Test opening and parsing parameter file."""
        result = open_parameters(mock_parameters_file)

        assert isinstance(result, dict)
        assert 'x array' in result
        assert 'time series' in result
        assert 'noise series' in result

    def test_open_parameters_file_not_found(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="was not found"):
            open_parameters("nonexistent_file.txt")

    def test_open_parameters_invalid_syntax(self, tmp_path):
        """Test that invalid syntax raises ValueError."""
        invalid_file = tmp_path / "invalid.txt"
        invalid_file.write_text("this is not a dictionary")

        with pytest.raises(ValueError, match="Error evaluating"):
            open_parameters(str(invalid_file))
