"""
Tests for dice.utils.units module.

Tests cover conversion functions, validation, abbreviations, and the
resolve_units boundary function.
"""

import pytest
import numpy as np

from dice.utils.units import (
    Dimension,
    VALID_LENGTH_UNITS,
    VALID_TIME_UNITS,
    LENGTH_ABBREVIATIONS,
    TIME_ABBREVIATIONS,
    PARAMETER_DIMENSIONS,
    validate_length_unit,
    validate_time_unit,
    convert_length,
    convert_time,
    convert_length_squared,
    convert_diffusion_coefficient,
    length_abbreviation,
    time_abbreviation,
    diffusion_coefficient_label,
    global_length_unit,
    global_time_unit,
    resolve_units,
    validate_output_units,
)


class TestValidation:
    """Test unit validation functions."""

    def test_valid_length_units(self):
        for unit in VALID_LENGTH_UNITS:
            validate_length_unit(unit)

    def test_invalid_length_unit(self):
        with pytest.raises(ValueError, match="Unknown length unit"):
            validate_length_unit("furlongs")

    def test_valid_time_units(self):
        for unit in VALID_TIME_UNITS:
            validate_time_unit(unit)

    def test_invalid_time_unit(self):
        with pytest.raises(ValueError, match="Unknown time unit"):
            validate_time_unit("fortnights")


class TestValidationNamesTheKey:
    """An output unit error names the key it was given for; a non-string is an unknown unit."""

    def test_key_named_in_the_message(self):
        with pytest.raises(ValueError, match="Unknown length unit 'furlong' for 'some_key'. Valid units: "):
            validate_length_unit('furlong', 'some_key')
        with pytest.raises(ValueError, match="Unknown time unit 'fortnight' for 'other_key'. Valid units: "):
            validate_time_unit('fortnight', 'other_key')

    @pytest.mark.parametrize('unit', [None, 5, ['micrometer']])
    def test_non_string_is_unknown(self, unit):
        with pytest.raises(ValueError, match="Unknown length unit"):
            validate_length_unit(unit)
        with pytest.raises(ValueError, match="Unknown time unit"):
            validate_time_unit(unit)


class TestValidateOutputUnits:
    """Only the diffusion coefficient output unit is configurable, as a pair."""

    def test_no_output_keys(self):
        validate_output_units({'length unit': 'micrometer'})

    def test_valid_pair(self):
        validate_output_units({'output_diffusion_length_unit': 'centimeter',
                               'output_diffusion_time_unit': 'second'})

    @pytest.mark.parametrize('present', ['output_diffusion_length_unit', 'output_diffusion_time_unit'])
    def test_one_of_the_pair_rejected(self, present):
        with pytest.raises(ValueError, match="Give both 'output_diffusion_length_unit' and "
                                             "'output_diffusion_time_unit', or neither."):
            validate_output_units({present: 'second' if 'time' in present else 'centimeter'})

    @pytest.mark.parametrize('pair, message', [
        (('furlong', 'second'), "Unknown length unit 'furlong' for 'output_diffusion_length_unit'"),
        (('centimeter', 'fortnight'), "Unknown time unit 'fortnight' for 'output_diffusion_time_unit'"),
        (('cm', 's'), "Unknown length unit 'cm' for 'output_diffusion_length_unit'"),
    ])
    def test_unknown_unit_names_the_key(self, pair, message):
        with pytest.raises(ValueError, match=message):
            validate_output_units({'output_diffusion_length_unit': pair[0],
                                   'output_diffusion_time_unit': pair[1]})

    @pytest.mark.parametrize('key', ['output_length_unit', 'output_time_unit'])
    def test_unsupported_key_rejected_with_the_alternative(self, key):
        with pytest.raises(ValueError) as error:
            validate_output_units({key: 'nanometer'})
        assert str(error.value).startswith(f"'{key}' is not supported: the summary and results files "
                                           "report lengths and times in the global 'length unit'")
        assert "'output_diffusion_length_unit' and 'output_diffusion_time_unit'" in str(error.value)


class TestConvertLength:
    """Test length unit conversions."""

    def test_same_unit_returns_exact(self):
        for unit in VALID_LENGTH_UNITS:
            assert convert_length(42.0, unit, unit) == 42.0

    def test_meter_to_centimeter(self):
        result = convert_length(1.0, 'meter', 'centimeter')
        assert np.isclose(result, 100.0, rtol=1e-12)

    def test_micrometer_to_nanometer(self):
        result = convert_length(1.0, 'micrometer', 'nanometer')
        assert np.isclose(result, 1000.0, rtol=1e-12)

    def test_nanometer_to_micrometer(self):
        result = convert_length(500.0, 'nanometer', 'micrometer')
        assert np.isclose(result, 0.5, rtol=1e-12)

    def test_angstrom_to_nanometer(self):
        result = convert_length(10.0, 'angstrom', 'nanometer')
        assert np.isclose(result, 1.0, rtol=1e-12)

    def test_round_trip_all_pairs(self):
        """Convert A->B->A and verify the value is preserved."""
        units = list(VALID_LENGTH_UNITS.keys())
        value = 3.14159
        for u1 in units:
            for u2 in units:
                converted = convert_length(value, u1, u2)
                back = convert_length(converted, u2, u1)
                assert np.isclose(back, value, rtol=1e-10), (
                    f"Round trip {u1}->{u2}->{u1} failed: {value} -> {converted} -> {back}"
                )

    def test_invalid_from_unit(self):
        with pytest.raises(ValueError):
            convert_length(1.0, 'invalid', 'meter')

    def test_invalid_to_unit(self):
        with pytest.raises(ValueError):
            convert_length(1.0, 'meter', 'invalid')

    def test_zero_value(self):
        assert convert_length(0.0, 'meter', 'nanometer') == 0.0

    def test_negative_value(self):
        result = convert_length(-5.0, 'micrometer', 'nanometer')
        assert np.isclose(result, -5000.0, rtol=1e-12)


class TestConvertTime:
    """Test time unit conversions."""

    def test_same_unit_returns_exact(self):
        for unit in VALID_TIME_UNITS:
            assert convert_time(42.0, unit, unit) == 42.0

    def test_second_to_nanosecond(self):
        result = convert_time(1.0, 'second', 'nanosecond')
        assert np.isclose(result, 1e9, rtol=1e-12)

    def test_nanosecond_to_picosecond(self):
        result = convert_time(1.0, 'nanosecond', 'picosecond')
        assert np.isclose(result, 1000.0, rtol=1e-12)

    def test_picosecond_to_nanosecond(self):
        result = convert_time(500.0, 'picosecond', 'nanosecond')
        assert np.isclose(result, 0.5, rtol=1e-12)

    def test_round_trip_all_pairs(self):
        units = list(VALID_TIME_UNITS.keys())
        value = 2.71828
        for u1 in units:
            for u2 in units:
                converted = convert_time(value, u1, u2)
                back = convert_time(converted, u2, u1)
                assert np.isclose(back, value, rtol=1e-10), (
                    f"Round trip {u1}->{u2}->{u1} failed: {value} -> {converted} -> {back}"
                )

    def test_invalid_from_unit(self):
        with pytest.raises(ValueError):
            convert_time(1.0, 'invalid', 'second')

    def test_invalid_to_unit(self):
        with pytest.raises(ValueError):
            convert_time(1.0, 'second', 'invalid')


class TestConvertLengthSquared:
    """Test length-squared (area) conversions."""

    def test_same_unit_returns_exact(self):
        for unit in VALID_LENGTH_UNITS:
            assert convert_length_squared(42.0, unit, unit) == 42.0

    def test_meter_squared_to_centimeter_squared(self):
        result = convert_length_squared(1.0, 'meter', 'centimeter')
        assert np.isclose(result, 10000.0, rtol=1e-12)

    def test_micrometer_squared_to_nanometer_squared(self):
        result = convert_length_squared(1.0, 'micrometer', 'nanometer')
        assert np.isclose(result, 1e6, rtol=1e-12)

    def test_nanometer_squared_to_micrometer_squared(self):
        result = convert_length_squared(1e6, 'nanometer', 'micrometer')
        assert np.isclose(result, 1.0, rtol=1e-12)

    def test_round_trip(self):
        value = 5.5
        result = convert_length_squared(
            convert_length_squared(value, 'micrometer', 'angstrom'),
            'angstrom', 'micrometer'
        )
        assert np.isclose(result, value, rtol=1e-10)


class TestConvertDiffusionCoefficient:
    """Test diffusion coefficient (length^2/time) conversions."""

    def test_same_units_returns_exact(self):
        assert convert_diffusion_coefficient(
            1.5, 'micrometer', 'nanosecond', 'micrometer', 'nanosecond'
        ) == 1.5

    def test_square_micrometer_per_ns_to_cm2_per_s(self):
        # 1 μm^2/ns = (1e-4 cm)^2 / (1e-9 s) = 1e-8 / 1e-9 = 10 cm^2/s
        # Wait, that's for slope. For D, the factor is the same.
        result = convert_diffusion_coefficient(
            1.0, 'micrometer', 'nanosecond', 'centimeter', 'second'
        )
        expected = (1e-4)**2 / 1e-9  # = 1e-8 / 1e-9 = 10.0
        assert np.isclose(result, expected, rtol=1e-10)

    def test_cm2_per_s_to_square_micrometer_per_ns(self):
        # Inverse of above
        result = convert_diffusion_coefficient(
            10.0, 'centimeter', 'second', 'micrometer', 'nanosecond'
        )
        assert np.isclose(result, 1.0, rtol=1e-10)

    def test_nm2_per_ps_to_square_micrometer_per_ns(self):
        # 1 nm^2/ps = (1e-3 μm)^2 / (1e-3 ns) = 1e-6/1e-3 = 1e-3 μm^2/ns
        result = convert_diffusion_coefficient(
            1.0, 'nanometer', 'picosecond', 'micrometer', 'nanosecond'
        )
        assert np.isclose(result, 1e-3, rtol=1e-10)

    def test_round_trip(self):
        value = 0.05
        intermediate = convert_diffusion_coefficient(
            value, 'micrometer', 'nanosecond', 'centimeter', 'second'
        )
        back = convert_diffusion_coefficient(
            intermediate, 'centimeter', 'second', 'micrometer', 'nanosecond'
        )
        assert np.isclose(back, value, rtol=1e-10)

    def test_invalid_length_unit(self):
        with pytest.raises(ValueError, match="Unknown length unit"):
            convert_diffusion_coefficient(1.0, 'invalid', 'second', 'meter', 'second')

    def test_invalid_time_unit(self):
        with pytest.raises(ValueError, match="Unknown time unit"):
            convert_diffusion_coefficient(1.0, 'meter', 'invalid', 'meter', 'second')

    def test_consistency_with_slope_to_diffusion_constant(self):
        """Verify the refactored slope_to_diffusion_constant matches."""
        from dice.utils.converters import slope_to_diffusion_constant

        slope = 2.5
        # slope_to_diffusion_constant converts slope -> D in cm^2/s
        d_legacy = slope_to_diffusion_constant(slope, 'micrometer', 'nanosecond')
        # Manual: D = slope/2, then convert
        d_manual = convert_diffusion_coefficient(
            slope / 2, 'micrometer', 'nanosecond', 'centimeter', 'second'
        )
        assert np.isclose(d_legacy, d_manual, rtol=1e-12)


class TestAbbreviations:
    """Test unit abbreviation functions."""

    def test_all_length_units_have_abbreviations(self):
        for unit in VALID_LENGTH_UNITS:
            abbr = length_abbreviation(unit)
            assert abbr != unit or unit == 'meter'  # meter -> m, not meter

    def test_all_time_units_have_abbreviations(self):
        for unit in VALID_TIME_UNITS:
            abbr = time_abbreviation(unit)
            assert abbr != unit or unit == 'second'

    def test_specific_abbreviations(self):
        assert length_abbreviation('nanometer') == 'nm'
        assert time_abbreviation('nanosecond') == 'ns'
        assert time_abbreviation('picosecond') == 'ps'

    def test_unknown_unit_returns_itself(self):
        assert length_abbreviation('unknown') == 'unknown'
        assert time_abbreviation('unknown') == 'unknown'

    def test_diffusion_coefficient_label(self):
        label = diffusion_coefficient_label('micrometer', 'nanosecond')
        assert 'm' in label  # contains micrometer abbreviation
        assert 'ns' in label
        assert '\u00b2' in label  # superscript 2


class TestParameterDimensions:
    """Test the parameter dimension mapping."""

    def test_length_parameters(self):
        assert PARAMETER_DIMENSIONS['fwhm_0'] == Dimension.LENGTH
        assert PARAMETER_DIMENSIONS['sigma_0'] == Dimension.LENGTH
        assert PARAMETER_DIMENSIONS['mu_0'] == Dimension.LENGTH
        assert PARAMETER_DIMENSIONS['spatial_width'] == Dimension.LENGTH
        assert PARAMETER_DIMENSIONS['diffusion_length'] == Dimension.LENGTH

    def test_time_parameters(self):
        assert PARAMETER_DIMENSIONS['lifetime'] == Dimension.TIME
        assert PARAMETER_DIMENSIONS['time_start'] == Dimension.TIME
        assert PARAMETER_DIMENSIONS['time_stop'] == Dimension.TIME

    def test_compound_parameters(self):
        assert PARAMETER_DIMENSIONS['diffusion_coefficient'] == Dimension.LENGTH_SQUARED_PER_TIME

    def test_length_squared_parameters(self):
        assert PARAMETER_DIMENSIONS['sigma2_0'] == Dimension.LENGTH_SQUARED


class TestResolveUnits:
    """Test the resolve_units boundary function."""

    def test_no_overrides_returns_unchanged(self):
        """Parameters without _unit keys should pass through unchanged."""
        params = {
            'length_unit': 'micrometer',
            'time_unit': 'nanosecond',
            'FWHM_0': 1.0,
            'spatial width': 10.0,
            'nominal lifetime (tau)': 5.0,
        }
        result = resolve_units(params)
        assert result['FWHM_0'] == 1.0
        assert result['spatial width'] == 10.0
        assert result['nominal lifetime (tau)'] == 5.0

    def test_fwhm_unit_override(self):
        """FWHM in nanometers should be converted to the global micrometer unit."""
        params = {
            'length_unit': 'micrometer',
            'time_unit': 'nanosecond',
            'FWHM_0': 500.0,
            'fwhm_0_unit': 'nanometer',
        }
        result = resolve_units(params)
        assert np.isclose(result['FWHM_0'], 0.5, rtol=1e-12)
        assert 'fwhm_0_unit' not in result

    def test_spatial_width_override(self):
        params = {
            'length_unit': 'micrometer',
            'time_unit': 'nanosecond',
            'spatial width': 0.01,
            'spatial_width_unit': 'millimeter',
        }
        result = resolve_units(params)
        # 0.01 mm = 10 μm
        assert np.isclose(result['spatial width'], 10.0, rtol=1e-12)
        assert 'spatial_width_unit' not in result

    def test_lifetime_unit_override(self):
        params = {
            'length_unit': 'micrometer',
            'time_unit': 'nanosecond',
            'nominal lifetime (tau)': 500.0,
            'lifetime_unit': 'picosecond',
        }
        result = resolve_units(params)
        # 500 ps = 0.5 ns
        assert np.isclose(result['nominal lifetime (tau)'], 0.5, rtol=1e-12)
        assert 'lifetime_unit' not in result

    def test_diffusion_coefficient_override(self):
        params = {
            'length_unit': 'micrometer',
            'time_unit': 'nanosecond',
            'nominal diffusion coefficient': 1e-5,
            'diffusion_coefficient_length_unit': 'centimeter',
            'diffusion_coefficient_time_unit': 'second',
        }
        result = resolve_units(params)
        # 1e-5 cm^2/s -> μm^2/ns
        expected = convert_diffusion_coefficient(
            1e-5, 'centimeter', 'second', 'micrometer', 'nanosecond'
        )
        assert np.isclose(result['nominal diffusion coefficient'], expected, rtol=1e-10)
        assert 'diffusion_coefficient_length_unit' not in result
        assert 'diffusion_coefficient_time_unit' not in result

    def test_time_range_override(self):
        params = {
            'length_unit': 'micrometer',
            'time_unit': 'nanosecond',
            'time range': [0, 1000, 10],
            'time_range_unit': 'picosecond',
        }
        result = resolve_units(params)
        # [0, 1000, 10] ps -> [0, 1, 10] ns (steps unchanged)
        assert np.isclose(result['time range'][0], 0.0, rtol=1e-12)
        assert np.isclose(result['time range'][1], 1.0, rtol=1e-12)
        assert result['time range'][2] == 10  # steps unchanged
        assert 'time_range_unit' not in result

    def test_time_series_override(self):
        params = {
            'length_unit': 'micrometer',
            'time_unit': 'nanosecond',
            'time series': [100, 500, 1000],
            'time_series_unit': 'picosecond',
        }
        result = resolve_units(params)
        # [100, 500, 1000] ps -> [0.1, 0.5, 1.0] ns
        assert np.isclose(result['time series'][0], 0.1, rtol=1e-12)
        assert np.isclose(result['time series'][1], 0.5, rtol=1e-12)
        assert np.isclose(result['time series'][2], 1.0, rtol=1e-12)
        assert 'time_series_unit' not in result

    def test_same_unit_override_no_conversion(self):
        """Override with the same unit as global should not change value."""
        params = {
            'length_unit': 'micrometer',
            'time_unit': 'nanosecond',
            'FWHM_0': 1.0,
            'fwhm_0_unit': 'micrometer',
        }
        result = resolve_units(params)
        assert result['FWHM_0'] == 1.0
        assert 'fwhm_0_unit' not in result

    def test_multiple_overrides(self):
        """Multiple parameters with different unit overrides."""
        params = {
            'length_unit': 'micrometer',
            'time_unit': 'nanosecond',
            'FWHM_0': 500.0,
            'fwhm_0_unit': 'nanometer',
            'spatial width': 10.0,
            # no override for spatial_width -- stays in μm
            'nominal lifetime (tau)': 500.0,
            'lifetime_unit': 'picosecond',
        }
        result = resolve_units(params)
        assert np.isclose(result['FWHM_0'], 0.5, rtol=1e-12)
        assert result['spatial width'] == 10.0
        assert np.isclose(result['nominal lifetime (tau)'], 0.5, rtol=1e-12)

    def test_legacy_key_format(self):
        """Works with legacy space-separated keys."""
        params = {
            'length unit': 'micrometer',
            'time unit': 'nanosecond',
            'FWHM_0': 500.0,
            'fwhm_0_unit': 'nanometer',
        }
        result = resolve_units(params)
        assert np.isclose(result['FWHM_0'], 0.5, rtol=1e-12)

    def test_diffusion_length_override(self):
        params = {
            'length_unit': 'micrometer',
            'time_unit': 'nanosecond',
            'nominal diffusion length': 100.0,
            'diffusion_length_unit': 'nanometer',
        }
        result = resolve_units(params)
        # 100 nm = 0.1 μm
        assert np.isclose(result['nominal diffusion length'], 0.1, rtol=1e-12)

    def test_mu_0_override(self):
        params = {
            'length_unit': 'micrometer',
            'time_unit': 'nanosecond',
            'mean_0': 500.0,
            'mu_0_unit': 'nanometer',
        }
        result = resolve_units(params)
        assert np.isclose(result['mean_0'], 0.5, rtol=1e-12)

    def test_partial_diffusion_coefficient_override_length_only(self):
        """Override only the length unit of diffusion coefficient."""
        params = {
            'length_unit': 'micrometer',
            'time_unit': 'nanosecond',
            'nominal diffusion coefficient': 0.01,
            'diffusion_coefficient_length_unit': 'nanometer',
            # no time override -> uses global nanosecond
        }
        result = resolve_units(params)
        # 0.01 nm^2/ns -> μm^2/ns
        expected = convert_diffusion_coefficient(
            0.01, 'nanometer', 'nanosecond', 'micrometer', 'nanosecond'
        )
        assert np.isclose(result['nominal diffusion coefficient'], expected, rtol=1e-10)

    def test_sigma2_0_override(self):
        params = {
            'length_unit': 'micrometer',
            'time_unit': 'nanosecond',
            'sigma^2_0': 250000.0,
            'sigma2_0_unit': 'nanometer',
        }
        result = resolve_units(params)
        # 250000 nm^2 = 0.25 μm^2
        assert np.isclose(result['sigma^2_0'], 0.25, rtol=1e-10)

    def test_time_start_unit_in_time_range_list(self):
        """time_start_unit converts the first element of a time range list."""
        params = {
            'length_unit': 'micrometer',
            'time_unit': 'nanosecond',
            'time range': [1000, 5, 10],
            'time_start_unit': 'picosecond',
        }
        result = resolve_units(params)
        # 1000 ps -> 1 ns; stop and steps unchanged
        assert np.isclose(result['time range'][0], 1.0, rtol=1e-12)
        assert result['time range'][1] == 5
        assert result['time range'][2] == 10
        assert 'time_start_unit' not in result

    def test_time_stop_unit_in_time_range_list(self):
        """time_stop_unit converts the second element of a time range list."""
        params = {
            'length_unit': 'micrometer',
            'time_unit': 'nanosecond',
            'time range': [0, 1000, 10],
            'time_stop_unit': 'picosecond',
        }
        result = resolve_units(params)
        # stop: 1000 ps -> 1 ns; start and steps unchanged
        assert result['time range'][0] == 0
        assert np.isclose(result['time range'][1], 1.0, rtol=1e-12)
        assert result['time range'][2] == 10
        assert 'time_stop_unit' not in result

    def test_mixed_time_start_stop_units_in_time_range(self):
        """Both time_start_unit and time_stop_unit with different source units."""
        params = {
            'length_unit': 'micrometer',
            'time_unit': 'nanosecond',
            'time range': [5000, 1, 10],
            'time_start_unit': 'picosecond',
            'time_stop_unit': 'microsecond',
        }
        result = resolve_units(params)
        # start: 5000 ps -> 5 ns; stop: 1 us -> 1000 ns
        assert np.isclose(result['time range'][0], 5.0, rtol=1e-12)
        assert np.isclose(result['time range'][1], 1000.0, rtol=1e-12)
        assert result['time range'][2] == 10
        assert 'time_start_unit' not in result
        assert 'time_stop_unit' not in result

    def test_time_start_unit_separate_key(self):
        """time_start_unit converts a separate time_start key (CLI path)."""
        params = {
            'length_unit': 'micrometer',
            'time_unit': 'nanosecond',
            'time_start': 1000.0,
            'time_start_unit': 'picosecond',
        }
        result = resolve_units(params)
        assert np.isclose(result['time_start'], 1.0, rtol=1e-12)
        assert 'time_start_unit' not in result

    def test_time_range_unit_and_time_series_unit_both_present(self):
        """time_range_unit should convert both; time_series_unit is skipped."""
        params = {
            'length_unit': 'micrometer',
            'time_unit': 'nanosecond',
            'time range': [0.0, 1000.0, 5],
            'time series': [0.0, 500.0, 1000.0],
            'time_range_unit': 'picosecond',
            'time_series_unit': 'femtosecond',  # should be ignored
        }
        result = resolve_units(params)
        # time_range_unit converts both time range and time series to nanosecond
        assert np.isclose(result['time range'][0], 0.0, atol=1e-12)
        assert np.isclose(result['time range'][1], 1.0, rtol=1e-12)
        # time series also converted by time_range_unit (ps -> ns), NOT by time_series_unit
        assert np.isclose(result['time series'][1], 0.5, rtol=1e-12)

    def test_time_range_unit_and_time_start_stop_units(self):
        """time_range_unit should win; time_start/stop_unit are skipped."""
        params = {
            'length_unit': 'micrometer',
            'time_unit': 'nanosecond',
            'time range': [0.0, 1000.0, 5],
            'time_range_unit': 'picosecond',
            'time_start_unit': 'femtosecond',
            'time_stop_unit': 'femtosecond',
        }
        result = resolve_units(params)
        # time_range_unit converts first; start/stop guards prevent double conversion
        assert np.isclose(result['time range'][0], 0.0, atol=1e-12)
        assert np.isclose(result['time range'][1], 1.0, rtol=1e-12)

    def test_missing_value_key_for_unit_override(self):
        """Unit override key present but no corresponding value key should not crash."""
        params = {
            'length_unit': 'micrometer',
            'time_unit': 'nanosecond',
            'FWHM_0_unit': 'nanometer',
            # No 'FWHM_0' key present
        }
        result = resolve_units(params)
        assert 'FWHM_0_unit' not in result
        assert 'FWHM_0' not in result


class TestGlobalUnits:
    """Global unit lookup shared by resolve_units and the output writers."""

    def test_defaults(self):
        assert global_length_unit({}) == 'micrometer'
        assert global_time_unit({}) == 'nanosecond'

    def test_legacy_keys(self):
        params = {'length unit': 'nanometer', 'time unit': 'picosecond'}
        assert global_length_unit(params) == 'nanometer'
        assert global_time_unit(params) == 'picosecond'

    def test_canonical_keys_take_precedence(self):
        params = {'length unit': 'nanometer', 'length_unit': 'angstrom',
                  'time unit': 'picosecond', 'time_unit': 'femtosecond'}
        assert global_length_unit(params) == 'angstrom'
        assert global_time_unit(params) == 'femtosecond'


class TestUnknownUnitKeys:
    """resolve_units rejects a key that looks like a unit key but that no parameter uses."""

    BASE = {'length unit': 'micrometer', 'time unit': 'nanosecond', 'FWHM_0': 500.0, 'spatial width': 5.0}

    @pytest.mark.parametrize('key, suggestion', [
        ('FWHM_0_units', 'FWHM_0_unit'),
        ('mean_0_unit', 'mu_0_unit'),
        ('fwhm_unit', 'fwhm_0_unit'),
        ('time range unit', 'time_range_unit'),
    ])
    def test_misspelled_unit_key_is_rejected_with_the_closest_key(self, key, suggestion):
        with pytest.raises(ValueError, match=f"Unknown unit key '{key}'. Did you mean '{suggestion}'"):
            resolve_units(self.BASE | {key: 'nanometer'})

    @pytest.mark.parametrize('first, second', [
        ('FWHM_0_unit', 'fwhm_0_unit'),
        ('spatial_width_unit', 'spatial width unit'),
        ('diffusion_length_unit', 'diffusion length unit'),
    ])
    def test_unit_key_in_two_spellings_is_rejected(self, first, second):
        """One of the two would be applied and the other ignored, even when they disagree."""
        with pytest.raises(ValueError, match=f"'{first}' and '{second}' are two spellings of one unit key"):
            resolve_units(self.BASE | {first: 'nanometer', second: 'micrometer'})

    def test_every_documented_unit_key_is_accepted(self):
        from dice.utils.units import GLOBAL_AND_OUTPUT_UNIT_KEYS, PER_PARAMETER_UNIT_KEYS
        time_keys = {'lifetime_unit', 'time_range_unit', 'time_series_unit', 'time_start_unit',
                     'time_stop_unit', 'diffusion_coefficient_time_unit', 'time unit', 'time_unit',
                     'output_diffusion_time_unit'}
        for key in PER_PARAMETER_UNIT_KEYS + GLOBAL_AND_OUTPUT_UNIT_KEYS:
            unit = 'nanosecond' if key in time_keys else 'micrometer'
            resolved = resolve_units(self.BASE | {key: unit})
            if key in PER_PARAMETER_UNIT_KEYS:
                assert key not in resolved

    def test_space_spelling_of_the_diffusion_length_unit_converts(self):
        resolved = resolve_units(self.BASE | {'nominal diffusion length': 500.0,
                                              'diffusion length unit': 'nanometer'})
        assert resolved['nominal diffusion length'] == pytest.approx(0.5)

    def test_keys_that_only_contain_the_word_unit_are_left_alone(self):
        resolved = resolve_units(self.BASE | {'units of measure note': 'kept'})
        assert resolved['units of measure note'] == 'kept'
