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
    resolve_units,
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

    def test_um2_per_ns_to_cm2_per_s(self):
        # 1 um^2/ns = (1e-4 cm)^2 / (1e-9 s) = 1e-8 / 1e-9 = 10 cm^2/s
        # Wait, that's for slope. For D, the factor is the same.
        result = convert_diffusion_coefficient(
            1.0, 'micrometer', 'nanosecond', 'centimeter', 'second'
        )
        expected = (1e-4)**2 / 1e-9  # = 1e-8 / 1e-9 = 10.0
        assert np.isclose(result, expected, rtol=1e-10)

    def test_cm2_per_s_to_um2_per_ns(self):
        # Inverse of above
        result = convert_diffusion_coefficient(
            10.0, 'centimeter', 'second', 'micrometer', 'nanosecond'
        )
        assert np.isclose(result, 1.0, rtol=1e-10)

    def test_nm2_per_ps_to_um2_per_ns(self):
        # 1 nm^2/ps = (1e-3 um)^2 / (1e-3 ns) = 1e-6/1e-3 = 1e-3 um^2/ns
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
        # 0.01 mm = 10 um
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
        # 1e-5 cm^2/s -> um^2/ns
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
            # no override for spatial_width -- stays in um
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
        # 100 nm = 0.1 um
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
        # 0.01 nm^2/ns -> um^2/ns
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
        # 250000 nm^2 = 0.25 um^2
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
