"""
Unit definitions and conversion functions for DICE simulations.

Provides conversion between physical units used in diffusion simulations.
Length units convert to meters as the base unit; time units convert to seconds.
This module is the single source of truth for unit definitions, conversion
factors, and display abbreviations.
"""

import difflib
from enum import Enum
from typing import Optional, Dict, Any, Union
import warnings


class Dimension(Enum):
    """Physical dimension categories for simulation parameters."""
    LENGTH = "length"
    TIME = "time"
    LENGTH_SQUARED = "length_squared"
    LENGTH_SQUARED_PER_TIME = "length_squared_per_time"
    DIMENSIONLESS = "dimensionless"


# Conversion factors from each unit to SI base units (meters, seconds).
# To convert value from unit A to unit B:
#   value_B = value_A * (FACTOR[A] / FACTOR[B])

VALID_LENGTH_UNITS: Dict[str, float] = {
    'meter': 1.0,
    'centimeter': 1e-2,
    'millimeter': 1e-3,
    'micrometer': 1e-6,
    'nanometer': 1e-9,
    'angstrom': 1e-10,
    'picometer': 1e-12,
}

VALID_TIME_UNITS: Dict[str, float] = {
    'second': 1.0,
    'millisecond': 1e-3,
    'microsecond': 1e-6,
    'nanosecond': 1e-9,
    'picosecond': 1e-12,
    'femtosecond': 1e-15,
    'attosecond': 1e-18,
}

# Display abbreviations for GUI labels and output formatting.
LENGTH_ABBREVIATIONS: Dict[str, str] = {
    'meter': 'm',
    'centimeter': 'cm',
    'millimeter': 'mm',
    'micrometer': '\u03bcm',   # μm
    'nanometer': 'nm',
    'angstrom': '\u00c5',      # Å
    'picometer': 'pm',
}

TIME_ABBREVIATIONS: Dict[str, str] = {
    'second': 's',
    'millisecond': 'ms',
    'microsecond': '\u03bcs',  # μs
    'nanosecond': 'ns',
    'picosecond': 'ps',
    'femtosecond': 'fs',
    'attosecond': 'as',
}

# Maps canonical parameter names to their physical dimension.
# Parameters not listed here are dimensionless.
PARAMETER_DIMENSIONS: Dict[str, Dimension] = {
    'fwhm_0': Dimension.LENGTH,
    'sigma_0': Dimension.LENGTH,
    'sigma2_0': Dimension.LENGTH_SQUARED,
    'mu_0': Dimension.LENGTH,
    'spatial_width': Dimension.LENGTH,
    'diffusion_length': Dimension.LENGTH,
    'diffusion_coefficient': Dimension.LENGTH_SQUARED_PER_TIME,
    'lifetime': Dimension.TIME,
    'time_start': Dimension.TIME,
    'time_stop': Dimension.TIME,
    'time_series': Dimension.TIME,
    'time_range': Dimension.TIME,
}


def validate_length_unit(unit: str, key: Optional[str] = None) -> None:
    """
    Validate that a string is a recognized length unit.

    Parameters
    ----------
    unit : str
        Unit name to validate.
    key : str, optional
        Parameter key the unit was given for, named in the error message.

    Raises
    ------
    ValueError
        If the unit is not in VALID_LENGTH_UNITS.
    """
    if not isinstance(unit, str) or unit not in VALID_LENGTH_UNITS:
        valid = ', '.join(sorted(VALID_LENGTH_UNITS.keys()))
        for_key = f" for '{key}'" if key else ""
        raise ValueError(
            f"Unknown length unit '{unit}'{for_key}. Valid units: {valid}"
        )


def validate_time_unit(unit: str, key: Optional[str] = None) -> None:
    """
    Validate that a string is a recognized time unit.

    Parameters
    ----------
    unit : str
        Unit name to validate.
    key : str, optional
        Parameter key the unit was given for, named in the error message.

    Raises
    ------
    ValueError
        If the unit is not in VALID_TIME_UNITS.
    """
    if not isinstance(unit, str) or unit not in VALID_TIME_UNITS:
        valid = ', '.join(sorted(VALID_TIME_UNITS.keys()))
        for_key = f" for '{key}'" if key else ""
        raise ValueError(
            f"Unknown time unit '{unit}'{for_key}. Valid units: {valid}"
        )


def convert_length(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert a length value between units.

    Parameters
    ----------
    value : float
        The value to convert.
    from_unit : str
        Source length unit name.
    to_unit : str
        Target length unit name.

    Returns
    -------
    float
        The converted value.

    Raises
    ------
    ValueError
        If either unit is not recognized.
    """
    if from_unit == to_unit:
        return value
    validate_length_unit(from_unit)
    validate_length_unit(to_unit)
    return value * VALID_LENGTH_UNITS[from_unit] / VALID_LENGTH_UNITS[to_unit]


def convert_time(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert a time value between units.

    Parameters
    ----------
    value : float
        The value to convert.
    from_unit : str
        Source time unit name.
    to_unit : str
        Target time unit name.

    Returns
    -------
    float
        The converted value.

    Raises
    ------
    ValueError
        If either unit is not recognized.
    """
    if from_unit == to_unit:
        return value
    validate_time_unit(from_unit)
    validate_time_unit(to_unit)
    return value * VALID_TIME_UNITS[from_unit] / VALID_TIME_UNITS[to_unit]


def convert_length_squared(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert a length-squared (area) value between length units.

    Parameters
    ----------
    value : float
        The value in from_unit^2.
    from_unit : str
        Source length unit name.
    to_unit : str
        Target length unit name.

    Returns
    -------
    float
        The converted value in to_unit^2.

    Raises
    ------
    ValueError
        If either unit is not recognized.
    """
    if from_unit == to_unit:
        return value
    validate_length_unit(from_unit)
    validate_length_unit(to_unit)
    factor = VALID_LENGTH_UNITS[from_unit] / VALID_LENGTH_UNITS[to_unit]
    return value * factor ** 2


def convert_diffusion_coefficient(
    value: float,
    from_length_unit: str,
    from_time_unit: str,
    to_length_unit: str,
    to_time_unit: str,
) -> float:
    """
    Convert a diffusion coefficient (length^2/time) between unit systems.

    Parameters
    ----------
    value : float
        Diffusion coefficient in from_length_unit^2 / from_time_unit.
    from_length_unit : str
        Source length unit name.
    from_time_unit : str
        Source time unit name.
    to_length_unit : str
        Target length unit name.
    to_time_unit : str
        Target time unit name.

    Returns
    -------
    float
        The converted diffusion coefficient in to_length_unit^2 / to_time_unit.

    Raises
    ------
    ValueError
        If any unit is not recognized.
    """
    if from_length_unit == to_length_unit and from_time_unit == to_time_unit:
        return value
    validate_length_unit(from_length_unit)
    validate_length_unit(to_length_unit)
    validate_time_unit(from_time_unit)
    validate_time_unit(to_time_unit)
    length_factor = VALID_LENGTH_UNITS[from_length_unit] / VALID_LENGTH_UNITS[to_length_unit]
    time_factor = VALID_TIME_UNITS[from_time_unit] / VALID_TIME_UNITS[to_time_unit]
    return value * (length_factor ** 2) / time_factor


def length_abbreviation(unit: str) -> str:
    """
    Get the display abbreviation for a length unit.

    Parameters
    ----------
    unit : str
        Full length unit name.

    Returns
    -------
    str
        Abbreviation (e.g., 'micrometer' -> 'μm').
    """
    return LENGTH_ABBREVIATIONS.get(unit, unit)


def time_abbreviation(unit: str) -> str:
    """
    Get the display abbreviation for a time unit.

    Parameters
    ----------
    unit : str
        Full time unit name.

    Returns
    -------
    str
        Abbreviation (e.g., 'nanosecond' -> 'ns').
    """
    return TIME_ABBREVIATIONS.get(unit, unit)


def diffusion_coefficient_label(length_unit: str, time_unit: str) -> str:
    """
    Build a display label for diffusion coefficient units.

    Parameters
    ----------
    length_unit : str
        Length unit name.
    time_unit : str
        Time unit name.

    Returns
    -------
    str
        Label like 'μm²/ns' or 'cm²/s'.
    """
    l_abbr = length_abbreviation(length_unit)
    t_abbr = time_abbreviation(time_unit)
    return f"{l_abbr}\u00b2/{t_abbr}"


def global_length_unit(parameters: Dict[str, Any]) -> str:
    """Global simulation length unit of a parameters dictionary (default 'micrometer')."""
    return parameters.get('length_unit', parameters.get('length unit', 'micrometer'))


def global_time_unit(parameters: Dict[str, Any]) -> str:
    """Global simulation time unit of a parameters dictionary (default 'nanosecond')."""
    return parameters.get('time_unit', parameters.get('time unit', 'nanosecond'))


# Keys that set the unit of the diffusion coefficients in the output files.
OUTPUT_DIFFUSION_LENGTH_UNIT_KEY = 'output_diffusion_length_unit'
OUTPUT_DIFFUSION_TIME_UNIT_KEY = 'output_diffusion_time_unit'

# Output unit keys no output file uses: the files report lengths and times in
# the global units, and their unlabeled results-file columns must stay so.
UNSUPPORTED_OUTPUT_UNIT_KEYS = ('output_length_unit', 'output_time_unit')


def validate_output_units(parameters: Dict[str, Any]) -> None:
    """
    Check the output unit keys of a parameters dictionary before a simulation.

    Parameters
    ----------
    parameters : dict
        Parameters dictionary.

    Raises
    ------
    ValueError
        If it has an output unit key that no output uses, only one of the two
        diffusion coefficient output unit keys, or an unknown unit name.
    """
    for key in UNSUPPORTED_OUTPUT_UNIT_KEYS:
        if key in parameters:
            raise ValueError(
                f"'{key}' is not supported: the summary and results files report lengths "
                "and times in the global 'length unit' and 'time unit'. To choose the unit "
                f"of the diffusion coefficients, use '{OUTPUT_DIFFUSION_LENGTH_UNIT_KEY}' "
                f"and '{OUTPUT_DIFFUSION_TIME_UNIT_KEY}'."
            )

    length = parameters.get(OUTPUT_DIFFUSION_LENGTH_UNIT_KEY)
    time = parameters.get(OUTPUT_DIFFUSION_TIME_UNIT_KEY)
    if (length is None) != (time is None):
        raise ValueError(
            f"Give both '{OUTPUT_DIFFUSION_LENGTH_UNIT_KEY}' and "
            f"'{OUTPUT_DIFFUSION_TIME_UNIT_KEY}', or neither."
        )
    if length is not None:
        validate_length_unit(length, OUTPUT_DIFFUSION_LENGTH_UNIT_KEY)
        validate_time_unit(time, OUTPUT_DIFFUSION_TIME_UNIT_KEY)


def resolve_units(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize all parameter values to the global simulation unit system.

    Reads per-parameter unit override keys (suffixed with '_unit') from the
    parameters dictionary and converts the corresponding values from their
    specified units to the global length_unit/time_unit. Returns a copy of
    the dictionary with converted values and override keys removed.

    Parameters without overrides are assumed to already be in the global units.

    Parameters
    ----------
    parameters : dict
        Parsed parameters dictionary. Must contain 'length_unit' and
        'time_unit' keys for the global simulation units. May contain
        per-parameter override keys like 'fwhm_0_unit', 'lifetime_unit', etc.

    Returns
    -------
    dict
        Copy of parameters with all values converted to global units and
        per-parameter unit override keys removed.
    """
    result = dict(parameters)

    target_length = global_length_unit(result)
    target_time = global_time_unit(result)

    # Length-dimension parameters and their override key mappings.
    # Each entry: (value_keys, override_keys)
    # value_keys are checked in order; the first one found is converted.
    # override_keys are checked in order; the first one found is used.
    length_params = [
        (['FWHM_0', 'fwhm_0'], ['FWHM_0_unit', 'fwhm_0_unit']),
        (['sigma_0'], ['sigma_0_unit']),
        (['mean_0', 'mu_0'], ['mu_0_unit']),
        (['spatial width', 'spatial_width'], ['spatial_width_unit', 'spatial width unit']),
        (['nominal diffusion length', 'diffusion_length'],
         ['diffusion_length_unit', 'diffusion length unit']),
    ]

    for value_keys, override_keys in length_params:
        matched_key = next((k for k in override_keys if k in result), None)
        if matched_key is not None:
            from_unit = result[matched_key]
            if from_unit != target_length:
                for vk in value_keys:
                    if vk in result:
                        result[vk] = convert_length(result[vk], from_unit, target_length)
            del result[matched_key]

    # sigma^2_0 is length-squared
    sigma2_override = 'sigma2_0_unit'
    if sigma2_override in result:
        from_unit = result[sigma2_override]
        if from_unit != target_length:
            for vk in ['sigma^2_0', 'sigma2_0']:
                if vk in result:
                    result[vk] = convert_length_squared(result[vk], from_unit, target_length)
        del result[sigma2_override]

    # Time-dimension parameters
    time_params = [
        (['nominal lifetime (tau)', 'lifetime'], ['lifetime_unit']),
    ]

    for value_keys, override_keys in time_params:
        matched_key = next((k for k in override_keys if k in result), None)
        if matched_key is not None:
            from_unit = result[matched_key]
            if from_unit != target_time:
                for vk in value_keys:
                    if vk in result:
                        result[vk] = convert_time(result[vk], from_unit, target_time)
            del result[matched_key]

    # Time range: convert start and stop if override present.
    # Track which values have been converted to prevent double conversion
    # when multiple overlapping time override keys are present.
    converted_time_range = False
    converted_time_series = False

    time_range_override = 'time_range_unit'
    if time_range_override in result:
        from_unit = result[time_range_override]
        if from_unit != target_time:
            # time range is [start, stop, steps] -- convert start and stop
            for vk in ['time range', 'time_range']:
                if vk in result and result[vk] is not None:
                    tr = list(result[vk])
                    tr[0] = convert_time(tr[0], from_unit, target_time)
                    tr[1] = convert_time(tr[1], from_unit, target_time)
                    result[vk] = tr
                    converted_time_range = True
            # time series is a list of time values
            for vk in ['time series', 'time_series']:
                if vk in result and result[vk] is not None:
                    result[vk] = [
                        convert_time(t, from_unit, target_time) for t in result[vk]
                    ]
                    converted_time_series = True
        del result[time_range_override]

    # Time series override (separate from time range)
    time_series_override = 'time_series_unit'
    if time_series_override in result:
        from_unit = result[time_series_override]
        if not converted_time_series and from_unit != target_time:
            for vk in ['time series', 'time_series']:
                if vk in result and result[vk] is not None:
                    result[vk] = [
                        convert_time(t, from_unit, target_time) for t in result[vk]
                    ]
        del result[time_series_override]

    # Time start/stop overrides (for GUI which sends these separately)
    for param, override_key, idx in [('time_start', 'time_start_unit', 0),
                                     ('time_stop', 'time_stop_unit', 1)]:
        if override_key in result:
            from_unit = result[override_key]
            if not converted_time_range and from_unit != target_time:
                if param in result:
                    result[param] = convert_time(result[param], from_unit, target_time)
                else:
                    # Value may be inside a time range list (GUI path)
                    for vk in ['time range', 'time_range']:
                        if vk in result and result[vk] is not None:
                            tr = list(result[vk])
                            tr[idx] = convert_time(tr[idx], from_unit, target_time)
                            result[vk] = tr
            del result[override_key]

    # Diffusion coefficient (compound unit: length^2/time)
    dc_length_override = 'diffusion_coefficient_length_unit'
    dc_time_override = 'diffusion_coefficient_time_unit'
    has_dc_override = dc_length_override in result or dc_time_override in result

    if has_dc_override:
        from_l = result.pop(dc_length_override, target_length)
        from_t = result.pop(dc_time_override, target_time)
        if from_l != target_length or from_t != target_time:
            for vk in ['nominal diffusion coefficient', 'diffusion_coefficient']:
                if vk in result:
                    result[vk] = convert_diffusion_coefficient(
                        result[vk], from_l, from_t, target_length, target_time
                    )

    reject_unknown_unit_keys(result)

    return result


# The per-parameter unit keys that resolve_units converts and removes.
PER_PARAMETER_UNIT_KEYS = (
    'FWHM_0_unit', 'fwhm_0_unit', 'sigma_0_unit', 'sigma2_0_unit', 'mu_0_unit',
    'spatial_width_unit', 'spatial width unit',
    'diffusion_length_unit', 'diffusion length unit',
    'diffusion_coefficient_length_unit', 'diffusion_coefficient_time_unit',
    'lifetime_unit', 'time_range_unit', 'time_series_unit',
    'time_start_unit', 'time_stop_unit',
)

# The unit keys that stay in the dictionary: the global units and the units of
# the diffusion coefficients in the output files.
GLOBAL_AND_OUTPUT_UNIT_KEYS = (
    'length unit', 'length_unit', 'time unit', 'time_unit',
    OUTPUT_DIFFUSION_LENGTH_UNIT_KEY, OUTPUT_DIFFUSION_TIME_UNIT_KEY,
)

_UNIT_KEY_ENDINGS = ('_unit', ' unit', '_units', ' units')


def reject_unknown_unit_keys(parameters: Dict[str, Any]) -> None:
    """
    Reject a key that looks like a unit key but that no parameter uses.

    A misspelled unit key would otherwise be ignored and its value read in
    the global unit: a width meant in nanometers would run in micrometers.

    Parameters
    ----------
    parameters : dict
        Parameters dictionary after the per-parameter unit keys are removed.

    Raises
    ------
    ValueError
        If a key ends like a unit key and is not one. The message names the
        closest unit key, if there is one.
    """
    known_keys = PER_PARAMETER_UNIT_KEYS + GLOBAL_AND_OUTPUT_UNIT_KEYS
    for key in parameters:
        if not isinstance(key, str) or not key.lower().endswith(_UNIT_KEY_ENDINGS):
            continue
        # validate_output_units rejects these two with its own explanation.
        if key in GLOBAL_AND_OUTPUT_UNIT_KEYS or key in UNSUPPORTED_OUTPUT_UNIT_KEYS:
            continue
        close = difflib.get_close_matches(key, known_keys, n=1)
        hint = f" Did you mean '{close[0]}'?" if close else ""
        raise ValueError(
            f"Unknown unit key '{key}'.{hint} The unit keys are: {', '.join(known_keys)}."
        )
