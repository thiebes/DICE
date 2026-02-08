"""
Canonical parameter key names and alias mappings for DICE.

This module provides a single source of truth for parameter names,
enabling both legacy (space-separated) and modern (snake_case) formats
to coexist during the migration period.
"""

from typing import Dict, Any


class ParameterKey:
    """Canonical parameter key names (snake_case convention)."""

    # Profile parameters
    SIGMA2_0 = "sigma2_0"
    AMPLITUDE_0 = "amplitude_0"
    MU_0 = "mu_0"
    FWHM_0 = "fwhm_0"
    SIGMA_0 = "sigma_0"

    # Physics/diffusion parameters
    DIFFUSION_COEFFICIENT = "diffusion_coefficient"
    LIFETIME = "lifetime"
    DIFFUSION_LENGTH = "diffusion_length"

    # Simulation control
    NUMBER_OF_RUNS = "number_of_runs"
    MULTIPROCESSING = "multiprocessing"
    RETAIN_PROFILE_DATA = "retain_profile_data"
    PROXIMITY_LEVEL = "proximity_level"

    # Spatial axis
    SPATIAL_WIDTH = "spatial_width"
    PIXEL_WIDTH = "pixel_width"
    X_ARRAY = "x_array"

    # Temporal axis
    TIME_SERIES = "time_series"
    TIME_RANGE = "time_range"

    # Noise parameters
    NOISE_VALUE = "noise_value"
    NOISE_SERIES = "noise_series"
    ESTIMATE_NOISE_FROM_DATA = "estimate_noise_from_data"
    NOISE_RANGE_RECIPROCAL = "noise_range_reciprocal"
    NOISE_RANGE_RECIPROCAL_LOG = "noise_range_reciprocal_log"

    # Per-parameter unit overrides
    FWHM_0_UNIT = "fwhm_0_unit"
    SIGMA_0_UNIT = "sigma_0_unit"
    SIGMA2_0_UNIT = "sigma2_0_unit"
    MU_0_UNIT = "mu_0_unit"
    SPATIAL_WIDTH_UNIT = "spatial_width_unit"
    DIFFUSION_LENGTH_UNIT = "diffusion_length_unit"
    DIFFUSION_COEFFICIENT_LENGTH_UNIT = "diffusion_coefficient_length_unit"
    DIFFUSION_COEFFICIENT_TIME_UNIT = "diffusion_coefficient_time_unit"
    LIFETIME_UNIT = "lifetime_unit"
    TIME_START_UNIT = "time_start_unit"
    TIME_STOP_UNIT = "time_stop_unit"
    TIME_SERIES_UNIT = "time_series_unit"
    TIME_RANGE_UNIT = "time_range_unit"

    # Output unit preferences
    OUTPUT_LENGTH_UNIT = "output_length_unit"
    OUTPUT_TIME_UNIT = "output_time_unit"
    OUTPUT_DIFFUSION_LENGTH_UNIT = "output_diffusion_length_unit"
    OUTPUT_DIFFUSION_TIME_UNIT = "output_diffusion_time_unit"

    # Output parameters
    FILENAME_SLUG = "filename_slug"
    LENGTH_UNIT = "length_unit"
    TIME_UNIT = "time_unit"
    IMAGE_TYPE = "image_type"
    IMAGE_WIDTH = "image_width"
    IMAGE_HEIGHT = "image_height"
    IMAGE_DPI = "image_dpi"
    IMAGE_FONT_SIZE = "image_font_size"
    IMAGE_TICK_LENGTH = "image_tick_length"
    IMAGE_TICK_WIDTH = "image_tick_width"
    IMAGE_NUMBINS = "image_numbins"
    IMAGE_X_LIM = "image_x_lim"


# Mapping from legacy keys (with spaces, special chars) to canonical keys
PARAMETER_ALIASES: Dict[str, str] = {
    # Simulation control
    "number of runs": ParameterKey.NUMBER_OF_RUNS,
    "retain profile data": ParameterKey.RETAIN_PROFILE_DATA,
    "proximity level": ParameterKey.PROXIMITY_LEVEL,

    # Profile parameters
    "sigma^2_0": ParameterKey.SIGMA2_0,
    "mean_0": ParameterKey.MU_0,  # alias for mu_0
    "FWHM_0": ParameterKey.FWHM_0,

    # Physics/diffusion parameters
    "nominal diffusion coefficient": ParameterKey.DIFFUSION_COEFFICIENT,
    "nominal lifetime (tau)": ParameterKey.LIFETIME,
    "nominal diffusion length": ParameterKey.DIFFUSION_LENGTH,

    # Spatial axis
    "spatial width": ParameterKey.SPATIAL_WIDTH,
    "pixel width": ParameterKey.PIXEL_WIDTH,
    "x array": ParameterKey.X_ARRAY,

    # Temporal axis
    "time series": ParameterKey.TIME_SERIES,
    "time range": ParameterKey.TIME_RANGE,

    # Noise parameters
    "noise value": ParameterKey.NOISE_VALUE,
    "noise series": ParameterKey.NOISE_SERIES,
    "estimate noise from data": ParameterKey.ESTIMATE_NOISE_FROM_DATA,
    "noise range, reciprocal": ParameterKey.NOISE_RANGE_RECIPROCAL,
    "noise range, reciprocal log": ParameterKey.NOISE_RANGE_RECIPROCAL_LOG,

    # Per-parameter unit overrides
    "FWHM_0_unit": ParameterKey.FWHM_0_UNIT,
    "sigma_0_unit": ParameterKey.SIGMA_0_UNIT,
    "sigma2_0_unit": ParameterKey.SIGMA2_0_UNIT,
    "mu_0_unit": ParameterKey.MU_0_UNIT,
    "spatial_width_unit": ParameterKey.SPATIAL_WIDTH_UNIT,
    "spatial width unit": ParameterKey.SPATIAL_WIDTH_UNIT,
    "diffusion_length_unit": ParameterKey.DIFFUSION_LENGTH_UNIT,
    "diffusion length unit": ParameterKey.DIFFUSION_LENGTH_UNIT,
    "diffusion_coefficient_length_unit": ParameterKey.DIFFUSION_COEFFICIENT_LENGTH_UNIT,
    "diffusion_coefficient_time_unit": ParameterKey.DIFFUSION_COEFFICIENT_TIME_UNIT,
    "lifetime_unit": ParameterKey.LIFETIME_UNIT,
    "time_start_unit": ParameterKey.TIME_START_UNIT,
    "time_stop_unit": ParameterKey.TIME_STOP_UNIT,
    "time_series_unit": ParameterKey.TIME_SERIES_UNIT,
    "time_range_unit": ParameterKey.TIME_RANGE_UNIT,
    "output_length_unit": ParameterKey.OUTPUT_LENGTH_UNIT,
    "output_time_unit": ParameterKey.OUTPUT_TIME_UNIT,
    "output_diffusion_length_unit": ParameterKey.OUTPUT_DIFFUSION_LENGTH_UNIT,
    "output_diffusion_time_unit": ParameterKey.OUTPUT_DIFFUSION_TIME_UNIT,

    # Output parameters
    "filename slug": ParameterKey.FILENAME_SLUG,
    "length unit": ParameterKey.LENGTH_UNIT,
    "time unit": ParameterKey.TIME_UNIT,
    "image type": ParameterKey.IMAGE_TYPE,
    "image width": ParameterKey.IMAGE_WIDTH,
    "image height": ParameterKey.IMAGE_HEIGHT,
    "image dpi": ParameterKey.IMAGE_DPI,
    "image font size": ParameterKey.IMAGE_FONT_SIZE,
    "image tick length": ParameterKey.IMAGE_TICK_LENGTH,
    "image tick width": ParameterKey.IMAGE_TICK_WIDTH,
    "image numbins": ParameterKey.IMAGE_NUMBINS,
    "image x_lim": ParameterKey.IMAGE_X_LIM,
}

# Reverse mapping: canonical keys to legacy format (for backward-compatible output)
LEGACY_OUTPUT_KEYS: Dict[str, str] = {
    ParameterKey.NUMBER_OF_RUNS: "number of runs",
    ParameterKey.RETAIN_PROFILE_DATA: "retain profile data",
    ParameterKey.PROXIMITY_LEVEL: "proximity level",
    ParameterKey.SIGMA2_0: "sigma^2_0",
    ParameterKey.DIFFUSION_COEFFICIENT: "nominal diffusion coefficient",
    ParameterKey.LIFETIME: "nominal lifetime (tau)",
    ParameterKey.DIFFUSION_LENGTH: "nominal diffusion length",
    ParameterKey.SPATIAL_WIDTH: "spatial width",
    ParameterKey.PIXEL_WIDTH: "pixel width",
    ParameterKey.X_ARRAY: "x array",
    ParameterKey.TIME_SERIES: "time series",
    ParameterKey.TIME_RANGE: "time range",
    ParameterKey.NOISE_VALUE: "noise value",
    ParameterKey.NOISE_SERIES: "noise series",
    ParameterKey.ESTIMATE_NOISE_FROM_DATA: "estimate noise from data",
    ParameterKey.NOISE_RANGE_RECIPROCAL: "noise range, reciprocal",
    ParameterKey.NOISE_RANGE_RECIPROCAL_LOG: "noise range, reciprocal log",
    ParameterKey.FILENAME_SLUG: "filename slug",
    ParameterKey.LENGTH_UNIT: "length unit",
    ParameterKey.TIME_UNIT: "time unit",
    ParameterKey.IMAGE_TYPE: "image type",
    ParameterKey.IMAGE_WIDTH: "image width",
    ParameterKey.IMAGE_HEIGHT: "image height",
    ParameterKey.IMAGE_DPI: "image dpi",
    ParameterKey.IMAGE_FONT_SIZE: "image font size",
    ParameterKey.IMAGE_TICK_LENGTH: "image tick length",
    ParameterKey.IMAGE_TICK_WIDTH: "image tick width",
    ParameterKey.IMAGE_NUMBINS: "image numbins",
    ParameterKey.IMAGE_X_LIM: "image x_lim",
}


def normalize_parameter_key(key: str) -> str:
    """
    Convert any parameter key variant to canonical form.

    Parameters
    ----------
    key : str
        Parameter key in any format (legacy or canonical).

    Returns
    -------
    str
        Canonical key name (snake_case).

    Examples
    --------
    >>> normalize_parameter_key("number of runs")
    'number_of_runs'
    >>> normalize_parameter_key("sigma^2_0")
    'sigma2_0'
    >>> normalize_parameter_key("number_of_runs")  # already canonical
    'number_of_runs'
    """
    return PARAMETER_ALIASES.get(key, key)


def normalize_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert all keys in a parameters dictionary to canonical form.

    Parameters
    ----------
    params : dict
        Parameters dictionary with keys in any format.

    Returns
    -------
    dict
        Parameters dictionary with all keys in canonical form.

    Examples
    --------
    >>> params = {"number of runs": 100, "sigma^2_0": 1.0}
    >>> normalize_parameters(params)
    {'number_of_runs': 100, 'sigma2_0': 1.0}
    """
    return {normalize_parameter_key(k): v for k, v in params.items()}


def to_legacy_key(canonical_key: str) -> str:
    """
    Convert a canonical key to its legacy format.

    Parameters
    ----------
    canonical_key : str
        Canonical parameter key name.

    Returns
    -------
    str
        Legacy key format, or original key if no mapping exists.

    Examples
    --------
    >>> to_legacy_key("number_of_runs")
    'number of runs'
    >>> to_legacy_key("sigma2_0")
    'sigma^2_0'
    """
    return LEGACY_OUTPUT_KEYS.get(canonical_key, canonical_key)


def add_legacy_keys(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add legacy key aliases to a parameters dictionary.

    This function adds legacy-format keys alongside canonical keys
    for backward compatibility with code expecting the old format.

    Parameters
    ----------
    params : dict
        Parameters dictionary with canonical keys.

    Returns
    -------
    dict
        Parameters dictionary with both canonical and legacy keys.
    """
    result = dict(params)
    for canonical, legacy in LEGACY_OUTPUT_KEYS.items():
        if canonical in params and legacy not in params:
            result[legacy] = params[canonical]
    return result
