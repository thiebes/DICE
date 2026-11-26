"""
Parameter loading and validation for DICE simulations.

This module provides functions for loading, parsing, and validating
simulation parameters from files and dictionaries.

Accepts both legacy (space-separated) and modern (snake_case) key names.
"""

import ast
import locale
import os
import warnings
import numpy as np
from typing import Dict, Any, List, Optional, Sequence, Union
from pathlib import Path

from ..utils.converters import fwhm_to_sigma2
from ..utils.axes import make_x_axis, make_time_axis, make_time_series
from ..utils.validators import validate_filename_slug, validate_integer, validate_limits
from ..utils.units import (
    global_length_unit, global_time_unit, validate_length_unit, validate_time_unit,
    validate_output_units,
)
from fft_cnr import fft_cnr
from ..core.diffusion import calculate_diffusion_length
from ..core.noise import (
    MIN_CNR_PEAK_FWHM, MINIMUM_PROFILE_POINTS, Seed, make_noise_distribution, peak_fwhm_samples,
)
from ..models.parameters import NoiseRange
from ..models.parameter_keys import (
    normalize_parameters, normalize_parameter_key, add_legacy_keys,
    ParameterKey, LEGACY_OUTPUT_KEYS
)

# Keys of the proximity-level parameter, retired from the simulation schema;
# still accepted in parameter files, with a warning.
RETIRED_PROXIMITY_KEYS = ('proximity level', 'proximity_level')

# Keys whose value is the path of the measured profile for the noise estimate.
NOISE_DATA_KEYS = ('estimate noise from data', 'estimate_noise_from_data')

# Defaults of the optional parameters, by the key the parameter files document.
OPTIONAL_PARAMETER_DEFAULTS = {
    'multiprocessing': 1,
    'filename slug': 'dice_simulation',
    'retain profile data': 0,
    'image type': 'png',
    'image width': 10,
    'image height': 6,
    'image dpi': 100,
}

# Keys of a noise range, from which each run draws its own noise level, each
# mapped to whether log10(1/sigma), rather than 1/sigma, is uniformly distributed.
NOISE_RANGE_KEYS = {
    'noise range, reciprocal': False,
    'noise_range_reciprocal': False,
    'noise range, reciprocal log': True,
    'noise_range_reciprocal_log': True,
}


def read_parameters_text(filename: Union[str, Path]) -> str:
    """
    Read the text of a parameters file.

    Parameter files are UTF-8, with or without a byte-order mark. A file that
    is not valid UTF-8 is read in the encoding of the locale, which is how an
    older text editor on Windows may have saved it.

    Parameters
    ----------
    filename : str or Path
        Path to the parameters file.

    Returns
    -------
    str
        The text of the file.
    """
    data = Path(filename).read_bytes()
    try:
        return data.decode('utf-8-sig')
    except UnicodeDecodeError:
        return data.decode(locale.getpreferredencoding(False))


def resolve_noise_data_path(
    parameters_dictionary: Dict[str, Any], base_dir: Union[str, Path]
) -> Dict[str, Any]:
    """
    Resolve a relative measured-profile path against the folder of its parameters file.

    A parameters file names the profile for 'estimate noise from data'
    relative to its own folder, so that it gives the same run from any working
    directory. An absolute path, or a value that is not a path string, is left
    unchanged.

    Parameters
    ----------
    parameters_dictionary : dict
        Parameters as read from the file.
    base_dir : str or Path
        Folder of the parameters file.

    Returns
    -------
    dict
        Copy of the parameters with the path made absolute.
    """
    resolved = dict(parameters_dictionary)
    for key in NOISE_DATA_KEYS:
        path = resolved.get(key)
        if isinstance(path, str) and path.strip() and not Path(path).is_absolute():
            resolved[key] = os.path.abspath(Path(base_dir) / path)
    return resolved


def open_parameters(filename: Union[str, Path], seed: Seed = None) -> Dict[str, Any]:
    """
    Read simulation parameters from a file.
    
    Parameters
    ----------
    filename : str or Path
        Path to the parameters file.
    seed : int or SeedSequence, optional
        Seed for the noise draws of the 'noise range' options, one for each
        run; None draws fresh entropy.
    
    Returns
    -------
    dict
        Parsed parameters dictionary. A relative 'estimate noise from data'
        path is resolved against the folder of the file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file content is not a valid dictionary.
    """
    try:
        parms_dict = ast.literal_eval(read_parameters_text(filename))
        # Resolve per-parameter unit overrides before parsing,
        # so that values are in the global unit system when the
        # parser converts between representations (e.g., FWHM -> sigma^2).
        from ..utils.units import resolve_units
        parms_dict = resolve_units(parms_dict)
        parms_dict = resolve_noise_data_path(parms_dict, Path(filename).parent)
        return parameter_parser(parms_dict, seed=seed)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {filename} was not found.")
    except SyntaxError as e:
        raise ValueError(f"Error evaluating the file's contents as a dictionary: {e}")
    except Exception as e:
        raise ValueError(f"An error occurred while parsing parameters: {e}") from e


def check_for_unique_key(parameters_dictionary: Dict[str, Any], keys: List[str]) -> str:
    """
    Check for exactly one key from a list in a dictionary.

    Handles both legacy and canonical key formats by normalizing keys
    before checking. Keys that normalize to the same canonical form
    are treated as the same parameter.

    Parameters
    ----------
    parameters_dictionary : dict
        Dictionary to check (can have legacy or canonical keys).
    keys : list
        List of possible keys (can mix legacy and canonical formats).

    Returns
    -------
    str
        The unique key that exists (in the format it appears in the dictionary).

    Raises
    ------
    ValueError
        If none or more than one distinct parameter is present.
    """
    if not isinstance(parameters_dictionary, dict):
        raise TypeError("The parameters_dictionary argument must be a dictionary.")
    if not isinstance(keys, list):
        raise TypeError("The keys argument must be a list.")

    # Group keys by their canonical form to find distinct parameters
    canonical_to_keys = {}
    for key in keys:
        canonical = normalize_parameter_key(key)
        if canonical not in canonical_to_keys:
            canonical_to_keys[canonical] = []
        canonical_to_keys[canonical].append(key)

    # Build canonical->actual key mapping for the dictionary
    dict_canonical = {}
    for dk in parameters_dictionary:
        canonical_dk = normalize_parameter_key(dk)
        if canonical_dk not in dict_canonical:
            dict_canonical[canonical_dk] = dk

    # Check which requested canonical keys exist in the dictionary
    found_params = {}
    for canonical in canonical_to_keys:
        if canonical in dict_canonical:
            found_params[canonical] = dict_canonical[canonical]

    if len(found_params) > 1:
        raise ValueError(f"More than one parameter provided for {list(found_params.values())}. Please provide only one.")
    elif len(found_params) == 0:
        raise ValueError(f"No parameter provided for {keys}.")
    else:
        return list(found_params.values())[0]


def handle_time_parameters(parameters_dictionary: Dict[str, Any]) -> None:
    """
    Process time-related parameters.

    Supports both legacy ('time range', 'time series') and canonical
    ('time_range', 'time_series') key formats.

    Parameters
    ----------
    parameters_dictionary : dict
        Dictionary containing time parameters.

    Updates the dictionary with 'time series' key: at least two time points
    in ascending order. A 'time series' is sorted, as the GUI sorts it.

    Raises
    ------
    TypeError
        If the number of steps of a 'time range' is not a whole number.
    ValueError
        If the time axis has fewer than two points or a time below 0, a
        'time range' does not run forward, or a 'time series' repeats a time
        or holds a value that is not a finite number.
    """
    unique_time_key = check_for_unique_key(
        parameters_dictionary,
        ['time range', 'time series', 'time_range', 'time_series']
    )

    if unique_time_key in ('time range', 'time_range'):
        try:
            t_start, t_end, t_steps = parameters_dictionary[unique_time_key]
        except ValueError as e:
            raise ValueError("The 'time range' must contain three values: start, end, and number of steps.") from e
        except TypeError as e:
            raise ValueError("The 'time range' must be a sequence with three numerical values.") from e

        if not all(isinstance(value, (int, float)) for value in [t_start, t_end, t_steps]):
            raise ValueError("The 'time range' values must be numeric.")

        # The fit of the diffusion coefficient is a line through the widths,
        # so it needs two time points or more.
        t_steps = validate_integer(t_steps, "The number of steps in 'time range'", min_val=2)
        if t_start >= t_end:
            raise ValueError(
                f"The start of 'time range' ({t_start:g}) must be less than its stop ({t_end:g})."
            )
        parameters_dictionary['time series'] = make_time_axis(t_start, t_end, t_steps)
        if unique_time_key in parameters_dictionary:
            del parameters_dictionary[unique_time_key]
    else:
        try:
            time_series = make_time_series(parameters_dictionary[unique_time_key])
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid 'time series': {e}") from e
        if len(time_series) < 2:
            raise ValueError(
                f"'time series' holds {len(time_series)} time point; "
                "the fit of the diffusion coefficient needs at least two time points."
            )
        parameters_dictionary[unique_time_key] = time_series

    # The profile spreads from time 0 (sigma^2 = sigma_0^2 + 2 D t), so no time is before it.
    time_key = 'time series' if 'time series' in parameters_dictionary else 'time_series'
    first_time = float(np.min(parameters_dictionary[time_key]))
    if first_time < 0:
        raise ValueError(f"The time axis starts at {first_time:g}; every time must be 0 or greater.")


def narrow_peak_warning(profile: Sequence[float], source: str) -> Optional[str]:
    """
    Warning for a measured profile whose peak is too narrow for fft-cnr to estimate its CNR well.

    Parameters
    ----------
    profile : sequence of float
        The measured profile, one value per pixel.
    source : str
        Where the profile came from (its file name), for the message.

    Returns
    -------
    str or None
        The message when the peak is narrower than MIN_CNR_PEAK_FWHM pixels
        at half maximum, otherwise None.
    """
    width = peak_fwhm_samples(profile)
    if not width < MIN_CNR_PEAK_FWHM:  # a nan width (no peak) is not narrow
        return None
    return (
        f"The peak of the measured profile '{source}' is {width:.1f} pixels wide at half "
        f"maximum. For peaks narrower than about {MIN_CNR_PEAK_FWHM:g} pixels, fft-cnr "
        "overestimates the noise, so the estimated initial CNR is probably too low and the "
        "simulated noise too high. In simulated Gaussian profiles with white noise, the "
        "estimate was 14% to 27% too low for a peak 4.6 pixels wide, and within 2.4% of the "
        "true CNR for peaks 9 to 14 pixels wide."
    )


def parse_noise_range(value: Any, logarithmic: bool) -> NoiseRange:
    """
    The NoiseRange that the value [low, high] of a 'noise range' parameter describes.

    Parameters
    ----------
    value : list or tuple
        The parameter value: two noise standard deviations, low < high.
    logarithmic : bool
        True for 'noise range, reciprocal log', False for 'noise range, reciprocal'.

    Raises
    ------
    ValueError
        If the value is not two numbers, or they are not positive and finite
        with low < high.
    """
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or not all(isinstance(bound, (int, float)) and not isinstance(bound, bool) for bound in value)):
        raise ValueError(f"Invalid noise range {value}. Must be two numeric values.")
    return NoiseRange(low=value[0], high=value[1], logarithmic=logarithmic)


def noise_range_from_parameters(parameters_dictionary: Dict[str, Any]) -> Optional[NoiseRange]:
    """
    The range from which each run draws its noise level, or None for fixed noise.

    Reads the 'noise range, reciprocal' or 'noise range, reciprocal log'
    parameter, in legacy or canonical form, of a raw or a parsed parameters
    dictionary.

    Raises
    ------
    ValueError
        If the range is malformed (parse_noise_range).
    """
    for key, logarithmic in NOISE_RANGE_KEYS.items():
        if key in parameters_dictionary:
            return parse_noise_range(parameters_dictionary[key], logarithmic)
    return None


def handle_noise_parameters(
    parameters_dictionary: Dict[str, Any], num_vals: int = 1, seed: Seed = None
) -> None:
    """
    Process noise-related parameters.

    Supports both legacy and canonical key formats.

    Parameters
    ----------
    parameters_dictionary : dict
        Dictionary containing noise parameters.
    num_vals : int
        Number of noise values to draw from a 'noise range': the number of
        runs, since each run draws its own.
    seed : int or SeedSequence, optional
        Seed for the 'noise range' draws; None draws fresh entropy.

    Updates the dictionary with 'noise series' key: the one noise value of
    every run, or for a 'noise range', num_vals values, one for each run.
    """
    unique_noise_key = check_for_unique_key(parameters_dictionary, [
        'noise range, reciprocal log', 'noise range, reciprocal',
        'noise value', 'estimate noise from data',
        'noise_range_reciprocal_log', 'noise_range_reciprocal',
        'noise_value', 'estimate_noise_from_data'
    ])

    try:
        if unique_noise_key in ('noise value', 'noise_value'):
            noise_value = parameters_dictionary[unique_noise_key]
            if not isinstance(noise_value, (int, float)):
                raise ValueError(f"Invalid noise value {noise_value}. Must be numeric.")
            if not np.isfinite(noise_value) or noise_value < 0:
                raise ValueError(f"'noise value' must be 0 or greater, got {noise_value}.")
            parameters_dictionary['noise series'] = [noise_value]

        elif unique_noise_key in ('estimate noise from data', 'estimate_noise_from_data'):
            try:
                noise_filename = parameters_dictionary[unique_noise_key]
                # A spreadsheet program may save a CSV file as UTF-8 with a
                # byte-order mark, which this encoding accepts and removes.
                t0_profile_strings = np.loadtxt(noise_filename, delimiter=',', encoding='utf-8-sig')
                t0_profile_y = t0_profile_strings.astype(float).tolist()
            except Exception as e:
                raise ValueError(
                    f"Error reading CSV for noise estimation: {noise_filename!r} could not be read ({e})."
                ) from e
            # fft-cnr and the narrow-peak check take one profile; a single
            # column also loads as one.
            if t0_profile_strings.ndim > 1:
                lines, values = t0_profile_strings.shape
                raise ValueError(
                    f"The measured profile {noise_filename!r} must be a single line of "
                    f"comma-separated values; it has {lines} lines of {values} values."
                )
            # The measured profile is in its own intensity units, so its noise_rms
            # cannot be reused directly. The simulated profile has peak amplitude
            # amplitude_0; its noise must reproduce the measured initial CNR.
            cnr_estimate = fft_cnr(t0_profile_y).cnr
            if not 0 < cnr_estimate < np.inf:
                raise ValueError(
                    f"Could not estimate a finite, positive CNR from {noise_filename!r} "
                    f"(estimate: {cnr_estimate})."
                )
            sigma_n = parameters_dictionary['amplitude_0'] / cnr_estimate
            parameters_dictionary['noise series'] = [sigma_n]
            message = narrow_peak_warning(t0_profile_y, noise_filename)
            if message:
                # stacklevel 3 names the caller of parameter_parser, like the retired-key warning.
                warnings.warn(message, stacklevel=3)

        elif unique_noise_key in NOISE_RANGE_KEYS:
            noise_range = parse_noise_range(
                parameters_dictionary[unique_noise_key], NOISE_RANGE_KEYS[unique_noise_key])
            parameters_dictionary['noise series'] = make_noise_distribution(
                noise_range.low, noise_range.high,
                validate_integer(num_vals, "'number of runs'", min_val=1),
                logarithmic=noise_range.logarithmic, seed=seed,
            )
    except KeyError as e:
        raise ValueError(f"The key {e} was not found in the parameters dictionary.") from e


def handle_profile_width_parameters(parameters_dictionary: Dict[str, Any]) -> float:
    """
    Extract and convert profile width to variance.

    Supports both legacy and canonical key formats.

    Parameters
    ----------
    parameters_dictionary : dict
        Dictionary containing profile width parameters.

    Returns
    -------
    float
        Variance (sigma^2) of the initial profile.
    """
    unique_width_key = check_for_unique_key(
        parameters_dictionary,
        ['FWHM_0', 'sigma_0', 'fwhm_0']
    )

    try:
        if unique_width_key in ('FWHM_0', 'fwhm_0'):
            fwhm = parameters_dictionary[unique_width_key]
            if not isinstance(fwhm, (int, float)):
                raise ValueError(f"Invalid FWHM_0 value: {fwhm}. Must be numeric.")
            if fwhm <= 0:
                raise ValueError(f"'FWHM_0' must be greater than 0, got {fwhm}.")
            return fwhm_to_sigma2(fwhm)
        else:  # sigma_0
            sigma = parameters_dictionary['sigma_0']
            if not isinstance(sigma, (int, float)):
                raise ValueError(f"Invalid sigma_0 value: {sigma}. Must be numeric.")
            if sigma <= 0:
                raise ValueError(f"'sigma_0' must be greater than 0, got {sigma}.")
            return sigma ** 2
    except KeyError as e:
        raise ValueError(f"The key {e} was not found in the parameters dictionary.") from e


def handle_diffusion_parameters(parameters_dictionary: Dict[str, Any]) -> None:
    """
    Process diffusion-related parameters.

    Supports both legacy and canonical key formats.

    Parameters
    ----------
    parameters_dictionary : dict
        Dictionary containing diffusion parameters.

    Updates the dictionary with complete diffusion parameters.
    """
    # Check for keys in both legacy and canonical formats
    has_diff_coeff = ('nominal diffusion coefficient' in parameters_dictionary or
                      'diffusion_coefficient' in parameters_dictionary)
    has_lifetime = ('nominal lifetime (tau)' in parameters_dictionary or
                    'lifetime' in parameters_dictionary)
    has_diff_length = ('nominal diffusion length' in parameters_dictionary or
                       'diffusion_length' in parameters_dictionary)

    def get_value(legacy_key, canonical_key):
        return parameters_dictionary.get(legacy_key, parameters_dictionary.get(canonical_key))

    if has_diff_coeff and has_lifetime:
        if has_diff_length:
            raise ValueError("Provide either 'nominal diffusion length' or both "
                           "'nominal diffusion coefficient' and 'nominal lifetime (tau)', not all three.")

        diff = get_value('nominal diffusion coefficient', 'diffusion_coefficient')
        tau = get_value('nominal lifetime (tau)', 'lifetime')

        if not isinstance(diff, (int, float)) or not isinstance(tau, (int, float)):
            raise ValueError("Both 'nominal diffusion coefficient' and 'nominal lifetime (tau)' must be numeric.")

        # A lifetime of 0 means no decay, so the diffusion length is infinite.
        parameters_dictionary['nominal diffusion length'] = calculate_diffusion_length(diff, tau)
        parameters_dictionary['nominal diffusion coefficient'] = diff
        parameters_dictionary['nominal lifetime (tau)'] = tau

    elif has_diff_length:
        if has_diff_coeff or has_lifetime:
            raise ValueError("Provide either 'nominal diffusion length' or both "
                           "'nominal diffusion coefficient' and 'nominal lifetime (tau)', not a combination.")

        ld = get_value('nominal diffusion length', 'diffusion_length')
        if not isinstance(ld, (int, float)):
            raise ValueError("'nominal diffusion length' must be numeric.")
        if not ld >= 0:
            raise ValueError(f"'nominal diffusion length' must be >= 0, got {ld}")

        # Set nominal values based on diffusion length
        parameters_dictionary['nominal diffusion coefficient'] = ld ** 2
        parameters_dictionary['nominal lifetime (tau)'] = 1
        parameters_dictionary['nominal diffusion length'] = ld
        # The lifetime of 1 is a convention, not a value of the file: the
        # time axis is then in units of the lifetime. The summary says so.
        parameters_dictionary[LIFETIME_ASSUMED_KEY] = True
    else:
        raise ValueError("Provide either 'nominal diffusion length', or both "
                       "'nominal diffusion coefficient' and 'nominal lifetime (tau)'.")


# Key of a parsed dictionary whose file gave only the diffusion length, so
# that the lifetime of 1 time unit is a convention.
LIFETIME_ASSUMED_KEY = 'lifetime assumed'


def check_retired_parameters(parameters_dictionary: Dict[str, Any]) -> List[str]:
    """Messages for keys a parameter file may still contain that no longer affect the simulation."""
    return [
        f"Parameter '{key}' no longer affects the simulation and is ignored. "
        "Proximity analysis is now post-hoc: the summary reports the fraction of "
        "estimates within 5%, 10%, 20%, and 50% of nominal, and other thresholds are "
        "available from dice.analysis.statistics.analyze_simulation_results"
        "(result, proximity_levels=[...]) or from the GUI's Analysis tab "
        "('Explore Accuracy Threshold')."
        for key in RETIRED_PROXIMITY_KEYS
        if key in parameters_dictionary
    ]


def parameter_parser(parameters_dictionary: Dict[str, Any], seed: Seed = None) -> Dict[str, Any]:
    """
    Parse and validate simulation parameters.

    Accepts both legacy (space-separated) and modern (snake_case) key names.
    Returns a dictionary with legacy keys for backward compatibility.

    'noise series' holds the noise value of every run, or with a 'noise
    range', one value drawn for each run; noise_range_from_parameters tells
    the two apart.

    Parameters
    ----------
    parameters_dictionary : dict
        Raw parameters dictionary (can use either legacy or canonical keys).
    seed : int or SeedSequence, optional
        Seed for the 'noise range' draws; None draws fresh entropy.

    Returns
    -------
    dict
        Processed parameters dictionary with legacy keys.

    Raises
    ------
    KeyError
        If required parameters are missing.
    ValueError
        If parameters have invalid values.
    """
    # Make a working copy - don't normalize yet to preserve original key format
    # The check_for_unique_key function handles both formats
    parameters_dictionary = dict(parameters_dictionary)

    for message in check_retired_parameters(parameters_dictionary):
        warnings.warn(message, stacklevel=2)

    # Check required keys (support both formats)
    required_keys_legacy = ['number of runs', 'spatial width', 'pixel width', 'mean_0', 'amplitude_0']
    required_keys_canonical = ['number_of_runs', 'spatial_width', 'pixel_width', 'mu_0', 'amplitude_0']

    for legacy_key, canonical_key in zip(required_keys_legacy, required_keys_canonical):
        if legacy_key not in parameters_dictionary and canonical_key not in parameters_dictionary:
            raise KeyError(f"The required parameter '{legacy_key}' (or '{canonical_key}') is missing.")
    
    # Process spatial axis (support both key formats)
    spatial_width = parameters_dictionary.get('spatial width', parameters_dictionary.get('spatial_width'))
    pixel_width = parameters_dictionary.get('pixel width', parameters_dictionary.get('pixel_width'))
    pixel_key = 'pixel width' if 'pixel width' in parameters_dictionary else 'pixel_width'
    parameters_dictionary[pixel_key] = pixel_width = validate_integer(pixel_width, "'pixel width'")
    if pixel_width < MINIMUM_PROFILE_POINTS:
        raise ValueError(
            f"'pixel width' (the number of pixels) is {pixel_width}; a profile needs at least "
            f"{MINIMUM_PROFILE_POINTS} pixels for the estimate of its contrast-to-noise ratio."
        )
    if not isinstance(spatial_width, (int, float)) or spatial_width <= 0:
        raise ValueError(f"'spatial width' must be a number greater than 0, got {spatial_width!r}.")
    parameters_dictionary['x array'] = make_x_axis(spatial_width, pixel_width)
    
    # Process time parameters
    handle_time_parameters(parameters_dictionary)
    
    # Process noise parameters; a noise range draws one value for each run
    runs_key = 'number of runs' if 'number of runs' in parameters_dictionary else 'number_of_runs'
    number_of_runs = validate_integer(parameters_dictionary[runs_key], "'number of runs'", min_val=1)
    parameters_dictionary[runs_key] = number_of_runs
    handle_noise_parameters(parameters_dictionary, num_vals=number_of_runs, seed=seed)
    
    # Process profile width
    parameters_dictionary['sigma^2_0'] = handle_profile_width_parameters(parameters_dictionary)
    
    # Process diffusion parameters
    handle_diffusion_parameters(parameters_dictionary)
    if not parameters_dictionary['nominal diffusion coefficient'] > 0:
        raise ValueError(
            "The nominal diffusion coefficient must be greater than 0 (it follows from 'nominal diffusion "
            "coefficient', or from 'nominal diffusion length'). The estimates are reported as ratios to it."
        )

    # The number of histogram bins is a count; the plot is drawn after the
    # simulation, so a value it cannot use is rejected here.
    for bins_key in ('image numbins', 'image_numbins'):
        if bins_key in parameters_dictionary:
            parameters_dictionary[bins_key] = validate_integer(
                parameters_dictionary[bins_key], "'image numbins'", min_val=1)
    
    # Set defaults for optional parameters. A value given under either
    # spelling of a key stands; the two spellings share one canonical key.
    for key, default in OPTIONAL_PARAMETER_DEFAULTS.items():
        if key not in parameters_dictionary and normalize_parameter_key(key) not in parameters_dictionary:
            parameters_dictionary[key] = default

    # Ensure both legacy and canonical keys are present for compatibility
    parameters_dictionary = add_legacy_keys(normalize_parameters(parameters_dictionary))

    return parameters_dictionary


def validate_parameters(params: Dict[str, Any]) -> None:
    """
    Validate simulation parameters.
    
    Parameters
    ----------
    params : dict
        Parsed parameters dictionary to validate.
    
    Raises
    ------
    ValueError
        If parameters are invalid.
    """
    # parameter_parser checks each value on its own (counts, widths, the
    # time axis, the noise). The checks here need several values, or
    # concern the outputs.

    # Outputs convert from the global units, so reject an unknown unit
    # before the simulation runs rather than when the results are written.
    validate_length_unit(global_length_unit(params))
    validate_time_unit(global_time_unit(params))
    validate_output_units(params)

    # The Gaussian fit looks for the center of a profile only in the
    # central 40% of the window. Outside it, the fit holds the center at
    # the limit and reports a wrong width without failing.
    center = params.get('mu_0', params.get('mean_0'))
    if center is not None and 'x array' in params:
        from ..core.fitting import center_search_limits
        low, high = center_search_limits(params['x array'])
        if not low <= center <= high:
            raise ValueError(
                f"'mean_0' is {center:g}, outside the central 40% of the spatial window "
                f"({low:g} to {high:g}), where the fit looks for the center of the profile. "
                "Move the profile toward the center, or increase 'spatial width'."
            )

    # The slug names the output folder and every output file, so it must
    # be a plain name; a path separator or '..' would write outside it.
    validate_filename_slug(
        params.get('filename slug', params.get('filename_slug', 'dice_simulation')),
        name="'filename slug'",
    )

    # The histogram is drawn after the simulation, so reject malformed
    # limits and an image type it cannot be written as here instead of
    # losing the plot at the end of a long run.
    if params.get('image x_lim') is not None:
        validate_limits(params['image x_lim'], "'image x_lim'")
    from ..visualization.histograms import validate_image_type
    validate_image_type(params.get('image type', params.get('image_type', 'png')))
