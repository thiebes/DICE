"""
Parameter loading and validation for DICE simulations.

This module provides functions for loading, parsing, and validating
simulation parameters from files and dictionaries.

Accepts both legacy (space-separated) and modern (snake_case) key names.
"""

import ast
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from ..utils.converters import fwhm_to_sigma2
from ..utils.axes import make_x_axis, make_time_axis
from ..core.noise import fft_cnr, make_noise_distribution
from ..models.parameters import SimulationParameters
from ..models.parameter_keys import (
    normalize_parameters, normalize_parameter_key, add_legacy_keys,
    ParameterKey, LEGACY_OUTPUT_KEYS
)


def open_parameters(filename: Union[str, Path]) -> Dict[str, Any]:
    """
    Read simulation parameters from a file.
    
    Parameters
    ----------
    filename : str or Path
        Path to the parameters file.
    
    Returns
    -------
    dict
        Parsed parameters dictionary.
    
    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file content is not a valid dictionary.
    """
    try:
        with open(filename, 'r') as f:
            parms_txt = f.read()
            parms_dict = ast.literal_eval(parms_txt)
            # Resolve per-parameter unit overrides before parsing,
            # so that values are in the global unit system when the
            # parser converts between representations (e.g., FWHM -> sigma^2).
            from ..utils.units import resolve_units
            parms_dict = resolve_units(parms_dict)
            result = parameter_parser(parms_dict)
        return result
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {filename} was not found.")
    except SyntaxError as e:
        raise ValueError(f"Error evaluating the file's contents as a dictionary: {e}")
    except Exception as e:
        raise Exception(f"An error occurred while parsing parameters: {e}")


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

    Updates the dictionary with 'time series' key.
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

        parameters_dictionary['time series'] = make_time_axis(t_start, t_end, int(t_steps))
        if unique_time_key in parameters_dictionary:
            del parameters_dictionary[unique_time_key]


def handle_noise_parameters(parameters_dictionary: Dict[str, Any], num_vals: int = 1) -> None:
    """
    Process noise-related parameters.

    Supports both legacy and canonical key formats.

    Parameters
    ----------
    parameters_dictionary : dict
        Dictionary containing noise parameters.
    num_vals : int
        Number of noise values to generate.

    Updates the dictionary with 'noise series' key.
    """
    unique_noise_key = check_for_unique_key(parameters_dictionary, [
        'noise range, reciprocal log', 'noise range, reciprocal',
        'noise value', 'estimate noise from data',
        'noise_range_reciprocal_log', 'noise_range_reciprocal',
        'noise_value', 'estimate_noise_from_data'
    ])

    def validate_noise_range(noise_range):
        if len(noise_range) != 2 or not all(isinstance(value, (int, float)) for value in noise_range):
            raise ValueError(f"Invalid noise range {noise_range}. Must be two numeric values.")

    try:
        if unique_noise_key in ('noise value', 'noise_value'):
            noise_value = parameters_dictionary[unique_noise_key]
            if not isinstance(noise_value, (int, float)):
                raise ValueError(f"Invalid noise value {noise_value}. Must be numeric.")
            parameters_dictionary['noise series'] = [noise_value]

        elif unique_noise_key in ('estimate noise from data', 'estimate_noise_from_data'):
            try:
                noise_filename = parameters_dictionary[unique_noise_key]
                t0_profile_strings = np.loadtxt(noise_filename, delimiter=',')
                t0_profile_y = t0_profile_strings.astype(float).tolist()
            except Exception as e:
                raise ValueError("Error reading CSV for noise estimation.") from e
            cnr_est = fft_cnr(t0_profile_y)
            sigma_n = 1.0 / cnr_est
            parameters_dictionary['noise series'] = [sigma_n]

        elif unique_noise_key in ('noise range, reciprocal log', 'noise_range_reciprocal_log'):
            noise_range = parameters_dictionary[unique_noise_key]
            validate_noise_range(noise_range)
            parameters_dictionary['noise series'] = make_noise_distribution(
                noise_range[0], noise_range[1], num_vals, logarithmic=True
            )

        elif unique_noise_key in ('noise range, reciprocal', 'noise_range_reciprocal'):
            noise_range = parameters_dictionary[unique_noise_key]
            validate_noise_range(noise_range)
            parameters_dictionary['noise series'] = make_noise_distribution(
                noise_range[0], noise_range[1], num_vals, logarithmic=False
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
            return fwhm_to_sigma2(fwhm)
        else:  # sigma_0
            sigma = parameters_dictionary['sigma_0']
            if not isinstance(sigma, (int, float)):
                raise ValueError(f"Invalid sigma_0 value: {sigma}. Must be numeric.")
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

        parameters_dictionary['nominal diffusion length'] = np.sqrt(diff * tau)
        parameters_dictionary['nominal diffusion coefficient'] = diff
        parameters_dictionary['nominal lifetime (tau)'] = tau

    elif has_diff_length:
        if has_diff_coeff or has_lifetime:
            raise ValueError("Provide either 'nominal diffusion length' or both "
                           "'nominal diffusion coefficient' and 'nominal lifetime (tau)', not a combination.")

        ld = get_value('nominal diffusion length', 'diffusion_length')
        if not isinstance(ld, (int, float)):
            raise ValueError("'nominal diffusion length' must be numeric.")

        # Set nominal values based on diffusion length
        parameters_dictionary['nominal diffusion coefficient'] = ld ** 2
        parameters_dictionary['nominal lifetime (tau)'] = 1
        parameters_dictionary['nominal diffusion length'] = ld
    else:
        raise ValueError("Provide either 'nominal diffusion length', or both "
                       "'nominal diffusion coefficient' and 'nominal lifetime (tau)'.")


def parameter_parser(parameters_dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse and validate simulation parameters.

    Accepts both legacy (space-separated) and modern (snake_case) key names.
    Returns a dictionary with legacy keys for backward compatibility.

    Parameters
    ----------
    parameters_dictionary : dict
        Raw parameters dictionary (can use either legacy or canonical keys).

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

    # Check required keys (support both formats)
    required_keys_legacy = ['number of runs', 'spatial width', 'pixel width', 'mean_0', 'amplitude_0']
    required_keys_canonical = ['number_of_runs', 'spatial_width', 'pixel_width', 'mu_0', 'amplitude_0']

    for legacy_key, canonical_key in zip(required_keys_legacy, required_keys_canonical):
        if legacy_key not in parameters_dictionary and canonical_key not in parameters_dictionary:
            raise KeyError(f"The required parameter '{legacy_key}' (or '{canonical_key}') is missing.")
    
    # Process spatial axis (support both key formats)
    spatial_width = parameters_dictionary.get('spatial width', parameters_dictionary.get('spatial_width'))
    pixel_width = parameters_dictionary.get('pixel width', parameters_dictionary.get('pixel_width'))
    parameters_dictionary['x array'] = make_x_axis(spatial_width, pixel_width)
    
    # Process time parameters
    handle_time_parameters(parameters_dictionary)
    
    # Process noise parameters
    handle_noise_parameters(parameters_dictionary, num_vals=1)
    
    # Process profile width
    parameters_dictionary['sigma^2_0'] = handle_profile_width_parameters(parameters_dictionary)
    
    # Process diffusion parameters
    handle_diffusion_parameters(parameters_dictionary)
    
    # Set defaults for optional parameters
    if 'multiprocessing' not in parameters_dictionary:
        parameters_dictionary['multiprocessing'] = 1
    
    if 'filename slug' not in parameters_dictionary:
        parameters_dictionary['filename slug'] = 'dice_simulation'
    
    # Don't set default proximity level - let CLI handle standard vs custom levels
    
    if 'retain profile data' not in parameters_dictionary:
        parameters_dictionary['retain profile data'] = 0
    
    # Image parameters
    if 'image type' not in parameters_dictionary:
        parameters_dictionary['image type'] = 'png'
    
    if 'image width' not in parameters_dictionary:
        parameters_dictionary['image width'] = 10
    
    if 'image height' not in parameters_dictionary:
        parameters_dictionary['image height'] = 6
    
    if 'image dpi' not in parameters_dictionary:
        parameters_dictionary['image dpi'] = 100

    # Ensure both legacy and canonical keys are present for compatibility
    parameters_dictionary = add_legacy_keys(normalize_parameters(parameters_dictionary))

    return parameters_dictionary


def create_simulation_parameters(params_dict: Dict[str, Any]):
    """
    Create SimulationParameters object from dictionary.
    
    Parameters
    ----------
    params_dict : dict
        Parsed parameters dictionary.
    
    Returns
    -------
    SimulationParameters
        Structured parameters object.
    """
    # This would create a SimulationParameters object
    # For now, just return the dictionary as compatibility layer
    return params_dict


def validate_parameters(params) -> None:
    """
    Validate simulation parameters.
    
    Parameters
    ----------
    params : dict or SimulationParameters
        Parameters to validate.
    
    Raises
    ------
    ValueError
        If parameters are invalid.
    """
    # Basic validation for dictionary format
    if isinstance(params, dict):
        if params.get('number of runs', 0) <= 0:
            raise ValueError("Number of runs must be positive.")
        
        if params.get('spatial width', 0) <= 0:
            raise ValueError("Spatial width must be positive.")
        
        if params.get('pixel width', 0) <= 0:
            raise ValueError("Pixel width must be positive.")
        
        return
    
    # Original validation for SimulationParameters object
    # (kept for future when proper classes are implemented)
    pass