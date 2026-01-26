"""
Parameter loading and validation for DICE simulations.

This module provides functions for loading, parsing, and validating
simulation parameters from files and dictionaries.
"""

import ast
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from ..utils.converters import fwhm_to_sigma2
from ..utils.axes import make_x_axis, make_time_axis
from ..core.noise import fft_cnr, make_noise_distribution
from ..models.parameters import SimulationParameters


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
    
    Parameters
    ----------
    parameters_dictionary : dict
        Dictionary to check.
    keys : list
        List of possible keys.
    
    Returns
    -------
    str
        The unique key that exists.
    
    Raises
    ------
    ValueError
        If none or more than one key is present.
    """
    if not isinstance(parameters_dictionary, dict):
        raise TypeError("The parameters_dictionary argument must be a dictionary.")
    if not isinstance(keys, list):
        raise TypeError("The keys argument must be a list.")
    
    existing_keys = [key for key in keys if key in parameters_dictionary]
    if len(existing_keys) > 1:
        raise ValueError(f"More than one parameter provided for {existing_keys}. Please provide only one.")
    elif len(existing_keys) == 0:
        raise ValueError(f"No parameter provided for {keys}.")
    else:
        return existing_keys[0]


def handle_time_parameters(parameters_dictionary: Dict[str, Any]) -> None:
    """
    Process time-related parameters.
    
    Parameters
    ----------
    parameters_dictionary : dict
        Dictionary containing time parameters.
    
    Updates the dictionary with 'time series' key.
    """
    unique_time_key = check_for_unique_key(parameters_dictionary, ['time range', 'time series'])
    
    if unique_time_key == 'time range':
        try:
            t_start, t_end, t_steps = parameters_dictionary[unique_time_key]
        except ValueError as e:
            raise ValueError("The 'time range' must contain three values: start, end, and number of steps.") from e
        except TypeError as e:
            raise ValueError("The 'time range' must be a sequence with three numerical values.") from e
        
        if not all(isinstance(value, (int, float)) for value in [t_start, t_end, t_steps]):
            raise ValueError("The 'time range' values must be numeric.")
        
        parameters_dictionary['time series'] = make_time_axis(t_start, t_end, int(t_steps))
        del parameters_dictionary[unique_time_key]


def handle_noise_parameters(parameters_dictionary: Dict[str, Any], num_vals: int = 1) -> None:
    """
    Process noise-related parameters.
    
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
        'noise value', 'estimate noise from data'
    ])
    
    def validate_noise_range(noise_range):
        if len(noise_range) != 2 or not all(isinstance(value, (int, float)) for value in noise_range):
            raise ValueError(f"Invalid noise range {noise_range}. Must be two numeric values.")
    
    try:
        if unique_noise_key == 'noise value':
            noise_value = parameters_dictionary[unique_noise_key]
            if not isinstance(noise_value, (int, float)):
                raise ValueError(f"Invalid noise value {noise_value}. Must be numeric.")
            parameters_dictionary['noise series'] = [noise_value]
            
        elif unique_noise_key == 'estimate noise from data':
            try:
                noise_filename = parameters_dictionary[unique_noise_key]
                t0_profile_strings = np.loadtxt(noise_filename, delimiter=',')
                t0_profile_y = t0_profile_strings.astype(float).tolist()
            except Exception as e:
                raise ValueError("Error reading CSV for noise estimation.") from e
            cnr_est = fft_cnr(t0_profile_y)
            sigma_n = 1.0 / cnr_est
            parameters_dictionary['noise series'] = [sigma_n]
            
        elif unique_noise_key == 'noise range, reciprocal log':
            noise_range = parameters_dictionary[unique_noise_key]
            validate_noise_range(noise_range)
            parameters_dictionary['noise series'] = make_noise_distribution(
                noise_range[0], noise_range[1], num_vals, logarithmic=True
            )
            
        elif unique_noise_key == 'noise range, reciprocal':
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
    
    Parameters
    ----------
    parameters_dictionary : dict
        Dictionary containing profile width parameters.
    
    Returns
    -------
    float
        Variance (sigma^2) of the initial profile.
    """
    unique_width_key = check_for_unique_key(parameters_dictionary, ['FWHM_0', 'sigma_0'])
    
    try:
        if unique_width_key == 'FWHM_0':
            fwhm = parameters_dictionary['FWHM_0']
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
    
    Parameters
    ----------
    parameters_dictionary : dict
        Dictionary containing diffusion parameters.
    
    Updates the dictionary with complete diffusion parameters.
    """
    keys = ['nominal diffusion coefficient', 'nominal lifetime (tau)', 'nominal diffusion length']
    provided_keys = [key for key in keys if key in parameters_dictionary]
    
    if 'nominal diffusion coefficient' in provided_keys and 'nominal lifetime (tau)' in provided_keys:
        if 'nominal diffusion length' in provided_keys:
            raise ValueError("Provide either 'nominal diffusion length' or both "
                           "'nominal diffusion coefficient' and 'nominal lifetime (tau)', not all three.")
        
        diff = parameters_dictionary['nominal diffusion coefficient']
        tau = parameters_dictionary['nominal lifetime (tau)']
        
        if not isinstance(diff, (int, float)) or not isinstance(tau, (int, float)):
            raise ValueError("Both 'nominal diffusion coefficient' and 'nominal lifetime (tau)' must be numeric.")
        
        parameters_dictionary['nominal diffusion length'] = np.sqrt(diff * tau)
        
    elif 'nominal diffusion length' in provided_keys:
        if len(provided_keys) > 1:
            raise ValueError("Provide either 'nominal diffusion length' or both "
                           "'nominal diffusion coefficient' and 'nominal lifetime (tau)', not a combination.")
        
        ld = parameters_dictionary['nominal diffusion length']
        if not isinstance(ld, (int, float)):
            raise ValueError("'nominal diffusion length' must be numeric.")
        
        # Set nominal values based on diffusion length
        parameters_dictionary['nominal diffusion coefficient'] = ld ** 2
        parameters_dictionary['nominal lifetime (tau)'] = 1
    else:
        raise ValueError("Provide either 'nominal diffusion length', or both "
                       "'nominal diffusion coefficient' and 'nominal lifetime (tau)'.")


def parameter_parser(parameters_dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse and validate simulation parameters.
    
    Parameters
    ----------
    parameters_dictionary : dict
        Raw parameters dictionary.
    
    Returns
    -------
    dict
        Processed parameters dictionary.
    
    Raises
    ------
    KeyError
        If required parameters are missing.
    ValueError
        If parameters have invalid values.
    """
    # Check required keys
    required_keys = ['number of runs', 'spatial width', 'pixel width', 'mean_0', 'amplitude_0']
    for key in required_keys:
        if key not in parameters_dictionary:
            raise KeyError(f"The required parameter '{key}' is missing.")
    
    # Process spatial axis
    parameters_dictionary['x array'] = make_x_axis(
        parameters_dictionary['spatial width'],
        parameters_dictionary['pixel width']
    )
    
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