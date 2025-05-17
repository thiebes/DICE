# Standard Python libraries
import ast
from typing import Any, Dict, List, Tuple

import numpy as np

from dice.utils import (
    make_x_axis,            make_time_axis,         make_noise_distribution,
    fwhm_to_sigma2,         fft_cnr,                
)

def open_parameters(filename):
    """
    Reads simulation parameters from a file, evaluates and parses them.

    Parameters:
    - filename: The name of the file containing the simulation parameters.

    Returns:
    - A dictionary with parsed and formatted simulation parameters.

    Raises:
    - FileNotFoundError: If the file does not exist.
    - ValueError: If the file content is not a valid dictionary.
    - Exception: Propagates any parsing errors from `parameter_parser`.
    """
    try:
        with open(filename, 'r') as f:
            parms_txt = f.read() # read the file
            parms_dict = ast.literal_eval(parms_txt) # evaluate the content literally
            result = parameter_parser(parms_dict) # parse the content
        return result
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {filename} was not found.")
    except SyntaxError as e:
        raise ValueError(f"Error evaluating the file's contents as a dictionary: {e}")
    except Exception as e:
        # Re-raise any exceptions from the parameter_parser or other unexpected issues
        raise Exception(f"An error occurred while parsing parameters: {e}")
    
def parameter_parser(parameters_dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses the parameters dictionary and sets up simulation parameters.

    Parameters:
    - parameters_dictionary: Dictionary with parameters for the simulation.

    Returns:
    - A dictionary with parsed and formatted simulation parameters.

    Raises:
    - KeyError: If a required parameter is missing from the dictionary.
    - ValueError: If there are conflicting or incorrect types of parameters.
    """
    # Ensure that these required keys are present 
    required_keys = ['number of runs', 'spatial width', 'pixel width', 'mean_0', 'amplitude_0']
    for key in required_keys:
        if key not in parameters_dictionary:
            raise KeyError(f"The required parameter '{key}' is missing from the parameters dictionary.")
    # Further checks for required parameters are performed within each helper function called below.

    num_noise = 1 # future use: parameters_dictionary['noise range, number of values']
    parameters_dictionary['x array'] = make_x_axis(
        parameters_dictionary['spatial width'], 
        parameters_dictionary['pixel width'], 
        parameters_dictionary['mean_0']
    )

    # Handle profile width and set up the Gaussian parameters
    width_0 = handle_profile_width_parameters(parameters_dictionary)
    amp_0, mean_0 = parameters_dictionary['amplitude_0'], parameters_dictionary['mean_0']
    parameters_dictionary['t0 Gaussian sigma^2, amplitude, mean'] = [width_0, amp_0, mean_0]

    # Delegate handling of other parameters to dedicated functions
    handle_time_parameters(parameters_dictionary)
    handle_noise_parameters(parameters_dictionary, num_noise)
    handle_diffusion_parameters(parameters_dictionary)

    return parameters_dictionary

def check_for_unique_key(parameters_dictionary: Dict[str, Any], keys: List[str]) -> str:
    """
    Checks for the presence of a unique key in a dictionary from a list of possible keys.

    Parameters:
    - parameters_dictionary: A dictionary where the check is to be performed.
    - keys: A list of keys, of which exactly one must exist in the dictionary.

    Returns:
    - The unique key that exists in the dictionary.

    Raises:
    - ValueError: If more than one of the specified keys is present in the dictionary,
                   or if none of the keys are present.
    - TypeError: If the inputs are not of the expected type (dictionary and list).
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
    Processes time-related parameters in the given dictionary by either creating
    a time series from a time range or verifying the presence of an explicit time series.

    Parameters:
    - parameters_dictionary: A dictionary containing the simulation parameters.

    Updates:
    - The 'parameters_dictionary' will be updated with a 'time series' key containing
      the calculated time series if 'time range' is provided. If 'time series' is provided,
      it is assumed to be correct and is left as-is.

    Raises:
    - ValueError: If 'time range' contains invalid data or is incomplete.
    """
    unique_time_key = check_for_unique_key(parameters_dictionary, ['time range', 'time series'])
    if unique_time_key == 'time range':
        try:
            t_start, t_end, t_steps = parameters_dictionary[unique_time_key]
        except ValueError as e:
            raise ValueError("The 'time range' must contain three values: start, end, and number of steps.") from e
        except TypeError as e:
            raise ValueError("The 'time range' must be a sequence (e.g., list or tuple) with three numerical values.") from e

        # check if t_start, t_end, t_steps have the expected types (e.g., numerical types)
        if not all(isinstance(value, (int, float)) for value in [t_start, t_end, t_steps]):
            raise ValueError("The 'time range' values must be numeric.")

        parameters_dictionary['time series'] = make_time_axis(t_start, t_end, t_steps)
        del parameters_dictionary[unique_time_key]
    
    # Else if 'time series' is provided, it's assumed to be valid and no action is taken.

def handle_noise_parameters(parameters_dictionary: Dict[str, Any], num_vals: int = 1) -> None:
    """
    Processes noise-related parameters in the provided dictionary based on the unique noise key.

    Parameters:
    - parameters_dictionary: A dictionary containing the noise parameters.
    - num_vals: The number of values to be in the noise series.

    Updates:
    - The 'parameters_dictionary' will be updated with a 'noise series' key containing
      the noise values based on the noise parameters provided by the user.

    Raises:
    - ValueError: If any of the provided noise parameters are invalid or missing.
    """
    unique_noise_key = check_for_unique_key(parameters_dictionary, [
        'noise range, reciprocal log', 'noise range, reciprocal',
        'noise value', 'estimate noise from data'
    ])

    # Function to validate noise range input
    def validate_noise_range(noise_range):
        if len(noise_range) != 2 or not all(isinstance(value, (int, float)) for value in noise_range):
            raise ValueError(f"Invalid noise range {noise_range}. It must be a sequence of two numeric values.")

    try:
        if unique_noise_key == 'noise value':
            noise_value = parameters_dictionary[unique_noise_key]
            if not isinstance(noise_value, (int, float)):
                raise ValueError(f"Invalid noise value {noise_value}. It must be a numeric value.")
            parameters_dictionary['noise series'] = [noise_value]
        elif unique_noise_key == 'estimate noise from data':
            try:
                # Assuming 'estimate noise from data' is a file path to the CSV
                noise_filename = parameters_dictionary[unique_noise_key]
                t0_profile_strings = np.loadtxt(noise_filename, delimiter=',')
                print(t0_profile_strings)
                t0_profile_y = t0_profile_strings.astype(float).tolist()
            except Exception as e:
                raise ValueError("Error reading CSV for noise estimation.") from e
            cnr_est = fft_cnr(t0_profile_y)
            sigma_n = np.power(cnr_est, -1)
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
    Extracts and converts the profile width parameter to Gaussian variance (sigma squared).

    Parameters:
    - parameters_dictionary: A dictionary containing the profile width parameters.

    Returns:
    - The converted profile width parameter as variance (sigma^2).

    Raises:
    - ValueError: If the profile width parameter is not provided or cannot be converted.
    """
    unique_width_key = check_for_unique_key(parameters_dictionary, ['FWHM_0', 'sigma_0'])

    try:
        if unique_width_key == 'FWHM_0':
            # Convert FWHM to variance.
            fwhm = parameters_dictionary['FWHM_0']
            if not isinstance(fwhm, (int, float)):
                raise ValueError(f"Invalid t0 FWHM value: {fwhm}. It must be a numeric value.")
            return fwhm_to_sigma2(fwhm)
        else:  # If the key is 't0 sigma'
            sigma = parameters_dictionary['sigma_0']
            if not isinstance(sigma, (int, float)):
                raise ValueError(f"Invalid sigma_0 value: {sigma}. It must be a numeric value.")
            return np.power(sigma, 2.)
    except KeyError as e:
        raise ValueError(f"The key {e} was not found in the parameters dictionary.") from e

def handle_diffusion_parameters(parameters_dictionary: Dict[str, Any]) -> None:
    """
    Handles the diffusion parameters by either calculating the diffusion length
    from the nominal diffusion coefficient and lifetime (tau), or vice-versa.

    Parameters:
    - parameters_dictionary: A dictionary containing the diffusion-related parameters.

    Raises:
    - ValueError: If an inconsistent set of parameters is provided.
    """
    keys = ['nominal diffusion coefficient', 'nominal lifetime (tau)', 'nominal diffusion length']
    provided_keys = [key for key in keys if key in parameters_dictionary]
    
    # Check if both 'nominal diffusion coefficient' and 'nominal lifetime (tau)' are provided
    if 'nominal diffusion coefficient' in provided_keys and 'nominal lifetime (tau)' in provided_keys:
        if 'nominal diffusion length' in provided_keys:
            raise ValueError("Please provide either 'nominal diffusion length' or both 'nominal diffusion coefficient' and 'nominal lifetime (tau)', not all three.")
        diff = parameters_dictionary['nominal diffusion coefficient']
        tau = parameters_dictionary['nominal lifetime (tau)']
        # Ensure numeric values are provided
        if not isinstance(diff, (int, float)) or not isinstance(tau, (int, float)):
            raise ValueError("Both 'nominal diffusion coefficient' and 'nominal lifetime (tau)' must be numeric.")
        parameters_dictionary['nominal diffusion length'] = np.sqrt(diff * tau)
        
    # Check if only 'diffusion length' is provided
    elif 'nominal diffusion length' in provided_keys:
        if len(provided_keys) > 1:
            raise ValueError("You provided 'nominal diffusion length' along with diffusion coefficient and/or lifetime. Please provide either 'diffusion length' or both 'diffusion coefficient' and 'lifetime (tau)', not a combination.")
        ld = parameters_dictionary['nominal diffusion length']
        # Ensure a numeric value is provided
        if not isinstance(ld, (int, float)):
            raise ValueError("'nominal diffusion length' must be numeric.")
        # Set nominal values of diffusion coefficient and lifetime based on diffusion length
        # Note there are infinite combinations that would be valid, but the end results would
        # be the same. Thus nom lifetime is arbitrarily set to 1, and nom diffusion coeff
        # is arbitrarily set to the square of the diffusion length. 
        parameters_dictionary['nominal diffusion coefficient'] = np.power(ld,2)
        parameters_dictionary['nominal lifetime (tau)'] = 1

    # In the case that nothing was provided:       
    else:
        raise ValueError("Provide nominal values for either 'nominal diffusion length', or both 'nominal diffusion coefficient' and 'nominal lifetime (tau)'.")
