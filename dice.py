#########################################################################
# Diffusion Insight Computation Engine (DICE) simulates optical         #
# measures of diffusion in optoelectronic semiconducting materials      #
# using experimental parameters, and evaluates the precision of         #
# composite fitting methods of estimating the diffusion coefficient.    #
#                                                                       #
# Copyright (C) 2023-2024 Joseph J. Thiebes                             #
#                                                                       #
# This software may be cited as follows:                                #
# Joseph J. Thiebes. (2024). DICE. Zenodo. DOI:10.5281/zenodo.10258192  #
#                                                                       #
# The concepts and methodologies underpinning this software were        #
# developed concurrently with the research findings presented in the    #
# paper referenced below. We strongly encourage users to consult the    #
# paper to gain comprehensive insights into the scientific and          #
# statistical principles that inform the functionality and application  #
# of this software.                                                     #
#                                                                       #
# Joseph J. Thiebes, Erik M. Grumstrup; Quantifying noise effects in    #
# optical measures of excited state transport. J. Chem. Phys. 28 March  #
# 2024; 160 (12): 124201. https://doi.org/10.1063/5.0190347             #
#                                                                       #
# This material is based upon work supported by the National Science    #
# Foundation under Grant No. 2154448. Any opinions, findings, and       #
# conclusions or recommendations expressed in this material are those   #
# of the author(s) and do not necessarily reflect the views of the      #
# National Science Foundation.                                          #
#                                                                       #
# This work is licensed under the Creative Commons Attribution 4.0      #
# International License. To view a copy of this license, visit          #
# http://creativecommons.org/licenses/by/4.0/ or send a letter to       #
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.          #
#                                                                       #
# This program is distributed in the hope that it will be useful,       #
# but WITHOUT ANY WARRANTY; without even the implied warranty of        #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                  #
#                                                                       #
# You should include a copy of the license or a link to it with         #
# every copy of the work you distribute. You can do this by             #
# including a link to the license in your README.md file or             #
# documentation.                                                        #
#########################################################################
# See the README.md file for information about how to use this tool.    #
#########################################################################

# Library imports
import numpy as np
from numpy.random import default_rng
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import statsmodels.api as sm
from joblib import Parallel, delayed

# Standard Python libraries
import ast
import os
import re
from typing import Any, Dict, List, Tuple

def make_x_axis(scan_width: float, scan_width_pixels: int, mu: float) -> np.ndarray:
    """
    Create an array representing an x-axis.

    Parameters:
    scan_width (float): Total width of x-axis in spatial units.
    scan_width_pixels (int): Total width of x-axis in pixels.
    mu (float): Center of x-axis aligned with center of point spread function.

    Returns:
    np.ndarray: Array of x-axis values.
    """
    if scan_width_pixels <= 0:
        raise ValueError("scan_width_pixels must be a positive integer")
    if scan_width <= 0:
        raise ValueError("scan_width must be a positive number")
    
    x_start = mu - scan_width / 2
    x_end = mu + scan_width / 2
    x_values = np.linspace(x_start, x_end, scan_width_pixels)
    return x_values

def make_time_axis(t_start: float, t_end: float, t_frames: int) -> np.ndarray:
    """
    Create an array representing a time axis.

    Parameters:
    t_start (float): Start timestamp.
    t_end (float): End timestamp.
    t_frames (int): Number of time frames.

    Returns:
    np.ndarray: Array of time frame values.
    """
    if t_frames <= 0:
        raise ValueError("Number of time frames must be a positive integer")
    if t_start >= t_end:
        raise ValueError("Start timestamp must be less than the end timestamp")
    
    return np.linspace(t_start, t_end, t_frames)

def sigma_to_fwhm(sigma: float) -> float:
    """
    Convert standard deviation (sigma) of a Gaussian to the full width at half maximum (FWHM).

    Parameters:
    sigma (float): Standard deviation of the Gaussian.

    Returns:
    float: FWHM of the Gaussian.
    """
    if sigma <= 0:
        raise ValueError("Sigma must be a positive number")
    return sigma * 2 * np.sqrt(2 * np.log(2))

def sigma2_to_fwhm(sigma2: float) -> float:
    """
    Convert variance (sigma^2) of a Gaussian to the full width at half maximum (FWHM).

    Parameters:
    sigma2 (float): Variance of the Gaussian.

    Returns:
    float: FWHM of the Gaussian.
    """
    if sigma2 <= 0:
        raise ValueError("Sigma squared must be a positive number")
    return sigma_to_fwhm(np.sqrt(sigma2))

def fwhm_to_sigma(fwhm: float) -> float:
    """
    Convert full width at half maximum (FWHM) of a Gaussian to standard deviation (sigma).

    Parameters:
    fwhm (float): Full width at half maximum of the Gaussian.

    Returns:
    float: Standard deviation (sigma) of the Gaussian.
    """
    if fwhm <= 0:
        raise ValueError("FWHM must be a positive number")
    return fwhm / (2 * np.sqrt(2 * np.log(2)))

def fwhm_to_sigma2(fwhm: float) -> float:
    """
    Convert full width at half maximum (FWHM) of a Gaussian to variance (sigma^2).

    Parameters:
    fwhm (float): Full width at half maximum of the Gaussian.

    Returns:
    float: Variance (sigma^2) of the Gaussian.
    """
    return fwhm_to_sigma(fwhm) ** 2

def gaussian(x: np.ndarray, mu: float, sig2: float, amp: float) -> np.ndarray:
    """
    Define a Gaussian function. Baseline is assumed to be zero.

    Parameters:
    x (np.ndarray): Array of x values.
    mu (float): Mean value of the Gaussian. Default: 0
    sig2 (float): Variance (sigma^2) of the Gaussian.
    amp (float): Amplitude of the Gaussian. Default: 1

    Returns:
    np.ndarray: Gaussian function y-values.
    """
    if sig2 <= 0:
        raise ValueError("Variance (sig2) must be a positive number")
    if amp < 0:
        raise ValueError("Amplitude (amp) must be non-negative")

    return amp * np.exp(-1 * np.power(x - mu, 2) / (2 * sig2))

def integrated_intensity(sig2: float, amp: float) -> float:
    """
    Calculate the integrated intensity (area under the curve) of a Gaussian function.

    Parameters:
    - sig2 (float): The variance of the Gaussian (sigma^2).
    - amp (float): The amplitude of the Gaussian peak.

    Returns:
    - float: The integrated intensity of the Gaussian.

    Raises:
    - ValueError: If any input is non-positive, as the variance and amplitude
                   must be positive for a valid Gaussian function.
    """
    # Check that the inputs are positive, which is necessary for a valid Gaussian
    if sig2 <= 0:
        raise ValueError("Variance (sigma^2) must be positive.")
    if amp <= 0:
        raise ValueError("Amplitude must be positive.")

    return amp * np.sqrt(2 * np.pi * sig2)


def slope_to_diffusion_constant(slope: float, l_unit: str, t_unit: str) -> float:
    """
    Convert the slope from a linear fit of mean squared displacement vs. time 
    -- (i.e., MSD(t) = sigma^2_t - sigma^2_0) --
    to a diffusion coefficient in conventional units [cm^2/s].

    Parameters:
    - slope (float): The slope from the MSD vs. time linear fit in user-provided units of length^2/time.
    - l_unit (str): The user-provided unit of length used in the slope (e.g., 'micrometer').
    - t_unit (str): The user-provided unit of time used in the slope (e.g., 'nanosecond').

    Returns:
    - float: The diffusion coefficient in units of cm^2/s.

    Raises:
    - ValueError: If the provided length or time units are not supported.
    """

    # Nested dictionaries for length and time conversion factors to centimeters and seconds
    conversion_factors = {
        'length': {
            'meter': 100,
            'centimeter': 1,
            'millimeter': 0.1,
            'micrometer': 1e-4,
            'nanometer': 1e-7,
            'angstrom': 1e-8,
            'picometer': 1e-10,
            # Add more length units here as needed
        },
        'time': {
            'second': 1,
            'millisecond': 1e-3,
            'microsecond': 1e-6,
            'nanosecond': 1e-9,
            'picosecond': 1e-12,
            'femtosecond': 1e-15,
            'attosecond': 1e-18,
            # Add more time units here as needed
        }
    }

    # Error handling for invalid units
    if l_unit not in conversion_factors['length']:
        raise ValueError(f"Invalid length unit '{l_unit}'. Please use one of the following: "
                         f"{', '.join(conversion_factors['length'].keys())}.")
    if t_unit not in conversion_factors['time']:
        raise ValueError(f"Invalid time unit '{t_unit}'. Please use one of the following: "
                         f"{', '.join(conversion_factors['time'].keys())}.")

    # Convert the slope to cm^2/s (note that 1 cm = 0.01 m)
    slope_cm2_per_s = slope * conversion_factors['length'][l_unit] ** 2 / conversion_factors['time'][t_unit]

    # Divide by 2 to get the diffusion coefficient in one dimension
    return slope_cm2_per_s / 2

def make_noise_distribution(noise_low, noise_high, num, logarithmic=False):
    """
    Create a distribution of noise sigmas, uniform in reciprocal space.
    Optionally, make the distribution uniform in log space as well.

    This is for the purpose of generating data that are uniformly distributed
    on a plot where one axis is CNR (inverse of noise) and possibly 
    logarithmic scale.

    Parameters:
    - noise_low (float): The lower bound for noise sigma values.
    - noise_high (float): The upper bound for noise sigma values.
    - num (int): The number of samples to generate.
    - logarithmic (bool, optional): Flag to generate the distribution in
      reciprocal log space. Defaults to False for linear space.

    Returns:
    - list: A list of noise sigma values.

    Raises:
    - ValueError: If the bounds are not positive or if the lower bound is
                   greater than or equal to the upper bound.
    """
    if noise_low <= 0 or noise_high <= 0:
        raise ValueError("Noise bounds must be positive.")
    if noise_low >= noise_high:
        raise ValueError("The lower bound must be less than the upper bound.")

    # Get the reciprocal of the noise range. Note this reverses their sequence.
    new_noise_high = 1 / noise_low
    new_noise_low = 1 / noise_high

    # Switch to logarithmic space if indicated
    if logarithmic:
        new_noise_low = np.log10(new_noise_low)
        new_noise_high = np.log10(new_noise_high)

    # Instantiate a PCG-64 pseudo-random number generator
    rng = default_rng()
    # Create the uniform distribution in the selected space
    noise_sigmas_inv = rng.uniform(low=new_noise_low, high=new_noise_high, size=num)

    # Transform the values back to direct space
    if logarithmic:
        noise_sigmas = np.power(10, -noise_sigmas_inv)
    else:
        noise_sigmas = 1 / noise_sigmas_inv

    return noise_sigmas.tolist()

#################################################################################
# Get parameters from file
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

#################################################################################
def print_and_append(summary_filename, text):
    """
    Prints the given text to the console and appends it to the specified file.
    
    Parameters:
    summary_filename (str): The path to the file where text should be appended.
    text (str): The text to be printed and appended to the file.
    """
    # Print the text to the console
    print(text)
    
    # Try appending the text to the specified file
    try:
        with open(summary_filename, 'a') as file1:
            file1.write(text + '\n')
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")

##################################
# model run management functions #
##################################

###########################################################
# Generate simulations using parameters
def dice_runner(parameters_filename):
    """
    Execute a series of simulations based on parameters provided in a file.

    This function reads simulation parameters from a file, initializes the simulation
    environment, and executes multiple simulation runs. It collates results from 
    individual simulations, performs analysis, and exports summary data.

    Parameters:
    - parameters_filename (str): Path to a text file containing simulation parameters.

    Returns:
    - result_dictionary (dict): A dictionary containing indices, parameters, run results,
                                collated results, analysis, and a filename slug for output.

    Raises:
    - FileNotFoundError: If the parameters file does not exist or is unreadable.
    - KeyError: If the expected keys are not present in the parameters file.
    - ValueError: If parameter values are not of the expected type or out of expected range.
    - IOError: If there is an error during file writing operations.

    Note:
    - This function uses multiprocessing if enabled in the parameters to speed up simulations.
    """

    # open and process the parameters
    parameters_dictionary = open_parameters(parameters_filename)

    # image parameters
    image_type = parameters_dictionary['image type']
    image_width = parameters_dictionary['image width']
    image_height = parameters_dictionary['image height']
    image_dpi = parameters_dictionary['image dpi']
    image_font_size = parameters_dictionary['image font size']
    image_tick_l = parameters_dictionary['image tick length']
    image_tick_w = parameters_dictionary['image tick width']
    image_numbins = parameters_dictionary['image numbins']
    image_x_lim = parameters_dictionary['image x_lim']

    # Units
    l_unit = parameters_dictionary['length unit']
    t_unit= parameters_dictionary['time unit']

    # Number of simulation runs per parameter set
    numruns = parameters_dictionary['number of runs']

    # Spatial parameters
    wid = parameters_dictionary['spatial width']
    pix = parameters_dictionary['pixel width']
    x_axis = parameters_dictionary['x array']

    # time axis
    t_axis = parameters_dictionary['time series']
    tix = len(t_axis)

    # Gaussian parameters for t0
    sig2_0, amp_0, mu_0 = parameters_dictionary['t0 Gaussian sigma^2, amplitude, mean']
    ld = parameters_dictionary['nominal diffusion length']
    diff = parameters_dictionary['nominal diffusion coefficient']
    tau = parameters_dictionary['nominal lifetime (tau)']

    # series of standard deviations of noise to be added
    noise_series = parameters_dictionary['noise series']
    noise_num = len(noise_series)

    # calculate the total number of runs
    runs_total = numruns * noise_num

    # do we keep all the profile data in memory or just the analysis results
    retain_profile_data = parameters_dictionary['retain profile data']

    # proximity level
    proximity_level = parameters_dictionary['proximity level']

    # initialize the result dictionary
    result_dictionary = {
        'indices': {
            'time axis': t_axis, 
            'x axis': x_axis,
            'noise sigmas': noise_series,
            'total runs': runs_total
        },
        'parameters':{
            'sigma^2_0': sig2_0,
            'amplitude_0': amp_0,
            'mu_0': mu_0,
            'scan width': wid,
            'scan pixels': pix,
            'nominal diffusion length': ld,
            'nominal diffusion coeff': diff,
            'nominal lifetime': tau,
            'length units': l_unit,
            'time units': t_unit,
        },
        'run results': {},
    }

    # aliases for subdictionaries
    indices = result_dictionary['indices']
    parameters = result_dictionary['parameters']

    # parameter text slugs for file naming
    file_prefix = parameters_dictionary['filename slug']
    ld_txt = str(round(parameters_dictionary['nominal diffusion length'], 3))
    cnr_txt = str(round(1/parameters_dictionary['noise series'][0], 3))
    pix_txt =  str(parameters_dictionary['pixel width'])
    tix_txt = str(len(parameters_dictionary['time series']))
    runs_txt = str(parameters_dictionary['number of runs'])

    # the filename incorporates user-specified text along with several parameters for reference
    filename_slug = file_prefix + '_LD-' + ld_txt + '_CNR-' + cnr_txt + "_px-" + pix_txt + "_tx-" + tix_txt + '_runs-' + runs_txt
    result_dictionary['filename slug'] = filename_slug
    
    summary_filename = filename_slug + '_summary.txt'
    result_dictionary['parameters']['summary filename'] = summary_filename
    
    result_filename = filename_slug + '_results.csv'
    result_dictionary['parameters']['result filename'] = result_filename

    image_type = parameters_dictionary['image type']
    image_filename = filename_slug + '_histogram.' + image_type
    result_dictionary['parameters']['image type'] = image_type
    
    # Make array of parameters for iterative simulations, indexed by run number.
    # For future use handling multiple noise values.
    run_numbers = list(range(runs_total))
    noise_list = np.concatenate([np.repeat(noise, numruns) for noise in noise_series])
    parameter_sets = zip(run_numbers, noise_list)
    
    # create summary text file and record parameters
    print_and_append(summary_filename, 'Running ' + str(indices['total runs']) + ' simulations with the following parameters (rounded):')
    print_and_append(summary_filename, '')
    print_and_append(summary_filename, 'Spatial width: ' + str(wid) + ' ' + l_unit)
    print_and_append(summary_filename, 'Pixel width: ' + str(pix) + ' pixels')
    print_and_append(summary_filename, 'Number of time frames: ' + str(tix) + ' frames')
    print_and_append(summary_filename, 'Noise stdev: ' + str(round(noise_series[0], 3)))
    print_and_append(summary_filename, 'Initial contrast-to-noise ratio (CNR): ' + str(round(1/noise_series[0], 3)))
    print_and_append(summary_filename, 'Initial profile sigma^2: ' + str(round(sig2_0, 3)) + ' ' + l_unit + '^2')
    print_and_append(summary_filename, 'Nominal diffusion length: ' + str(round(ld, 3)) + ' ' + l_unit)
    print_and_append(summary_filename, 'Nominal diffusion coeff: ' + str(round(diff, 5)) + ' ' + l_unit + '^2 per ' + t_unit )
    print_and_append(summary_filename, 'Nominal lifetime: ' + str(tau) + ' ' + t_unit)
    print_and_append(summary_filename, '')

    if parameters_dictionary['multiprocessing'] == 1:
        print_and_append(summary_filename, 'Starting simulation with multiprocessing')
        print_and_append(summary_filename, '')
        result = Parallel(n_jobs=-1)(delayed(scan_runner)(indices, parameters, ld, diff, tau, this_noise_sigma, this_run, retain_profile_data) for this_run, this_noise_sigma in parameter_sets)
    else:
        print_and_append(summary_filename, 'Starting simulation without multiprocessing')
        print_and_append(summary_filename, '')
        result = [scan_runner(indices, parameters, ld, diff, tau, this_noise_sigma, this_run, retain_profile_data) for this_run, this_noise_sigma in parameter_sets]
 
    # update the result dictionary
    [result_dictionary['run results'].update(this_result) for this_result in result]

    # Collate data from runs
    print_and_append(summary_filename, 'Simulation completed. Collating results.')
    print_and_append(summary_filename, '')
    collated_results = pd.DataFrame(
        np.asarray(
            [
            [result_dictionary['run results'][run]['run'], 
            result_dictionary['run results'][run]['run parameters']['nominal diffusion coefficient'],
            result_dictionary['run results'][run]['run parameters']['nominal lifetime'],
            result_dictionary['run results'][run]['run parameters']['nominal diffusion length'],
            1/result_dictionary['run results'][run]['run parameters']['noise stdev'], 
            result_dictionary['run results'][run]['cnr_0 estimate'],
            result_dictionary['run results'][run]['nominal profiles']['parameters_t']['sigma^2_t'][0],
            result_dictionary['run results'][run]['noisy profile fits']['sigma^2_t estimates'][0],
            result_dictionary['run results'][run]['diffusion']['unweighted fit']['MSD_t slope estimate'],
            result_dictionary['run results'][run]['diffusion']['unweighted fit']['MSD_t slope std error'],
            result_dictionary['run results'][run]['diffusion']['unweighted fit']['intercept estimate'],
            result_dictionary['run results'][run]['diffusion']['unweighted fit']['intercept standard error'],
            result_dictionary['run results'][run]['diffusion']['weighted fit']['MSD_t slope estimate'],
            result_dictionary['run results'][run]['diffusion']['weighted fit']['MSD_t slope std error'],
            result_dictionary['run results'][run]['diffusion']['weighted fit']['intercept estimate'],
            result_dictionary['run results'][run]['diffusion']['weighted fit']['intercept standard error'],
            ] 
            for run in result_dictionary['run results'].keys()
            ]),
            columns=['run number', 
                    'nominal diffusion coeff', 'nominal lifetime', 'nominal diffusion length', 
                    'nominal CNR', 'estimated CNR', 
                    'nominal sigma^2_0', 'estimated sigma^2_0',
                    'unweighted fit diffusion slope', 'unweighted fit diffusion slope stderr', 
                    'unweighted fit intercept', 'unweighted fit intercept stderr', 
                    'weighted fit diffusion slope', 'weighted fit diffusion slope stderr',
                    'weighted fit intercept', 'weighted fit intercept stderr', 
                    ]
        )

    if len(t_axis) > 1:
        print_and_append(summary_filename, 'Converting diffusion constants to conventional units')
        print_and_append(summary_filename,'')

        # get the fitted slopes with user defined units
        fit_wls_slopes = [slope for slope in collated_results['weighted fit diffusion slope']]
        fit_wls_slope_stderrs = [stderr for stderr in collated_results['weighted fit diffusion slope stderr']]
        fit_ols_slopes = [slope for slope in collated_results['unweighted fit diffusion slope']]
        fit_ols_slope_stderrs = [stderr for stderr in collated_results['unweighted fit diffusion slope stderr']]

        # convert the nominal diffusion coefficient to a MSD(t) slope, and then
        # convert to diffusion constant in conventional units of cm^2 s^-1
        nom_slopes = [diff * 2. for diff in collated_results['nominal diffusion coeff']]
        collated_results['nominal diffusion coeff [cm^2/s]'] = [
            slope_to_diffusion_constant(slope, l_unit, t_unit) for slope in nom_slopes
        ]
        # convert fitted slopes and std errors to cm^2/s
        collated_results['unweighted fit diffusion coeff [cm^2/s]'] = [
            slope_to_diffusion_constant(slope, l_unit, t_unit) for slope in fit_ols_slopes
        ]
        collated_results['unweighted fit diffusion stderr [cm^2/s]'] = [
            slope_to_diffusion_constant(slope, l_unit, t_unit) for slope in fit_ols_slope_stderrs
        ]
        collated_results['weighted fit diffusion coeff [cm^2/s]'] = [
            slope_to_diffusion_constant(slope, l_unit, t_unit) for slope in fit_wls_slopes
        ]
        collated_results['weighted fit diffusion stderr [cm^2/s]'] = [
            slope_to_diffusion_constant(stderr, l_unit, t_unit) for stderr in fit_wls_slope_stderrs
        ]

        print_and_append(summary_filename, 'Analyzing precision and accuracy')
        print_and_append(summary_filename,'')
        result_dictionary['analysis'] = estimates_precision(collated_results, proximity_level)
        ols_proxpct = result_dictionary['analysis']['% fits within proximity']['unweighted fit']
        wls_proxpct = result_dictionary['analysis']['% fits within proximity']['weighted fit']
        print_and_append(summary_filename,'Portion of fits where D_estimate / D_nominal = 1 ± ' + str(proximity_level) + ':')
        print_and_append(summary_filename, '-- Unweighted fit: ' + str(round(ols_proxpct, 2)))
        print_and_append(summary_filename, '-- Weighted fit: ' + str(round(wls_proxpct, 2)))
        print_and_append(summary_filename,'')

    else:
        print_and_append(summary_filename, 'No diffusion estimates: only one time frame')

    #store collated results
    result_dictionary['collated results'] = collated_results

    # export file of collated results
    # filename includes several parameters for identification
    print_and_append(summary_filename, 'Exporting result data and histogram.')

    collated_results.to_csv(result_filename, index = False)
    plot_accuracy_histogram(
        result_dictionary,
        proximity = proximity_level,
        filename = image_filename,
        image_type = image_type,
        width = image_width,
        height = image_height,
        dpi = image_dpi,
        font_size = image_font_size,
        tick_length = image_tick_l,
        tick_width = image_tick_w,
        num_bins = image_numbins,
        x_lim = image_x_lim
    )

    print_and_append(summary_filename, '-- Summary file: ' + summary_filename)
    print_and_append(summary_filename, '-- Collated CSV file: ' + result_filename)
    print_and_append(summary_filename, '-- Histogram image file ' + image_filename)
    print_and_append(summary_filename,'')

    print_and_append(summary_filename, 'Done!')
    return result_dictionary

#################################################################################
# Create one temporal series of Gaussian profiles, 
# add noise to each profile in the series,
# perform Gaussian fits and extract the sigma^2 parameter and its stderr,
# and perform a linear fit for the series of fitted sigma^2 parameters.
def scan_runner(indices, parameters, ld, this_diff, this_tau, this_noise, this_run, retain_profile_data):
    """
    Execute a simulation scan and generate a dictionary of results including Gaussian sigma^2 fits
    and linear MSD(t) fits for a given set of parameters.

    A "scan" or "run" refers to a single run of the model, which produces a 
    temporally evolved series of Gaussian distributions with decay, diffusion, and noise.

    Parameters:
    - indices (dict): A dictionary containing 'x axis' and 'time axis' as keys with corresponding values.
    - parameters (dict): A dictionary with the initial simulation parameters such as 't0_sigma^2', 't0_amplitude', etc.
    - ld (float): Nominal diffusion length for the current run.
    - this_diff (float): Nominal diffusion coefficient for the current run.
    - this_tau (float): Nominal lifetime for the current run.
    - this_noise (float): Stdev of normally-distributed noise to be added to the simulated profiles.
    - this_run (int): Identifier for the current simulation run.
    - retain_profile_data (bool): Flag indicating whether to retain profile data in the results.

    Returns:
    - result_dictionary (dict): A comprehensive dictionary containing the results of the simulation
                                run including profile fits, CNR estimates, and diffusion fits.

    Raises:
    - ValueError: If input parameters are out of the expected ranges or in incorrect formats.
    - RuntimeError: If the computation fails due to external library errors or internal logic errors.

    Notes:
    - The function is part of a larger simulation suite and expects specific input formats.
    - If 'retain_profile_data' is False, profile data will be removed from the result to save memory.
    """
    # run a single scan and generate a comprehensive dictionary of results
    # result dictionary is produced for each scan and placed as a subdictionary for the run

    # alias for brevity
    x_axis = indices['x axis']
    time_axis = indices['time axis']

    # initialize the run subdictionary and store the parameters for this run
    result_dictionary = {'run_' + str(this_run): {
                            'run': this_run,
                            'run parameters': {
                                'nominal diffusion coefficient': this_diff,
                                'nominal lifetime': this_tau,
                                'nominal diffusion length': ld, 
                                'noise stdev': this_noise
                            }
                            }
                        }
    
    # make a dictionary of parameters to pass for diffusion and decay generation
    diff_decay_parameters = {
        'x axis': x_axis,
        'time axis': time_axis,
        'sigma^2_0': parameters['sigma^2_0'],
        'amplitude_0': parameters['amplitude_0'],
        'mu_0': parameters['mu_0'],
        'nominal diffusion coefficient': this_diff,
        'nominal lifetime': this_tau,
    }
    # generate pure Gaussian PSF profiles
    nominal_profiles = make_diffusion_decay(diff_decay_parameters)

    # add noise to the profiles
    noisy_profiles = add_noise(this_noise, nominal_profiles['y_values_t'])
    # estimate the CNR at t0
    CNR_0_est = fft_cnr(noisy_profiles['y_values_t'][0])

    # fit the noisy profiles
    noisy_profile_fits = gauss_fitting(x_axis, noisy_profiles['y_values_t'])

    # get the fittted sigmas and stderrs of sigmas of the Gaussians
    gaussfit_sigma2_t = noisy_profile_fits['sigma^2_t estimates']
    gaussfit_sigma2_t_stderrs = list(zip(
        noisy_profile_fits['sigma^2_t estimates'], 
        noisy_profile_fits['sigma^2_t standard errors']
    ))

    if len(time_axis) > 1:
        # calculate the weights for diffusion fitting
        # -- if sigma or stdev are zero, that is bad, so give zero weight
        # -- otherwise, weight is calculated as:
        # -- normalized reciprocal of the relative variance of the fitted MSD
        weights = [np.power(stderr / sigma2, -2.) if (sigma2 !=0 and stderr != 0) else 0 for sigma2,stderr in gaussfit_sigma2_t_stderrs]
        # normalize so the sum of the weights is unity
        weights = [weight/np.sum(weights) for weight in weights]

        # get the unweighted (OLS) and weighted (WLS) fits of the MSD (change in sigma^2)
        ols_fit = diffusion_ols_fit(time_axis, gaussfit_sigma2_t)
        ols_slope_est = ols_fit['MSD_t slope estimate']
        ols_intercept_est = ols_fit['intercept estimate']
        ols_slope_stderr = ols_fit['MSD_t slope std error']
        ols_intercept_stderr = ols_fit['intercept standard error']

        wls_fit = diffusion_wls_fit(time_axis, gaussfit_sigma2_t, weights)
        wls_slope_est = wls_fit['MSD_t slope estimate']
        wls_intercept_est = wls_fit['intercept estimate']
        wls_slope_stderr = wls_fit['MSD_t slope std error']
        wls_intercept_stderr = wls_fit['intercept standard error']

    else:
        weights = 'N/A'
        weights = 'N/A'
        ols_fit = 'N/A'
        ols_slope_est = 'N/A'
        ols_intercept_est = 'N/A'
        ols_slope_stderr = 'N/A'
        ols_intercept_stderr = 'N/A'

        wls_fit = 'N/A'
        wls_slope_est = 'N/A'
        wls_intercept_est = 'N/A'
        wls_slope_stderr = 'N/A'
        wls_intercept_stderr = 'N/A'

    # delete the profile data if not retaining
    if retain_profile_data != 1:
        del noisy_profiles['y_values_t']
        del nominal_profiles['y_values_t']

    result_dictionary['run_' + str(this_run)].update({
        'nominal profiles': nominal_profiles, 
        'noisy profiles': noisy_profiles,
        'noisy profile fits': noisy_profile_fits,
        'cnr_0 estimate': CNR_0_est,
        'diffusion': {
            'unweighted fit': {
                'MSD_t slope estimate': ols_slope_est,
                'MSD_t slope std error': ols_slope_stderr,
                'intercept estimate': ols_intercept_est,
                'intercept standard error': ols_intercept_stderr,
                },
            'weighted fit': {
                'MSD_t slope estimate': wls_slope_est,
                'MSD_t slope std error': wls_slope_stderr,
                'intercept estimate': wls_intercept_est,
                'intercept standard error': wls_intercept_stderr,
                'weights': weights,
                },
            }
        })
    return result_dictionary

############################################################
# functions that create Gaussians with decay and diffusion #
############################################################

# A "scan" refers to a single run of the model, which produces
# a set of Gaussian distributions with decay, diffusion, and noise.
#
# Within a given scan, the noise amount added is the same for all 
# Gaussians, but because the signal is decaying and diffusing,
# the CNR is also diminishing over the course of the scan.
#
# Any units can be used, but note that they should be the same units 
# used throughout. So for example if you use ns for the lifetime, then
# anywhere there is an input of arbitrary time units, you must use ns.
# At the outset, you will specify your units so that they will be 
# handled and converted appropriately throughout.

#################################################################################
# Make a temporal series of Gaussians with diffusion and decay
def make_diffusion_decay(parameters):
    """
    Generate diffusion and decay profiles for Gaussian point spread functions (PSF) over time.

    Parameters:
    - parameters (dict): Dictionary containing the following keys:
        'x axis': array_like, positions at which to evaluate the Gaussian.
        'time axis': array_like, time points for the decay and diffusion simulation.
        'sigma^2_0': float, initial variance of the Gaussian at t=0.
        'amplitude_0': float, initial amplitude of the Gaussian at t=0.
        'mu_0': float, initial mean position of the Gaussian at t=0.
        'nominal diffusion coefficient': float, nominal diffusion coefficient.
        'nominal lifetime': float, nominal lifetime.

    Returns:
    - result_dictionary (dict): Dictionary with the calculated parameters and profiles at each time point.

    Raises:
    - ValueError: If the input parameters are not in the expected ranges or missing required keys.
    """

    # aliases of relevant parameters
    x_axis = parameters['x axis']
    time_axis = parameters['time axis']

    t0_sigma2 = parameters['sigma^2_0']
    t0_amplitude = parameters['amplitude_0']
    t0_mu = parameters['mu_0']

    this_diff = parameters['nominal diffusion coefficient']
    this_tau = parameters['nominal lifetime']

    # initialize result dictionary for this scan
    result_dictionary = {
        'parameters_t': {
            'amplitude_t': [], 'sigma^2_t': [], 'fwhm_t': [], 'mu_t':[], 'integrated intensity_t': []}
    }
    
    # calculate initial integrated intensity
    t0_ii = integrated_intensity(t0_sigma2, t0_amplitude)
    
    # calculate integrated intensities with decay
    ii_t = kinetic_decay_intensities(t0_ii, this_tau, time_axis)
    result_dictionary['parameters_t']['integrated intensity_t'] = ii_t

    # calculate sigmas with diffusion
    sig2_t = diffusion_sigma2_t(this_diff, t0_sigma2, time_axis)
    result_dictionary['parameters_t']['sigma^2_t'] = sig2_t

    # calcuilate and store fwhms with diffusion
    result_dictionary['parameters_t']['fwhm_t'] = [sigma2_to_fwhm(this_sig2) for this_sig2 in sig2_t]

    # calculate amplitudes from intensities and sigmas
    intensity_sig2_t = list(zip(ii_t,sig2_t)) # array of decay intensities and diffusion sigmas for iterating
    amp_t = [intensity / np.sqrt(2. * np.pi * sig2) for (intensity,sig2) in intensity_sig2_t]
    result_dictionary['parameters_t']['amplitude_t'] = amp_t

    # calculate y-values of Gaussians for each time point
    time_amp_sig2_t = list(zip(time_axis, amp_t, sig2_t))               # array of times, amplitudes, and sigmas for iterating
    y_values = []                                                       # initialize y-values array
    for this_time, this_amp, this_sig2 in time_amp_sig2_t:              # iterate over each time point
        this_gaussian = gaussian(x_axis, t0_mu, this_sig2, this_amp)    # make the gaussian for this time
        y_values.append(this_gaussian)                                  # store the gaussian profile

    result_dictionary.update({'y_values_t': y_values})

    return result_dictionary

def kinetic_decay_intensities(initial_integrated_intensity, tau, t_values):
    """
    Calculate the kinetic decay of integrated intensities over time.
    Integrated because it is the sum of intensity under the Gaussian curve.

    Parameters:
    - initial_integrated_intensity: float, the initial intensity value before decay.
    - tau: float, the decay time constant. A value of 0 implies no decay.
    - t_values: array_like, the time points at which to calculate the decayed intensities.

    Returns:
    - y_values: list, the intensities at each time point after applying the decay function.

    Raises:
    - ValueError: If `initial_integrated_intensity` is negative, or `tau` is negative.
    """
    if initial_integrated_intensity < 0:
        raise ValueError("Initial integrated intensity must be non-negative.")
    if tau < 0:
        raise ValueError("Decay time constant (tau) must be non-negative.")

    t_values = np.asarray(t_values)
    if tau == 0:
        y_values = np.full_like(t_values, initial_integrated_intensity)
    else:
        y_values = initial_integrated_intensity * np.exp(-t_values / tau)

    return y_values

def diffusion_sigma2_t(diffusion_coeff, sigma2_0, t_values):
    """
    Calculate the variance of a Gaussian PSF over time considering diffusion.

    Parameters:
    - diffusion_coeff: float, the nominal diffusion coefficient.
    - sigma2_0: float, the initial variance of the Gaussian PSF.
    - t_values: array_like, the time points at which to calculate the time-evolved variance.

    Returns:
    - sig2_t: list, the variance at each time point accounting for diffusion.

    Raises:
    - ValueError: If `diffusion_coeff` or `sigma2_0` is negative.
    """
    if diffusion_coeff < 0:
        raise ValueError("Diffusion coefficient must be non-negative.")
    if sigma2_0 < 0:
        raise ValueError("Initial PSF variance (sigma^2) must be non-negative.")

    t_values = np.asarray(t_values)
    if diffusion_coeff == 0:
        sig2_t = np.full_like(t_values, sigma2_0)
    else:
        sig2_t = sigma2_0 + 2 * diffusion_coeff * t_values

    return sig2_t
 
####################################################
# functions that add white noise to pure Gaussians #
####################################################

def add_noise(noise_sigma, nominal_profiles):
    """
    Add normally distributed noise to an array of nominal Gaussian profiles.

    Parameters:
    - noise_sigma: float, standard deviation of the noise.
    - nominal_profiles: array_like, the nominal Gaussian profiles without noise.

    Returns:
    - noisy_signal: dict, containing the noisy profiles in 'y_values_t'.
    """
    # Convert nominal_profiles to a NumPy array if not already
    nominal_profiles = np.array(nominal_profiles)
    
    # Generate noise for all profiles at once
    noise = np.random.normal(0, noise_sigma, nominal_profiles.shape)
    
    # Add noise to the nominal profiles
    noisy_profiles = nominal_profiles + noise

    noisy_signal = {'y_values_t': noisy_profiles}
    return noisy_signal

##################################################
# functions that analyze scans with fitting etc. #
##################################################

#################################################################################
# Gaussian fitting of noisy PSFs
def gauss_fitting(x_axis, noisy_profiles):
    """
    Fits Gaussian function to noisy data at each time point.

    This function takes a set of noisy Gaussian profiles and fits a Gaussian
    function to each profile using non-linear least squares optimization. The
    parameter estimates and their standard errors of the variance of the
    Gaussian are stored and returned in a dictionary.

    Parameters:
    - x_axis: (array-like) The x-values over which the profiles are defined.
    - noisy_profiles: (list of array-like) The y-values of the noisy profiles
      for each time point to be fitted.

    Returns:
    - fit_dictionary: (dict) A dictionary containing the fitted parameter
      estimates and the standard deviations for each time point.

    Raises:
    - ValueError: If any parameter bounds are invalid or inverted.
    """

    # initialize results dictionary
    fit_dictionary = {
        'sigma^2_t estimates': [], 
        'sigma^2_t standard errors': [], 
        } 
    
    # array of times and noisy gaussians
    profiles = list(noisy_profiles)

    for this_profile in profiles:
        # this fitting algorithm is ignorant of the input parameters
        
        ##############################################
        # Set guesses and bounds.                   
        # Handle errors so the script keeps running.
        
        # get x min, max and width
        xpix = len(x_axis)
        xmin,xmax = np.min(x_axis), np.max(x_axis)
        xwid = np.abs(xmax - xmin)

        # get index of max abs amp
        max_amp_idx = np.argmax(this_profile)

        # guesses
        mu0 = 0                         # mu guess: 0
        sigma2_0 = np.power(xwid / 4,2) # sigma^2 guess: 1/16 of squared scan width
        a0 = this_profile[max_amp_idx]  # amplitude guess: max value

        # set bounds of mu0 to the central fifth of the window
        fifthwidth = xwid / 5
        mu0_min = xmin + 2 * fifthwidth
        mu0_max = xmin + 3 * fifthwidth
        if mu0_min > mu0_max:
            print('error: mu0 bounds are inverted')
            break

        # set bounds of sigma2_0
        sigma2_min = np.power(xwid / xpix,2)                # sigma^2 minimum is 1 pixel
        sigma2_max = np.power(xwid,2)                       # sigma^2 maximum is the entire window width
        if sigma2_min > sigma2_max:
            print('error: sigma^2 bounds are inverted')
            break

        # set bounds of amp0
        a0_max = 2 * a0                             # maximum is 2x the max y-value
        a0_min = 0                                  # minimum is 0
        if a0_min > a0_max:
            print('error: a0 bounds are inverted')
            break

        p0 = [mu0, sigma2_0, a0]                    # guesses
        bounds_min = [mu0_min, sigma2_min, a0_min]  # lower bounds
        bounds_max = [mu0_max, sigma2_max, a0_max]  # upper bounds
        #############################################
        # Do the fit.
        [parms, covars] = curve_fit(
            gaussian, x_axis, this_profile, 
            p0 = p0,
            bounds=(bounds_min, bounds_max),
            maxfev=5000)

        ###################################################################
        # get the parameter estimates, covariances, variances, and stdevs
        fit_dictionary['sigma^2_t estimates'].append(parms[1])

        coefficient_variance_table = np.diag(covars) # diagonalize the covariance table to get parameter variances
        coefficient_stdev_table = np.sqrt(coefficient_variance_table) # stdev is square root of variance
        fit_dictionary['sigma^2_t standard errors'].append(coefficient_stdev_table[1])

    return fit_dictionary

def fft_cnr(noisy_profile):
    """
    Estimates the Contrast-to-Noise Ratio (CNR) for a noisy profile using FFT.

    The CNR is estimated by transforming the normalized profile using FFT,
    identifying peaks and minima, and calculating the noise level from the
    RMS of the modulus of the FFT beyond the first minimum following the first peak.

    Parameters:
    - noisy_profile: (array-like) The noisy profile to analyze.

    Returns:
    - cnr_estimate: (float) The estimated CNR for the profile.
    """

    # normalize against peak maximum
    profile_max = np.max(noisy_profile)
    this_profile_norm = noisy_profile / profile_max

    # FFT transform
    transform = np.fft.rfft(this_profile_norm, norm='ortho')  # Orthogonally normalized single-sided FFT
    fft_modulus = np.abs(transform)                           # Modulus of complex FFT values
    fft_modulus /= 2                                          # correct for the double counting of noise power

    # Prepend zero to ensure the first peak is found if it is at the edge
    fft_modulus = np.insert(fft_modulus, 0, 0)

    # Find peaks and minima in the FFT modulus
    peaks, _ = find_peaks(fft_modulus)
    neg_fft_modulus = -fft_modulus
    minima, _ = find_peaks(neg_fft_modulus)

    # undo preparatory changes    
    peaks = [peak - 1 for peak in peaks]        # subtract 1 from peaks indices
    minima = [peak - 1 for peak in minima]      # subtract 1 from minima indices
    fft_modulus = np.delete(fft_modulus, 0)     # remove leading 0 from modulus

    # get the index of the first minimum to the right of the first peak
    first_peak_idx = peaks[0]
    noise_start_idx = np.min([a for a in minima if a - first_peak_idx >= 0])

    # make noise array from first minimum to the end
    noise_regime = fft_modulus[noise_start_idx:]

    # get the noise estimate as root mean squared displacement
    noise_est = np.sqrt(np.mean(np.power(noise_regime, 2.)))

    # calculate and store cnr estimate
    cnr_estimate = np.round(1 / noise_est, 2)
        
    return cnr_estimate

def diffusion_ols_fit(time_axis, gaussfit_sigma2_t):
    """
    Estimates the Mean Squared Displacement (MSD) over time for a scan using 
    Ordinary Least Squares (OLS). This is useful in diffusion studies, where 
    the MSD is expected to linearly increase with time for a diffusive process.

    Parameters:
    - time_axis: (array-like) The time points for each measurement. These should 
                 be evenly spaced for accurate OLS fitting.
    - gaussfit_sigma2_t: (array-like) Squared width (variance) from Gaussian fits 
                         at each time point. This represents the displacement data 
                         for OLS fitting.

    Returns:
    - result: (dict) A dictionary containing key results from the OLS fit:
        'MSD_t slope estimate': The slope of the MSD versus time plot, which is 
                                an estimate proportional to the diffusion coefficient
                                in a linear diffusion process.
        'intercept estimate': The intercept of the MSD versus time plot, typically 
                              close to zero for a well-centered diffusion process.
        'MSD_t slope std error': The standard error of the slope estimate, 
                                 proportional to the uncertainty in the diffusion 
                                 coefficient estimate.
        'intercept standard error': The standard error of the intercept estimate, 
                                    providing a measure of the fit's precision 
                                    at the origin (time = 0).
    """
    if len(gaussfit_sigma2_t) < 2:
        raise ValueError("gaussfit_sigma2_t must contain multiple variances to fit.")
    if len(time_axis) != len(gaussfit_sigma2_t):
        raise ValueError("time_axis and gaussfit_sigma2_t must be of the same length.")
    
    # get delta of variances of Gaussian fits
    delta_vars = gaussfit_sigma2_t - gaussfit_sigma2_t[0]

    # Prepare the design matrix for OLS by adding a constant term for intercept
    design_matrix = sm.add_constant(time_axis)

    # Fit the model
    ols_model = sm.OLS(delta_vars, design_matrix).fit()

    return {
        'MSD_t slope estimate': ols_model.params[1],
        'intercept estimate': ols_model.params[0],
        'MSD_t slope std error': ols_model.bse[1],
        'intercept standard error': ols_model.bse[0],
    }

def diffusion_wls_fit(time_axis, gaussfit_sigma2_t, weights):
    """
    Estimates the diffusion coefficient for a scan using Weighted Least Squares (WLS).

    Parameters:
    - time_axis: (array-like) The time points for each measurement.
    - gaussfit_sigma2_t: (array-like) Variances of Gaussian fits at each time point.
    - weights: (array-like) Weights to apply to each measurement.

    Returns:
    - result: (dict) A dictionary containing the slope ('slope') and the standard error
                     of the slope ('std error') as the estimation of the diffusion coefficient.
    """
    if len(time_axis) != len(weights):
        raise ValueError("The length of weights must match the number of measurements.")
    
    # get delta of variances of Gaussian fits
    delta_vars = gaussfit_sigma2_t - gaussfit_sigma2_t[0]

    # Prepare the design matrix for WLS by adding a constant term for intercept
    design_matrix = sm.add_constant(time_axis)

    # Fit the model using WLS
    wls_model = sm.WLS(delta_vars, design_matrix, weights=weights).fit()

    result = {
        'MSD_t slope estimate': wls_model.params[1],
        'intercept estimate': wls_model.params[0],
        'MSD_t slope std error': wls_model.bse[1],
        'intercept standard error': wls_model.bse[0],
    }

    return result

def estimates_precision(df, proximity_level):
    """
    Calculate the percentage of simulations where the estimated diffusion fits
    are within a specified proximity to the nominal diffusion value.

    Parameters:
    - df: (DataFrame) DataFrame containing diffusion data.
    - proximity_level: (float) The acceptable proximity level around the nominal value.

    Returns:
    - result: (dict) Dictionary with the percentage of fits within the specified proximity.
    """
    if len(df) == 0:
        raise ValueError("The DataFrame is empty.")

    p_low = 1 - proximity_level
    p_high = 1 + proximity_level

    # Vectorized operations to calculate ratios
    df['d_wls_over_d_nom'] = df['weighted fit diffusion coeff [cm^2/s]'] / df['nominal diffusion coeff [cm^2/s]']
    df['d_ols_over_d_nom'] = df['unweighted fit diffusion coeff [cm^2/s]'] / df['nominal diffusion coeff [cm^2/s]']

    # Conditions to check if values are within the proximity level
    wls_within = df['d_wls_over_d_nom'].between(p_low, p_high)
    ols_within = df['d_ols_over_d_nom'].between(p_low, p_high)

    # Calculate percentages
    wls_portion_pct = 100 * wls_within.sum() / len(df)
    ols_portion_pct = 100 * ols_within.sum() / len(df)

    result = {
        '% fits within proximity': {
            'weighted fit': wls_portion_pct,
            'unweighted fit': ols_portion_pct,
        }
    }

    return result
    
#################################################################################
# Load result files, collate, and analyze precision
def load_files(file_path, file_match, cnr_low, cnr_high, precision_levels, num_bins):
    """
    Loads, collates, and analyzes precision from diffusion study result files.

    Parameters:
    - file_path: Path to the directory containing files.
    - file_match: String to match the beginning of filenames.
    - cnr_low: Low end of CNR range for these files.
    - cnr_high: High end of CNR range for these files.
    - precision_levels: Array of proximities to nominal values as a fraction.
    - num_bins: Number of CNR bins for sorting.

    Returns:
    - Dictionary with keys 'all' for full DataFrame and 'precision counts' for precision analysis.
    """

    # get the directory listing
    try:
        dir_list = os.listdir(file_path)
    except FileNotFoundError:
        return {'error': 'Directory not found'}

    # find matching files and report how many were found
    these_files = [re.findall('(?:^'+file_match+'.+)', s) for s in dir_list]
    these_files = [x[0] for x in these_files if x != []]
    print(f'Found {len(these_files)} matching files.')

    # initialize result dataframe
    df = pd.DataFrame()

    # load the file(s) and label pixels and time frames
    for filename in these_files:
        # load each file as a data frame
        try:
            df_this_file = pd.read_csv(os.path.join(file_path, filename))
        except Exception as e:
            print(f'Error reading {filename}: {e}')
            continue
        # get the number of pixels and time frames from the filename
        this_pix = re.findall(r"px-(\d+)", filename)
        this_tix = re.findall(r"tx-(\d+)", filename)
        if this_pix and this_tix:
            df_this_file['number of pixels'] = int(this_pix[0])
            df_this_file['number of time frames'] = int(this_tix[0])
        else:
            print(f'Filename format incorrect: {filename}')
            continue
        # append the result dataframe with this file data
        df = pd.concat([df, df_this_file], ignore_index=True)

    # older format has columns:
    # ['run num', 'diff nom', 'tau nom', 'ld nom', 'cnr', 'sigma2_0 nom',
    #    'sigma2_0 est', 'ols fit', 'wls fit', 'diff nom cm2/s',
    #    'wls diff cm2/s', 'ols diff cm2/s'],

    # fix older format
    if 'diff nom cm2/s' in df.columns:
       df = df.rename(
           columns={
               'run num': 'run number',
               'diff nom': 'nominal diffusion coeff',
               'tau nom': 'nominal lifetime',
               'ld nom': 'nominal diffusion length',
               'cnr': 'nominal CNR',
               'sigma2_0 nom': 'nominal sigma^2_0',
               'sigma2_0 est': 'estimated sigma^2_0',
               'ols fit': 'unweighted fit diffusion slope',
               'wls fit': 'weighted fit diffusion slope',
               'diff nom cm2/s': 'nominal diffusion coeff [cm^2/s]', 
               'wls diff cm2/s': 'weighted fit diffusion coeff [cm^2/s]',
               'ols diff cm2/s': 'unweighted fit diffusion coeff [cm^2/s]',
               }
            )

    # Calculations for relative proximity
    df['d_wls_over_d_nom'] = df['weighted fit diffusion coeff [cm^2/s]'] / df['nominal diffusion coeff [cm^2/s]']
    df['d_ols_over_d_nom'] = df['unweighted fit diffusion coeff [cm^2/s]'] / df['nominal diffusion coeff [cm^2/s]']

    # Verify CNR range
    if df['nominal CNR'].min() < cnr_low or df['nominal CNR'].max() > cnr_high:
        return {'error': 'nominal CNR values exist outside the expected range'}

    # CNR bin calculations
    cnr_bins = np.power(10, np.linspace(np.log10(cnr_low), np.log10(cnr_high), num_bins))
    df['CNR bins'] = pd.cut(df['nominal CNR'], bins=cnr_bins, include_lowest=True)
    df['CNR bin mid'] = [ival.mid for ival in df['CNR bins']]

    # Unique values for precision counts
    cnr_bin_mid_unique = df['CNR bin mid'].unique()
    ld_unique = df['nominal diffusion length'].unique()

    # Abbreviated results for speed
    results_brief = df[['CNR bin mid', 'nominal diffusion length', 'd_wls_over_d_nom', 'd_ols_over_d_nom']]

    # Get counts of results within precision proximity using precision_counts function
    precision_counts_data = precision_counts(
        results_brief, 
        cnr_bin_mid_unique, 
        ld_unique,
        precision_levels)

    # Return full results and precision counts
    result = {
        'all': df, 
        'precision counts': precision_counts_data
    }

    return result

# count the number of scans that fall within the designated precision levels
def precision_counts(cnr_ld_prox, cnr_bin_mid_unique, ld_unique, precision_levels):
    """
    Analyzes the precision of weighted and unweighted fits in diffusion studies by 
    calculating the proximity of estimated diffusion coefficients to their nominal values.

    Parameters:
    - cnr_ld_prox: DataFrame containing data for analysis. It should have four columns:
                  'CNR bin mid', 'nominal diffusion length', 'd_wls_over_d_nom', and 
                  'd_ols_over_d_nom', where the last two columns represent the proximity 
                  values (D_est/D_nom) for weighted (WLS) and unweighted (OLS) fits.
    - cnr_bin_mid_unique: List of unique middle values of CNR bins. These values are used 
                          to categorize the data into different CNR ranges.
    - ld_unique: List of unique diffusion lengths. These are used to categorize the data 
                 based on the diffusion length.
    - precision_levels: List of precision levels to analyze. These are fractions representing 
                        how close the estimated diffusion coefficient needs to be to the nominal 
                        value to be considered precise. For example, 0.1 (or 10%) means the 
                        estimated value should be within 10% of the nominal value.

    Returns:
    - A dictionary with keys formatted as 'precision proximity < X%', where X is the precision 
      level percentage. Each key maps to a DataFrame containing the analysis results for that 
      precision level. The DataFrames include counts and statistics (mean, standard deviation, 
      and difference in standard deviations between WLS and OLS fits) of the fits that fall within 
      the specified precision proximity for each combination of CNR bin middle value and diffusion length.
    """

    # initialize counts dictionary and subdictionaries to store results
    counts = {}

    # step through the precision levels to collate results
    # by precision level, cnr, ld, pix, and tix values
    for p in precision_levels:
        p_str = str(np.round(100*p))          # string to label the precision level
        print('Counting for precision within ' + p_str + '% of nominal value')
        these_counts = []   # initialize array of counts

        for cnr in cnr_bin_mid_unique:
            for ld in ld_unique:
                # store the records matching precision level, cnr, ld, pix, and tix value
                these_matches = cnr_ld_prox[
                    (cnr_ld_prox['CNR bin mid'] == cnr) & 
                    (cnr_ld_prox['nominal diffusion length'] == ld)
                    ]

                this_total = len(these_matches) # total number of matches
                ols_proxcount = len(these_matches[np.abs(these_matches['d_ols_over_d_nom'] - 1) < p])
                wls_proxcount = len(these_matches[np.abs(these_matches['d_wls_over_d_nom'] - 1) < p])

                # Calculations for mean and standard deviation of proximity ratios
                wls_prox_mean = np.mean(these_matches['d_wls_over_d_nom'])
                wls_prox_std  = np.std(these_matches['d_wls_over_d_nom'])
                ols_prox_mean = np.mean(these_matches['d_ols_over_d_nom'])
                ols_prox_std  = np.std(these_matches['d_ols_over_d_nom'])

                # Calculate the difference between the stdevs of wls and ols D/D0 values.
                # Higher values indicate that weighted fits are more precise than unweighted.
                # Negative values would indicate that unweighted fit is more precise than weighted.
                # - Note that better precision doesn't necessarily mean better accuracy:
                # - If there is bias, a more precise distribution may nevertheless be
                # - far from the nominal value, and may evben have fewer fits within proximity
                # - of the nominal value than a wider distribution would. Evaluation of the 
                # - spread of fits along with the mean D_est/D_nom is thus warranted.
                difference_ols_wls_std = ols_prox_std - wls_prox_std

                if this_total == 0:
                    # if none matched within precision proximity, report error
                    print('No results found for CNR = ' + str(cnr) + ', LD = ' + str(ld))
                    # flag the record for no matches
                    these_counts.append(
                        {'nominal CNR': cnr, 
                            'nominal diffusion length': ld, 
                            'weighted fits percent in proximity': -1, 
                            'unweighted fits percent in proximity': -1, 
                            'total in bin': -1,
                            'weighted fits mean D_est/D_nom': -1,
                            'unweighted fits mean D_est/D_nom': -1,
                            'weighted fits stdev D_est/D_nom':  -1,
                            'unweighted fits stdev D_est/D_nom':  -1,
                            'difference in weighted and unweighted stdev': -1,
                            })
                else:
                    these_counts.append(
                        {'nominal CNR': cnr, 
                            'nominal diffusion length': ld, 
                            'weighted fits percent in proximity': 100 * wls_proxcount/this_total, 
                            'unweighted fits percent in proximity': 100 * ols_proxcount/this_total, 
                            'total in bin': this_total,
                            'weighted fits mean D_est/D_nom': wls_prox_mean,
                            'unweighted fits mean D_est/D_nom': ols_prox_mean,
                            'weighted fits stdev D_est/D_nom':  wls_prox_std,
                            'unweighted fits stdev D_est/D_nom':  ols_prox_std,
                            'difference in weighted and unweighted stdev': difference_ols_wls_std,
                            })

        counts[f'precision proximity < {p_str}%'] = pd.DataFrame(these_counts)

    return counts

##########################
# plotting functions #
##########################

def colordefs():
    """
    Defines a set of colors for plotting purposes. The color choices are made with 
    considerations for color blindness accessibility and clarity in black & white printing.

    Returns:
    - A dictionary of color names mapped to their hexadecimal color codes.
    """
    return {
        'dice_blue': '#003f7f',    # Montana State blue, good contrast and colorblind safe
        'dice_gold': '#f7941e',    # Montana State gold, vibrant and distinguishable in grayscale
        'dice_green': '#0cce6b',   # Bright green, good visibility and colorblind safe
        'dice_gradient': sns.cubehelix_palette(start=1, rot=0.9, gamma=1.0, hue=1, light=0.75, dark=0.20, reverse=True, as_cmap=True)
    }

def plot_accuracy_histogram(
        simulation_result: dict, # Results from the simulation
        proximity: float,        # Accuracy & precision: nominal proximity threshold 
        filename: str,           # Base name for the output file
        image_type: str,         # Image format (e.g., 'png', 'jpg')
        width: float,            # Width of the image in cm
        height: float,           # Height of the image in cm
        dpi:float,               # Resolution of the image in dots per inch
        font_size: float,        # Font size in points, for labels
        tick_length: float,      # tick length in points
        tick_width: float,       # tick width in points
        num_bins: int,           # Number of bins in the histogram
        x_lim=None               # Optional x-axis limits
    ):
    
    """
    Exports a histogram of diffusion accuracy values (D_est/D_nom) from simulation results. 
    """

    # convert width and height from cm to inches
    inch = 1/2.54
    width, height = width * inch, height * inch

    # colors
    color_definitions = colordefs()
    dice_blue = color_definitions['dice_blue']
    dice_gold = color_definitions['dice_gold']
    
    # Verify required data is present
    if 'collated results' not in simulation_result or 'd_wls_over_d_nom' not in simulation_result['collated results']:
        raise ValueError("Required data not found in simulation_result.")

    # Convert list of values to NumPy array
    dest_over_d0 = np.array(simulation_result['collated results']['d_wls_over_d_nom'])
    
    # Initialize array to flag accuracy ratio values in proximity
    dest_d0_proximity_flag = np.abs(dest_over_d0 - 1) <= proximity

    # Calculate and report percentage of values that are within proximity threshold
    proxpct = 100 * np.sum(dest_d0_proximity_flag) / len(dest_d0_proximity_flag)
    print(f'Percent of D estimates within {proximity * 100:.2f}% of nominal: {proxpct:.1f}')

    fig, ax = plt.subplots(layout='constrained', figsize = (width,height))

    n_dd0, bins_dd0, patches_dd0 = ax.hist(
        dest_over_d0,
        bins=num_bins, density = True,
        color=dice_gold, edgecolor='w',
        )

    binspace_dd0 = np.linspace(bins_dd0[0], bins_dd0[-1], 100)
    mu_dd0 = np.mean(dest_over_d0)
    sigma_dd0 = np.std(dest_over_d0)
    y_dd0 = norm.pdf(binspace_dd0, mu_dd0, sigma_dd0)

    ax.set_xlabel('$D_{est}/D_{nom}$', fontsize = font_size)
    ax.set_ylabel('Probability density', fontsize = font_size)

    # default limits of x: 99.97% confidence interval
    if x_lim is None:
        x_lim = [mu_dd0 - 3 * sigma_dd0, mu_dd0 + 3 * sigma_dd0]
    ax.set_xlim(x_lim)
    
    ax.tick_params(axis='both', which='both', 
                   labelsize=font_size,
                   direction='in', 
                   length=tick_length, 
                   width=tick_width,
                   # left=False, labelleft=False,
                )

    ax.plot(binspace_dd0, y_dd0, 
            color = dice_blue, linewidth=2,
            label = 'mean ' + str(np.round(mu_dd0,3)) + 
                    '\nmedian ' + str(np.round(np.median(dest_over_d0),3)) + 
                    '\nstdev ' + str(np.round(sigma_dd0,3))
            )

    ax.legend(fontsize=font_size, handlelength=0, 
            labelspacing = 2, frameon=False)

    # make the image background white and opaque
    fig.patch.set_facecolor('w')
    fig.patch.set_alpha(1)

    try:
        # Export the image
        export_file = f"{filename}"
        plt.savefig(export_file, dpi=dpi, format=image_type)
    except Exception as e:
        print(f"Error saving file: {e}")

    plt.close(fig)  # Close the figure to free up memory
