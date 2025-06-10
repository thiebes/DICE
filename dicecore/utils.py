import ast
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.random import default_rng
from scipy.signal import find_peaks # Added for fft_cnr

def fft_cnr(noisy_profile: np.ndarray) -> float:
    """
    Estimates the Contrast-to-Noise Ratio (CNR) for a noisy profile using FFT.

    The CNR is estimated by transforming the normalized profile using FFT,
    identifying peaks and minima, and calculating the noise level from the
    RMS of the modulus of the FFT beyond the first minimum following the first peak.

    Parameters:
    - noisy_profile: (array-like) The noisy profile to analyze.

    Returns:
    - cnr_estimate: (float) The estimated CNR for the profile. Returns np.nan if estimation fails.
    """
    if noisy_profile is None or len(noisy_profile) == 0:
        return np.nan

    profile_max = np.max(noisy_profile)
    if np.isclose(profile_max, 0): # Avoid division by zero if max is close to 0
        return np.nan
        
    this_profile_norm = noisy_profile / profile_max

    transform = np.fft.rfft(this_profile_norm, norm='ortho')
    fft_modulus = np.abs(transform)
    # fft_modulus /= 2 # Correction for double counting noise power - this might be overcorrection or context dependent

    # Ensure there are enough points for peak finding
    if len(fft_modulus) < 3: # find_peaks requires at least 3 points
        return np.nan

    peaks, _ = find_peaks(fft_modulus)
    minima, _ = find_peaks(-fft_modulus) # Find peaks in negative to get minima

    if len(peaks) == 0: # No peak found
        # Fallback: estimate noise from overall std dev of FFT modulus if no clear signal peak
        # This is a rough estimate and might not be accurate if signal is strong
        if len(fft_modulus) > 1:
             noise_est = np.sqrt(np.mean(np.power(fft_modulus[1:], 2.))) # Exclude DC component
             if np.isclose(noise_est, 0): return np.inf # Avoid division by zero
             return 1.0 / noise_est if noise_est else np.nan
        return np.nan


    first_peak_idx = peaks[0]
    
    # Find first minimum after the first peak
    valid_minima = minima[minima > first_peak_idx]
    if len(valid_minima) == 0:
        # If no minimum after first peak, use end of spectrum for noise estimation
        # or handle as error / specific case, e.g. if signal fills spectrum
        noise_start_idx = first_peak_idx + 1 # Start noise estimation just after the peak
        if noise_start_idx >= len(fft_modulus): # Check if index is out of bounds
             # This case implies the peak is at the very end or the spectrum is too short.
             # Fallback to a simpler noise estimation or return NaN.
             if len(fft_modulus[first_peak_idx+1:]) > 0 : # if there is anything after peak
                 noise_est = np.sqrt(np.mean(np.power(fft_modulus[first_peak_idx+1:], 2.)))
             else: # if not, maybe use whole spectrum excluding DC as noise (very rough)
                  noise_est = np.sqrt(np.mean(np.power(fft_modulus[1:], 2.))) if len(fft_modulus)>1 else np.nan
             if noise_est is np.nan or np.isclose(noise_est, 0) : return np.nan if noise_est is np.nan else np.inf
             return 1.0 / noise_est

    else:
        noise_start_idx = valid_minima[0]

    if noise_start_idx >= len(fft_modulus): # Check if noise_start_idx is valid
        # This can happen if the minimum found is at the very end or spectrum is short.
        # Fallback: Consider noise from overall std dev or a segment if possible.
        # This part might need refinement based on expected signal characteristics.
        noise_est = np.sqrt(np.mean(np.power(fft_modulus[first_peak_idx+1:], 2.))) if len(fft_modulus[first_peak_idx+1:]) > 0 else np.nan
        if noise_est is np.nan or np.isclose(noise_est, 0): return np.nan if noise_est is np.nan else np.inf
        return 1.0 / noise_est

    noise_regime = fft_modulus[noise_start_idx:]
    if len(noise_regime) == 0: # No data points in noise regime
        # Fallback: use points after first peak if available
        noise_regime = fft_modulus[first_peak_idx+1:]
        if len(noise_regime) == 0: # Still no points, cannot estimate noise
            return np.nan


    noise_est = np.sqrt(np.mean(np.power(noise_regime, 2.)))
    if np.isclose(noise_est, 0):
        return np.inf # Effectively infinite CNR if noise is zero

    cnr_estimate = 1.0 / noise_est
        
    return cnr_estimate

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
                # print(t0_profile_strings) # Commented out print
                t0_profile_y = t0_profile_strings.astype(float).tolist()
            except Exception as e:
                raise ValueError("Error reading CSV for noise estimation.") from e
            cnr_est = fft_cnr(t0_profile_y) # Relies on fft_cnr from .fitting
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
        else:  # If the key is 'sigma_0'
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
    
    if 'nominal diffusion coefficient' in provided_keys and 'nominal lifetime (tau)' in provided_keys:
        if 'nominal diffusion length' in provided_keys:
            raise ValueError("Please provide either 'nominal diffusion length' or both 'nominal diffusion coefficient' and 'nominal lifetime (tau)', not all three.")
        diff = parameters_dictionary['nominal diffusion coefficient']
        tau = parameters_dictionary['nominal lifetime (tau)']
        if not isinstance(diff, (int, float)) or not isinstance(tau, (int, float)):
            raise ValueError("Both 'nominal diffusion coefficient' and 'nominal lifetime (tau)' must be numeric.")
        parameters_dictionary['nominal diffusion length'] = np.sqrt(diff * tau)
        
    elif 'nominal diffusion length' in provided_keys:
        if len(provided_keys) > 1:
            raise ValueError("You provided 'nominal diffusion length' along with diffusion coefficient and/or lifetime. Please provide either 'diffusion length' or both 'diffusion coefficient' and 'lifetime (tau)', not a combination.")
        ld = parameters_dictionary['nominal diffusion length']
        if not isinstance(ld, (int, float)):
            raise ValueError("'nominal diffusion length' must be numeric.")
        parameters_dictionary['nominal diffusion coefficient'] = np.power(ld,2)
        parameters_dictionary['nominal lifetime (tau)'] = 1
      
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
    required_keys = ['number of runs', 'spatial width', 'pixel width', 'mean_0', 'amplitude_0']
    for key in required_keys:
        if key not in parameters_dictionary:
            raise KeyError(f"The required parameter '{key}' is missing from the parameters dictionary.")

    num_noise = 1 # future use: parameters_dictionary.get('noise range, number of values', 1)
    parameters_dictionary['x array'] = make_x_axis(
        parameters_dictionary['spatial width'], 
        parameters_dictionary['pixel width'], 
        parameters_dictionary['mean_0']
    )

    width_0 = handle_profile_width_parameters(parameters_dictionary)
    amp_0, mean_0 = parameters_dictionary['amplitude_0'], parameters_dictionary['mean_0']
    parameters_dictionary['t0 Gaussian sigma^2, amplitude, mean'] = [width_0, amp_0, mean_0]

    handle_time_parameters(parameters_dictionary)
    handle_noise_parameters(parameters_dictionary, num_noise) # Relies on fft_cnr from .fitting
    handle_diffusion_parameters(parameters_dictionary)

    return parameters_dictionary

def print_and_append(summary_filename, text):
    """
    Prints the given text to the console and appends it to the specified file.
    
    Parameters:
    summary_filename (str): The path to the file where text should be appended.
    text (str): The text to be printed and appended to the file.
    """
    print(text)
    
    try:
        with open(summary_filename, 'a') as file1:
            file1.write(text + '\n')
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")
