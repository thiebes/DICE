from typing import cast, Any

import numpy as np
import seaborn as sns
import statsmodels.api as sm
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from numpy.random import default_rng

def print_and_append(summary_filename, text, gui_message_callback=None):
    """
    Prints the given text to the console, appends it to the specified file,
    and optionally calls a GUI message callback.
    
    Parameters:
    summary_filename (str): The path to the file where text should be appended.
    text (str): The text to be printed and appended to the file.
    gui_message_callback (callable, optional): A function to call with the text message for GUI display.
    """
    # Print the text to the console
    print(text)

    # Call the GUI message callback if provided
    if gui_message_callback and callable(gui_message_callback):
        try:
            gui_message_callback(text)
        except Exception as e:
            print(f"Error in gui_message_callback: {e}") # Avoid callback errors stopping simulation
    
    # Try appending the text to the specified file
    try:
        with open(summary_filename, 'a') as file1:
            file1.write(text + '\n')
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")

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

'''
A "scan" refers to a single run of the model, which produces
a set of Gaussian distributions with decay, diffusion, and noise.

Within a given scan, the noise amount added is the same for all 
Gaussians, but because the signal is decaying and diffusing,
the CNR is also diminishing over the course of the scan.

Any units can be used, but note that they should be the same units 
used throughout. So for example if you use ns for the lifetime, then
anywhere there is an input of arbitrary time units, you must use ns.
At the outset, you will specify your units so that they will be 
handled and converted appropriately throughout.
'''

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
    result_dictionary: dict[str, Any] = {
        'parameters_t': {
            'amplitude_t': [], 'sigma^2_t': [], 'fwhm_t': [], 'mu_t':[], 'integrated intensity_t': []}
    }
    
    # calculate initial integrated intensity
    t0_ii = integrated_intensity(t0_sigma2, t0_amplitude)
    
    # calculate integrated intensities with decay
    ii_t = kinetic_decay_intensities(t0_ii, this_tau, time_axis)
    result_dictionary['parameters_t']['integrated intensity_t'] = cast(Any, ii_t)

    # calculate sigmas with diffusion
    sig2_t = diffusion_sigma2_t(this_diff, t0_sigma2, time_axis)
    result_dictionary['parameters_t']['sigma^2_t'] = cast(Any, sig2_t)

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

    result_dictionary['y_values_t'] = y_values

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