import numpy as np
from .utils import sigma2_to_fwhm # For make_diffusion_decay

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

def make_diffusion_decay(parameters: dict):
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
    result_dictionary['parameters_t']['fwhm_t'] = [sigma2_to_fwhm(this_sig2) for this_sig2 in sig2_t] # Uses sigma2_to_fwhm from .utils

    # calculate amplitudes from intensities and sigmas
    intensity_sig2_t = list(zip(ii_t,sig2_t)) # array of decay intensities and diffusion sigmas for iterating
    amp_t = [intensity / np.sqrt(2. * np.pi * sig2) if sig2 > 0 else 0 for (intensity,sig2) in intensity_sig2_t] # handle sig2=0 case
    result_dictionary['parameters_t']['amplitude_t'] = amp_t

    # calculate y-values of Gaussians for each time point
    # time_amp_sig2_t = list(zip(time_axis, amp_t, sig2_t)) # Not directly used, values used in loop
    y_values = []                                                       # initialize y-values array
    for idx, this_time in enumerate(time_axis):              # iterate over each time point
        this_amp = amp_t[idx]
        this_sig2 = sig2_t[idx]
        this_gaussian = gaussian(x_axis, t0_mu, this_sig2, this_amp)    # make the gaussian for this time
        y_values.append(this_gaussian)                                  # store the gaussian profile

    result_dictionary.update({'y_values_t': y_values})

    return result_dictionary

def kinetic_decay_intensities(initial_integrated_intensity: float, tau: float, t_values: np.ndarray) -> np.ndarray:
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
    if tau == 0: # or np.isclose(tau, 0): # handles potential float inaccuracies
        y_values = np.full_like(t_values, initial_integrated_intensity, dtype=float)
    else:
        y_values = initial_integrated_intensity * np.exp(-t_values / tau)

    return y_values

def diffusion_sigma2_t(diffusion_coeff: float, sigma2_0: float, t_values: np.ndarray) -> np.ndarray:
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
    if sigma2_0 < 0: # Allow sigma2_0 to be zero
        raise ValueError("Initial PSF variance (sigma^2) must be non-negative.")

    t_values = np.asarray(t_values)
    #if diffusion_coeff == 0: # or np.isclose(diffusion_coeff, 0):
    #    sig2_t = np.full_like(t_values, sigma2_0, dtype=float)
    #else:
    sig2_t = sigma2_0 + 2 * diffusion_coeff * t_values

    return sig2_t
 
def add_noise(noise_sigma: float, nominal_profiles: np.ndarray) -> dict:
    """
    Add normally distributed noise to an array of nominal Gaussian profiles.

    Parameters:
    - noise_sigma: float, standard deviation of the noise.
    - nominal_profiles: array_like, the nominal Gaussian profiles without noise.

    Returns:
    - noisy_signal: dict, containing the noisy profiles in 'y_values_t'.
    """
    # Convert nominal_profiles to a NumPy array if not already
    nominal_profiles_arr = np.array(nominal_profiles)
    
    # Generate noise for all profiles at once
    noise = np.random.normal(0, noise_sigma, nominal_profiles_arr.shape)
    
    # Add noise to the nominal profiles
    noisy_profiles_arr = nominal_profiles_arr + noise

    noisy_signal = {'y_values_t': noisy_profiles_arr}
    return noisy_signal
