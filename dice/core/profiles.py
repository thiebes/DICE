"""
Gaussian profile generation and evolution functions.

This module provides functions for creating and evolving Gaussian profiles
over time, including diffusion and exponential decay effects.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from ..utils.validators import validate_numeric, validate_array_like
from ..utils.converters import sigma2_to_fwhm


def gaussian(x: np.ndarray, mu: float, sig2: float, amp: float) -> np.ndarray:
    """
    Generate a Gaussian function with zero baseline.
    
    Parameters
    ----------
    x : np.ndarray
        Array of x values.
    mu : float
        Mean value of the Gaussian.
    sig2 : float
        Variance (sigma^2) of the Gaussian.
    amp : float
        Amplitude of the Gaussian.
    
    Returns
    -------
    np.ndarray
        Gaussian function y-values.
    
    Raises
    ------
    ValueError
        If sig2 <= 0 or amp < 0.
    """
    sig2 = validate_numeric(sig2, "sig2", min_val=0, allow_inf=False)
    amp = validate_numeric(amp, "amp", min_val=0, allow_inf=False)
    x = validate_array_like(x, "x")
    
    if sig2 == 0:
        raise ValueError("Variance (sig2) must be positive, not zero")
    
    return amp * np.exp(-1 * np.power(x - mu, 2) / (2 * sig2))


def integrated_intensity(sig2: float, amp: float) -> float:
    """
    Calculate the integrated intensity (area under the curve) of a Gaussian.
    
    For a Gaussian, the integrated intensity is: amp * sqrt(2 * pi * sig2)
    
    Parameters
    ----------
    sig2 : float
        The variance of the Gaussian (sigma^2).
    amp : float
        The amplitude of the Gaussian peak.
    
    Returns
    -------
    float
        The integrated intensity of the Gaussian.
    
    Raises
    ------
    ValueError
        If sig2 <= 0 or amp <= 0.
    """
    sig2 = validate_numeric(sig2, "sig2", min_val=0, allow_inf=False)
    amp = validate_numeric(amp, "amp", min_val=0, allow_inf=False)
    
    if sig2 == 0:
        raise ValueError("Variance (sig2) must be positive, not zero")
    if amp == 0:
        raise ValueError("Amplitude must be positive, not zero")
    
    return amp * np.sqrt(2 * np.pi * sig2)


def kinetic_decay_intensities(initial_integrated_intensity: float, 
                             tau: float, t_values: np.ndarray) -> np.ndarray:
    """
    Calculate the kinetic decay of integrated intensities over time.
    
    Models single-exponential decay: I(t) = I0 * exp(-t/tau)
    
    Parameters
    ----------
    initial_integrated_intensity : float
        The initial intensity value before decay.
    tau : float
        The decay time constant. A value of 0 implies no decay.
    t_values : np.ndarray
        The time points at which to calculate the decayed intensities.
    
    Returns
    -------
    np.ndarray
        The intensities at each time point after applying the decay function.
    
    Raises
    ------
    ValueError
        If initial_integrated_intensity < 0 or tau < 0.
    """
    initial_integrated_intensity = validate_numeric(
        initial_integrated_intensity, 
        "initial_integrated_intensity", 
        min_val=0
    )
    tau = validate_numeric(tau, "tau", min_val=0)
    t_values = validate_array_like(t_values, "t_values")
    
    if tau == 0:
        # No decay
        return np.full_like(t_values, initial_integrated_intensity, dtype=float)
    else:
        return initial_integrated_intensity * np.exp(-t_values / tau)


def diffusion_sigma2_t(diffusion_coeff: float, sigma2_0: float, 
                      t_values: np.ndarray) -> np.ndarray:
    """
    Calculate the variance (sigma^2) evolution due to diffusion.
    
    For 1D Fickian diffusion: sigma^2(t) = sigma^2(0) + 2*D*t
    
    Parameters
    ----------
    diffusion_coeff : float
        The diffusion coefficient (D).
    sigma2_0 : float
        The initial variance at t=0.
    t_values : np.ndarray
        The time points at which to calculate the variance.
    
    Returns
    -------
    np.ndarray
        The variance values at each time point.
    
    Raises
    ------
    ValueError
        If diffusion_coeff < 0 or sigma2_0 <= 0.
    """
    diffusion_coeff = validate_numeric(diffusion_coeff, "diffusion_coeff", min_val=0)
    sigma2_0 = validate_numeric(sigma2_0, "sigma2_0", min_val=0)
    t_values = validate_array_like(t_values, "t_values")
    
    if sigma2_0 == 0:
        raise ValueError("Initial variance (sigma2_0) must be positive, not zero")
    
    # Check for negative time values
    if np.any(t_values < 0):
        raise ValueError("All time values must be non-negative")
    
    return sigma2_0 + 2 * diffusion_coeff * t_values


def make_diffusion_decay(parameters: Dict) -> Dict:
    """
    Generate diffusion and decay profiles for Gaussian PSF over time.
    
    This function creates a time series of Gaussian profiles that undergo
    both diffusion (spreading) and exponential decay (amplitude decrease).
    
    Parameters
    ----------
    parameters : dict
        Dictionary containing the following keys:
        - 'x axis': array_like, positions at which to evaluate the Gaussian
        - 'time axis': array_like, time points for the simulation
        - 'sigma^2_0': float, initial variance of the Gaussian at t=0
        - 'amplitude_0': float, initial amplitude of the Gaussian at t=0
        - 'mu_0': float, initial mean position of the Gaussian at t=0
        - 'nominal diffusion coefficient': float, diffusion coefficient
        - 'nominal lifetime': float, decay lifetime (tau)
    
    Returns
    -------
    dict
        Dictionary with the calculated parameters and profiles at each time point:
        - 'parameters_t': Time-dependent parameters (amplitude, sigma^2, FWHM, etc.)
        - 'y_values_t': List of Gaussian profiles at each time point
    
    Raises
    ------
    ValueError
        If the input parameters are not in the expected ranges or missing required keys.
    KeyError
        If required keys are missing from the parameters dictionary.
    """
    # Extract and validate parameters
    try:
        x_axis = validate_array_like(parameters['x axis'], 'x axis')
        time_axis = validate_array_like(parameters['time axis'], 'time axis')
        
        t0_sigma2 = validate_numeric(parameters['sigma^2_0'], 'sigma^2_0', min_val=0)
        t0_amplitude = validate_numeric(parameters['amplitude_0'], 'amplitude_0', min_val=0)
        t0_mu = validate_numeric(parameters['mu_0'], 'mu_0')
        
        this_diff = validate_numeric(
            parameters['nominal diffusion coefficient'], 
            'diffusion coefficient', 
            min_val=0
        )
        this_tau = validate_numeric(
            parameters['nominal lifetime'], 
            'lifetime', 
            min_val=0
        )
    except KeyError as e:
        raise KeyError(f"Missing required parameter: {e}")
    
    if t0_sigma2 == 0:
        raise ValueError("Initial variance (sigma^2_0) must be positive, not zero")
    
    # Initialize result dictionary
    result_dictionary = {
        'parameters_t': {
            'amplitude_t': [],
            'sigma^2_t': [],
            'fwhm_t': [],
            'mu_t': [],
            'integrated intensity_t': []
        }
    }
    
    # Calculate initial integrated intensity
    t0_ii = integrated_intensity(t0_sigma2, t0_amplitude)
    
    # Calculate integrated intensities with decay
    ii_t = kinetic_decay_intensities(t0_ii, this_tau, time_axis)
    result_dictionary['parameters_t']['integrated intensity_t'] = ii_t.tolist()
    
    # Calculate sigmas with diffusion
    sig2_t = diffusion_sigma2_t(this_diff, t0_sigma2, time_axis)
    result_dictionary['parameters_t']['sigma^2_t'] = sig2_t.tolist()
    
    # Calculate and store FWHMs with diffusion
    result_dictionary['parameters_t']['fwhm_t'] = [
        sigma2_to_fwhm(this_sig2) for this_sig2 in sig2_t
    ]
    
    # Calculate amplitudes from intensities and sigmas
    # Amplitude = Integrated_Intensity / sqrt(2 * pi * sigma^2)
    amp_t = []
    for intensity, sig2 in zip(ii_t, sig2_t):
        if sig2 > 0:
            amp = intensity / np.sqrt(2.0 * np.pi * sig2)
        else:
            amp = 0.0
        amp_t.append(amp)
    result_dictionary['parameters_t']['amplitude_t'] = amp_t
    
    # Store the mean position (constant for pure diffusion)
    result_dictionary['parameters_t']['mu_t'] = [t0_mu] * len(time_axis)
    
    # Calculate y-values of Gaussians for each time point
    y_values = []
    for t_idx, (this_time, this_amp, this_sig2) in enumerate(
        zip(time_axis, amp_t, sig2_t)
    ):
        if this_sig2 > 0:
            this_gaussian = gaussian(x_axis, t0_mu, this_sig2, this_amp)
        else:
            this_gaussian = np.zeros_like(x_axis)
        y_values.append(this_gaussian)
    
    result_dictionary['y_values_t'] = y_values
    
    return result_dictionary