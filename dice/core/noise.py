"""
Noise generation and CNR (Contrast-to-Noise Ratio) estimation functions.

This module provides functions for adding noise to profiles,
generating noise distributions, and estimating CNR from noisy data.
"""

import numpy as np
from numpy.random import default_rng
from fft_cnr import fft_cnr
from typing import Union, List, Dict, Optional
from ..utils.validators import validate_numeric, validate_array_like, validate_integer


def add_noise(nominal_profiles: np.ndarray, noise_sigma: float, 
              seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Add normally distributed noise to nominal Gaussian profiles.
    
    Parameters
    ----------
    nominal_profiles : np.ndarray
        The nominal Gaussian profiles without noise.
        Can be 1D (single profile) or 2D (multiple profiles).
    noise_sigma : float
        Standard deviation of the noise.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    dict
        Dictionary containing the noisy profiles in 'y_values_t'.
    
    Raises
    ------
    ValueError
        If noise_sigma < 0.
    """
    nominal_profiles = validate_array_like(nominal_profiles, "nominal_profiles")
    noise_sigma = validate_numeric(noise_sigma, "noise_sigma", min_val=0)
    
    # Set random seed if provided
    rng = default_rng(seed)
    
    # Generate noise for all profiles at once
    noise = rng.normal(0, noise_sigma, nominal_profiles.shape)
    
    # Add noise to the nominal profiles
    noisy_profiles = nominal_profiles + noise
    
    return {'y_values_t': noisy_profiles}


def make_noise_distribution(noise_low: float, noise_high: float, 
                          num: int, logarithmic: bool = False,
                          seed: Optional[int] = None) -> List[float]:
    """
    Create a distribution of noise sigmas, uniform in reciprocal space.
    
    This generates noise values that are uniformly distributed when
    plotted as CNR (1/noise), optionally with logarithmic scaling.
    
    Parameters
    ----------
    noise_low : float
        The lower bound for noise sigma values.
    noise_high : float
        The upper bound for noise sigma values.
    num : int
        The number of samples to generate.
    logarithmic : bool, optional
        Flag to generate the distribution in reciprocal log space.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    list of float
        List of noise sigma values.
    
    Raises
    ------
    ValueError
        If bounds are not positive or if lower bound >= upper bound.
    """
    noise_low = validate_numeric(noise_low, "noise_low", min_val=0)
    noise_high = validate_numeric(noise_high, "noise_high", min_val=0)
    num = validate_integer(num, "num", min_val=1)
    
    if noise_low == 0 or noise_high == 0:
        raise ValueError("Noise bounds must be positive (not zero)")
    if noise_low >= noise_high:
        raise ValueError("The lower bound must be less than the upper bound")
    
    # Get the reciprocal of the noise range (reverses order)
    cnr_high = 1 / noise_low  # High noise -> Low CNR
    cnr_low = 1 / noise_high  # Low noise -> High CNR
    
    # Switch to logarithmic space if indicated
    if logarithmic:
        cnr_low = np.log10(cnr_low)
        cnr_high = np.log10(cnr_high)
    
    # Create random number generator
    rng = default_rng(seed)
    
    # Create uniform distribution in CNR space
    cnr_values = rng.uniform(low=cnr_low, high=cnr_high, size=num)
    
    # Transform back to noise space
    if logarithmic:
        noise_sigmas = 1 / np.power(10, cnr_values)
    else:
        noise_sigmas = 1 / cnr_values
    
    return noise_sigmas.tolist()


def estimate_noise_from_profile(profile: np.ndarray,
                               method: str = 'fft') -> float:
    """
    Estimate noise standard deviation from a profile.
    
    Parameters
    ----------
    profile : np.ndarray
        The profile to analyze.
    method : str, optional
        Method to use ('fft' or 'high_freq'). Default is 'fft'.
    
    Returns
    -------
    float
        Estimated noise standard deviation.
    
    Raises
    ------
    ValueError
        If method is not recognized.
    """
    profile = validate_array_like(profile, "profile", ndim=1)
    
    if method == 'fft':
        result = fft_cnr(profile)
        return result.noise_rms
    
    elif method == 'high_freq':
        # Alternative: estimate from high-frequency components
        # Apply high-pass filter (difference filter)
        if len(profile) < 2:
            raise ValueError("Profile too short for high-frequency analysis")
        
        # Calculate differences (high-pass filter)
        diff = np.diff(profile)
        
        # Estimate noise as std of differences, scaled by sqrt(2)
        # (since diff amplifies noise by sqrt(2))
        noise_est = np.std(diff) / np.sqrt(2)
        
        return noise_est
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'fft' or 'high_freq'")


