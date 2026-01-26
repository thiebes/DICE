"""
Noise generation and CNR (Contrast-to-Noise Ratio) estimation functions.

This module provides functions for adding noise to profiles,
generating noise distributions, and estimating CNR from noisy data.
"""

import numpy as np
from numpy.random import default_rng
from scipy.signal import find_peaks
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


def fft_cnr(noisy_profile: np.ndarray) -> float:
    """
    Estimate the Contrast-to-Noise Ratio (CNR) using FFT analysis.
    
    The CNR is estimated by:
    1. Normalizing the profile to unit amplitude
    2. Computing the FFT
    3. Finding the first local minimum after the signal peak
    4. Calculating noise as RMS of high-frequency components
    5. CNR = 1 / noise_level for unit-amplitude signal
    
    Parameters
    ----------
    noisy_profile : np.ndarray
        The noisy profile to analyze (1D array).
    
    Returns
    -------
    float
        The estimated CNR for the profile.
    
    Raises
    ------
    ValueError
        If the profile is empty or constant.
    """
    noisy_profile = validate_array_like(noisy_profile, "noisy_profile", ndim=1)
    
    if len(noisy_profile) < 3:
        raise ValueError("Profile must have at least 3 points for CNR estimation")
    
    # Check for constant profile
    profile_max = np.max(noisy_profile)
    profile_min = np.min(noisy_profile)
    
    if profile_max == profile_min:
        raise ValueError("Cannot estimate CNR from constant profile")
    
    # Normalize against peak maximum
    this_profile_norm = noisy_profile / profile_max
    
    # FFT transform
    # Orthogonally normalized single-sided FFT
    transform = np.fft.rfft(this_profile_norm, norm='ortho')
    fft_modulus = np.abs(transform)
    
    # Prepend zero to ensure the first peak is found if it is at the edge
    fft_modulus = np.insert(fft_modulus, 0, 0)
    
    # Find peaks and minima in the FFT modulus
    peaks, _ = find_peaks(fft_modulus)
    neg_fft_modulus = -fft_modulus
    minima, _ = find_peaks(neg_fft_modulus)
    
    # Undo preparatory changes
    peaks = peaks - 1  # Adjust peak indices
    minima = minima - 1  # Adjust minima indices
    fft_modulus = np.delete(fft_modulus, 0)  # Remove leading 0
    
    # Handle edge cases
    if len(peaks) == 0:
        # No clear peak found, use half the spectrum as noise
        noise_start_idx = len(fft_modulus) // 2
    else:
        first_peak_idx = peaks[0]
        
        # Find minima after the first peak
        minima_after_peak = minima[minima > first_peak_idx]
        
        if len(minima_after_peak) == 0:
            # No minimum after peak, use point after peak
            noise_start_idx = min(first_peak_idx + 1, len(fft_modulus) - 1)
        else:
            noise_start_idx = minima_after_peak[0]
    
    # Make noise array from first minimum to the end
    noise_regime = fft_modulus[noise_start_idx:]
    
    if len(noise_regime) == 0:
        # Fallback: use last quarter of spectrum
        noise_regime = fft_modulus[3*len(fft_modulus)//4:]
    
    # Get the noise estimate as root mean squared
    noise_est = np.sqrt(np.mean(np.power(noise_regime, 2)))
    
    # Prevent division by zero
    if noise_est == 0:
        return float('inf')
    
    # Calculate and return CNR estimate
    cnr_estimate = np.round(1 / noise_est, 2)
    
    return cnr_estimate


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
        # Use FFT-based CNR estimation
        cnr = fft_cnr(profile)
        # For unit amplitude, noise_sigma = 1/CNR
        # For actual amplitude A, noise_sigma = A/CNR
        amplitude = np.max(np.abs(profile))
        return amplitude / cnr
    
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


