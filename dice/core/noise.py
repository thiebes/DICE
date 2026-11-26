"""
Noise generation and CNR (Contrast-to-Noise Ratio) estimation functions.

This module provides functions for adding noise to profiles,
generating noise distributions, and estimating CNR from noisy data.
"""

import warnings

import numpy as np
from numpy.random import SeedSequence, default_rng
from fft_cnr import fft_cnr
from typing import Union, List, Dict, Optional, Sequence
from ..utils.validators import validate_numeric, validate_array_like, validate_integer
from ..utils.converters import fwhm_to_sigma2, sigma2_to_fwhm
from .profiles import gaussian

# A reproducible seed: an integer, or a SeedSequence spawned from one. None draws fresh entropy.
Seed = Optional[Union[int, SeedSequence]]

# Narrowest peak (FWHM, in samples) whose CNR fft-cnr estimates well. In
# simulated Gaussian profiles with white noise, its estimate was within 2.4%
# of the true CNR for peaks 9 to 14 samples wide, and 14% to 27% too low for
# a peak 4.6 samples wide.
MIN_CNR_PEAK_FWHM = 9.0

# fft-cnr needs this many points for its estimate of the noise spectrum, and
# each run estimates the CNR of its first profile.
MINIMUM_PROFILE_POINTS = 16


def add_noise(nominal_profiles: np.ndarray, noise_sigma: float, 
              seed: Seed = None) -> Dict[str, np.ndarray]:
    """
    Add normally distributed noise to nominal Gaussian profiles.
    
    Parameters
    ----------
    nominal_profiles : np.ndarray
        The nominal Gaussian profiles without noise.
        Can be 1D (single profile) or 2D (multiple profiles).
    noise_sigma : float
        Standard deviation of the noise.
    seed : int or SeedSequence, optional
        Seed for reproducibility; None draws fresh entropy.
    
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


def nominal_cnr(amplitude: float, noise_sigma: float) -> float:
    """Initial CNR of a profile of peak amplitude `amplitude` under white noise of SD `noise_sigma`.

    CNR is amplitude / noise_sigma (docs/glossary.md); it is infinite for a noise-free profile.
    """
    return amplitude / noise_sigma if noise_sigma > 0 else float('inf')


def make_noise_distribution(noise_low: float, noise_high: float, 
                          num: int, logarithmic: bool = False,
                          seed: Seed = None) -> List[float]:
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
    seed : int or SeedSequence, optional
        Seed for reproducibility; None draws fresh entropy.
    
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


def _half_maximum_width(deviation: np.ndarray, peak: int) -> float:
    """Width, in samples, between the half-maximum crossings on either side of `peak`.

    Each crossing is interpolated linearly between the samples around it; a
    side that stays above half maximum to the end of the profile ends there.
    """
    half = deviation[peak] / 2
    left = peak
    while left > 0 and deviation[left - 1] > half:
        left -= 1
    left_crossing = (
        left - (deviation[left] - half) / (deviation[left] - deviation[left - 1]) if left > 0 else 0.0
    )
    right = peak
    while right < deviation.size - 1 and deviation[right + 1] > half:
        right += 1
    right_crossing = (
        right + (deviation[right] - half) / (deviation[right] - deviation[right + 1])
        if right < deviation.size - 1 else deviation.size - 1.0
    )
    return float(right_crossing - left_crossing)


def peak_fwhm_samples(profile: Union[np.ndarray, Sequence[float]]) -> float:
    """
    Full width at half maximum, in samples, of the largest peak of a measured profile.

    The width comes from a least-squares fit of a Gaussian on a constant
    baseline, which uses every sample and so is not thrown off by the noise
    at the peak or at the half-maximum points. The fit starts from the
    half-maximum crossings of the largest excursion from the edge baseline
    (the median of the outer quarter of the samples at each end), and that
    crossing width is returned if the fit fails or the profile has no more
    samples than the fit has parameters. A downward peak is measured like an
    upward one.

    Parameters
    ----------
    profile : array-like
        The measured profile, one value per sample (pixel).

    Returns
    -------
    float
        FWHM in samples; nan for a profile with no excursion from its baseline.
    """
    from scipy.optimize import OptimizeWarning, curve_fit

    y = validate_array_like(profile, "profile", ndim=1).astype(float)
    n = y.size
    margin = max(1, n // 4)
    deviation = y - np.median(np.concatenate([y[:margin], y[-margin:]]))
    peak = int(np.argmax(np.abs(deviation)))
    if deviation[peak] == 0:
        return np.nan
    deviation *= np.sign(deviation[peak])
    crossing_width = _half_maximum_width(deviation, peak)

    def gaussian_on_baseline(x, mu, sig2, amp, offset):
        return gaussian(x, mu, sig2, amp) + offset

    initial = [peak, fwhm_to_sigma2(max(crossing_width, 1.0)), deviation[peak], 0.0]
    if n <= len(initial):
        return crossing_width
    bounds = ([0.0, 1e-6, 0.0, -np.inf], [n - 1.0, float(n) ** 2, np.inf, np.inf])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', OptimizeWarning)
            fitted, _ = curve_fit(gaussian_on_baseline, np.arange(n, dtype=float), deviation,
                                  p0=initial, bounds=bounds, maxfev=5000)
    except (RuntimeError, ValueError, TypeError):
        return crossing_width
    return float(sigma2_to_fwhm(fitted[1]))
