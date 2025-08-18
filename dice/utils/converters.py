"""
Unit conversion functions for Gaussian profile parameters.

This module provides conversions between different representations of
Gaussian width parameters (FWHM, sigma, sigma^2).
"""

import numpy as np
from typing import Union


def sigma_to_fwhm(sigma: float) -> float:
    """
    Convert standard deviation (sigma) of a Gaussian to the full width at half maximum (FWHM).
    
    The relationship is: FWHM = sigma * 2 * sqrt(2 * ln(2))
    
    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian.
    
    Returns
    -------
    float
        FWHM of the Gaussian.
    
    Raises
    ------
    ValueError
        If sigma is not positive.
    """
    if sigma <= 0:
        raise ValueError("Sigma must be a positive number")
    return sigma * 2 * np.sqrt(2 * np.log(2))


def sigma2_to_fwhm(sigma2: float) -> float:
    """
    Convert variance (sigma^2) of a Gaussian to the full width at half maximum (FWHM).
    
    Parameters
    ----------
    sigma2 : float
        Variance of the Gaussian.
    
    Returns
    -------
    float
        FWHM of the Gaussian.
    
    Raises
    ------
    ValueError
        If sigma2 is not positive.
    """
    if sigma2 <= 0:
        raise ValueError("Sigma squared must be a positive number")
    return sigma_to_fwhm(np.sqrt(sigma2))


def fwhm_to_sigma(fwhm: float) -> float:
    """
    Convert full width at half maximum (FWHM) of a Gaussian to standard deviation (sigma).
    
    The relationship is: sigma = FWHM / (2 * sqrt(2 * ln(2)))
    
    Parameters
    ----------
    fwhm : float
        Full width at half maximum of the Gaussian.
    
    Returns
    -------
    float
        Standard deviation (sigma) of the Gaussian.
    
    Raises
    ------
    ValueError
        If FWHM is not positive.
    """
    if fwhm <= 0:
        raise ValueError("FWHM must be a positive number")
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def fwhm_to_sigma2(fwhm: float) -> float:
    """
    Convert full width at half maximum (FWHM) of a Gaussian to variance (sigma^2).
    
    Parameters
    ----------
    fwhm : float
        Full width at half maximum of the Gaussian.
    
    Returns
    -------
    float
        Variance (sigma^2) of the Gaussian.
    
    Raises
    ------
    ValueError
        If FWHM is not positive.
    """
    return fwhm_to_sigma(fwhm) ** 2


def slope_to_diffusion_constant(slope: float, l_unit: str, t_unit: str) -> float:
    """
    Convert the slope from a linear fit of mean squared displacement vs. time 
    to a diffusion coefficient in conventional units [cm^2/s].
    
    The MSD in 1D follows: MSD(t) = 2*D*t, so D = slope/2
    
    Parameters
    ----------
    slope : float
        The slope from the MSD vs. time linear fit in user-provided units of length^2/time.
    l_unit : str
        The user-provided unit of length used in the slope (e.g., 'micrometer').
    t_unit : str
        The user-provided unit of time used in the slope (e.g., 'nanosecond').
    
    Returns
    -------
    float
        The diffusion coefficient in units of cm^2/s.
    
    Raises
    ------
    ValueError
        If the provided length or time units are not supported.
    """
    # Conversion factors to centimeters and seconds
    conversion_factors = {
        'length': {
            'meter': 100,
            'centimeter': 1,
            'millimeter': 0.1,
            'micrometer': 1e-4,
            'nanometer': 1e-7,
            'angstrom': 1e-8,
            'picometer': 1e-10,
        },
        'time': {
            'second': 1,
            'millisecond': 1e-3,
            'microsecond': 1e-6,
            'nanosecond': 1e-9,
            'picosecond': 1e-12,
            'femtosecond': 1e-15,
            'attosecond': 1e-18,
        }
    }
    
    # Error handling for invalid units
    if l_unit not in conversion_factors['length']:
        raise ValueError(f"Invalid length unit '{l_unit}'. Please use one of the following: "
                         f"{', '.join(conversion_factors['length'].keys())}.")
    if t_unit not in conversion_factors['time']:
        raise ValueError(f"Invalid time unit '{t_unit}'. Please use one of the following: "
                         f"{', '.join(conversion_factors['time'].keys())}.")
    
    # Convert the slope to cm^2/s
    slope_cm2_per_s = slope * conversion_factors['length'][l_unit] ** 2 / conversion_factors['time'][t_unit]
    
    # Divide by 2 to get the diffusion coefficient in one dimension
    return slope_cm2_per_s / 2