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


def calculate_pixel_size(spatial_width: float, pixel_count: int) -> float:
    """
    Calculate the physical size of each pixel.

    Parameters
    ----------
    spatial_width : float
        Total spatial width of the observation window.
    pixel_count : int
        Number of pixels across the spatial domain.

    Returns
    -------
    float
        Size of each pixel in the same units as spatial_width.

    Raises
    ------
    ValueError
        If spatial_width <= 0 or pixel_count <= 0.
    """
    if spatial_width <= 0:
        raise ValueError("Spatial width must be positive")
    if pixel_count <= 0:
        raise ValueError("Pixel count must be positive")
    return spatial_width / pixel_count


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
    from .units import convert_diffusion_coefficient

    # slope is in l_unit^2/t_unit; D = slope/2 in the same units.
    # Convert D from (l_unit^2/t_unit) to (cm^2/s).
    d_user_units = slope / 2
    return convert_diffusion_coefficient(
        d_user_units, l_unit, t_unit, 'centimeter', 'second'
    )