"""
Diffusion calculations and related functions.

This module provides functions for calculating diffusion coefficients,
diffusion lengths, and mean squared displacement (MSD).
"""

import numpy as np
from typing import Union, Optional
from ..utils.validators import validate_numeric, validate_array_like


def calculate_diffusion_coefficient(slope: float, dimensions: int = 1) -> float:
    """
    Calculate diffusion coefficient from MSD vs. time slope.
    
    The relationship between MSD slope and diffusion coefficient is:
    - 1D: MSD = 2*D*t, so D = slope/2
    - 2D: MSD = 4*D*t, so D = slope/4
    - 3D: MSD = 6*D*t, so D = slope/6
    
    Parameters
    ----------
    slope : float
        Slope from linear fit of MSD vs. time.
    dimensions : int, optional
        Number of dimensions (1, 2, or 3). Default is 1.
    
    Returns
    -------
    float
        Diffusion coefficient.
    
    Raises
    ------
    ValueError
        If slope < 0 or dimensions not in [1, 2, 3].
    """
    slope = validate_numeric(slope, "slope", min_val=0)
    
    if dimensions not in [1, 2, 3]:
        raise ValueError(f"Dimensions must be 1, 2, or 3, got {dimensions}")
    
    return slope / (2 * dimensions)


def calculate_diffusion_length(diffusion_coeff: float, lifetime: float) -> float:
    """
    Calculate the characteristic diffusion length.
    
    The diffusion length is defined as: L_D = sqrt(D * tau)
    
    Parameters
    ----------
    diffusion_coeff : float
        Diffusion coefficient (D).
    lifetime : float
        Excited state lifetime (tau).
    
    Returns
    -------
    float
        Diffusion length.
    
    Raises
    ------
    ValueError
        If diffusion_coeff < 0 or lifetime < 0.
    """
    diffusion_coeff = validate_numeric(diffusion_coeff, "diffusion_coeff", min_val=0)
    lifetime = validate_numeric(lifetime, "lifetime", min_val=0)
    
    return np.sqrt(diffusion_coeff * lifetime)


def calculate_msd(sigma2_t: np.ndarray, sigma2_0: Optional[float] = None) -> np.ndarray:
    """
    Calculate mean squared displacement from variance evolution.
    
    MSD(t) = sigma^2(t) - sigma^2(0)
    
    Parameters
    ----------
    sigma2_t : np.ndarray
        Variance values at different time points.
    sigma2_0 : float, optional
        Initial variance. If None, uses first element of sigma2_t.
    
    Returns
    -------
    np.ndarray
        Mean squared displacement at each time point.
    
    Raises
    ------
    ValueError
        If sigma2_t is empty or sigma2_0 < 0.
    """
    sigma2_t = validate_array_like(sigma2_t, "sigma2_t")
    
    if sigma2_0 is None:
        if len(sigma2_t) == 0:
            raise ValueError("sigma2_t cannot be empty when sigma2_0 is not provided")
        sigma2_0 = sigma2_t[0]
    else:
        sigma2_0 = validate_numeric(sigma2_0, "sigma2_0", min_val=0)
    
    msd = sigma2_t - sigma2_0
    
    # MSD should be non-negative (accounting for numerical errors)
    if np.any(msd < -1e-10):
        raise ValueError("Calculated MSD contains significantly negative values")
    
    # Clean up tiny negative values from numerical errors
    msd = np.maximum(msd, 0)
    
    return msd


def estimate_diffusion_from_msd(time: np.ndarray, msd: np.ndarray, 
                               dimensions: int = 1) -> tuple:
    """
    Estimate diffusion coefficient from MSD data using linear regression.
    
    Performs a simple linear fit of MSD vs. time through the origin.
    
    Parameters
    ----------
    time : np.ndarray
        Time points.
    msd : np.ndarray
        Mean squared displacement values.
    dimensions : int, optional
        Number of dimensions (1, 2, or 3). Default is 1.
    
    Returns
    -------
    tuple
        (diffusion_coefficient, slope, r_squared)
        - diffusion_coefficient: Estimated D value
        - slope: Slope of MSD vs. time
        - r_squared: Coefficient of determination
    
    Raises
    ------
    ValueError
        If arrays have different lengths or insufficient data points.
    """
    time = validate_array_like(time, "time")
    msd = validate_array_like(msd, "msd")
    
    if len(time) != len(msd):
        raise ValueError("time and msd arrays must have the same length")
    
    if len(time) < 2:
        raise ValueError("Need at least 2 points for linear regression")
    
    if dimensions not in [1, 2, 3]:
        raise ValueError(f"Dimensions must be 1, 2, or 3, got {dimensions}")
    
    # Simple linear regression through origin
    # MSD = slope * t, where slope = 2*D (in 1D)
    slope = np.sum(time * msd) / np.sum(time * time)
    
    # Calculate R-squared
    msd_pred = slope * time
    ss_res = np.sum((msd - msd_pred) ** 2)
    ss_tot = np.sum((msd - np.mean(msd)) ** 2)
    
    if ss_tot == 0:
        r_squared = 1.0 if ss_res == 0 else 0.0
    else:
        r_squared = 1 - (ss_res / ss_tot)
    
    diffusion_coefficient = calculate_diffusion_coefficient(slope, dimensions)
    
    return diffusion_coefficient, slope, r_squared


def calculate_peclet_number(velocity: float, length_scale: float, 
                           diffusion_coeff: float) -> float:
    """
    Calculate the Péclet number for transport phenomena.
    
    The Péclet number is the ratio of advective to diffusive transport:
    Pe = v * L / D
    
    Parameters
    ----------
    velocity : float
        Characteristic velocity.
    length_scale : float
        Characteristic length scale.
    diffusion_coeff : float
        Diffusion coefficient.
    
    Returns
    -------
    float
        Péclet number.
    
    Raises
    ------
    ValueError
        If any parameter is negative or diffusion_coeff is zero.
    """
    velocity = validate_numeric(velocity, "velocity", min_val=0)
    length_scale = validate_numeric(length_scale, "length_scale", min_val=0)
    diffusion_coeff = validate_numeric(diffusion_coeff, "diffusion_coeff", min_val=0)
    
    if diffusion_coeff == 0:
        raise ValueError("Diffusion coefficient cannot be zero")
    
    return (velocity * length_scale) / diffusion_coeff


def einstein_relation(diffusion_coeff: float, temperature: float, 
                      viscosity: Optional[float] = None,
                      radius: Optional[float] = None) -> float:
    """
    Apply the Einstein relation for diffusion.
    
    For particles in a fluid: D = k_B * T / (6 * pi * eta * r)
    Or mobility: mu = D / (k_B * T)
    
    Parameters
    ----------
    diffusion_coeff : float
        Diffusion coefficient.
    temperature : float
        Temperature in Kelvin.
    viscosity : float, optional
        Dynamic viscosity of the medium.
    radius : float, optional
        Particle radius (for Stokes-Einstein relation).
    
    Returns
    -------
    float
        Either mobility (if viscosity/radius not provided) or 
        expected diffusion coefficient (if they are).
    
    Notes
    -----
    k_B = 1.380649e-23 J/K (Boltzmann constant)
    """
    k_B = 1.380649e-23  # Boltzmann constant in J/K
    
    diffusion_coeff = validate_numeric(diffusion_coeff, "diffusion_coeff", min_val=0)
    temperature = validate_numeric(temperature, "temperature", min_val=0)
    
    if temperature == 0:
        raise ValueError("Temperature cannot be zero")
    
    if viscosity is not None and radius is not None:
        # Calculate expected D from Stokes-Einstein
        viscosity = validate_numeric(viscosity, "viscosity", min_val=0)
        radius = validate_numeric(radius, "radius", min_val=0)
        
        if viscosity == 0 or radius == 0:
            raise ValueError("Viscosity and radius cannot be zero")
        
        return k_B * temperature / (6 * np.pi * viscosity * radius)
    else:
        # Calculate mobility
        return diffusion_coeff / (k_B * temperature)