"""
Axis generation functions for spatial and temporal dimensions.

This module provides functions to create coordinate arrays for
spatial and temporal axes used in diffusion simulations.
"""

import numpy as np
from typing import Union
from .validators import validate_numeric, validate_integer


def make_x_axis(scan_width: float, scan_width_pixels: int, mu: float = 0.0) -> np.ndarray:
    """
    Create an array representing a spatial x-axis.
    
    The axis is centered at mu with total width scan_width, discretized
    into scan_width_pixels points.
    
    Parameters
    ----------
    scan_width : float
        Total width of x-axis in spatial units.
    scan_width_pixels : int
        Total number of pixels (points) on the x-axis.
    mu : float, optional
        Center position of the x-axis. Default is 0.
    
    Returns
    -------
    np.ndarray
        Array of x-axis values.
    
    Raises
    ------
    ValueError
        If scan_width_pixels <= 0 or scan_width <= 0.
    
    Examples
    --------
    >>> x = make_x_axis(10.0, 101, mu=0)
    >>> len(x)
    101
    >>> x[50]  # Center point
    0.0
    """
    scan_width_pixels = validate_integer(scan_width_pixels, "scan_width_pixels", min_val=1)
    scan_width = validate_numeric(scan_width, "scan_width", min_val=0, allow_inf=False)
    
    if scan_width == 0:
        raise ValueError("scan_width must be greater than 0")
    
    mu = validate_numeric(mu, "mu", allow_inf=False)
    
    x_start = mu - scan_width / 2
    x_end = mu + scan_width / 2
    x_values = np.linspace(x_start, x_end, scan_width_pixels)
    
    return x_values


def make_time_axis(t_start: float, t_end: float, t_frames: int) -> np.ndarray:
    """
    Create an array representing a time axis with evenly spaced points.
    
    Parameters
    ----------
    t_start : float
        Start timestamp.
    t_end : float
        End timestamp.
    t_frames : int
        Number of time frames.
    
    Returns
    -------
    np.ndarray
        Array of time values.
    
    Raises
    ------
    ValueError
        If t_frames <= 0 or t_start >= t_end.
    
    Examples
    --------
    >>> t = make_time_axis(0, 10, 11)
    >>> len(t)
    11
    >>> t[0], t[-1]
    (0.0, 10.0)
    """
    t_frames = validate_integer(t_frames, "t_frames", min_val=1)
    t_start = validate_numeric(t_start, "t_start", allow_inf=False)
    t_end = validate_numeric(t_end, "t_end", allow_inf=False)
    
    # Special case: single frame at a specific time
    if t_frames == 1 and t_start == t_end:
        return np.array([t_start])
    
    if t_start >= t_end:
        raise ValueError(f"t_start ({t_start}) must be less than t_end ({t_end})")
    
    return np.linspace(t_start, t_end, t_frames)


def make_time_series(time_points: Union[list, np.ndarray]) -> np.ndarray:
    """
    Create a time axis from explicit time points.
    
    This function validates and converts a list of time points into
    a sorted numpy array suitable for time-series analysis.
    
    Parameters
    ----------
    time_points : list or np.ndarray
        Explicit time values.
    
    Returns
    -------
    np.ndarray
        Sorted array of time values.
    
    Raises
    ------
    ValueError
        If time_points is empty or contains non-finite values.
    
    Examples
    --------
    >>> t = make_time_series([0.1, 0.3, 0.5, 0.7, 0.9])
    >>> len(t)
    5
    >>> t[2]
    0.5
    """
    from .validators import validate_array_like
    
    time_array = validate_array_like(time_points, "time_points", ndim=1)
    
    # Check for finite values
    if not np.all(np.isfinite(time_array)):
        raise ValueError("All time points must be finite values")
    
    # Sort the array (in case it's not already sorted)
    sorted_array = np.sort(time_array)
    
    # Check for duplicates
    if len(np.unique(sorted_array)) != len(sorted_array):
        raise ValueError("Time points must be unique")
    
    return sorted_array


def make_spatial_grid(x_width: float, y_width: float, 
                      x_pixels: int, y_pixels: int,
                      center: tuple = (0.0, 0.0)) -> tuple:
    """
    Create a 2D spatial grid (for future 2D extensions).
    
    Parameters
    ----------
    x_width : float
        Width of the grid in x-direction.
    y_width : float
        Width of the grid in y-direction.
    x_pixels : int
        Number of pixels in x-direction.
    y_pixels : int
        Number of pixels in y-direction.
    center : tuple, optional
        Center coordinates (x0, y0) of the grid.
    
    Returns
    -------
    tuple of np.ndarray
        (X, Y) meshgrid arrays.
    
    Notes
    -----
    This function is included for future 2D diffusion simulations
    but is not currently used in the 1D implementation.
    """
    x_axis = make_x_axis(x_width, x_pixels, mu=center[0])
    y_axis = make_x_axis(y_width, y_pixels, mu=center[1])
    
    X, Y = np.meshgrid(x_axis, y_axis, indexing='xy')
    
    return X, Y