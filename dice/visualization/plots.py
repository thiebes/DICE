"""
Plotting functions for DICE visualization.

This module provides functions for creating plots and visualizations
of simulation results, profiles, and statistical analyses.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path


def get_color_definitions() -> Dict[str, Union[str, Any]]:
    """
    Define color scheme for DICE plots.
    
    Colors are chosen with considerations for color blindness accessibility
    and clarity in black & white printing.
    
    Returns
    -------
    dict
        Dictionary mapping color names to hex codes or colormap objects.
    
    Examples
    --------
    >>> colors = get_color_definitions()
    >>> colors['dice_blue']
    '#003f7f'
    """
    return {
        'dice_blue': '#003f7f',    # Montana State blue, good contrast and colorblind safe
        'dice_gold': '#f7941e',    # Montana State gold, vibrant and distinguishable in grayscale
        'dice_green': '#0cce6b',   # Bright green, good visibility and colorblind safe
        'dice_gradient': sns.cubehelix_palette(
            start=1, rot=0.9, gamma=1.0, hue=1, 
            light=0.75, dark=0.20, reverse=True, as_cmap=True
        )
    }


def plot_gaussian_profile(
    x_axis: np.ndarray,
    profile: np.ndarray,
    title: str = "Gaussian Profile",
    xlabel: str = "Position",
    ylabel: str = "Intensity",
    filename: Optional[str] = None,
    **plot_kwargs
) -> plt.Figure:
    """
    Plot a single Gaussian profile.
    
    Parameters
    ----------
    x_axis : np.ndarray
        Spatial coordinates.
    profile : np.ndarray
        Profile intensities.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    filename : str, optional
        If provided, save plot to this filename.
    **plot_kwargs
        Additional plotting parameters (width, height, dpi, etc.)
    
    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    
    Examples
    --------
    >>> x = np.linspace(-10, 10, 201)
    >>> y = np.exp(-x**2 / 2)
    >>> fig = plot_gaussian_profile(x, y, title="Test Profile")
    """
    # Extract plot parameters
    width = plot_kwargs.get('width', 10.0)  # cm
    height = plot_kwargs.get('height', 6.0)  # cm  
    dpi = plot_kwargs.get('dpi', 100)
    font_size = plot_kwargs.get('font_size', 12)
    image_type = plot_kwargs.get('image_type', 'png')
    
    # Convert cm to inches
    inch = 1/2.54
    width_in, height_in = width * inch, height * inch
    
    # Get colors
    colors = get_color_definitions()
    
    # Create figure
    fig, ax = plt.subplots(layout='constrained', figsize=(width_in, height_in))
    
    # Plot profile
    ax.plot(x_axis, profile, color=colors['dice_blue'], linewidth=2)
    
    # Formatting
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.set_title(title, fontsize=font_size)
    ax.tick_params(axis='both', labelsize=font_size, direction='in')
    
    # Make background white and opaque
    fig.patch.set_facecolor('w')
    fig.patch.set_alpha(1)
    
    # Save if filename provided
    if filename:
        try:
            plt.savefig(filename, dpi=dpi, format=image_type)
        except Exception as e:
            print(f"Error saving plot: {e}")
    
    return fig


def plot_profile_evolution(
    x_axis: np.ndarray,
    time_axis: np.ndarray,
    profiles: List[np.ndarray],
    title: str = "Profile Evolution",
    xlabel: str = "Position",
    ylabel: str = "Intensity",
    time_unit: str = "ns",
    filename: Optional[str] = None,
    **plot_kwargs
) -> plt.Figure:
    """
    Plot evolution of Gaussian profiles over time.
    
    Parameters
    ----------
    x_axis : np.ndarray
        Spatial coordinates.
    time_axis : np.ndarray
        Time points.
    profiles : list of np.ndarray
        List of profile intensities at each time point.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    time_unit : str
        Unit for time axis (for legend).
    filename : str, optional
        If provided, save plot to this filename.
    **plot_kwargs
        Additional plotting parameters.
    
    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    
    Examples
    --------
    >>> x = np.linspace(-10, 10, 201)
    >>> t = np.array([0, 1, 2, 3])
    >>> profiles = [np.exp(-x**2 / (2*(1+0.5*ti))) for ti in t]
    >>> fig = plot_profile_evolution(x, t, profiles)
    """
    # Extract plot parameters
    width = plot_kwargs.get('width', 12.0)  # cm
    height = plot_kwargs.get('height', 8.0)  # cm
    dpi = plot_kwargs.get('dpi', 100)
    font_size = plot_kwargs.get('font_size', 12)
    image_type = plot_kwargs.get('image_type', 'png')
    
    # Convert cm to inches
    inch = 1/2.54
    width_in, height_in = width * inch, height * inch
    
    # Get colors
    colors = get_color_definitions()
    
    # Create figure
    fig, ax = plt.subplots(layout='constrained', figsize=(width_in, height_in))
    
    # Plot profiles with color gradient
    cmap = colors['dice_gradient']
    n_profiles = len(profiles)
    
    for i, (t, profile) in enumerate(zip(time_axis, profiles)):
        color = cmap(i / max(1, n_profiles - 1))
        ax.plot(x_axis, profile, color=color, linewidth=2, 
                label=f't = {t:.1f} {time_unit}')
    
    # Formatting
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.set_title(title, fontsize=font_size)
    ax.tick_params(axis='both', labelsize=font_size, direction='in')
    
    # Legend (only if not too many profiles)
    if n_profiles <= 10:
        ax.legend(fontsize=font_size-2, frameon=False)
    
    # Make background white and opaque
    fig.patch.set_facecolor('w')
    fig.patch.set_alpha(1)
    
    # Save if filename provided
    if filename:
        try:
            plt.savefig(filename, dpi=dpi, format=image_type)
        except Exception as e:
            print(f"Error saving plot: {e}")
    
    return fig


def plot_diffusion_msd(
    time_axis: np.ndarray,
    sigma2_values: np.ndarray,
    sigma2_errors: Optional[np.ndarray] = None,
    ols_fit: Optional[Dict] = None,
    wls_fit: Optional[Dict] = None,
    title: str = "Mean Squared Displacement",
    xlabel: str = "Time",
    ylabel: str = "σ² - σ²₀",
    time_unit: str = "ns",
    length_unit: str = "µm",
    filename: Optional[str] = None,
    **plot_kwargs
) -> plt.Figure:
    """
    Plot mean squared displacement (MSD) analysis.
    
    Parameters
    ----------
    time_axis : np.ndarray
        Time points.
    sigma2_values : np.ndarray
        Measured variance values.
    sigma2_errors : np.ndarray, optional
        Error bars for variance measurements.
    ols_fit : dict, optional
        OLS fit results with 'slope' and 'intercept' keys.
    wls_fit : dict, optional
        WLS fit results with 'slope' and 'intercept' keys.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    time_unit : str
        Unit for time axis.
    length_unit : str
        Unit for length measurements.
    filename : str, optional
        If provided, save plot to this filename.
    **plot_kwargs
        Additional plotting parameters.
    
    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    # Extract plot parameters
    width = plot_kwargs.get('width', 10.0)  # cm
    height = plot_kwargs.get('height', 8.0)  # cm
    dpi = plot_kwargs.get('dpi', 100)
    font_size = plot_kwargs.get('font_size', 12)
    image_type = plot_kwargs.get('image_type', 'png')
    
    # Convert cm to inches
    inch = 1/2.54
    width_in, height_in = width * inch, height * inch
    
    # Get colors
    colors = get_color_definitions()
    
    # Create figure
    fig, ax = plt.subplots(layout='constrained', figsize=(width_in, height_in))
    
    # Calculate MSD (subtract initial variance)
    if len(sigma2_values) > 0:
        msd = sigma2_values - sigma2_values[0]
    else:
        msd = sigma2_values
    
    # Plot data points
    if sigma2_errors is not None:
        ax.errorbar(time_axis, msd, yerr=sigma2_errors, 
                   fmt='o', color=colors['dice_blue'], 
                   capsize=3, label='Data')
    else:
        ax.plot(time_axis, msd, 'o', color=colors['dice_blue'], 
                markersize=6, label='Data')
    
    # Plot fits if provided
    if ols_fit is not None:
        fit_line = ols_fit['slope'] * time_axis + ols_fit['intercept']
        ax.plot(time_axis, fit_line, '--', color=colors['dice_gold'],
                linewidth=2, label=f"OLS: D = {ols_fit['slope']/2:.3f}")
    
    if wls_fit is not None:
        fit_line = wls_fit['slope'] * time_axis + wls_fit['intercept']
        ax.plot(time_axis, fit_line, '-', color=colors['dice_green'],
                linewidth=2, label=f"WLS: D = {wls_fit['slope']/2:.3f}")
    
    # Formatting
    ax.set_xlabel(f"{xlabel} ({time_unit})", fontsize=font_size)
    ax.set_ylabel(f"{ylabel} ({length_unit}²)", fontsize=font_size)
    ax.set_title(title, fontsize=font_size)
    ax.tick_params(axis='both', labelsize=font_size, direction='in')
    ax.legend(fontsize=font_size, frameon=False)
    
    # Make background white and opaque
    fig.patch.set_facecolor('w')
    fig.patch.set_alpha(1)
    
    # Save if filename provided
    if filename:
        try:
            plt.savefig(filename, dpi=dpi, format=image_type)
        except Exception as e:
            print(f"Error saving plot: {e}")
    
    return fig


def plot_cnr_dependence(
    cnr_values: np.ndarray,
    precision_values: np.ndarray,
    title: str = "Precision vs. CNR",
    xlabel: str = "Contrast-to-Noise Ratio",
    ylabel: str = "Precision (%)",
    filename: Optional[str] = None,
    **plot_kwargs
) -> plt.Figure:
    """
    Plot precision as a function of contrast-to-noise ratio.
    
    Parameters
    ----------
    cnr_values : np.ndarray
        CNR values.
    precision_values : np.ndarray
        Precision percentages.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    filename : str, optional
        If provided, save plot to this filename.
    **plot_kwargs
        Additional plotting parameters.
    
    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    # Extract plot parameters
    width = plot_kwargs.get('width', 10.0)  # cm
    height = plot_kwargs.get('height', 8.0)  # cm
    dpi = plot_kwargs.get('dpi', 100)
    font_size = plot_kwargs.get('font_size', 12)
    image_type = plot_kwargs.get('image_type', 'png')
    
    # Convert cm to inches
    inch = 1/2.54
    width_in, height_in = width * inch, height * inch
    
    # Get colors
    colors = get_color_definitions()
    
    # Create figure
    fig, ax = plt.subplots(layout='constrained', figsize=(width_in, height_in))
    
    # Plot data
    ax.plot(cnr_values, precision_values, 'o-', 
            color=colors['dice_blue'], linewidth=2, markersize=6)
    
    # Formatting
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.set_title(title, fontsize=font_size)
    ax.tick_params(axis='both', labelsize=font_size, direction='in')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    # Make background white and opaque
    fig.patch.set_facecolor('w')
    fig.patch.set_alpha(1)
    
    # Save if filename provided
    if filename:
        try:
            plt.savefig(filename, dpi=dpi, format=image_type)
        except Exception as e:
            print(f"Error saving plot: {e}")
    
    return fig


def close_figure(fig: plt.Figure) -> None:
    """
    Close a matplotlib figure to free memory.
    
    Parameters
    ----------
    fig : plt.Figure
        The figure to close.
    
    Examples
    --------
    >>> fig = plt.figure()
    >>> close_figure(fig)
    """
    plt.close(fig)


def close_all_figures() -> None:
    """
    Close all open matplotlib figures.
    
    Examples
    --------
    >>> close_all_figures()
    """
    plt.close('all')


def configure_matplotlib_style(
    font_size: int = 12,
    font_family: str = 'serif'
) -> None:
    """
    Configure matplotlib style for consistent plotting.
    
    Parameters
    ----------
    font_size : int
        Default font size for plots.
    font_family : str
        Font family to use ('serif', 'sans-serif', etc.).
    
    Examples
    --------
    >>> configure_matplotlib_style(font_size=14)
    """
    plt.rcParams.update({
        'font.size': font_size,
        'font.family': font_family,
        'axes.linewidth': 1,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'legend.frameon': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })