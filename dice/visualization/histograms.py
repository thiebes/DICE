"""
Histogram plotting functions for DICE statistical analysis.

This module provides functions for creating histograms and statistical
distribution plots from simulation results.
"""

import os
import matplotlib

# Use non-interactive backend for thread safety in GUI/CI contexts
# Allow override via MPLBACKEND environment variable
if os.environ.get('MPLBACKEND') is None:
    matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from .plots import get_color_definitions


def plot_accuracy_histogram(
    simulation_result: Dict[str, Any],
    proximity: float,
    filename: str,
    image_type: str = 'png',
    width: float = 10.0,
    height: float = 6.0,
    dpi: float = 100,
    font_size: float = 12,
    tick_length: float = 4,
    tick_width: float = 1,
    num_bins: int = 50,
    x_lim: Optional[List[float]] = None
) -> plt.Figure:
    """
    Create a histogram of diffusion accuracy values (D_est/D_nom) from simulation results.
    
    This function extracts weighted least squares diffusion coefficient estimates,
    normalizes them by the nominal value, and creates a histogram with statistical
    overlay showing the distribution characteristics.
    
    Parameters
    ----------
    simulation_result : dict
        Results dictionary containing 'collated results' with 'd_wls_over_d_nom' key.
    proximity : float
        Proximity threshold for accuracy assessment (e.g., 0.1 for ±10%).
    filename : str
        Output filename for the histogram image.
    image_type : str
        Image format ('png', 'jpg', 'svg', etc.).
    width : float
        Width of the image in cm.
    height : float
        Height of the image in cm.
    dpi : float
        Resolution in dots per inch.
    font_size : float
        Font size for labels and text.
    tick_length : float
        Length of axis ticks in points.
    tick_width : float
        Width of axis ticks in points.
    num_bins : int
        Number of histogram bins.
    x_lim : list, optional
        X-axis limits [min, max]. If None, uses 99.97% confidence interval.
    
    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    
    Raises
    ------
    ValueError
        If required data is not found in simulation_result.
    
    Examples
    --------
    >>> result = {'collated results': {'d_wls_over_d_nom': [0.98, 1.02, 0.99, 1.01]}}
    >>> fig = plot_accuracy_histogram(result, 0.1, 'accuracy.png')
    """
    # Verify required data is present
    if 'collated results' not in simulation_result:
        raise ValueError("'collated results' not found in simulation_result")
    
    collated = simulation_result['collated results']
    if 'd_wls_over_d_nom' not in collated:
        raise ValueError("'d_wls_over_d_nom' not found in collated results")

    # Convert width and height from cm to inches
    inch = 1/2.54
    width_in, height_in = width * inch, height * inch

    # Get color definitions
    colors = get_color_definitions()
    dice_blue = colors['dice_blue']
    dice_gold = colors['dice_gold']
    
    # Convert list of values to NumPy array
    dest_over_d0 = np.array(collated['d_wls_over_d_nom'])
    
    # Initialize array to flag accuracy ratio values in proximity
    dest_d0_proximity_flag = np.abs(dest_over_d0 - 1) <= proximity
    
    # Calculate and report percentage of values within proximity threshold
    proxpct = 100 * np.sum(dest_d0_proximity_flag) / len(dest_d0_proximity_flag)
    print(f'Percent of D estimates within {proximity * 100:.2f}% of nominal: {proxpct:.1f}')

    # Create figure
    fig, ax = plt.subplots(layout='constrained', figsize=(width_in, height_in))

    # Create histogram
    n_dd0, bins_dd0, patches_dd0 = ax.hist(
        dest_over_d0,
        bins=num_bins, 
        density=True,
        color=dice_gold, 
        edgecolor='w',
        alpha=0.7
    )

    # Calculate normal distribution overlay
    binspace_dd0 = np.linspace(bins_dd0[0], bins_dd0[-1], 100)
    mu_dd0 = np.mean(dest_over_d0)
    sigma_dd0 = np.std(dest_over_d0)
    y_dd0 = norm.pdf(binspace_dd0, mu_dd0, sigma_dd0)

    # Set labels
    ax.set_xlabel('$D_{est}/D_{nom}$', fontsize=font_size)
    ax.set_ylabel('Probability density', fontsize=font_size)

    # Set x-axis limits (default: 99.97% confidence interval)
    if x_lim is None:
        x_lim = [mu_dd0 - 3 * sigma_dd0, mu_dd0 + 3 * sigma_dd0]
    ax.set_xlim(x_lim)
    
    # Configure tick parameters
    ax.tick_params(
        axis='both', 
        which='both', 
        labelsize=font_size,
        direction='in', 
        length=tick_length, 
        width=tick_width
    )

    # Plot normal distribution overlay with statistics
    ax.plot(
        binspace_dd0, y_dd0, 
        color=dice_blue, 
        linewidth=2,
        label=(f'mean {np.round(mu_dd0, 3)}\n'
               f'median {np.round(np.median(dest_over_d0), 3)}\n'
               f'stdev {np.round(sigma_dd0, 3)}')
    )

    # Add legend
    ax.legend(
        fontsize=font_size, 
        handlelength=0, 
        labelspacing=2, 
        frameon=False
    )

    # Make background white and opaque
    fig.patch.set_facecolor('w')
    fig.patch.set_alpha(1)

    # Save the figure
    try:
        plt.savefig(filename, dpi=dpi, format=image_type)
        print(f"Accuracy histogram saved as: {filename}")
    except Exception as e:
        print(f"Error saving file: {e}")

    return fig


def plot_diffusion_coefficient_histogram(
    diffusion_estimates: np.ndarray,
    nominal_value: float,
    proximity: float = 0.1,
    filename: Optional[str] = None,
    title: str = "Diffusion Coefficient Distribution",
    **plot_kwargs
) -> plt.Figure:
    """
    Create histogram of diffusion coefficient estimates.
    
    Parameters
    ----------
    diffusion_estimates : np.ndarray
        Array of diffusion coefficient estimates.
    nominal_value : float
        Nominal (true) diffusion coefficient value.
    proximity : float
        Proximity threshold for accuracy assessment.
    filename : str, optional
        Output filename if saving is desired.
    title : str
        Plot title.
    **plot_kwargs
        Additional plotting parameters.
    
    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    # Extract plot parameters
    width = plot_kwargs.get('width', 10.0)  # cm
    height = plot_kwargs.get('height', 6.0)  # cm
    dpi = plot_kwargs.get('dpi', 100)
    font_size = plot_kwargs.get('font_size', 12)
    num_bins = plot_kwargs.get('num_bins', 30)
    image_type = plot_kwargs.get('image_type', 'png')
    
    # Convert cm to inches
    inch = 1/2.54
    width_in, height_in = width * inch, height * inch
    
    # Get colors
    colors = get_color_definitions()
    
    # Create figure
    fig, ax = plt.subplots(layout='constrained', figsize=(width_in, height_in))
    
    # Calculate statistics
    mean_val = np.mean(diffusion_estimates)
    std_val = np.std(diffusion_estimates)
    median_val = np.median(diffusion_estimates)
    
    # Calculate precision
    within_proximity = np.abs(diffusion_estimates - nominal_value) <= proximity * nominal_value
    precision_pct = 100 * np.sum(within_proximity) / len(diffusion_estimates)
    
    # Create histogram
    n, bins, patches = ax.hist(
        diffusion_estimates,
        bins=num_bins,
        density=True,
        color=colors['dice_gold'],
        edgecolor='white',
        alpha=0.7,
        label=f'Data (n={len(diffusion_estimates)})'
    )
    
    # Normal distribution overlay
    x_norm = np.linspace(bins[0], bins[-1], 100)
    y_norm = norm.pdf(x_norm, mean_val, std_val)
    ax.plot(x_norm, y_norm, color=colors['dice_blue'], linewidth=2,
            label=f'Normal fit')
    
    # Vertical line at nominal value
    ax.axvline(nominal_value, color=colors['dice_green'], linestyle='--', 
               linewidth=2, label=f'Nominal: {nominal_value:.3f}')
    
    # Formatting
    ax.set_xlabel('Diffusion Coefficient', fontsize=font_size)
    ax.set_ylabel('Probability Density', fontsize=font_size)
    ax.set_title(title, fontsize=font_size)
    ax.tick_params(axis='both', labelsize=font_size, direction='in')
    
    # Add statistics text box
    stats_text = (f'Mean: {mean_val:.4f}\n'
                  f'Median: {median_val:.4f}\n'
                  f'Std: {std_val:.4f}\n'
                  f'Precision: {precision_pct:.1f}%')
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=font_size-1,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend(fontsize=font_size, frameon=False)
    
    # Make background white and opaque
    fig.patch.set_facecolor('w')
    fig.patch.set_alpha(1)
    
    # Save if filename provided
    if filename:
        try:
            plt.savefig(filename, dpi=dpi, format=image_type)
            print(f"Diffusion coefficient histogram saved as: {filename}")
        except Exception as e:
            print(f"Error saving file: {e}")
    
    return fig


def plot_cnr_histogram(
    cnr_estimates: np.ndarray,
    filename: Optional[str] = None,
    title: str = "CNR Distribution",
    **plot_kwargs
) -> plt.Figure:
    """
    Create histogram of contrast-to-noise ratio estimates.
    
    Parameters
    ----------
    cnr_estimates : np.ndarray
        Array of CNR estimates.
    filename : str, optional
        Output filename if saving is desired.
    title : str
        Plot title.
    **plot_kwargs
        Additional plotting parameters.
    
    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    # Extract plot parameters
    width = plot_kwargs.get('width', 10.0)  # cm
    height = plot_kwargs.get('height', 6.0)  # cm
    dpi = plot_kwargs.get('dpi', 100)
    font_size = plot_kwargs.get('font_size', 12)
    num_bins = plot_kwargs.get('num_bins', 25)
    image_type = plot_kwargs.get('image_type', 'png')
    
    # Convert cm to inches
    inch = 1/2.54
    width_in, height_in = width * inch, height * inch
    
    # Get colors
    colors = get_color_definitions()
    
    # Create figure
    fig, ax = plt.subplots(layout='constrained', figsize=(width_in, height_in))
    
    # Calculate statistics
    mean_val = np.mean(cnr_estimates)
    median_val = np.median(cnr_estimates)
    std_val = np.std(cnr_estimates)
    
    # Create histogram
    n, bins, patches = ax.hist(
        cnr_estimates,
        bins=num_bins,
        density=True,
        color=colors['dice_blue'],
        edgecolor='white',
        alpha=0.7
    )
    
    # Vertical lines for statistics
    ax.axvline(mean_val, color=colors['dice_gold'], linestyle='-', 
               linewidth=2, label=f'Mean: {mean_val:.1f}')
    ax.axvline(median_val, color=colors['dice_green'], linestyle='--', 
               linewidth=2, label=f'Median: {median_val:.1f}')
    
    # Formatting
    ax.set_xlabel('Contrast-to-Noise Ratio', fontsize=font_size)
    ax.set_ylabel('Probability Density', fontsize=font_size)
    ax.set_title(title, fontsize=font_size)
    ax.tick_params(axis='both', labelsize=font_size, direction='in')
    ax.legend(fontsize=font_size, frameon=False)
    
    # Add statistics text
    stats_text = f'n = {len(cnr_estimates)}\nσ = {std_val:.2f}'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            fontsize=font_size-1,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Make background white and opaque
    fig.patch.set_facecolor('w')
    fig.patch.set_alpha(1)
    
    # Save if filename provided
    if filename:
        try:
            plt.savefig(filename, dpi=dpi, format=image_type)
            print(f"CNR histogram saved as: {filename}")
        except Exception as e:
            print(f"Error saving file: {e}")
    
    return fig


def plot_precision_vs_parameter(
    parameter_values: np.ndarray,
    precision_values: np.ndarray,
    parameter_name: str = "Parameter",
    filename: Optional[str] = None,
    title: Optional[str] = None,
    **plot_kwargs
) -> plt.Figure:
    """
    Plot precision as a function of a parameter (e.g., CNR, noise, etc.).
    
    Parameters
    ----------
    parameter_values : np.ndarray
        Array of parameter values.
    precision_values : np.ndarray
        Array of precision values (percentages).
    parameter_name : str
        Name of the parameter for axis labeling.
    filename : str, optional
        Output filename if saving is desired.
    title : str, optional
        Plot title. If None, generated automatically.
    **plot_kwargs
        Additional plotting parameters.
    
    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    if title is None:
        title = f"Precision vs. {parameter_name}"
    
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
    
    # Plot data
    ax.plot(parameter_values, precision_values, 'o-',
            color=colors['dice_blue'], linewidth=2, markersize=6)
    
    # Formatting
    ax.set_xlabel(parameter_name, fontsize=font_size)
    ax.set_ylabel('Precision (%)', fontsize=font_size)
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
            print(f"Precision plot saved as: {filename}")
        except Exception as e:
            print(f"Error saving file: {e}")
    
    return fig