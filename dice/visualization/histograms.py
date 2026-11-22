"""
Histogram plotting functions for DICE statistical analysis.

This module provides functions for creating histograms and statistical
distribution plots from simulation results.
"""

import logging
import os
import matplotlib

# Use non-interactive backend for thread safety in GUI/CI contexts
# Allow override via MPLBACKEND environment variable
if os.environ.get('MPLBACKEND') is None:
    matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from scipy.stats import norm
from typing import Dict, List, Optional, Sequence, Tuple, Union, Any
from pathlib import Path

logger = logging.getLogger(__name__)
from .plots import get_color_definitions
from ..analysis.statistics import sample_standard_deviation
from ..utils.validators import validate_limits

# Parameter-file key -> keyword argument of plot_accuracy_histogram.
_HISTOGRAM_OPTION_KEYS = {
    'image type': 'image_type',
    'image width': 'width',
    'image height': 'height',
    'image dpi': 'dpi',
    'image font size': 'font_size',
    'image tick length': 'tick_length',
    'image tick width': 'tick_width',
    'image numbins': 'num_bins',
    'image x_lim': 'x_lim',
}


def save_figure(fig, filename: Union[str, Path], dpi: float, image_type: str) -> None:
    """
    Save a figure, or raise if its file cannot be written.

    The error of the save (an OSError that names the path and the cause)
    reaches the caller, so no caller can report a plot that was not written.
    The figure is closed first: the caller then gets no figure to close.
    """
    try:
        fig.savefig(filename, dpi=dpi, format=image_type)
    except Exception:
        plt.close(fig)
        raise


def supported_image_types() -> List[str]:
    """Image types the histogram can be written as: the formats of matplotlib's savefig."""
    from matplotlib.backend_bases import FigureCanvasBase
    return sorted(FigureCanvasBase.get_supported_filetypes())


def validate_image_type(image_type: Any) -> str:
    """
    Check that the histogram can be written as this image type.

    Parameters
    ----------
    image_type : str
        The 'image type' parameter, for example 'png'; case does not matter.

    Returns
    -------
    str
        The image type, unchanged.

    Raises
    ------
    ValueError
        If matplotlib cannot write the type; the message lists the types it can write.
    """
    supported = supported_image_types()
    if not isinstance(image_type, str) or image_type.lower() not in supported:
        raise ValueError(
            f"Unknown image type '{image_type}'. Valid types: {', '.join(supported)}"
        )
    return image_type


def histogram_options(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the plot_accuracy_histogram keyword arguments from a parameters dictionary.

    The command-line interface and the GUI both call this function, so a
    parameter file gives the same figure in either one. A key that is absent
    (or None) is left out, and plot_accuracy_histogram applies its own default.

    Parameters
    ----------
    parameters : dict
        Parsed simulation parameters with the legacy 'image ...' keys.

    Returns
    -------
    dict
        Keyword arguments for plot_accuracy_histogram.

    Raises
    ------
    ValueError
        If 'image x_lim' is not two increasing finite numbers.
    """
    options = {
        argument: parameters[key]
        for key, argument in _HISTOGRAM_OPTION_KEYS.items()
        if parameters.get(key) is not None
    }

    if 'x_lim' in options:
        options['x_lim'] = validate_limits(options['x_lim'], "'image x_lim'")

    return options


def counts_beyond_limits(values: np.ndarray, limits) -> Tuple[int, int]:
    """Number of finite values below the lower limit and above the upper limit of an axis."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    low, high = min(limits), max(limits)
    return int(np.count_nonzero(finite < low)), int(np.count_nonzero(finite > high))


def note_counts_beyond_axis(ax, values: np.ndarray, limits, along: str, font_size: float) -> None:
    """
    Write on a plot how many values lie beyond the limits of one of its axes.

    The default limits of the accuracy plots span three standard deviations
    around the mean, so a few estimates can fall outside the picture. The
    note stands just outside the frame, at the end of the axis it refers to;
    there is none when every value is drawn.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to annotate.
    values : np.ndarray
        The plotted values along the axis.
    limits : sequence of two floats
        The limits of that axis.
    along : str
        'x' for the horizontal axis, 'y' for the vertical one.
    font_size : float
        Font size of the figure.
    """
    below, above = counts_beyond_limits(values, limits)
    if along == 'x':
        if below:
            ax.text(0.0, 1.01, f'{below} below this range', transform=ax.transAxes,
                    ha='left', va='bottom', fontsize=font_size)
        if above:
            ax.text(1.0, 1.01, f'{above} above this range', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=font_size)
    else:
        notes = ([f'{above} above this range'] if above else []) + ([f'{below} below this range'] if below else [])
        if notes:
            ax.text(1.0, 1.01, ', '.join(notes), transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=font_size)


def plot_accuracy_histogram(
    simulation_result: Dict[str, Any],
    filename: str,
    proximity: Optional[float] = None,
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
        Results dictionary containing 'collated results' with 'd_est_over_d_nom' key.
    filename : str
        Output filename for the histogram image.
    proximity : float, optional
        Proximity threshold for accuracy assessment (e.g., 0.1 for ±10%).
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
    >>> result = {'collated results': {'d_est_over_d_nom': [0.98, 1.02, 0.99, 1.01]}}
    >>> fig = plot_accuracy_histogram(result, 'accuracy.png')
    """
    # Verify required data is present
    if 'collated results' not in simulation_result:
        raise ValueError("'collated results' not found in simulation_result")
    
    collated = simulation_result['collated results']
    if 'd_est_over_d_nom' not in collated:
        raise ValueError("'d_est_over_d_nom' not found in collated results")

    # Convert width and height from cm to inches
    inch = 1/2.54
    width_in, height_in = width * inch, height * inch

    # Get color definitions
    colors = get_color_definitions()
    dice_blue = colors['dice_blue']
    dice_gold = colors['dice_gold']
    
    # Convert list of values to NumPy array
    dest_over_d0 = np.array(collated['d_est_over_d_nom'])
    
    # Optionally flag and report accuracy ratio values within proximity
    if proximity is not None:
        dest_d0_proximity_flag = np.abs(dest_over_d0 - 1) <= proximity
        proxpct = 100 * np.sum(dest_d0_proximity_flag) / len(dest_d0_proximity_flag)
        logger.info(f'Percent of D estimates within {proximity * 100:.2f}% of nominal: {proxpct:.1f}')

    # Create figure
    fig, ax = plt.subplots(layout='constrained', figsize=(width_in, height_in))

    # Create histogram with probability on y-axis
    n_dd0, bins_dd0, patches_dd0 = ax.hist(
        dest_over_d0,
        bins=num_bins,
        weights=np.ones(len(dest_over_d0)) / len(dest_over_d0),
        color=dice_gold,
        edgecolor='w',
        alpha=0.7
    )

    # Calculate statistics
    mu_dd0 = np.mean(dest_over_d0)
    sigma_dd0 = sample_standard_deviation(dest_over_d0)

    # Set labels
    ax.set_xlabel('$D_{est}/D_{nom}$', fontsize=font_size)
    ax.set_ylabel('Probability', fontsize=font_size)

    # Set x-axis limits (default: 99.97% confidence interval)
    if x_lim is None:
        if sigma_dd0 > 0:
            x_lim = [mu_dd0 - 3 * sigma_dd0, mu_dd0 + 3 * sigma_dd0]
        else:
            x_lim = [mu_dd0 - 0.1, mu_dd0 + 0.1]
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

    # Mark the nominal value, where the ratio is 1. When the axis does not
    # reach 1 (a biased simulation, or limits from the parameters), the legend
    # says on which side of the picture it lies.
    low, high = sorted(x_lim)
    if low <= 1.0 <= high:
        ax.axvline(1.0, color=colors['dice_green'], linewidth=1.5, linestyle='--', label='nominal')
    else:
        side = 'below' if 1.0 < low else 'above'
        ax.plot([], [], ' ', color=colors['dice_green'], label=f'nominal (1) is {side} this range')

    # Plot normal distribution overlay when variance is nonzero. The legend
    # is text only; each label has the color of what it names. The curve has
    # the mean and the standard deviation of the estimates; the median is a
    # property of the estimates alone.
    mean_label = f'mean {np.round(mu_dd0, 3)}'
    stdev_label = f'stdev {np.round(sigma_dd0, 3)}'
    median_label = f'median {np.round(np.median(dest_over_d0), 3)}'
    if sigma_dd0 > 0:
        binspace_dd0 = np.linspace(bins_dd0[0], bins_dd0[-1], 100)
        bin_width = bins_dd0[1] - bins_dd0[0]
        y_dd0 = norm.pdf(binspace_dd0, mu_dd0, sigma_dd0) * bin_width
        ax.plot(
            binspace_dd0, y_dd0,
            color=dice_blue,
            linewidth=2,
            label=f'normal curve\n{mean_label}\n{stdev_label}'
        )
        ax.plot([], [], ' ', color='black', label=median_label)
    else:
        ax.plot([], [], ' ', color='black', label=f'{mean_label}\n{median_label}\n{stdev_label}')

    # Estimates beyond the limits of the axis are not drawn, so count them.
    note_counts_beyond_axis(ax, dest_over_d0, x_lim, along='x', font_size=font_size)

    # Add legend
    ax.legend(
        fontsize=font_size,
        handlelength=0,
        labelspacing=1,
        labelcolor='linecolor',
        frameon=False
    )

    # Make background white and opaque
    fig.patch.set_facecolor('w')
    fig.patch.set_alpha(1)

    # Save the figure
    save_figure(fig, filename, dpi, image_type)
    logger.info("Accuracy histogram saved as: %s", filename)

    return fig


def accuracy_scatter_options(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the plot_accuracy_vs_cnr keyword arguments from a parameters dictionary.

    The 'image ...' keys apply as they do to the histogram (histogram_options),
    except 'image numbins', which a scatter plot does not use, and
    'image x_lim', the limits of D_est/D_nom, which set the y-axis of the
    scatter plot.

    Raises
    ------
    ValueError
        If 'image x_lim' is not two increasing finite numbers.
    """
    options = histogram_options(parameters)
    options.pop('num_bins', None)
    if 'x_lim' in options:
        options['y_lim'] = options.pop('x_lim')
    return options


def plot_accuracy_vs_cnr(
    cnr_values: Sequence[float],
    d_est_over_d_nom: Sequence[float],
    filename: str,
    log_cnr: bool = False,
    image_type: str = 'png',
    width: float = 10.0,
    height: float = 6.0,
    dpi: float = 100,
    font_size: float = 12,
    tick_length: float = 4,
    tick_width: float = 1,
    y_lim: Optional[List[float]] = None
) -> plt.Figure:
    """
    Scatter plot of the diffusion accuracy (D_est/D_nom) of each run against its initial CNR.

    A simulation with a noise range draws this plot in place of the accuracy
    histogram. A dashed line marks D_est/D_nom = 1.

    Parameters
    ----------
    cnr_values : sequence of float
        Nominal initial CNR of each run.
    d_est_over_d_nom : sequence of float
        Each run's diffusion coefficient estimate over the nominal value, in
        the order of cnr_values.
    filename : str
        Output filename for the image.
    log_cnr : bool
        Whether the CNR axis is logarithmic, as suits a CNR drawn uniformly
        in its logarithm.
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
    y_lim : list, optional
        Limits [min, max] of the D_est/D_nom axis. If None, the mean ± 3
        standard deviations of the finite ratios, as on the histogram's x-axis.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.

    Raises
    ------
    ValueError
        If cnr_values and d_est_over_d_nom differ in length.
    """
    cnr = np.asarray(cnr_values, dtype=float)
    ratios = np.asarray(d_est_over_d_nom, dtype=float)
    if cnr.shape != ratios.shape:
        raise ValueError(
            f"cnr_values and d_est_over_d_nom differ in length: {cnr.size} and {ratios.size}")

    # Convert width and height from cm to inches
    inch = 1/2.54
    colors = get_color_definitions()
    fig, ax = plt.subplots(layout='constrained', figsize=(width * inch, height * inch))

    ax.scatter(cnr, ratios, s=12, color=colors['dice_blue'], alpha=0.5, edgecolors='none')
    ax.axhline(1.0, color=colors['dice_gold'], linewidth=2, linestyle='--')

    if log_cnr:
        # Plain numbers at each power of ten, and also at 2 and 5 times each
        # one when the CNR spans two decades or less, so that a range of one
        # decade has more than two labels.
        ax.set_xscale('log')
        positive_cnr = cnr[np.isfinite(cnr) & (cnr > 0)]
        decades = np.log10(positive_cnr.max() / positive_cnr.min()) if positive_cnr.size else 0.0
        ax.xaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0) if decades <= 2 else (1.0,)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('Initial CNR', fontsize=font_size)
    ax.set_ylabel('$D_{est}/D_{nom}$', fontsize=font_size)

    # Default y-axis limits: the 99.97% interval of a normal distribution, as for the histogram
    finite_ratios = ratios[np.isfinite(ratios)]
    if y_lim is None and finite_ratios.size > 0:
        mean, std = np.mean(finite_ratios), sample_standard_deviation(finite_ratios)
        y_lim = [mean - 3 * std, mean + 3 * std] if std > 0 else [mean - 0.1, mean + 0.1]
    if y_lim is not None:
        ax.set_ylim(y_lim)
        # Estimates beyond the limits of the axis are not drawn, so count them.
        note_counts_beyond_axis(ax, finite_ratios, y_lim, along='y', font_size=font_size)

    ax.tick_params(
        axis='both',
        which='both',
        labelsize=font_size,
        direction='in',
        length=tick_length,
        width=tick_width
    )

    # Make background white and opaque
    fig.patch.set_facecolor('w')
    fig.patch.set_alpha(1)

    save_figure(fig, filename, dpi, image_type)
    logger.info("Accuracy scatter plot saved as: %s", filename)

    return fig


def plot_diffusion_coefficient_histogram(
    diffusion_estimates: np.ndarray,
    nominal_value: float,
    proximity: Optional[float] = None,
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
    
    # Guard against empty input
    if len(diffusion_estimates) == 0:
        ax.set_xlabel('Diffusion Coefficient', fontsize=font_size)
        ax.set_ylabel('Probability', fontsize=font_size)
        ax.set_title(title, fontsize=font_size)
        fig.patch.set_facecolor('w')
        fig.patch.set_alpha(1)
        return fig

    # Calculate statistics
    mean_val = np.mean(diffusion_estimates)
    std_val = sample_standard_deviation(diffusion_estimates)
    median_val = np.median(diffusion_estimates)

    # Calculate precision if proximity threshold provided
    if proximity is not None:
        within_proximity = np.abs(diffusion_estimates - nominal_value) <= proximity * nominal_value
        precision_pct = 100 * np.sum(within_proximity) / len(diffusion_estimates)
    else:
        precision_pct = None

    # Create histogram with probability on y-axis
    n, bins, patches = ax.hist(
        diffusion_estimates,
        bins=num_bins,
        weights=np.ones(len(diffusion_estimates)) / len(diffusion_estimates),
        color=colors['dice_gold'],
        edgecolor='white',
        alpha=0.7,
        label=f'Data (n={len(diffusion_estimates)})'
    )

    # Normal distribution overlay scaled to probability (skip when variance is zero)
    if std_val > 0:
        x_norm = np.linspace(bins[0], bins[-1], 100)
        bin_width = bins[1] - bins[0]
        y_norm = norm.pdf(x_norm, mean_val, std_val) * bin_width
        ax.plot(x_norm, y_norm, color=colors['dice_blue'], linewidth=2,
                label='Normal fit')

    # Vertical line at nominal value
    ax.axvline(nominal_value, color=colors['dice_green'], linestyle='--',
               linewidth=2, label=f'Nominal: {nominal_value:.3f}')

    # Formatting
    ax.set_xlabel('Diffusion Coefficient', fontsize=font_size)
    ax.set_ylabel('Probability', fontsize=font_size)
    ax.set_title(title, fontsize=font_size)
    ax.tick_params(axis='both', labelsize=font_size, direction='in')
    
    # Add statistics text box
    stats_text = (f'Mean: {mean_val:.4f}\n'
                  f'Median: {median_val:.4f}\n'
                  f'Std: {std_val:.4f}')
    if precision_pct is not None:
        stats_text += f'\nPrecision: {precision_pct:.1f}%'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=font_size-1,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend(fontsize=font_size, frameon=False)
    
    # Make background white and opaque
    fig.patch.set_facecolor('w')
    fig.patch.set_alpha(1)
    
    # Save if filename provided
    if filename:
        save_figure(fig, filename, dpi, image_type)
        logger.info("Diffusion coefficient histogram saved as: %s", filename)
    
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

    # Guard against empty input
    if len(cnr_estimates) == 0:
        ax.set_xlabel('Contrast-to-Noise Ratio', fontsize=font_size)
        ax.set_ylabel('Probability', fontsize=font_size)
        ax.set_title(title, fontsize=font_size)
        fig.patch.set_facecolor('w')
        fig.patch.set_alpha(1)
        return fig

    # Calculate statistics
    mean_val = np.mean(cnr_estimates)
    median_val = np.median(cnr_estimates)
    std_val = sample_standard_deviation(cnr_estimates)

    # Create histogram with probability on y-axis
    n, bins, patches = ax.hist(
        cnr_estimates,
        bins=num_bins,
        weights=np.ones(len(cnr_estimates)) / len(cnr_estimates),
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
    ax.set_ylabel('Probability', fontsize=font_size)
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
        save_figure(fig, filename, dpi, image_type)
        logger.info("CNR histogram saved as: %s", filename)
    
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
            logger.info("Precision plot saved as: %s", filename)
        except Exception as e:
            logger.warning("Error saving file: %s", e)
    
    return fig