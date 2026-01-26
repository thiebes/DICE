"""
Visualization package for DICE.

This package provides plotting and visualization functions for DICE simulation
results, including histograms, profiles, and statistical plots.
"""

from .plots import (
    get_color_definitions,
    plot_gaussian_profile,
    plot_profile_evolution,
    plot_diffusion_msd,
    plot_cnr_dependence,
    close_figure,
    close_all_figures,
    configure_matplotlib_style,
)

from .histograms import (
    plot_accuracy_histogram,
    plot_diffusion_coefficient_histogram,
    plot_cnr_histogram,
    plot_precision_vs_parameter,
)

__all__ = [
    # Color and style
    'get_color_definitions',
    'configure_matplotlib_style',
    
    # Basic plotting
    'plot_gaussian_profile',
    'plot_profile_evolution',
    'plot_diffusion_msd',
    'plot_cnr_dependence',
    
    # Histograms and distributions
    'plot_accuracy_histogram',
    'plot_diffusion_coefficient_histogram',
    'plot_cnr_histogram',
    'plot_precision_vs_parameter',
    
    # Utility functions
    'close_figure',
    'close_all_figures',
]