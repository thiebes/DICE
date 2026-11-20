"""
Utility functions for DICE package.
"""

from .converters import (
    sigma_to_fwhm,
    sigma2_to_fwhm,
    fwhm_to_sigma,
    fwhm_to_sigma2,
    calculate_pixel_size,
    slope_to_diffusion_constant,
)
from .validators import (
    validate_numeric,
    validate_array_like,
    validate_integer,
    validate_string_choice,
    validate_file_path,
)
from .axes import (
    make_x_axis,
    make_time_axis,
    make_time_series,
    make_spatial_grid,
)

__all__ = [
    # Converters
    "sigma_to_fwhm",
    "sigma2_to_fwhm",
    "fwhm_to_sigma",
    "fwhm_to_sigma2",
    "calculate_pixel_size",
    "slope_to_diffusion_constant",
    # Validators
    "validate_numeric",
    "validate_array_like",
    "validate_integer",
    "validate_string_choice",
    "validate_file_path",
    # Axes
    "make_x_axis",
    "make_time_axis",
    "make_time_series",
    "make_spatial_grid",
]