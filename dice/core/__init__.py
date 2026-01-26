"""
Core scientific algorithms for DICE package.

This module contains the fundamental computational functions for:
- Gaussian profile generation and evolution
- Diffusion calculations
- Noise modeling and CNR estimation
- Fitting algorithms
"""

from .profiles import (
    gaussian,
    integrated_intensity,
    make_diffusion_decay,
    kinetic_decay_intensities,
    diffusion_sigma2_t,
)
from .diffusion import (
    calculate_diffusion_coefficient,
    calculate_diffusion_length,
    calculate_msd,
    estimate_diffusion_from_msd,
    calculate_peclet_number,
    einstein_relation,
)
from .noise import (
    add_noise,
    make_noise_distribution,
    fft_cnr,
    estimate_noise_from_profile,
)
from .fitting import (
    gauss_fitting,
    diffusion_ols_fit,
    diffusion_wls_fit,
    fit_gaussian_profile,
    calculate_fit_weights,
)

__all__ = [
    # Profiles
    "gaussian",
    "integrated_intensity",
    "make_diffusion_decay",
    "kinetic_decay_intensities",
    "diffusion_sigma2_t",
    # Diffusion
    "calculate_diffusion_coefficient",
    "calculate_diffusion_length",
    "calculate_msd",
    "estimate_diffusion_from_msd",
    "calculate_peclet_number",
    "einstein_relation",
    # Noise
    "add_noise",
    "make_noise_distribution",
    "fft_cnr",
    "estimate_noise_from_profile",
    # Fitting
    "gauss_fitting",
    "diffusion_ols_fit",
    "diffusion_wls_fit",
    "fit_gaussian_profile",
    "calculate_fit_weights",
]