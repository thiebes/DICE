"""
Core scientific algorithms for DICE package.

This module contains the fundamental computational functions for:
- Gaussian profile generation and evolution
- Diffusion calculations
- Noise modeling and CNR estimation
- Fitting algorithms
"""

# Lazy imports (PEP 562) so that importing lightweight submodules
# (e.g. dice.core.diffusion) does not pull in scipy via noise/fitting.
_LAZY_IMPORTS = {
    # profiles
    "gaussian": ".profiles",
    "integrated_intensity": ".profiles",
    "make_diffusion_decay": ".profiles",
    "kinetic_decay_intensities": ".profiles",
    "diffusion_sigma2_t": ".profiles",
    # diffusion
    "calculate_diffusion_coefficient": ".diffusion",
    "calculate_diffusion_length": ".diffusion",
    "calculate_msd": ".diffusion",
    "estimate_diffusion_from_msd": ".diffusion",
    "calculate_peclet_number": ".diffusion",
    "einstein_relation": ".diffusion",
    # noise (scipy.signal)
    "add_noise": ".noise",
    "make_noise_distribution": ".noise",
    "fft_cnr": ".noise",
    "estimate_noise_from_profile": ".noise",
    # fitting (scipy.optimize, statsmodels)
    "gauss_fitting": ".fitting",
    "diffusion_ols_fit": ".fitting",
    "diffusion_wls_fit": ".fitting",
    "fit_gaussian_profile": ".fitting",
    "calculate_wls_weights": ".fitting",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
    "calculate_wls_weights",
]