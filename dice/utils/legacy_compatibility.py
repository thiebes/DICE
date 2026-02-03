"""
Legacy compatibility layer for DICE.

This module provides adapters and converters for maintaining backward
compatibility with the original monolithic dice.py implementation.

.. deprecated::
    All classes in this module are deprecated. Use the modern dataclasses
    from :mod:`dice.models.parameters` instead.
"""

from typing import Dict, Any
import warnings
import numpy as np


class LegacyProfileParameters:
    """
    Minimal profile parameters for legacy compatibility.

    .. deprecated::
        Use :class:`dice.models.parameters.GaussianParameters` instead.
        Migration: ``GaussianParameters(amplitude=amplitude_0, sigma2=sigma2_0, mu=mu_0)``
    """

    def __init__(self, sigma2_0: float, amplitude_0: float, mu_0: float):
        warnings.warn(
            "LegacyProfileParameters is deprecated. "
            "Use dice.models.parameters.GaussianParameters instead. "
            "Migration: GaussianParameters(amplitude=amplitude_0, sigma2=sigma2_0, mu=mu_0)",
            DeprecationWarning,
            stacklevel=2
        )
        self.sigma2_0 = sigma2_0
        self.amplitude_0 = amplitude_0
        self.mu_0 = mu_0


class LegacyPhysicsParameters:
    """
    Minimal physics parameters for legacy compatibility.

    .. deprecated::
        Physics parameters are now top-level attributes on
        :class:`dice.models.parameters.SimulationParameters`.
    """

    def __init__(self, diffusion_coefficient: float, lifetime: float,
                 diffusion_length: float):
        warnings.warn(
            "LegacyPhysicsParameters is deprecated. "
            "Physics parameters are now top-level attributes on SimulationParameters.",
            DeprecationWarning,
            stacklevel=2
        )
        self.diffusion_coefficient = diffusion_coefficient
        self.lifetime = lifetime
        self.diffusion_length = diffusion_length


class LegacySimulationParameters:
    """
    Minimal simulation parameters for legacy compatibility.

    This class provides a lightweight parameter container that mimics
    the structure expected by the new modular functions but can be
    created from legacy dictionary format.

    .. deprecated::
        Use :class:`dice.models.parameters.SimulationParameters` instead.
        Use ``SimulationParameters.from_legacy(params_dict)`` for migration.
    """

    def __init__(self, profile: LegacyProfileParameters,
                 physics: LegacyPhysicsParameters):
        warnings.warn(
            "LegacySimulationParameters is deprecated. "
            "Use dice.models.parameters.SimulationParameters instead. "
            "Use SimulationParameters.from_legacy(params_dict) for migration.",
            DeprecationWarning,
            stacklevel=2
        )
        self.profile = profile
        self.physics = physics


def create_parameters_from_legacy(
    parameters_dict: Dict[str, Any],
    diffusion_coefficient: float,
    lifetime: float,
    diffusion_length: float
) -> LegacySimulationParameters:
    """
    Create simulation parameters from legacy dictionary format.
    
    Parameters
    ----------
    parameters_dict : dict
        Legacy parameters dictionary with keys like 'sigma^2_0', 'amplitude_0', etc.
    diffusion_coefficient : float
        Nominal diffusion coefficient.
    lifetime : float
        Nominal lifetime (tau).
    diffusion_length : float
        Nominal diffusion length.
    
    Returns
    -------
    LegacySimulationParameters
        Parameters object compatible with new modular functions.
    
    Examples
    --------
    >>> params_dict = {
    ...     'sigma^2_0': 1.0,
    ...     'amplitude_0': 1.0,
    ...     'mu_0': 0.0
    ... }
    >>> params = create_parameters_from_legacy(params_dict, 0.5, 10.0, np.sqrt(5.0))
    >>> params.profile.sigma2_0
    1.0
    """
    profile = LegacyProfileParameters(
        sigma2_0=parameters_dict['sigma^2_0'],
        amplitude_0=parameters_dict['amplitude_0'],
        mu_0=parameters_dict.get('mu_0', parameters_dict.get('mean_0', 0.0))  # Handle both naming conventions
    )
    
    physics = LegacyPhysicsParameters(
        diffusion_coefficient=diffusion_coefficient,
        lifetime=lifetime,
        diffusion_length=diffusion_length
    )
    
    return LegacySimulationParameters(profile=profile, physics=physics)


def convert_legacy_result_to_dict(result, run_id: int, 
                                 retain_profile_data: bool = False) -> Dict:
    """
    Convert new RunResult to legacy dictionary format.
    
    Parameters
    ----------
    result : RunResult
        Result from run_single_simulation.
    run_id : int
        Run identifier.
    retain_profile_data : bool
        Whether to include profile data in output.
    
    Returns
    -------
    dict
        Legacy-formatted result dictionary.
    
    Notes
    -----
    This function maintains backward compatibility with the original
    scan_runner output format. It will be deprecated once all code
    is migrated to use the new RunResult format directly.
    """
    legacy_result = {
        f'run_{run_id}': {
            'run': run_id,
            'run parameters': {
                'nominal diffusion coefficient': result.nominal_diffusion_coefficient,
                'nominal lifetime': result.nominal_lifetime,
                'nominal diffusion length': result.nominal_diffusion_length,
                'noise stdev': result.noise_sigma
            },
            'cnr_0 estimate': result.cnr_0_estimate,
            'nominal profiles': {
                'parameters_t': {
                    'sigma^2_t': [result.nominal_sigma2_0]
                }
            },
            'noisy profile fits': {
                'sigma^2_t estimates': [result.estimated_sigma2_0] if result.estimated_sigma2_0 else []
            },
            'diffusion': {
                'unweighted fit': {
                    'MSD_t slope estimate': result.ols_slope or 'N/A',
                    'MSD_t slope std error': result.ols_slope_stderr or 'N/A',
                    'intercept estimate': result.ols_intercept or 'N/A',
                    'intercept standard error': result.ols_intercept_stderr or 'N/A',
                },
                'weighted fit': {
                    'MSD_t slope estimate': result.wls_slope or 'N/A',
                    'MSD_t slope std error': result.wls_slope_stderr or 'N/A',
                    'intercept estimate': result.wls_intercept or 'N/A',
                    'intercept standard error': result.wls_intercept_stderr or 'N/A',
                    'weights': result.weights if hasattr(result, 'weights') else 'N/A',
                }
            }
        }
    }
    
    # Add profile data if retained
    if retain_profile_data and hasattr(result, 'nominal_profiles'):
        legacy_result[f'run_{run_id}']['nominal profiles']['y_values_t'] = result.nominal_profiles
        legacy_result[f'run_{run_id}']['noisy profiles'] = {'y_values_t': result.noisy_profiles}
    
    return legacy_result


# Mock parameter classes for testing
# These are intentionally simple and should only be used in tests
class MockProfileParameters:
    """
    Mock profile parameters for testing only.

    .. deprecated::
        Use pytest fixtures from ``tests/conftest.py`` instead.
        See ``default_gaussian_params`` fixture.
    """
    def __init__(self):
        warnings.warn(
            "MockProfileParameters is deprecated. "
            "Use pytest fixtures from tests/conftest.py instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.sigma2_0 = 1.0
        self.amplitude_0 = 1.0
        self.mu_0 = 0.0


class MockPhysicsParameters:
    """
    Mock physics parameters for testing only.

    .. deprecated::
        Use pytest fixtures from ``tests/conftest.py`` instead.
        See ``default_simulation_params`` fixture.
    """
    def __init__(self):
        warnings.warn(
            "MockPhysicsParameters is deprecated. "
            "Use pytest fixtures from tests/conftest.py instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.diffusion_coefficient = 1.0
        self.lifetime = 10.0
        self.diffusion_length = np.sqrt(10.0)


class MockSimulationParameters:
    """
    Mock simulation parameters for testing only.

    This provides a minimal implementation that doesn't require
    complex initialization, suitable for unit tests.

    .. deprecated::
        Use pytest fixtures from ``tests/conftest.py`` instead.
        See ``default_simulation_params`` fixture for modern dataclass equivalent.
    """
    def __init__(self):
        warnings.warn(
            "MockSimulationParameters is deprecated. "
            "Use pytest fixtures from tests/conftest.py instead. "
            "See default_simulation_params fixture.",
            DeprecationWarning,
            stacklevel=2
        )
        # Suppress nested deprecation warnings for internal mock classes
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.profile = MockProfileParameters()
            self.physics = MockPhysicsParameters()
        self.num_runs = 100
        self.noise_values = [0.05]