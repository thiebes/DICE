"""
Parameter containers that the simulation engine takes as input.

The engine reads the initial profile from ``parameters.profile`` and the
nominal transport values from ``parameters.physics``.
``create_parameters_from_legacy`` builds that object from a parsed parameters
dictionary; the command-line interface, the GUI and ``dice.dice_runner`` all
call it.
"""

from typing import Dict, Any
import numpy as np


class LegacyProfileParameters:
    """Initial Gaussian profile: its variance, amplitude and mean."""

    def __init__(self, sigma2_0: float, amplitude_0: float, mu_0: float):
        self.sigma2_0 = sigma2_0
        self.amplitude_0 = amplitude_0
        self.mu_0 = mu_0


class LegacyPhysicsParameters:
    """Nominal diffusion coefficient, lifetime and diffusion length."""

    def __init__(self, diffusion_coefficient: float, lifetime: float,
                 diffusion_length: float):
        self.diffusion_coefficient = diffusion_coefficient
        self.lifetime = lifetime
        self.diffusion_length = diffusion_length


class LegacySimulationParameters:
    """
    Simulation parameters in the structure the engine reads.

    Attributes
    ----------
    profile : LegacyProfileParameters
        The initial profile.
    physics : LegacyPhysicsParameters
        The nominal transport values.
    """

    def __init__(self, profile: LegacyProfileParameters,
                 physics: LegacyPhysicsParameters):
        self.profile = profile
        self.physics = physics


def create_parameters_from_legacy(
    parameters_dict: Dict[str, Any],
    diffusion_coefficient: float,
    lifetime: float,
    diffusion_length: float
) -> LegacySimulationParameters:
    """
    Create the engine's simulation parameters from a parsed parameters dictionary.

    Parameters
    ----------
    parameters_dict : dict
        Parsed parameters dictionary with the keys 'sigma^2_0', 'amplitude_0'
        and 'mu_0' (or 'mean_0').
    diffusion_coefficient : float
        Nominal diffusion coefficient.
    lifetime : float
        Nominal lifetime (tau).
    diffusion_length : float
        Nominal diffusion length.

    Returns
    -------
    LegacySimulationParameters
        Parameters object that run_monte_carlo_simulation takes.

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


# Fixed parameter objects for unit tests of the engine.
class MockProfileParameters:
    """A unit-variance, unit-amplitude profile centered at zero, for tests."""

    def __init__(self):
        self.sigma2_0 = 1.0
        self.amplitude_0 = 1.0
        self.mu_0 = 0.0


class MockPhysicsParameters:
    """Transport values D = 1 and tau = 10, for tests."""

    def __init__(self):
        self.diffusion_coefficient = 1.0
        self.lifetime = 10.0
        self.diffusion_length = np.sqrt(10.0)


class MockSimulationParameters:
    """Simulation parameters with fixed values in the structure the engine reads, for tests."""

    def __init__(self):
        self.profile = MockProfileParameters()
        self.physics = MockPhysicsParameters()
        self.num_runs = 100
        self.noise_values = [0.05]
