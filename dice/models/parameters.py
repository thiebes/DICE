"""
Parameter dataclasses for DICE simulations.

``SimulationParameters`` is the object that the simulation engine takes as
input. The engine reads the initial profile from ``parameters.profile`` and
the nominal transport values from ``parameters.physics``.
``simulation_parameters_from_dict`` builds that object from a parsed
parameters dictionary; the command-line interface, the GUI and
``dice.dice_runner`` all call it.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class OutputUnitPreferences:
    """
    Units for output display.

    When a field is None, the output uses the global simulation unit or
    the default output convention (cm^2/s for diffusion coefficients).

    Attributes
    ----------
    length : str, optional
        Length unit for output display.
    time : str, optional
        Time unit for output display.
    diffusion_length : str, optional
        Length unit for diffusion coefficient output (defaults to 'centimeter').
    diffusion_time : str, optional
        Time unit for diffusion coefficient output (defaults to 'second').
    """
    length: Optional[str] = None
    time: Optional[str] = None
    diffusion_length: Optional[str] = None
    diffusion_time: Optional[str] = None


@dataclass(frozen=True)
class NoiseRange:
    """
    Range of noise standard deviations from which each run draws its own value.

    The reciprocal 1/sigma, which is the initial CNR for an amplitude of 1, is
    uniformly distributed between 1/high and 1/low. With logarithmic=True,
    its base-10 logarithm is uniformly distributed instead.

    Attributes
    ----------
    low : float
        Smallest noise standard deviation (the highest CNR).
    high : float
        Largest noise standard deviation (the lowest CNR).
    logarithmic : bool
        Whether log10(1/sigma), rather than 1/sigma, is uniformly distributed.

    Raises
    ------
    ValueError
        If a bound is not a positive, finite number, or low is not less than high.
    """
    low: float
    high: float
    logarithmic: bool = False

    def __post_init__(self):
        bounds = [self.low, self.high]
        if not all(np.isfinite(bound) and bound > 0 for bound in bounds):
            raise ValueError(f"Invalid noise range {bounds}. Both values must be positive and finite.")
        if self.low >= self.high:
            raise ValueError(
                f"Invalid noise range {bounds}. The first value (low) must be less than the second (high)."
            )


@dataclass
class ProfileParameters:
    """Initial Gaussian profile: its variance, amplitude and mean."""
    sigma2_0: float
    amplitude_0: float
    mu_0: float


@dataclass
class PhysicsParameters:
    """Nominal diffusion coefficient, lifetime and diffusion length."""
    diffusion_coefficient: float
    lifetime: float
    diffusion_length: float


@dataclass
class SimulationParameters:
    """
    Simulation parameters in the structure the engine reads.

    Attributes
    ----------
    profile : ProfileParameters
        The initial profile.
    physics : PhysicsParameters
        The nominal transport values.
    """
    profile: ProfileParameters
    physics: PhysicsParameters


def simulation_parameters_from_dict(
    parameters_dict: Dict[str, Any],
    diffusion_coefficient: float,
    lifetime: float,
    diffusion_length: float
) -> SimulationParameters:
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
    SimulationParameters
        Parameters object that run_monte_carlo_simulation takes.

    Examples
    --------
    >>> params_dict = {
    ...     'sigma^2_0': 1.0,
    ...     'amplitude_0': 1.0,
    ...     'mu_0': 0.0
    ... }
    >>> params = simulation_parameters_from_dict(params_dict, 0.5, 10.0, np.sqrt(5.0))
    >>> params.profile.sigma2_0
    1.0
    """
    profile = ProfileParameters(
        sigma2_0=parameters_dict['sigma^2_0'],
        amplitude_0=parameters_dict['amplitude_0'],
        mu_0=parameters_dict.get('mu_0', parameters_dict.get('mean_0', 0.0))  # Handle both naming conventions
    )

    physics = PhysicsParameters(
        diffusion_coefficient=diffusion_coefficient,
        lifetime=lifetime,
        diffusion_length=diffusion_length
    )

    return SimulationParameters(profile=profile, physics=physics)
