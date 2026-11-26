"""
Data models for DICE package.

This module provides dataclasses for organizing simulation parameters,
results, and profile data.
"""

from .parameters import (
    SimulationParameters,
    ProfileParameters,
    PhysicsParameters,
    NoiseRange,
    simulation_parameters_from_dict,
)
from .results import (
    MonteCarloOutput,
    RunResult,
)
from .profiles import (
    ProfileData,
    TimeSeriesProfile,
    GaussianProfile,
)
from .estimators import (
    DEFAULT_ESTIMATOR_KEYS,
    DISPLAY_ORDER,
    ESTIMATORS,
    EstimatorSpec,
    FrameFits,
    SlopeFit,
    get_estimator,
)

__all__ = [
    # Parameters
    "SimulationParameters",
    "ProfileParameters",
    "PhysicsParameters",
    "NoiseRange",
    "simulation_parameters_from_dict",
    # Results
    "MonteCarloOutput",
    "RunResult",
    # Profiles
    "ProfileData",
    "TimeSeriesProfile",
    "GaussianProfile",
    # Estimators
    "DEFAULT_ESTIMATOR_KEYS",
    "DISPLAY_ORDER",
    "ESTIMATORS",
    "EstimatorSpec",
    "FrameFits",
    "SlopeFit",
    "get_estimator",
]