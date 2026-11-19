"""
Data models for DICE package.

This module provides dataclasses for organizing simulation parameters,
results, and profile data.
"""

from .parameters import (
    SimulationParameters,
    GaussianParameters,
    NoiseParameters,
    SpatialParameters,
    TemporalParameters,
    OutputParameters,
)
from .results import (
    MonteCarloOutput,
    ProcessedSimulationResult,
    RunResult,
    StatisticalAnalysis,
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
    "GaussianParameters",
    "NoiseParameters",
    "SpatialParameters",
    "TemporalParameters",
    "OutputParameters",
    # Results
    "MonteCarloOutput",
    "ProcessedSimulationResult",
    "RunResult",
    "StatisticalAnalysis",
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