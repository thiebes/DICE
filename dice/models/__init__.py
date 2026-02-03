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
    SimulationResults,  # Deprecated alias
)
from .profiles import (
    ProfileData,
    TimeSeriesProfile,
    GaussianProfile,
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
    "SimulationResults",  # Deprecated alias
    # Profiles
    "ProfileData",
    "TimeSeriesProfile",
    "GaussianProfile",
]