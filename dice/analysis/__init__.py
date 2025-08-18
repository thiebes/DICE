"""
Analysis modules for DICE simulations.

This package contains modules for running Monte Carlo simulations
and performing statistical analysis on the results.
"""

from .simulation import (
    run_single_simulation,
    run_monte_carlo_simulation,
    scan_runner_compatibility,
    calculate_wls_weights,
)

from .statistics import (
    calculate_precision,
    calculate_accuracy_metrics,
    estimates_precision,
    analyze_simulation_results,
    precision_counts,
    calculate_confidence_intervals,
    analyze_cnr_dependence,
)

__all__ = [
    # Simulation functions
    'run_single_simulation',
    'run_monte_carlo_simulation',
    'scan_runner_compatibility',
    'calculate_wls_weights',
    # Statistics functions
    'calculate_precision',
    'calculate_accuracy_metrics',
    'estimates_precision',
    'analyze_simulation_results',
    'precision_counts',
    'calculate_confidence_intervals',
    'analyze_cnr_dependence',
]