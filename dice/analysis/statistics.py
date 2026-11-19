"""
Statistical analysis functions for DICE simulation results.

This module provides functions for analyzing the precision and accuracy
of diffusion coefficient estimates from Monte Carlo simulations.
"""

import dataclasses

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from ..core.noise import nominal_cnr
from ..models.estimators import in_column_order
from ..models.results import MonteCarloOutput


def calculate_precision(
    estimates: np.ndarray,
    nominal_value: float,
    proximity_level: float = 0.1
) -> float:
    """
    Calculate the fraction of estimates within a proximity level of nominal.
    
    Parameters
    ----------
    estimates : np.ndarray
        Array of estimated values.
    nominal_value : float
        The true/nominal value.
    proximity_level : float
        Fractional proximity level (e.g., 0.1 for ±10%).
    
    Returns
    -------
    float
        Fraction of estimates within proximity (0-1).
    """
    if len(estimates) == 0 or nominal_value == 0:
        return 0.0
    
    # Calculate ratios
    ratios = estimates / nominal_value
    
    # Check which are within proximity
    lower_bound = 1 - proximity_level
    upper_bound = 1 + proximity_level
    within_proximity = np.logical_and(ratios >= lower_bound, ratios <= upper_bound)
    
    return np.mean(within_proximity)


def calculate_accuracy_metrics(
    estimates: np.ndarray,
    nominal_value: float
) -> Dict[str, float]:
    """
    Calculate accuracy metrics for estimates.
    
    Parameters
    ----------
    estimates : np.ndarray
        Array of estimated values.
    nominal_value : float
        The true/nominal value.
    
    Returns
    -------
    dict
        Dictionary containing accuracy metrics.
    """
    if len(estimates) == 0:
        return {
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'bias': np.nan,
            'relative_bias': np.nan,
            'rmse': np.nan,
            'relative_rmse': np.nan
        }
    
    mean_est = np.mean(estimates)
    median_est = np.median(estimates)
    std_est = np.std(estimates)
    
    # Calculate bias
    bias = mean_est - nominal_value
    relative_bias = bias / nominal_value if nominal_value != 0 else np.nan
    
    # Calculate RMSE
    errors = estimates - nominal_value
    rmse = np.sqrt(np.mean(errors**2))
    relative_rmse = rmse / nominal_value if nominal_value != 0 else np.nan
    
    return {
        'mean': mean_est,
        'median': median_est,
        'std': std_est,
        'bias': bias,
        'relative_bias': relative_bias,
        'rmse': rmse,
        'relative_rmse': relative_rmse
    }


def analyze_simulation_results(
    result: MonteCarloOutput,
    proximity_levels: Optional[List[float]] = None
) -> Dict:
    """
    Comprehensive analysis of simulation results.
    
    Parameters
    ----------
    result : MonteCarloOutput
        Simulation results to analyze.
    proximity_levels : List[float], optional
        List of proximity levels to test. If None, uses [0.05, 0.1, 0.2, 0.5].
    
    Returns
    -------
    dict
        Dictionary containing comprehensive analysis.
    """
    if proximity_levels is None:
        proximity_levels = [0.05, 0.1, 0.2, 0.5]

    # Diffusion coefficient estimates (slope / 2 in 1D) of each estimator that ran,
    # keyed by estimator in column order. An estimator whose every fit failed is kept.
    keys = [spec.key for spec in in_column_order(result.estimator_keys)]
    d_estimates = {key: result.d_estimates(key) for key in keys}
    nominal_d = result.parameters.physics.diffusion_coefficient

    # Calculate precision at different levels, keyed by percentage ('10%', '2.5%').
    # Formatted, not truncated: 0.29 * 100 is 28.999... in floating point.
    precision_results = {}
    for level in proximity_levels:
        precision_results[f'{level * 100:g}%'] = {
            key: calculate_precision(d_estimates[key], nominal_d, level) for key in keys
        }

    # Calculate accuracy metrics
    accuracy = {key: calculate_accuracy_metrics(d_estimates[key], nominal_d) for key in keys}

    # Analyze by noise level if multiple noise values
    noise_analysis = {}
    if len(result.noise_values) > 1:
        amplitude_0 = result.parameters.profile.amplitude_0
        for noise_val in result.noise_values:
            # Filter runs for this noise value
            noise_runs = [r for r in result.run_results if r.noise_sigma == noise_val]
            noise_result = dataclasses.replace(
                result, run_results=noise_runs, num_runs=len(noise_runs))
            noise_d = {key: noise_result.d_estimates(key) for key in keys}

            # Every estimator's precision, then every estimator's bias
            entry = {
                'cnr': nominal_cnr(amplitude_0, noise_val),
                'num_runs': len(noise_runs),
            }
            for key in keys:
                entry[f'{key}_precision_10%'] = calculate_precision(noise_d[key], nominal_d, 0.1)
            for key in keys:
                entry[f'{key}_bias'] = (
                    np.mean(noise_d[key]) - nominal_d if len(noise_d[key]) > 0 else np.nan)
            noise_analysis[f'noise_{noise_val}'] = entry

    statistics = {
        key: {
            'num_successful': len(d_estimates[key]),
            'num_failed': result.num_runs - len(d_estimates[key]),
            'success_rate': len(d_estimates[key]) / result.num_runs if result.num_runs > 0 else 0
        }
        for key in keys
    }

    return {
        'summary': {
            'total_runs': result.num_runs,
            'nominal_diffusion_coefficient': nominal_d,
            'nominal_lifetime': result.parameters.physics.lifetime,
            'nominal_diffusion_length': result.parameters.physics.diffusion_length,
        },
        'precision': precision_results,
        'accuracy': accuracy,
        'noise_analysis': noise_analysis,
        'statistics': statistics,
    }


def calculate_confidence_intervals(
    estimates: np.ndarray,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence intervals for estimates.
    
    Parameters
    ----------
    estimates : np.ndarray
        Array of estimates.
    confidence_level : float
        Confidence level (e.g., 0.95 for 95%).
    
    Returns
    -------
    tuple
        (lower_bound, upper_bound) of confidence interval.
    """
    if len(estimates) == 0:
        return (np.nan, np.nan)
    
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    return (
        np.percentile(estimates, lower_percentile),
        np.percentile(estimates, upper_percentile)
    )
