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
from ..models.results import MonteCarloOutput, RunResult

# A simulation with a noise range is analyzed in bins of nominal CNR with equal
# numbers of runs: MAX_CNR_BINS bins, or fewer so that each has at least
# MIN_RUNS_PER_CNR_BIN runs (see cnr_bin_count).
MAX_CNR_BINS = 5
MIN_RUNS_PER_CNR_BIN = 20


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
        Dictionary containing comprehensive analysis. For a simulation with a
        noise range, the key 'cnr_bins' adds the statistics of each bin of
        nominal CNR (analyze_cnr_bins); the other statistics pool all runs.
    """
    if proximity_levels is None:
        proximity_levels = [0.05, 0.1, 0.2, 0.5]

    # Estimators that ran, in column order. An estimator whose every fit failed is kept.
    keys = [spec.key for spec in in_column_order(result.estimator_keys)]
    nominal_d = result.parameters.physics.diffusion_coefficient
    estimates = estimate_statistics(result, keys, proximity_levels)

    # Analyze by noise level if multiple fixed noise levels
    noise_analysis = {}
    if len(result.noise_values) > 1 and result.noise_range is None:
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

    analysis = {
        'summary': {
            'total_runs': result.num_runs,
            'nominal_diffusion_coefficient': nominal_d,
            'nominal_lifetime': result.parameters.physics.lifetime,
            'nominal_diffusion_length': result.parameters.physics.diffusion_length,
        },
        'precision': estimates['precision'],
        'accuracy': estimates['accuracy'],
        'noise_analysis': noise_analysis,
        'statistics': estimates['statistics'],
    }
    if result.noise_range is not None:
        analysis['cnr_bins'] = analyze_cnr_bins(result, keys, proximity_levels)
    return analysis


def estimate_statistics(
    result: MonteCarloOutput, keys: List[str], proximity_levels: List[float]
) -> Dict[str, Dict]:
    """
    Precision, accuracy, and fit success of the D estimates of each estimator.

    Parameters
    ----------
    result : MonteCarloOutput
        The runs to analyze.
    keys : List[str]
        Estimator keys, in the order of the output.
    proximity_levels : List[float]
        Fractional proximity levels for the precision (0.1 for ±10%).

    Returns
    -------
    dict
        'precision': the fraction of each estimator's estimates within each
        proximity level, keyed by percentage ('10%', '2.5%'), then estimator;
        'accuracy': calculate_accuracy_metrics of each estimator;
        'statistics': each estimator's number of successful and failed fits
        and its success rate.
    """
    # Diffusion coefficient estimates (slope / 2 in 1D) of each estimator
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

    statistics = {
        key: {
            'num_successful': len(d_estimates[key]),
            'num_failed': result.num_runs - len(d_estimates[key]),
            'success_rate': len(d_estimates[key]) / result.num_runs if result.num_runs > 0 else 0
        }
        for key in keys
    }

    return {'precision': precision_results, 'accuracy': accuracy, 'statistics': statistics}


def cnr_bin_count(run_count: int) -> int:
    """
    Number of CNR bins for a simulation with a noise range of run_count runs.

    MAX_CNR_BINS bins, or fewer so that each bin has at least
    MIN_RUNS_PER_CNR_BIN runs; one bin when there are fewer than
    2 * MIN_RUNS_PER_CNR_BIN runs.
    """
    return max(1, min(MAX_CNR_BINS, run_count // MIN_RUNS_PER_CNR_BIN))


def runs_by_cnr_bin(result: MonteCarloOutput) -> List[List[RunResult]]:
    """
    The runs grouped into cnr_bin_count bins of increasing nominal CNR.

    The runs are sorted by nominal CNR (amplitude_0 / noise sigma), runs of
    equal CNR in run order, and split into consecutive bins whose numbers of
    runs differ by at most one; the first bins take the extra runs. No runs
    give no bins.
    """
    amplitude_0 = result.parameters.profile.amplitude_0
    ordered = sorted(result.run_results, key=lambda run: nominal_cnr(amplitude_0, run.noise_sigma))
    if not ordered:
        return []
    return [
        [ordered[index] for index in indices]
        for indices in np.array_split(np.arange(len(ordered)), cnr_bin_count(len(ordered)))
    ]


def analyze_cnr_bins(
    result: MonteCarloOutput, keys: List[str], proximity_levels: List[float]
) -> List[Dict]:
    """
    Statistics of the D estimates in each bin of nominal CNR (runs_by_cnr_bin).

    For a simulation with a noise range, in which the statistics of all runs
    together would mix noise regimes.

    Parameters
    ----------
    result : MonteCarloOutput
        The runs to analyze.
    keys : List[str]
        Estimator keys, in the order of the output.
    proximity_levels : List[float]
        Fractional proximity levels for the precision (0.1 for ±10%).

    Returns
    -------
    list of dict
        One entry per bin, in order of increasing CNR: 'cnr_min' and
        'cnr_max', the lowest and highest nominal CNR of the bin's runs;
        'num_runs'; and the 'precision', 'accuracy', and 'statistics' of
        estimate_statistics for the bin's runs.
    """
    amplitude_0 = result.parameters.profile.amplitude_0
    bins = []
    for runs in runs_by_cnr_bin(result):
        bin_result = dataclasses.replace(
            result, run_results=runs, num_runs=len(runs), noise_values=[run.noise_sigma for run in runs])
        cnr = [nominal_cnr(amplitude_0, run.noise_sigma) for run in runs]
        bins.append({
            'cnr_min': min(cnr),
            'cnr_max': max(cnr),
            'num_runs': len(runs),
            **estimate_statistics(bin_result, keys, proximity_levels),
        })
    return bins


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
