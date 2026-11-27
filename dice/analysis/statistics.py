"""
Statistical analysis functions for DICE simulation results.

This module provides functions for analyzing the precision and accuracy
of diffusion coefficient estimates from Monte Carlo simulations.
"""

import dataclasses

import numpy as np
from typing import Dict, List, Tuple, Optional
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


def proportion_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Confidence interval of a proportion estimated from a count.

    The fraction of estimates within a proximity level comes from a finite
    number of runs: 1 of 20 is not a proportion of exactly 5%. The interval
    is the score interval for a binomial proportion (the Wilson interval).
    Unlike the interval from the standard error, it stays inside 0 to 1 and
    has a width when the count is 0 or equals the total.

    Parameters
    ----------
    successes : int
        Number of estimates within the proximity level.
    total : int
        Number of estimates.
    confidence : float
        Confidence level, 0.95 for a 95% interval.

    Returns
    -------
    tuple of float
        Lower and upper limit of the proportion (0-1); NaN for no estimates.
    """
    if total <= 0:
        return (np.nan, np.nan)
    # Imported here so that importing this module does not load statsmodels.
    from statsmodels.stats.proportion import proportion_confint

    lower, upper = proportion_confint(successes, total, alpha=1 - confidence, method='wilson')
    # At a count of none or of all, the interval ends exactly at 0 or at 1;
    # rounding must not move that end.
    return (0.0 if successes == 0 else float(lower), 1.0 if successes == total else float(upper))


def sample_standard_deviation(values: np.ndarray) -> float:
    """
    Sample standard deviation of the values (divisor n - 1).

    The estimates of a simulation are a sample of the estimates the
    simulated measurement can give, so their spread is estimated with the
    divisor n - 1. The summary, the plots and the GUI all report this value.

    Returns
    -------
    float
        The standard deviation; NaN for fewer than two values, which have none.
    """
    values = np.asarray(values, dtype=float)
    return float(np.std(values, ddof=1)) if values.size > 1 else float('nan')


# Resamples of the bootstrap interval of the standard deviation.
BOOTSTRAP_RESAMPLES = 10_000
# Largest number of resampled values held in memory at once.
BOOTSTRAP_BATCH_VALUES = 4_000_000


def standard_deviation_interval(
    values: np.ndarray, rng: np.random.Generator, confidence: float = 0.95,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> Tuple[float, float]:
    """
    Percentile bootstrap interval of the sample standard deviation.

    The estimates at a low CNR have heavy tails, for which the normal-theory
    interval s / sqrt(2(n - 1)) is too narrow, so the interval comes from the
    standard deviations of resamples drawn with replacement.

    Parameters
    ----------
    values : np.ndarray
        The estimates.
    rng : numpy.random.Generator
        Source of the resampling; a seeded one repeats the interval.
    confidence : float
        Coverage of the interval.
    resamples : int
        Number of resamples.

    Returns
    -------
    tuple of float
        Lower and upper limit; NaN for fewer than two values.
    """
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return float('nan'), float('nan')
    batch = max(1, min(resamples, BOOTSTRAP_BATCH_VALUES // values.size))
    spreads = np.empty(resamples)
    for start in range(0, resamples, batch):
        stop = min(start + batch, resamples)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        spreads[start:stop] = values[indices].std(axis=1, ddof=1)
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(spreads, [tail, 1 - tail])
    return float(lower), float(upper)


def calculate_accuracy_metrics(
    estimates: np.ndarray,
    nominal_value: float,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """
    Calculate accuracy metrics for estimates.
    
    Parameters
    ----------
    estimates : np.ndarray
        Array of estimated values.
    nominal_value : float
        The true/nominal value.
    rng : numpy.random.Generator, optional
        Source of the resampling of the interval of the standard deviation.
        Without it, the interval is not computed.
    
    Returns
    -------
    dict
        Dictionary containing accuracy metrics. 'std' is the sample standard
        deviation of the estimates (sample_standard_deviation), and
        'standard_error' the standard error of their mean ('std' divided by
        the square root of the number of estimates), which is also the
        standard error of the bias, because the nominal value is exact. Both
        are NaN for fewer than two estimates. With rng, 'std_interval' is
        the 95% bootstrap interval of 'std' (standard_deviation_interval),
        and 'relative_std_interval' the same divided by the nominal value.
    """
    if len(estimates) == 0:
        empty = {
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'standard_error': np.nan,
            'bias': np.nan,
            'relative_bias': np.nan,
            'rmse': np.nan,
            'relative_rmse': np.nan
        }
        if rng is not None:
            empty['std_interval'] = [np.nan, np.nan]
            empty['relative_std_interval'] = [np.nan, np.nan]
        return empty
    
    mean_est = np.mean(estimates)
    median_est = np.median(estimates)
    std_est = sample_standard_deviation(estimates)
    
    # Calculate bias
    bias = mean_est - nominal_value
    relative_bias = bias / nominal_value if nominal_value != 0 else np.nan
    
    # Calculate RMSE
    errors = estimates - nominal_value
    rmse = np.sqrt(np.mean(errors**2))
    relative_rmse = rmse / nominal_value if nominal_value != 0 else np.nan
    
    standard_error = std_est / np.sqrt(len(estimates))

    metrics = {
        'mean': mean_est,
        'median': median_est,
        'std': std_est,
        'standard_error': standard_error,
        'bias': bias,
        'relative_bias': relative_bias,
        'rmse': rmse,
        'relative_rmse': relative_rmse
    }
    if rng is not None:
        lower, upper = standard_deviation_interval(estimates, rng)
        metrics['std_interval'] = [lower, upper]
        metrics['relative_std_interval'] = (
            [lower / nominal_value, upper / nominal_value] if nominal_value != 0 else [np.nan, np.nan])
    return metrics


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
    # One stream for every bootstrap of this analysis, in a fixed order, so
    # that a seeded simulation repeats its intervals.
    rng = np.random.default_rng(result.analysis_seed)
    estimates = estimate_statistics(result, keys, proximity_levels, rng)

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
        'precision_intervals': estimates['precision_intervals'],
        'accuracy': estimates['accuracy'],
        'noise_analysis': noise_analysis,
        'statistics': estimates['statistics'],
    }
    if result.noise_range is not None:
        analysis['cnr_bins'] = analyze_cnr_bins(result, keys, proximity_levels, rng)
    return analysis


def estimate_statistics(
    result: MonteCarloOutput, keys: List[str], proximity_levels: List[float],
    rng: Optional[np.random.Generator] = None,
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
    rng : numpy.random.Generator, optional
        Source of the bootstrap interval of each standard deviation; without
        it, the intervals are not computed (calculate_accuracy_metrics).

    Returns
    -------
    dict
        'precision': the fraction of each estimator's estimates within each
        proximity level, keyed by percentage ('10%', '2.5%'), then estimator;
        'precision_intervals': for the same keys, the count behind each
        fraction ('count' of 'total' estimates) and its 95% interval
        ('lower', 'upper'; proportion_interval);
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
    precision_intervals = {}
    for level in proximity_levels:
        label = f'{level * 100:g}%'
        precision_results[label] = {
            key: calculate_precision(d_estimates[key], nominal_d, level) for key in keys
        }
        precision_intervals[label] = {
            key: precision_interval(precision_results[label][key], len(d_estimates[key])) for key in keys
        }

    # Calculate accuracy metrics
    accuracy = {key: calculate_accuracy_metrics(d_estimates[key], nominal_d, rng) for key in keys}

    statistics = {
        key: {
            'num_successful': len(d_estimates[key]),
            'num_failed': result.num_runs - len(d_estimates[key]),
            'success_rate': len(d_estimates[key]) / result.num_runs if result.num_runs > 0 else 0
        }
        for key in keys
    }

    return {'precision': precision_results, 'precision_intervals': precision_intervals,
            'accuracy': accuracy, 'statistics': statistics}


def precision_interval(fraction: float, total: int) -> Dict[str, float]:
    """
    The count behind a fraction of estimates, and the 95% interval of the fraction.

    Parameters
    ----------
    fraction : float
        Fraction of the estimates within a proximity level (calculate_precision).
    total : int
        Number of estimates the fraction was taken of.

    Returns
    -------
    dict
        'count' and 'total', and the 'lower' and 'upper' limit of the 95%
        interval of the fraction (proportion_interval).
    """
    count = int(round(fraction * total))
    lower, upper = proportion_interval(count, total)
    return {'count': count, 'total': total, 'lower': lower, 'upper': upper}


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
    result: MonteCarloOutput, keys: List[str], proximity_levels: List[float],
    rng: Optional[np.random.Generator] = None,
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
        estimate_statistics for the bin's runs. rng is passed on to
        estimate_statistics, bin by bin in this order.
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
            **estimate_statistics(bin_result, keys, proximity_levels, rng),
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
