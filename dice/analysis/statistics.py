"""
Statistical analysis functions for DICE simulation results.

This module provides functions for analyzing the precision and accuracy
of diffusion coefficient estimates from Monte Carlo simulations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
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


def estimates_precision(
    df: pd.DataFrame,
    proximity_level: float
) -> Dict:
    """
    Calculate precision of diffusion coefficient estimates.
    
    Legacy compatibility function that works with DataFrame format.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing simulation results.
    proximity_level : float
        Acceptable proximity level (e.g., 0.1 for ±10%).
    
    Returns
    -------
    dict
        Dictionary with precision percentages.
    """
    if len(df) == 0:
        raise ValueError("The DataFrame is empty.")

    # Work on a copy to avoid SettingWithCopyWarning
    df = df.copy()

    p_low = 1 - proximity_level
    p_high = 1 + proximity_level

    # Find diffusion coefficient columns by prefix (unit label may vary)
    def _find_column(prefix):
        matches = [c for c in df.columns if c.startswith(prefix)]
        return matches[0] if matches else None

    wls_col = _find_column('weighted fit diffusion coeff')
    ols_col = _find_column('unweighted fit diffusion coeff')
    nom_col = _find_column('nominal diffusion coeff [')

    # Calculate ratios for WLS and OLS
    if wls_col and nom_col:
        df['d_wls_over_d_nom'] = df[wls_col] / df[nom_col]
        wls_within = df['d_wls_over_d_nom'].between(p_low, p_high)
        wls_portion_pct = 100 * wls_within.sum() / len(df)
    else:
        wls_portion_pct = 0.0

    if ols_col and nom_col:
        df['d_ols_over_d_nom'] = df[ols_col] / df[nom_col]
        ols_within = df['d_ols_over_d_nom'].between(p_low, p_high)
        ols_portion_pct = 100 * ols_within.sum() / len(df)
    else:
        ols_portion_pct = 0.0
    
    return {
        '% fits within proximity': {
            'weighted fit': wls_portion_pct,
            'unweighted fit': ols_portion_pct,
        }
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
    # Extract OLS and WLS estimates
    ols_slopes = []
    wls_slopes = []
    
    for run in result.run_results:
        if run.ols_slope is not None:
            ols_slopes.append(run.ols_slope)
        if run.wls_slope is not None:
            wls_slopes.append(run.wls_slope)
    
    ols_slopes = np.array(ols_slopes)
    wls_slopes = np.array(wls_slopes)
    
    # Convert slopes to diffusion coefficients (divide by 2 for 1D)
    ols_d_estimates = ols_slopes / 2
    wls_d_estimates = wls_slopes / 2
    nominal_d = result.parameters.physics.diffusion_coefficient
    
    # Calculate precision at different levels
    precision_results = {}
    for level in proximity_levels:
        precision_results[f'{int(level*100)}%'] = {
            'ols': calculate_precision(ols_d_estimates, nominal_d, level),
            'wls': calculate_precision(wls_d_estimates, nominal_d, level)
        }
    
    # Calculate accuracy metrics
    ols_accuracy = calculate_accuracy_metrics(ols_d_estimates, nominal_d)
    wls_accuracy = calculate_accuracy_metrics(wls_d_estimates, nominal_d)
    
    # Analyze by noise level if multiple noise values
    noise_analysis = {}
    if len(result.noise_values) > 1:
        for noise_val in result.noise_values:
            # Filter runs for this noise value
            noise_runs = [r for r in result.run_results if r.noise_sigma == noise_val]
            
            noise_ols = np.array([r.ols_slope/2 for r in noise_runs 
                                 if r.ols_slope is not None])
            noise_wls = np.array([r.wls_slope/2 for r in noise_runs 
                                 if r.wls_slope is not None])
            
            noise_analysis[f'noise_{noise_val}'] = {
                'cnr': 1.0 / noise_val if noise_val > 0 else np.inf,
                'num_runs': len(noise_runs),
                'ols_precision_10%': calculate_precision(noise_ols, nominal_d, 0.1),
                'wls_precision_10%': calculate_precision(noise_wls, nominal_d, 0.1),
                'ols_bias': np.mean(noise_ols) - nominal_d if len(noise_ols) > 0 else np.nan,
                'wls_bias': np.mean(noise_wls) - nominal_d if len(noise_wls) > 0 else np.nan,
            }
    
    return {
        'summary': {
            'total_runs': result.num_runs,
            'nominal_diffusion_coefficient': nominal_d,
            'nominal_lifetime': result.parameters.physics.lifetime,
            'nominal_diffusion_length': result.parameters.physics.diffusion_length,
        },
        'precision': precision_results,
        'accuracy': {
            'ols': ols_accuracy,
            'wls': wls_accuracy
        },
        'noise_analysis': noise_analysis,
        'statistics': {
            'ols': {
                'num_successful': len(ols_d_estimates),
                'num_failed': result.num_runs - len(ols_d_estimates),
                'success_rate': len(ols_d_estimates) / result.num_runs if result.num_runs > 0 else 0
            },
            'wls': {
                'num_successful': len(wls_d_estimates),
                'num_failed': result.num_runs - len(wls_d_estimates),
                'success_rate': len(wls_d_estimates) / result.num_runs if result.num_runs > 0 else 0
            }
        }
    }


def precision_counts(
    df: pd.DataFrame,
    cnr_bins: np.ndarray,
    proximity_levels: List[float]
) -> Dict:
    """
    Count precision at different CNR bins and proximity levels.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with simulation results.
    cnr_bins : np.ndarray
        CNR bin edges.
    proximity_levels : List[float]
        Proximity levels to test.
    
    Returns
    -------
    dict
        Precision counts by CNR and proximity level.
    """
    results = {}
    
    # Bin the data by CNR
    df['cnr_bin'] = pd.cut(df['nominal CNR'], bins=cnr_bins)
    
    for proximity in proximity_levels:
        prox_key = f'proximity_{int(proximity*100)}%'
        results[prox_key] = {}
        
        for cnr_bin in df['cnr_bin'].unique():
            if pd.isna(cnr_bin):
                continue
            
            bin_data = df[df['cnr_bin'] == cnr_bin]
            if len(bin_data) == 0:
                continue
            
            # Calculate precision for this bin
            precision = estimates_precision(bin_data, proximity)
            
            results[prox_key][str(cnr_bin)] = {
                'count': len(bin_data),
                'ols_precision': precision['% fits within proximity']['unweighted fit'],
                'wls_precision': precision['% fits within proximity']['weighted fit']
            }
    
    return results


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


def analyze_cnr_dependence(
    df: pd.DataFrame,
    cnr_column: str = 'nominal CNR',
    d_est_column: str = 'weighted fit diffusion coeff [cm^2/s]',
    d_nom_column: str = 'nominal diffusion coeff [cm^2/s]'
) -> Dict:
    """
    Analyze how estimation accuracy depends on CNR.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with simulation results.
    cnr_column : str
        Column name for CNR values.
    d_est_column : str
        Column name for estimated diffusion coefficients.
    d_nom_column : str
        Column name for nominal diffusion coefficient.
    
    Returns
    -------
    dict
        Analysis of CNR dependence.
    """
    # Group by unique CNR values
    cnr_groups = df.groupby(cnr_column)
    
    results = {}
    for cnr, group in cnr_groups:
        d_estimates = group[d_est_column].values
        d_nominal = group[d_nom_column].iloc[0]
        
        # Calculate metrics for this CNR
        accuracy = calculate_accuracy_metrics(d_estimates, d_nominal)
        precision_10 = calculate_precision(d_estimates, d_nominal, 0.1)
        ci_lower, ci_upper = calculate_confidence_intervals(d_estimates / d_nominal)
        
        results[float(cnr)] = {
            'num_runs': len(group),
            'mean_ratio': accuracy['mean'] / d_nominal if d_nominal != 0 else np.nan,
            'std_ratio': accuracy['std'] / d_nominal if d_nominal != 0 else np.nan,
            'precision_10%': precision_10,
            'ci_95%_lower': ci_lower,
            'ci_95%_upper': ci_upper
        }
    
    return results