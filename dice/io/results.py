"""
Result export and file I/O for DICE simulations.

This module provides functions for exporting simulation results to various
formats including CSV, text summaries, and structured data files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, TextIO
import json
import csv

from ..models.results import SimulationResult, RunResult
from ..utils.converters import slope_to_diffusion_constant


def print_and_append(filename: Union[str, Path], message: str, print_to_console: bool = True) -> None:
    """
    Print message to console and append to file.
    
    Parameters
    ----------
    filename : str or Path
        File to append to.
    message : str
        Message to print and save.
    print_to_console : bool
        Whether to print to console.
    """
    if print_to_console:
        print(message)
    
    with open(filename, 'a') as f:
        f.write(message + '\n')


def export_collated_results(
    results: SimulationResult,
    filename: Union[str, Path],
    length_unit: str = 'micrometer',
    time_unit: str = 'nanosecond'
) -> pd.DataFrame:
    """
    Export collated results to CSV file.
    
    Parameters
    ----------
    results : SimulationResult
        Simulation results to export.
    filename : str or Path
        Output CSV filename.
    length_unit : str
        Unit of length used in simulation.
    time_unit : str
        Unit of time used in simulation.
    
    Returns
    -------
    pd.DataFrame
        DataFrame of collated results.
    """
    # Create list of dictionaries for DataFrame
    data_rows = []
    
    for run in results.run_results:
        row = {
            'run number': run.run_id,
            'nominal diffusion coeff': run.nominal_diffusion_coefficient,
            'nominal lifetime': run.nominal_lifetime,
            'nominal diffusion length': run.nominal_diffusion_length,
            'nominal CNR': 1.0 / run.noise_sigma if run.noise_sigma > 0 else np.inf,
            'estimated CNR': run.cnr_0_estimate,
            'nominal sigma^2_0': run.nominal_sigma2_0,
            'estimated sigma^2_0': run.estimated_sigma2_0,
        }
        
        # Add OLS results if available
        if run.ols_slope is not None:
            row.update({
                'unweighted fit diffusion slope': run.ols_slope,
                'unweighted fit diffusion slope stderr': run.ols_slope_stderr,
                'unweighted fit intercept': run.ols_intercept,
                'unweighted fit intercept stderr': run.ols_intercept_stderr,
                'unweighted fit diffusion coeff [cm^2/s]': slope_to_diffusion_constant(
                    run.ols_slope, length_unit, time_unit
                ),
                'unweighted fit diffusion stderr [cm^2/s]': slope_to_diffusion_constant(
                    run.ols_slope_stderr, length_unit, time_unit
                ) if run.ols_slope_stderr else None,
            })
        
        # Add WLS results if available
        if run.wls_slope is not None:
            row.update({
                'weighted fit diffusion slope': run.wls_slope,
                'weighted fit diffusion slope stderr': run.wls_slope_stderr,
                'weighted fit intercept': run.wls_intercept,
                'weighted fit intercept stderr': run.wls_intercept_stderr,
                'weighted fit diffusion coeff [cm^2/s]': slope_to_diffusion_constant(
                    run.wls_slope, length_unit, time_unit
                ),
                'weighted fit diffusion stderr [cm^2/s]': slope_to_diffusion_constant(
                    run.wls_slope_stderr, length_unit, time_unit
                ) if run.wls_slope_stderr else None,
            })
        
        # Add nominal diffusion coefficient in cm^2/s
        row['nominal diffusion coeff [cm^2/s]'] = slope_to_diffusion_constant(
            run.nominal_diffusion_coefficient * 2,  # Convert to slope
            length_unit, time_unit
        )
        
        data_rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data_rows)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    
    return df


def write_summary_file(
    results: SimulationResult,
    filename: Union[str, Path],
    parameters: Dict[str, Any],
    analysis: Optional[Dict[str, Any]] = None,
    length_unit: str = 'micrometer',
    time_unit: str = 'nanosecond'
) -> None:
    """
    Write comprehensive summary file.
    
    Parameters
    ----------
    results : SimulationResult
        Simulation results.
    filename : str or Path
        Output filename.
    parameters : dict
        Simulation parameters.
    analysis : dict, optional
        Analysis results.
    length_unit : str
        Unit of length.
    time_unit : str
        Unit of time.
    """
    with open(filename, 'w') as f:
        # Write header
        f.write("DICE Simulation Summary\n")
        f.write("=" * 50 + "\n\n")
        
        # Write parameters
        f.write("Simulation Parameters:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Number of runs:\n")
        f.write(f"    {results.num_runs}\n")
        f.write(f"Spatial width:\n")
        f.write(f"    {parameters.get('spatial width', 'N/A')} {length_unit}\n")
        f.write(f"Pixel width:\n")
        f.write(f"    {parameters.get('pixel width', 'N/A')} pixels\n")
        f.write(f"Time frames:\n")
        f.write(f"    {len(parameters.get('time series', []))} frames\n")
        f.write(f"Noise standard deviation values:\n")
        f.write(f"    {results.noise_values}\n")
        f.write(f"Initial CNR (Contrast-to-Noise Ratio):\n")
        f.write(f"    {1.0/results.noise_values[0] if results.noise_values[0] > 0 else 'inf':.3f}\n")
        f.write(f"Initial profile squared width (sigma^2):\n")
        f.write(f"    {results.parameters.profile.sigma2_0:.3f} {length_unit}^2\n")
        f.write(f"Nominal diffusion length:\n")
        f.write(f"    {results.parameters.physics.diffusion_length:.3f} {length_unit}\n")
        f.write(f"Nominal diffusion coefficient:\n")
        f.write(f"    {results.parameters.physics.diffusion_coefficient:.5f} {length_unit}^2/{time_unit}\n")
        f.write(f"Nominal lifetime:\n")
        f.write(f"    {results.parameters.physics.lifetime} {time_unit}\n")
        f.write("\n")
        
        # Write analysis results if available
        if analysis:
            f.write("Analysis Results:\n")
            f.write("-" * 30 + "\n")
            
            if 'precision' in analysis:
                f.write("Precision of Diffusion Coefficient Estimates:\n")
                for level, values in analysis['precision'].items():
                    f.write(f" Fraction of estimates within +/-{level} of nominal:\n")
                    f.write(f"    Ordinary (unweighted) Least Squares:\n")
                    f.write(f"        {values['ols']:.2%}\n")
                    f.write(f"    Weighted Least Squares:\n")
                    f.write(f"        {values['wls']:.2%}\n")
            
            if 'accuracy' in analysis:
                f.write("\nStatistical Accuracy of Diffusion Coefficient Estimates:\n")
                for method in ['ols', 'wls']:
                    if method in analysis['accuracy']:
                        metrics = analysis['accuracy'][method]
                        method_name = "Ordinary (unweighted) Least Squares" if method == 'ols' else "Weighted Least Squares"
                        f.write(f"  {method_name}:\n")
                        f.write(f"    Mean diffusion coefficient estimate:\n")
                        f.write(f"        {metrics.get('mean', 'N/A'):.5f}\n")
                        f.write(f"    Standard deviation:\n")
                        f.write(f"        {metrics.get('std', 'N/A'):.5f}\n")
                        f.write(f"    Mean error:\n")
                        f.write(f"        {metrics.get('bias', 'N/A'):.5f}\n")
                        f.write(f"    Root Mean Square Error:\n")
                        f.write(f"        {metrics.get('rmse', 'N/A'):.5f}\n")
            
            f.write("\n")
        
        # Write completion message
        f.write("Simulation completed successfully.\n")


def export_to_json(
    results: SimulationResult,
    filename: Union[str, Path],
    include_profile_data: bool = False
) -> None:
    """
    Export results to JSON format.
    
    Parameters
    ----------
    results : SimulationResult
        Simulation results.
    filename : str or Path
        Output JSON filename.
    include_profile_data : bool
        Whether to include profile data.
    """
    # Convert results to dictionary
    data = {
        'parameters': {
            'num_runs': results.num_runs,
            'noise_values': results.noise_values,
            'diffusion_coefficient': results.parameters.physics.diffusion_coefficient,
            'lifetime': results.parameters.physics.lifetime,
            'diffusion_length': results.parameters.physics.diffusion_length,
            'sigma2_0': results.parameters.profile.sigma2_0,
            'amplitude_0': results.parameters.profile.amplitude_0,
            'mu_0': results.parameters.profile.mu_0,
        },
        'runs': []
    }
    
    for run in results.run_results:
        run_data = {
            'run_id': run.run_id,
            'noise_sigma': run.noise_sigma,
            'cnr_0_estimate': run.cnr_0_estimate,
            'estimated_sigma2_0': run.estimated_sigma2_0,
            'ols_slope': run.ols_slope,
            'ols_slope_stderr': run.ols_slope_stderr,
            'wls_slope': run.wls_slope,
            'wls_slope_stderr': run.wls_slope_stderr,
        }
        
        if include_profile_data and hasattr(run, 'nominal_profiles'):
            run_data['nominal_profiles'] = run.nominal_profiles.tolist() if isinstance(run.nominal_profiles, np.ndarray) else run.nominal_profiles
            run_data['noisy_profiles'] = run.noisy_profiles.tolist() if isinstance(run.noisy_profiles, np.ndarray) else run.noisy_profiles
        
        data['runs'].append(run_data)
    
    # Write JSON file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def load_results_csv(filename: Union[str, Path]) -> pd.DataFrame:
    """
    Load results from CSV file.
    
    Parameters
    ----------
    filename : str or Path
        CSV file to load.
    
    Returns
    -------
    pd.DataFrame
        Loaded results.
    """
    return pd.read_csv(filename)


def load_multiple_results(
    directory: Union[str, Path],
    pattern: str = "*.csv"
) -> pd.DataFrame:
    """
    Load and concatenate multiple result files.
    
    Parameters
    ----------
    directory : str or Path
        Directory containing result files.
    pattern : str
        Glob pattern for matching files.
    
    Returns
    -------
    pd.DataFrame
        Concatenated results.
    """
    directory = Path(directory)
    files = list(directory.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {directory}")
    
    # Load all files
    dfs = []
    for file in files:
        try:
            df = pd.read_csv(file)
            df['source_file'] = file.name
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not dfs:
        raise ValueError("No valid CSV files could be loaded")
    
    # Concatenate all DataFrames
    return pd.concat(dfs, ignore_index=True)


def export_legacy_format(
    results: SimulationResult,
    result_dictionary: Dict[str, Any],
    csv_filename: Union[str, Path],
    summary_filename: Union[str, Path],
    length_unit: str = 'micrometer',
    time_unit: str = 'nanosecond'
) -> None:
    """
    Export results in legacy DICE format for backward compatibility.
    
    Parameters
    ----------
    results : SimulationResult
        Modern format results.
    result_dictionary : dict
        Legacy format result dictionary.
    csv_filename : str or Path
        Output CSV filename.
    summary_filename : str or Path
        Output summary filename.
    length_unit : str
        Unit of length.
    time_unit : str
        Unit of time.
    """
    # Export CSV using the modern method
    df = export_collated_results(results, csv_filename, length_unit, time_unit)
    
    # Write legacy summary format
    with open(summary_filename, 'w') as f:
        f.write(f"Running {results.num_runs} simulations with the following parameters (rounded):\n")
        f.write("\n")
        
        # Extract parameters from result_dictionary for legacy format
        params = result_dictionary.get('parameters', {})
        f.write(f"Spatial width: {params.get('spatial width', 'N/A')} {length_unit}\n")
        f.write(f"Pixel width: {params.get('pixel width', 'N/A')} pixels\n")
        f.write(f"Number of time frames: {len(params.get('time series', []))} frames\n")
        f.write(f"Noise stdev: {results.noise_values[0]:.3f}\n")
        f.write(f"Initial contrast-to-noise ratio (CNR): {1.0/results.noise_values[0]:.3f}\n")
        f.write(f"Initial profile sigma^2: {results.parameters.profile.sigma2_0:.3f} {length_unit}^2\n")
        f.write(f"Nominal diffusion length: {results.parameters.physics.diffusion_length:.3f} {length_unit}\n")
        f.write(f"Nominal diffusion coeff: {results.parameters.physics.diffusion_coefficient:.5f} {length_unit}^2 per {time_unit}\n")
        f.write(f"Nominal lifetime: {results.parameters.physics.lifetime} {time_unit}\n")
        f.write("\n")
        
        # Add analysis results if available
        if 'analysis' in result_dictionary:
            analysis = result_dictionary['analysis']
            if '% fits within proximity' in analysis:
                proximity = params.get('proximity level', 0.1)
                f.write(f"Portion of fits where D_estimate / D_nominal = 1 ± {proximity}:\n")
                f.write(f"-- Unweighted fit: {analysis['% fits within proximity']['unweighted fit']:.2f}\n")
                f.write(f"-- Weighted fit: {analysis['% fits within proximity']['weighted fit']:.2f}\n")
                f.write("\n")
        
        f.write("Simulation completed. Collating results.\n")
        f.write("\n")
        f.write("Exporting result data and histogram.\n")
        f.write(f"-- Summary file: {summary_filename}\n")
        f.write(f"-- Collated CSV file: {csv_filename}\n")
        f.write("\n")
        f.write("Done!\n")