"""
Result export and file I/O for DICE simulations.

This module provides functions for exporting simulation results to various
formats including CSV, text summaries, and structured data files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, TextIO
import json
import csv

from ..core.noise import nominal_cnr
from ..models.estimators import ESTIMATORS, in_display_order
from ..models.results import MonteCarloOutput, RunResult
from ..models.parameters import OutputUnitPreferences
from ..utils.converters import slope_to_diffusion_constant
from ..utils.units import (
    convert_diffusion_coefficient, length_abbreviation, time_abbreviation,
    diffusion_coefficient_label, OUTPUT_DIFFUSION_LENGTH_UNIT_KEY, OUTPUT_DIFFUSION_TIME_UNIT_KEY,
)


def output_unit_preferences(parameters: Dict[str, Any]) -> OutputUnitPreferences:
    """
    Output units that a parameters dictionary sets.

    Only the unit of the diffusion coefficients can be set, with the keys
    'output_diffusion_length_unit' and 'output_diffusion_time_unit';
    dice.utils.units.validate_output_units rejects the other output unit keys.
    """
    return OutputUnitPreferences(
        diffusion_length=parameters.get(OUTPUT_DIFFUSION_LENGTH_UNIT_KEY),
        diffusion_time=parameters.get(OUTPUT_DIFFUSION_TIME_UNIT_KEY),
    )


def summary_diffusion_units(
    length_unit: str, time_unit: str, output_units: Optional[OutputUnitPreferences] = None,
) -> Tuple[str, str]:
    """Length and time units of the summary's diffusion coefficients: the output units, else the global units."""
    if output_units is None:
        return length_unit, time_unit
    return output_units.diffusion_length or length_unit, output_units.diffusion_time or time_unit


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
    
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def export_collated_results(
    results: MonteCarloOutput,
    filename: Union[str, Path],
    length_unit: str = 'micrometer',
    time_unit: str = 'nanosecond',
    output_units: Optional[OutputUnitPreferences] = None,
) -> pd.DataFrame:
    """
    Export collated results to CSV file.

    Parameters
    ----------
    results : MonteCarloOutput
        Simulation results to export.
    filename : str or Path
        Output CSV filename.
    length_unit : str
        Unit of length used in simulation.
    time_unit : str
        Unit of time used in simulation.
    output_units : OutputUnitPreferences, optional
        Preferred units for output columns. When None, diffusion
        coefficients are exported in cm^2/s (default behavior).

    Returns
    -------
    pd.DataFrame
        DataFrame of collated results.
    """
    # Determine output units for diffusion coefficient
    out_l = 'centimeter'
    out_t = 'second'
    if output_units is not None:
        if output_units.diffusion_length is not None:
            out_l = output_units.diffusion_length
        if output_units.diffusion_time is not None:
            out_t = output_units.diffusion_time

    # Build column label for diffusion coefficient
    d_unit_label = diffusion_coefficient_label(out_l, out_t)

    def _convert_slope_to_d(slope):
        """Convert MSD slope in simulation units to D in output units."""
        d_sim = slope / 2
        return convert_diffusion_coefficient(
            d_sim, length_unit, time_unit, out_l, out_t
        )

    amplitude_0 = results.parameters.profile.amplitude_0

    # Create list of dictionaries for DataFrame
    data_rows = []

    for run in results.run_results:
        row = {
            'run number': run.run_id,
            'nominal diffusion coeff': run.nominal_diffusion_coefficient,
            'nominal lifetime': run.nominal_lifetime,
            'nominal diffusion length': run.nominal_diffusion_length,
            'nominal CNR': nominal_cnr(amplitude_0, run.noise_sigma),
            'estimated CNR': run.cnr_0_estimate,
            'nominal sigma^2_0': run.nominal_sigma2_0,
            'estimated sigma^2_0': run.estimated_sigma2_0,
        }

        # One block of columns per estimator with a fit for this run, in registry order
        for spec in ESTIMATORS:
            fit = run.slope_fit(spec.key)
            if fit is not None:
                label = spec.csv_label
                row.update({
                    f'{label} diffusion slope': fit.slope,
                    f'{label} diffusion slope stderr': fit.slope_stderr,
                    f'{label} intercept': fit.intercept,
                    f'{label} intercept stderr': fit.intercept_stderr,
                    f'{label} diffusion coeff [{d_unit_label}]': _convert_slope_to_d(fit.slope),
                    f'{label} diffusion stderr [{d_unit_label}]': _convert_slope_to_d(fit.slope_stderr) if fit.slope_stderr else None,
                })

        # Add nominal diffusion coefficient in output units
        row[f'nominal diffusion coeff [{d_unit_label}]'] = convert_diffusion_coefficient(
            run.nominal_diffusion_coefficient, length_unit, time_unit, out_l, out_t
        )

        data_rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(data_rows)

    # Save to CSV
    df.to_csv(filename, index=False)

    return df


def write_summary_file(
    results: MonteCarloOutput,
    filename: Union[str, Path],
    parameters: Dict[str, Any],
    analysis: Optional[Dict[str, Any]] = None,
    length_unit: str = 'micrometer',
    time_unit: str = 'nanosecond',
    output_units: Optional[OutputUnitPreferences] = None,
) -> None:
    """
    Write comprehensive summary file.

    Parameters
    ----------
    results : MonteCarloOutput
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
    output_units : OutputUnitPreferences, optional
        Unit of the diffusion coefficient values (its diffusion_length and
        diffusion_time fields). When None, they are in the simulation units.
        Lengths and times are always in the simulation units.
    """
    # The unit of every diffusion coefficient value: the nominal value and the
    # statistics of the estimates, except the unitless relative standard error.
    d_length_unit, d_time_unit = summary_diffusion_units(length_unit, time_unit, output_units)
    d_unit = f"{d_length_unit}^2/{d_time_unit}"

    def in_d_unit(value: float) -> float:
        """A diffusion coefficient in simulation units, converted to d_unit."""
        return convert_diffusion_coefficient(value, length_unit, time_unit, d_length_unit, d_time_unit)

    with open(filename, 'w', encoding='utf-8') as f:
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
        time_series = parameters.get('time series', [])
        f.write(f"Time frames:\n")
        f.write(f"    {len(time_series)} frames\n")
        if len(time_series) > 0:
            f.write(f"Time range:\n")
            f.write(f"    {time_series[0]:.3f} to {time_series[-1]:.3f} {time_unit}\n")
        f.write(f"Noise standard deviation values:\n")
        f.write(f"    {results.noise_values}\n")
        f.write(f"Initial CNR (Contrast-to-Noise Ratio):\n")
        f.write(f"    {nominal_cnr(results.parameters.profile.amplitude_0, results.noise_values[0]):.3f}\n")
        f.write(f"Initial profile squared width (sigma^2):\n")
        f.write(f"    {results.parameters.profile.sigma2_0:.3f} {length_unit}^2\n")
        f.write(f"Nominal diffusion length:\n")
        f.write(f"    {results.parameters.physics.diffusion_length:.3f} {length_unit}\n")
        f.write(f"Nominal diffusion coefficient:\n")
        f.write(f"    {in_d_unit(results.parameters.physics.diffusion_coefficient):.5f} {d_unit}\n")
        f.write(f"Nominal lifetime:\n")
        f.write(f"    {results.parameters.physics.lifetime} {time_unit}\n")
        f.write("\n")
        
        # Write analysis results if available
        if analysis:
            f.write("Analysis Results:\n")
            f.write("-" * 30 + "\n")

            if 'accuracy' in analysis:
                f.write("Diffusion Coefficient Estimate Statistics:\n")
                for spec in in_display_order(analysis['accuracy']):
                    metrics = analysis['accuracy'][spec.key]
                    f.write(f"  {spec.summary_name}:\n")
                    f.write(f"    Standard deviation:\n")
                    f.write(f"        {in_d_unit(metrics.get('std', np.nan)):.5f} {d_unit}\n")
                    nominal_d = analysis.get('summary', {}).get('nominal_diffusion_coefficient', None)
                    if nominal_d and nominal_d != 0:
                        rse = metrics.get('std', 0) / nominal_d
                        f.write(f"    Relative standard error:\n")
                        f.write(f"        {rse:.3f}\n")
                    f.write(f"    Mean estimate:\n")
                    f.write(f"        {in_d_unit(metrics.get('mean', np.nan)):.5f} {d_unit}\n")
                    f.write(f"    Bias (mean - nominal):\n")
                    f.write(f"        {in_d_unit(metrics.get('bias', np.nan)):.5f} {d_unit}\n")
                    f.write(f"    Root Mean Square Error:\n")
                    f.write(f"        {in_d_unit(metrics.get('rmse', np.nan)):.5f} {d_unit}\n")

            if 'precision' in analysis:
                f.write("\nPrecision at standard thresholds:\n")
                for level, values in analysis['precision'].items():
                    f.write(f"  Fraction within ±{level} of nominal:\n")
                    # One line in display order, e.g. "    WLS: 19.00%    OLS: 15.00%"
                    f.write("".join(
                        f"    {spec.short_name}: {values[spec.key]:.2%}"
                        for spec in in_display_order(values)
                    ) + "\n")

            f.write("\n")
        
        # Write completion message
        f.write("Simulation completed successfully.\n")


def export_to_json(
    results: MonteCarloOutput,
    filename: Union[str, Path],
    include_profile_data: bool = False
) -> None:
    """
    Export results to JSON format.
    
    Parameters
    ----------
    results : MonteCarloOutput
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
    results: MonteCarloOutput,
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
    results : MonteCarloOutput
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
    with open(summary_filename, 'w', encoding='utf-8') as f:
        f.write(f"Running {results.num_runs} simulations with the following parameters (rounded):\n")
        f.write("\n")
        
        # Extract parameters from result_dictionary for legacy format
        params = result_dictionary.get('parameters', {})
        f.write(f"Spatial width: {params.get('spatial width', 'N/A')} {length_unit}\n")
        f.write(f"Pixel width: {params.get('pixel width', 'N/A')} pixels\n")
        f.write(f"Number of time frames: {len(params.get('time series', []))} frames\n")
        f.write(f"Noise stdev: {results.noise_values[0]:.3f}\n")
        f.write(f"Initial contrast-to-noise ratio (CNR): {nominal_cnr(results.parameters.profile.amplitude_0, results.noise_values[0]):.3f}\n")
        f.write(f"Initial profile sigma^2: {results.parameters.profile.sigma2_0:.3f} {length_unit}^2\n")
        f.write(f"Nominal diffusion length: {results.parameters.physics.diffusion_length:.3f} {length_unit}\n")
        f.write(f"Nominal diffusion coeff: {results.parameters.physics.diffusion_coefficient:.5f} {length_unit}^2 per {time_unit}\n")
        f.write(f"Nominal lifetime: {results.parameters.physics.lifetime} {time_unit}\n")
        f.write("\n")
        
        # Add analysis results if available
        if 'analysis' in result_dictionary:
            analysis = result_dictionary['analysis']
            if '% fits within proximity' in analysis:
                for level_pct, values in analysis['% fits within proximity'].items():
                    if isinstance(values, dict):
                        f.write(f"Portion of fits where D_estimate / D_nominal = 1 ± {level_pct}:\n")
                        f.write(f"-- Unweighted fit: {values.get('unweighted fit', 0):.2f}\n")
                        f.write(f"-- Weighted fit: {values.get('weighted fit', 0):.2f}\n")
                f.write("\n")
        
        f.write("Simulation completed. Collating results.\n")
        f.write("\n")
        f.write("Exporting result data and histogram.\n")
        f.write(f"-- Summary file: {summary_filename}\n")
        f.write(f"-- Collated CSV file: {csv_filename}\n")
        f.write("\n")
        f.write("Done!\n")