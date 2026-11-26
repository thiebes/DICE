"""
Result export and file I/O for DICE simulations.

This module provides functions for exporting simulation results to CSV
files and text summaries, and for loading result CSV files.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence, Tuple, Union, TextIO
import csv

from ..core.noise import nominal_cnr
from ..models.estimators import ESTIMATORS, in_display_order
from ..models.results import MonteCarloOutput, RunResult
from ..models.parameters import OutputUnitPreferences
from ..utils.converters import slope_to_diffusion_constant
from ..utils.units import (
    convert_diffusion_coefficient, length_abbreviation, time_abbreviation,
    diffusion_coefficient_label, global_length_unit, global_time_unit,
    OUTPUT_DIFFUSION_LENGTH_UNIT_KEY, OUTPUT_DIFFUSION_TIME_UNIT_KEY,
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


def format_quantity(value: float, decimals: int, reference: Optional[float] = None) -> str:
    """
    Format a number for the summary with a fixed number of decimals, or in
    scientific notation when fixed decimals would show fewer than two
    significant digits of `reference` (default: the value itself).

    Passing one reference for a group of values (for example the nominal
    diffusion coefficient for all diffusion-coefficient lines) keeps the group
    in one format. Zero, infinite and NaN references keep the fixed format.
    """
    reference = value if reference is None else reference
    if np.isfinite(reference) and reference != 0 and abs(reference) < 10.0 ** (1 - decimals):
        return f"{value:.{decimals}e}"
    return f"{value:.{decimals}f}"


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


# Results-file column of the nominal diffusion coefficient, before its unit.
NOMINAL_D_COLUMN = 'nominal diffusion coeff'


def results_column(columns: Sequence[str], name: str) -> str:
    """
    The full name of the results-file column that holds `name`.

    A column carries its unit in its name, for example 'nominal lifetime
    [ns]' or 'weighted fit diffusion slope (2D) [μm²/ns]', so the full name
    depends on the units of the simulation. This function finds it from the
    name without the unit, which keeps a script independent of the units:

        slope = df[results_column(df.columns, 'weighted fit diffusion slope')]

    A results file of DICE 1.2.0, whose columns have no unit, is read with
    the same call.

    Parameters
    ----------
    columns : sequence of str
        The column names of a results file.
    name : str
        The column name without '(2D)' and without the unit in brackets.

    Returns
    -------
    str
        The matching column. The nominal diffusion coefficient can have two
        columns, in the units of the simulation and in the output units;
        the first, in the units of the simulation, is returned.

    Raises
    ------
    KeyError
        If no column holds `name`.
    """
    for column in columns:
        if column == name or column.startswith((f'{name} [', f'{name} (2D) [')):
            return column
    raise KeyError(f"The results file has no column '{name}'. Its columns are: {list(columns)}")


def column_unit(column: str) -> Optional[str]:
    """The unit in the brackets at the end of a results-file column name; None if it names no unit."""
    if column.endswith(']') and ' [' in column:
        return column[column.rindex(' [') + 2:-1]
    return None


def export_collated_results(
    results: MonteCarloOutput,
    filename: Union[str, Path],
    length_unit: str,
    time_unit: str,
    output_units: Optional[OutputUnitPreferences] = None,
) -> pd.DataFrame:
    """
    Export collated results to CSV file.

    One row per run. A column whose value has a unit names it in brackets:
    the units of the simulation for the nominal values, the variances and
    the fits of variance against time, and the output units for the
    diffusion coefficients. The slope of that fit is twice the diffusion
    coefficient, which its column name states ('... diffusion slope (2D)').
    results_column finds a column from its name without the unit.

    For a simulation with a noise range, the column 'noise standard
    deviation', before 'nominal CNR', holds each run's noise level, in the
    units of the signal.

    Parameters
    ----------
    results : MonteCarloOutput
        Simulation results to export.
    filename : str or Path
        Output CSV filename.
    length_unit : str
        Unit of length the results are in: the global length unit of the
        simulation. The results do not carry their units, so it has no default.
    time_unit : str
        Unit of time the results are in: the global time unit of the simulation.
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

    # Unit labels of the columns: the output unit of the diffusion
    # coefficients, and the units of the simulation for everything else.
    d_unit_label = diffusion_coefficient_label(out_l, out_t)
    simulation_d_label = diffusion_coefficient_label(length_unit, time_unit)
    length_label = length_abbreviation(length_unit)
    time_label = time_abbreviation(time_unit)
    area_label = f"{length_label}\u00b2"

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
            f'{NOMINAL_D_COLUMN} [{simulation_d_label}]': run.nominal_diffusion_coefficient,
            f'nominal lifetime [{time_label}]': run.nominal_lifetime,
            f'nominal diffusion length [{length_label}]': run.nominal_diffusion_length,
        }
        # With a noise range each run has its own noise level, so it gets a column.
        if results.noise_range is not None:
            row['noise standard deviation'] = run.noise_sigma
        row.update({
            'nominal CNR': nominal_cnr(amplitude_0, run.noise_sigma),
            'estimated CNR': run.cnr_0_estimate,
            f'nominal sigma^2_0 [{area_label}]': run.nominal_sigma2_0,
            f'estimated sigma^2_0 [{area_label}]': run.estimated_sigma2_0,
        })

        # One block of columns per estimator with a fit for this run, in registry order
        for spec in ESTIMATORS:
            fit = run.slope_fit(spec.key)
            if fit is not None:
                label = spec.csv_label
                row.update({
                    f'{label} diffusion slope (2D) [{simulation_d_label}]': fit.slope,
                    f'{label} diffusion slope stderr [{simulation_d_label}]': fit.slope_stderr,
                    f'{label} intercept [{area_label}]': fit.intercept,
                    f'{label} intercept stderr [{area_label}]': fit.intercept_stderr,
                    f'{label} diffusion coeff [{d_unit_label}]': _convert_slope_to_d(fit.slope),
                    f'{label} diffusion stderr [{d_unit_label}]': _convert_slope_to_d(fit.slope_stderr) if fit.slope_stderr else None,
                })

        # The nominal diffusion coefficient in the output units; when these
        # are the units of the simulation, the first column already holds it.
        if d_unit_label != simulation_d_label:
            row[f'{NOMINAL_D_COLUMN} [{d_unit_label}]'] = convert_diffusion_coefficient(
                run.nominal_diffusion_coefficient, length_unit, time_unit, out_l, out_t
            )

        data_rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(data_rows)

    # Save to CSV, as UTF-8 with a byte-order mark: the column names hold
    # 'μ' and '²', which a spreadsheet program reads correctly only with it.
    df.to_csv(filename, index=False, encoding='utf-8-sig')

    return df


def fraction_text(fraction: float, interval: Optional[Dict[str, float]] = None,
                  interval_label: str = "") -> str:
    """
    A fraction of the estimates as text, with its count and 95% interval when they are known.

    Parameters
    ----------
    fraction : float
        Fraction of the estimates within a proximity level (0-1).
    interval : dict, optional
        'count', 'total', 'lower' and 'upper' of the fraction
        (dice.analysis.statistics.precision_interval).

    Returns
    -------
    str
        For example "19.0% (19 of 100; 12.5% to 27.8%)", or "19.0%" without
        an interval, or "no estimates" when the fraction is of none. One
        decimal, because no simulation of practical size fixes the fraction
        more closely.
    """
    if interval is None:
        return f"{fraction:.1%}"
    if not interval['total']:
        return "no estimates"
    return (f"{fraction:.1%} ({interval['count']} of {interval['total']}; "
            f"{interval_label}{interval['lower']:.1%} to {interval['upper']:.1%})")


def write_summary_file(
    results: MonteCarloOutput,
    filename: Union[str, Path],
    parameters: Dict[str, Any],
    analysis: Optional[Dict[str, Any]] = None,
    length_unit: Optional[str] = None,
    time_unit: Optional[str] = None,
    output_units: Optional[OutputUnitPreferences] = None,
) -> None:
    """
    Write comprehensive summary file.

    For a simulation with a noise range, the summary reports the number of
    noise values drawn, their range and the range of the initial CNR, and the
    statistics of each bin of initial CNR (analysis['cnr_bins']) in place of
    the statistics of all runs together.

    Parameters
    ----------
    results : MonteCarloOutput
        Simulation results.
    filename : str or Path
        Output filename.
    parameters : dict
        Simulation parameters.
    analysis : dict, optional
        Analysis results, as from analyze_simulation_results.
    length_unit : str, optional
        Unit of length the results are in. When None, the global length unit
        of `parameters`.
    time_unit : str, optional
        Unit of time the results are in. When None, the global time unit of
        `parameters`.
    output_units : OutputUnitPreferences, optional
        Unit of the diffusion coefficient values (its diffusion_length and
        diffusion_time fields). When None, they are in the simulation units.
        Lengths and times are always in the simulation units.
    """
    if length_unit is None:
        length_unit = global_length_unit(parameters)
    if time_unit is None:
        time_unit = global_time_unit(parameters)

    # The unit of every diffusion coefficient value: the nominal value and the
    # statistics of the estimates, except the unitless relative standard deviation.
    d_length_unit, d_time_unit = summary_diffusion_units(length_unit, time_unit, output_units)
    d_unit = f"{d_length_unit}^2/{d_time_unit}"

    def in_d_unit(value: float) -> float:
        """A diffusion coefficient in simulation units, converted to d_unit."""
        return convert_diffusion_coefficient(value, length_unit, time_unit, d_length_unit, d_time_unit)

    nominal_d_in_unit = in_d_unit(results.parameters.physics.diffusion_coefficient)

    def d_text(value: float) -> str:
        """A diffusion coefficient in d_unit, in the format the nominal value sets."""
        return format_quantity(in_d_unit(value), 5, reference=nominal_d_in_unit)

    amplitude_0 = results.parameters.profile.amplitude_0
    noise_range = results.noise_range
    # The lowest CNR of a noise range sets the format of all its CNR values.
    lowest_cnr = nominal_cnr(amplitude_0, noise_range.high) if noise_range is not None else None

    def cnr_text(value: float) -> str:
        """An initial CNR of a simulation with a noise range, in the format its lowest CNR sets."""
        return format_quantity(value, 3, reference=lowest_cnr)

    def write_noise_range(f: TextIO) -> None:
        """The noise sigma and initial CNR drawn for the runs, and the range they were drawn from."""
        drawn = results.noise_values

        def sigma_text(value: float) -> str:
            return format_quantity(value, 4, reference=noise_range.low)

        # CNR is amplitude_0 / sigma, so the highest noise gives the lowest CNR.
        distribution = 'uniform in log10 of the CNR' if noise_range.logarithmic else 'uniform in CNR'
        f.write("Noise standard deviation, drawn for each run:\n")
        f.write(f"    {len(drawn)} values from {sigma_text(min(drawn))} to {sigma_text(max(drawn))}"
                f" (range {sigma_text(noise_range.low)} to {sigma_text(noise_range.high)})\n")
        f.write("Initial CNR (Contrast-to-Noise Ratio), drawn for each run:\n")
        f.write(f"    {cnr_text(nominal_cnr(amplitude_0, max(drawn)))} to "
                f"{cnr_text(nominal_cnr(amplitude_0, min(drawn)))} (range {cnr_text(lowest_cnr)} to "
                f"{cnr_text(nominal_cnr(amplitude_0, noise_range.low))}, {distribution})\n")

    nominal_d = (analysis or {}).get('summary', {}).get('nominal_diffusion_coefficient', None)

    def write_estimate_statistics(f: TextIO, statistics: Dict[str, Any], indent: str) -> None:
        """The accuracy and precision of each estimator's D estimates, each line after `indent`."""
        if 'accuracy' in statistics:
            f.write(f"{indent}Diffusion Coefficient Estimate Statistics:\n")
            for spec in in_display_order(statistics['accuracy']):
                metrics = statistics['accuracy'][spec.key]
                f.write(f"{indent}  {spec.summary_name}:\n")
                if 'statistics' in statistics:
                    fits = statistics['statistics'][spec.key]
                    runs = fits['num_successful'] + fits['num_failed']
                    f.write(f"{indent}    Estimates:\n")
                    f.write(f"{indent}        {fits['num_successful']} of {runs} runs "
                            f"({fits['num_failed']} without an estimate)\n")
                f.write(f"{indent}    Standard deviation:\n")
                f.write(f"{indent}        {d_text(metrics['std'])} {d_unit}\n")
                if nominal_d and nominal_d != 0:
                    relative_standard_deviation = metrics['std'] / nominal_d
                    f.write(f"{indent}    Relative standard deviation:\n")
                    f.write(f"{indent}        {relative_standard_deviation:.3f}\n")
                f.write(f"{indent}    Mean estimate:\n")
                f.write(f"{indent}        {d_text(metrics['mean'])} {d_unit}\n")
                if 'standard_error' in metrics:
                    f.write(f"{indent}    Standard error of the mean and of the bias:\n")
                    f.write(f"{indent}        {d_text(metrics['standard_error'])} {d_unit}\n")
                f.write(f"{indent}    Bias (mean - nominal):\n")
                f.write(f"{indent}        {d_text(metrics['bias'])} {d_unit}\n")
                if nominal_d and nominal_d != 0:
                    # The form in which the 2024 paper reports the bias (its Fig. 7);
                    # the Analysis tab of the GUI shows it as "Mean D_est / D_nom".
                    f.write(f"{indent}    Mean estimate / nominal:\n")
                    f.write(f"{indent}        {metrics['mean'] / nominal_d:.3f}\n")
                f.write(f"{indent}    Root Mean Square Error:\n")
                f.write(f"{indent}        {d_text(metrics['rmse'])} {d_unit}\n")

        if 'precision' in statistics:
            intervals = statistics.get('precision_intervals', {})
            f.write(f"\n{indent}Precision at standard thresholds:\n")
            if intervals:
                f.write(f"{indent}  (each fraction with its count of estimates and its 95% interval)\n")
            for level, values in statistics['precision'].items():
                f.write(f"{indent}  Fraction within ±{level} of nominal:\n")
                # One line in display order, e.g.
                # "    WLS: 19.0% (19 of 100; 12.5% to 27.8%)    OLS: 15.0% (15 of 100; 9.3% to 23.3%)"
                f.write(indent + "".join(
                    f"    {spec.short_name}: {fraction_text(values[spec.key], intervals.get(level, {}).get(spec.key))}"
                    for spec in in_display_order(values)
                ) + "\n")

    with open(filename, 'w', encoding='utf-8') as f:
        # Write header
        f.write("DICE Simulation Summary\n")
        f.write("=" * 50 + "\n\n")

        # What reproduces the simulation
        from .. import __version__
        f.write("DICE version:\n")
        f.write(f"    {__version__}\n")
        f.write("Seed:\n")
        if results.seed is None:
            f.write("    none (another simulation with these parameters draws other noise)\n")
        else:
            f.write(f"    {results.seed}\n")
        f.write("\n")
        
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
            time_scale = max(abs(time_series[0]), abs(time_series[-1]))
            f.write(f"    {format_quantity(time_series[0], 3, time_scale)} to "
                    f"{format_quantity(time_series[-1], 3, time_scale)} {time_unit}\n")
        if noise_range is None:
            f.write(f"Noise standard deviation values:\n")
            f.write(f"    {results.noise_values}\n")
            f.write(f"Initial CNR (Contrast-to-Noise Ratio):\n")
            f.write(f"    {nominal_cnr(amplitude_0, results.noise_values[0]):.3f}\n")
        else:
            write_noise_range(f)
        f.write(f"Initial profile squared width (sigma^2):\n")
        f.write(f"    {format_quantity(results.parameters.profile.sigma2_0, 3)} {length_unit}^2\n")
        f.write(f"Nominal diffusion length:\n")
        f.write(f"    {format_quantity(results.parameters.physics.diffusion_length, 3)} {length_unit}\n")
        f.write(f"Nominal diffusion coefficient:\n")
        f.write(f"    {d_text(results.parameters.physics.diffusion_coefficient)} {d_unit}\n")
        f.write(f"Nominal lifetime:\n")
        from .parameters import LIFETIME_ASSUMED_KEY
        assumed = (" (assumed: the file gives only the diffusion length, so its times are in units "
                   "of the lifetime)") if parameters.get(LIFETIME_ASSUMED_KEY) else ""
        f.write(f"    {results.parameters.physics.lifetime} {time_unit}{assumed}\n")
        f.write("\n")
        
        # Write analysis results if available
        if analysis:
            f.write("Analysis Results:\n")
            f.write("-" * 30 + "\n")

            if noise_range is None:
                write_estimate_statistics(f, analysis, indent="")
            else:
                # Statistics pooled over a range of CNR would mix noise regimes.
                f.write("Statistics in bins of initial CNR with equal numbers of runs:\n")
                for cnr_bin in analysis['cnr_bins']:
                    f.write(f"\nInitial CNR {cnr_text(cnr_bin['cnr_min'])} to "
                            f"{cnr_text(cnr_bin['cnr_max'])} ({cnr_bin['num_runs']} runs):\n")
                    write_estimate_statistics(f, cnr_bin, indent="  ")

            f.write("\n")
        
        # Write completion message
        f.write(f"{completion_message(results)}\n")


def file_error_text(error: OSError) -> str:
    """The file and the reason of a failed read or write, for a message: 'C:\\...\\run.csv: Permission denied'."""
    if error.filename and error.strerror:
        return f"{error.filename}: {error.strerror}"
    return str(error)


def completion_message(results: MonteCarloOutput) -> str:
    """The last line of the summary, of the console output and of the GUI's status after a simulation."""
    if results.has_estimates():
        return "Simulation completed successfully."
    return "Simulation completed, but no run gave an estimate of the diffusion coefficient."


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
