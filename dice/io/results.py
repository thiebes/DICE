"""
Result export and file I/O for DICE simulations.

This module provides functions for exporting simulation results to CSV
files and text summaries, and for loading result CSV files.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ..core.noise import nominal_cnr
from ..models.estimators import ESTIMATORS, in_display_order
from ..models.results import MonteCarloOutput
from ..models.parameters import OutputUnitPreferences
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


# The proximity level of the fraction in the result block (0.1 for ±10%).
HEADLINE_LEVEL = '10%'


def table_lines(rows: List[List[str]], indent: str = "") -> List[str]:
    """Rows of cells as aligned text: each column as wide as its widest cell, two spaces apart."""
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    return [(indent + "  ".join(cell.ljust(width) for cell, width in zip(row, widths))).rstrip()
            for row in rows]


def _headline(analysis: Dict[str, Any]) -> Tuple[str, List[str]]:
    """The opening of the result line ("Result, <fit>, <runs>") and the parts of the result.

    The analysis has at least one estimator in 'accuracy' (result_lines and
    result_sentence check it).
    """
    spec = in_display_order(analysis['accuracy'])[0]
    # The name without a note such as ", for comparison".
    opening = f"Result, {spec.summary_name.split(', ')[0]} ({spec.short_name})"
    metrics = analysis['accuracy'][spec.key]
    fits = analysis.get('statistics', {}).get(spec.key)
    if fits is not None:
        opening += f", {fits['num_successful']} of {fits['num_successful'] + fits['num_failed']} runs"
        if fits['num_successful'] == 0:
            return opening, ["no estimate from this fit"]
    elif not np.isfinite(metrics['mean']):
        return opening, ["no estimate from this fit"]
    nominal = analysis.get('summary', {}).get('nominal_diffusion_coefficient')
    parts = []
    if HEADLINE_LEVEL in analysis.get('precision', {}):
        fraction = analysis['precision'][HEADLINE_LEVEL][spec.key]
        parts.append(f"{fraction:.1%} of estimates within ±{HEADLINE_LEVEL} of nominal")
    if nominal:
        relative_sd = metrics['std'] / nominal
        parts.append(f"relative standard deviation {relative_sd:.3f}" if np.isfinite(relative_sd)
                     else "relative standard deviation not defined for one estimate")
        parts.append(f"mean estimate / nominal {metrics['mean'] / nominal:.3f}")
    return opening, parts


def result_lines(results: MonteCarloOutput, analysis: Dict[str, Any]) -> List[str]:
    """
    The result block at the top of the summary file.

    For the first estimator in display order (the weighted fit): the fraction
    of its estimates within ±10% of nominal, their relative standard
    deviation, and the mean estimate / nominal. A simulation with a noise
    range has no such block, because its statistics pooled over the range of
    CNR would mix noise regimes.
    """
    if not results.has_estimates():
        return ["No run gave an estimate of the diffusion coefficient."]
    if results.noise_range is not None:
        return ["The statistics below are given for each bin of initial contrast-to-noise ratio (CNR);",
                "pooled over the range of CNR, they would mix noise regimes."]
    if not analysis.get('accuracy'):
        return []
    opening, parts = _headline(analysis)
    return [f"{opening}:"] + [f"  {part}" for part in parts] if parts else []


def result_sentence(results: MonteCarloOutput, analysis: Dict[str, Any]) -> Optional[str]:
    """The result block in one line, for the end of the console output; None when there is no result to state."""
    if not results.has_estimates():
        return None
    if results.noise_range is not None:
        return ("Result: the summary gives the statistics for each bin of initial "
                "contrast-to-noise ratio (CNR).")
    if not analysis.get('accuracy'):
        return None
    opening, parts = _headline(analysis)
    return f"{opening}: {'; '.join(parts)}." if parts else None


def _fraction_cell(fraction: float, interval: Optional[Dict[str, float]]) -> str:
    """A fraction of the estimates as a table cell: "16.0% (16 of 100)", or "no estimates" when it is of none.

    Without the interval, which holds the count, the cell is the percent alone.
    """
    if interval is None:
        return f"{fraction:.1%}"
    if not interval['total']:
        return "no estimates"
    return f"{fraction:.1%} ({interval['count']} of {interval['total']})"


def _interval_cell(limits: Sequence[float], text) -> str:
    """The limits of an interval as "lower to upper", or empty when they are not finite."""
    lower, upper = limits
    if not (np.isfinite(lower) and np.isfinite(upper)):
        return ""
    return f"{text(lower)} to {text(upper)}"


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

    def noise_range_rows() -> List[List[str]]:
        """The noise sigma and initial CNR drawn for the runs, and the range they were drawn from."""
        drawn = results.noise_values

        def sigma_text(value: float) -> str:
            return format_quantity(value, 4, reference=noise_range.low)

        # CNR is amplitude_0 / sigma, so the highest noise gives the lowest CNR.
        distribution = 'uniform in log10 of the CNR' if noise_range.logarithmic else 'uniform in CNR'
        return [
            ["Noise standard deviation, drawn for each run",
             f"{len(drawn)} values from {sigma_text(min(drawn))} to {sigma_text(max(drawn))}"
             f" (range {sigma_text(noise_range.low)} to {sigma_text(noise_range.high)})"],
            ["Initial CNR (Contrast-to-Noise Ratio), drawn for each run",
             f"{cnr_text(nominal_cnr(amplitude_0, max(drawn)))} to "
             f"{cnr_text(nominal_cnr(amplitude_0, min(drawn)))} (range {cnr_text(lowest_cnr)} to "
             f"{cnr_text(nominal_cnr(amplitude_0, noise_range.low))}, {distribution})"],
        ]

    nominal_d = (analysis or {}).get('summary', {}).get('nominal_diffusion_coefficient', None)

    def statistics_table(statistics: Dict[str, Any]) -> List[str]:
        """The statistics of each estimator's D estimates: one row per statistic, one column per estimator."""
        if 'accuracy' not in statistics:
            return []
        specs = in_display_order(statistics['accuracy'])
        accuracy = {spec.key: statistics['accuracy'][spec.key] for spec in specs}
        rows = [[""] + [spec.short_name for spec in specs]]

        def add(label: str, cell) -> None:
            cells = [cell(spec.key, accuracy[spec.key]) for spec in specs]
            # An interval row with no interval (no estimates, or fewer than two) is left out.
            if label.startswith("  ") and not any(cells):
                return
            rows.append([label] + cells)

        if 'statistics' in statistics:
            fits = statistics['statistics']
            add("Runs with an estimate", lambda key, _: (
                f"{fits[key]['num_successful']} of {fits[key]['num_successful'] + fits[key]['num_failed']}"))
        intervals = statistics.get('precision_intervals', {})
        for level, values in statistics.get('precision', {}).items():
            level_intervals = intervals.get(level, {})
            add(f"Within ±{level} of nominal",
                lambda key, _: _fraction_cell(values[key], level_intervals.get(key)))
            if level_intervals:
                add("  95% interval", lambda key, _: (
                    f"{level_intervals[key]['lower']:.1%} to {level_intervals[key]['upper']:.1%}"
                    if level_intervals[key]['total'] else ""))
        if nominal_d:
            # The mean ratio is the form in which the 2024 paper reports the bias
            # (its Fig. 7); the Analysis tab of the GUI shows it as "Mean D_est / D_nom".
            add("Mean estimate / nominal", lambda key, m: f"{m['mean'] / nominal_d:.3f}")
            add("Relative standard deviation", lambda key, m: f"{m['std'] / nominal_d:.3f}")
            if all('relative_std_interval' in m for m in accuracy.values()):
                add("  95% interval", lambda key, m: _interval_cell(
                    m['relative_std_interval'], lambda value: f"{value:.3f}"))
        add("Standard deviation", lambda key, m: d_text(m['std']))
        if all('std_interval' in m for m in accuracy.values()):
            add("  95% interval", lambda key, m: _interval_cell(m['std_interval'], d_text))
        add("Mean estimate", lambda key, m: d_text(m['mean']))
        add("Bias (mean - nominal)", lambda key, m: d_text(m['bias']))
        if all('standard_error' in m for m in accuracy.values()):
            add("  standard error", lambda key, m: d_text(m['standard_error']))
        add("Root mean square error", lambda key, m: d_text(m['rmse']))
        return table_lines(rows)

    def table_notes(statistics: Dict[str, Any]) -> List[str]:
        """What the names, the unit and the intervals of the tables are."""
        from ..analysis.statistics import BOOTSTRAP_RESAMPLES, MIN_BOOTSTRAP_ESTIMATES
        specs = in_display_order(statistics['accuracy'])
        notes = [" ".join(f"{spec.short_name}: {spec.summary_name}." for spec in specs),
                 f"Diffusion coefficients in {d_unit}."]
        if any('std_interval' in metrics for metrics in statistics['accuracy'].values()):
            notes.append("Intervals are 95%: for a fraction, the Wilson score interval; for a standard "
                         f"deviation, a bootstrap interval from {BOOTSTRAP_RESAMPLES:,} resamples, given from "
                         f"{MIN_BOOTSTRAP_ESTIMATES} estimates on, because with fewer it is too narrow.")
        elif statistics.get('precision_intervals'):
            notes.append("Intervals are 95% Wilson score intervals of the fractions.")
        return notes

    lines = []
    slug = parameters.get('filename slug')
    lines += [f"DICE simulation summary: {slug}" if slug else "DICE simulation summary", ""]

    # The result, then the statistics behind it
    if analysis:
        block = result_lines(results, analysis)
        lines += block + [""] if block else []
        if noise_range is None:
            table = statistics_table(analysis)
            noted = analysis
        else:
            table = []
            for cnr_bin in analysis['cnr_bins']:
                table += ([f"Initial CNR {cnr_text(cnr_bin['cnr_min'])} to {cnr_text(cnr_bin['cnr_max'])} "
                           f"({cnr_bin['num_runs']} runs)"] + statistics_table(cnr_bin) + [""])
            table = table[:-1]
            noted = analysis['cnr_bins'][0] if analysis['cnr_bins'] else {}
        if table:
            lines += table + [""] + table_notes(noted) + [""]

    # The parameters of the simulation
    rows = [["Number of runs", f"{results.num_runs}"],
            ["Spatial width", f"{parameters.get('spatial width', 'N/A')} {length_unit}"],
            ["Pixel width", f"{parameters.get('pixel width', 'N/A')} pixels"]]
    time_series = parameters.get('time series', [])
    rows.append(["Time frames", f"{len(time_series)} frames"])
    if len(time_series) > 0:
        time_scale = max(abs(time_series[0]), abs(time_series[-1]))
        rows.append(["Time range", f"{format_quantity(time_series[0], 3, time_scale)} to "
                                   f"{format_quantity(time_series[-1], 3, time_scale)} {time_unit}"])
    if noise_range is None:
        rows.append(["Noise standard deviation values", ", ".join(str(value) for value in results.noise_values)])
        rows.append(["Initial CNR (Contrast-to-Noise Ratio)",
                     f"{nominal_cnr(amplitude_0, results.noise_values[0]):.3f}"])
    else:
        rows += noise_range_rows()
    rows.append(["Initial profile squared width (sigma^2)",
                 f"{format_quantity(results.parameters.profile.sigma2_0, 3)} {length_unit}^2"])
    rows.append(["Nominal diffusion length",
                 f"{format_quantity(results.parameters.physics.diffusion_length, 3)} {length_unit}"])
    rows.append(["Nominal diffusion coefficient",
                 f"{d_text(results.parameters.physics.diffusion_coefficient)} {d_unit}"])
    rows.append(["Nominal lifetime", f"{results.parameters.physics.lifetime} {time_unit}"])
    from .parameters import LIFETIME_ASSUMED_KEY
    if parameters.get(LIFETIME_ASSUMED_KEY):
        rows.append(["", "(assumed: the file gives only the diffusion length, so its times are in units "
                         "of the lifetime)"])
    lines += ["Simulation parameters"] + table_lines(rows, indent="  ") + [""]

    # What reproduces the simulation
    from .. import __version__
    if results.seed is None:
        lines.append(f"DICE {__version__}, no seed (another simulation with these parameters draws other noise)")
    else:
        lines.append(f"DICE {__version__}, seed {results.seed}")

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")


def file_error_text(error: OSError) -> str:
    """The file and the reason of a failed read or write, for a message: 'C:\\...\\run.csv: Permission denied'."""
    if error.filename and error.strerror:
        return f"{error.filename}: {error.strerror}"
    return str(error)


def completion_message(results: MonteCarloOutput) -> str:
    """The GUI's status after a simulation, and the console's warning when no run gave an estimate."""
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
