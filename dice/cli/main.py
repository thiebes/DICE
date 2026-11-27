"""
Main command-line interface for DICE.

This module provides the main entry point for the DICE command-line interface,
orchestrating argument parsing, parameter loading, simulation execution, and output.
"""

import dataclasses
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from .. import __version__
from .arguments import parse_arguments, show_parameter_summary, handle_special_arguments
from ..io.parameters import noise_range_from_parameters, open_parameters, validate_parameters
from ..analysis.simulation import run_monte_carlo_simulation
from ..core.noise import Seed, nominal_cnr
from ..analysis.statistics import analyze_simulation_results
from ..io.results import (
    completion_message, export_collated_results, file_error_text, output_unit_preferences, result_sentence,
    write_summary_file,
)
from ..models.results import MonteCarloOutput
from ..visualization.histograms import (
    accuracy_scatter_options, histogram_options, plot_accuracy_histogram, plot_accuracy_vs_cnr,
    remove_plots_of_earlier_runs, use_file_backend,
)
from ..utils.units import diffusion_coefficient_label, global_length_unit, global_time_unit


def setup_output_directory(output_dir: Optional[str], parameters: Dict[str, Any]) -> Path:
    """
    Set up the output directory for results.
    
    Parameters
    ----------
    output_dir : str, optional
        Output directory from command line. If None, uses output/slug structure.
    parameters : dict
        Simulation parameters dictionary.
    
    Returns
    -------
    Path
        Path to the output directory.
    """
    if output_dir:
        output_path = Path(output_dir)
    else:
        # Use filename slug from parameters or default, under output/ folder
        slug = parameters.get('filename slug', 'dice_output')
        output_path = Path.cwd() / 'output' / slug
    
    # Create directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    return output_path


def create_progress_callback(verbose: bool, quiet: bool, interval: int = 100):
    """
    Create a progress callback function for simulation updates.
    
    Parameters
    ----------
    verbose : bool
        Whether to show verbose output.
    quiet : bool
        Whether to suppress output.
    interval : int
        A message is printed each time the number of finished runs passes a
        multiple of this number, and after the last run.
    
    Returns
    -------
    callable or None
        Progress callback function or None if not needed.
    """
    if quiet or not verbose:
        return None
    
    last_reported = 0

    def progress_callback(current: int, total: int):
        # A parallel simulation reports at counts that follow the clock, so
        # the test is whether a multiple of the interval has been passed.
        nonlocal last_reported
        if current // interval > last_reported // interval or current == total:
            last_reported = current
            percent = (current / total) * 100
            print(f"Progress: {current}/{total} runs completed ({percent:.1f}%)")
    
    return progress_callback


def run_simulation(parameters: Dict[str, Any], args, seed: Seed = None) -> MonteCarloOutput:
    """
    Execute the Monte Carlo simulation.

    Parameters
    ----------
    parameters : dict
        Simulation parameters.
    args : Namespace
        Command-line arguments.
    seed : int or SeedSequence, optional
        Root seed for the runs' noise; None draws fresh entropy.

    Returns
    -------
    dict
        Simulation results.
    """
    if not args.quiet:
        print("Starting DICE simulation...")
    
    # Create axes
    x_axis = parameters['x array']
    time_axis = parameters['time series']
    
    # --multiprocessing N: 0 runs serially, -1 uses all cores, N > 0 uses N workers.
    if args.multiprocessing is not None:
        use_parallel = args.multiprocessing != 0
        n_jobs = args.multiprocessing if use_parallel else -1
    else:
        use_parallel = parameters.get('multiprocessing', 1) != 0
        n_jobs = -1
    
    # Create progress callback
    progress_callback = create_progress_callback(
        args.verbose, args.quiet, 
        getattr(args, 'progress_interval', 100)
    )
    
    # Convert legacy parameters format to modern structure
    from ..models.parameters import simulation_parameters_from_dict
    
    # Create simulation parameters object
    sim_params = simulation_parameters_from_dict(
        parameters_dict=parameters,
        diffusion_coefficient=parameters['nominal diffusion coefficient'],
        lifetime=parameters['nominal lifetime (tau)'],
        diffusion_length=parameters['nominal diffusion length']
    )
    
    # With a noise range, 'noise series' holds one noise value for each run.
    noise_range = noise_range_from_parameters(parameters)

    # Run simulation
    start_time = time.time()
    
    if not args.quiet:
        print(f"Running {parameters['number of runs']} simulations...")
        if args.verbose:
            print(f"  Spatial points: {len(x_axis)}")
            print(f"  Time points: {len(time_axis)}")
            if noise_range is None:
                print(f"  Noise levels: {len(parameters['noise series'])}")
            else:
                print(f"  Noise levels: one for each run, from {noise_range.low} to {noise_range.high}")
            print(f"  Multiprocessing: {use_parallel}")
    
    # Execute Monte Carlo simulation
    result = run_monte_carlo_simulation(
        parameters=sim_params,
        x_axis=x_axis,
        time_axis=time_axis,
        noise_values=parameters['noise series'],
        num_runs=parameters['number of runs'],
        multiprocessing=use_parallel,
        # The --retain-profiles flag or the parameter-file key turns retention on.
        retain_profile_data=(
            getattr(args, 'retain_profiles', False)
            or parameters.get('retain profile data', 0) != 0
        ),
        progress_callback=progress_callback,
        seed=seed,
        n_jobs=n_jobs,
        noise_range=noise_range,
    )
    
    end_time = time.time()
    
    if not args.quiet:
        print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    return result


def save_results(result: MonteCarloOutput, parameters: Dict[str, Any],
                 output_dir: Path, args) -> Tuple[List[Path], Dict[str, Any]]:
    """
    Save simulation results to files.
    
    Parameters
    ----------
    result : dict
        Simulation results.
    parameters : dict
        Simulation parameters.
    output_dir : Path
        Output directory.
    args : Namespace
        Command-line arguments.

    Returns
    -------
    tuple
        The files written, in the order written, and the analysis of the results.
    """
    written = []
    length_unit = global_length_unit(parameters)
    time_unit = global_time_unit(parameters)
    output_units = output_unit_preferences(parameters)

    if not args.quiet:
        print("Saving results...")

    # Analyze results with standard proximity levels
    analysis = analyze_simulation_results(result)
    
    
    # Export format
    export_format = getattr(args, 'export_format', 'csv')
    
    # Export results  
    if export_format in ['csv', 'both']:
        csv_file = output_dir / f"{parameters.get('filename slug', 'dice_results')}.csv"
        # The export function writes the CSV file.
        export_collated_results(
            result, str(csv_file), length_unit=length_unit, time_unit=time_unit,
            output_units=output_units,
        )
        written.append(csv_file)
    
    if export_format in ['json', 'both']:
        # For JSON, we'll create our own export since the function doesn't support it
        json_file = output_dir / f"{parameters.get('filename slug', 'dice_results')}.json"
        import json
        
        # Convert result to JSON-serializable format
        json_data = {
            'analysis': analysis,
            'parameters': {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in parameters.items()},
            'summary': {
                'dice_version': __version__,
                'seed': result.seed,
                'total_runs': len(result.run_results),
                'noise_values': result.noise_values,
                # The parameters and the statistics of the analysis are in
                # the units of the simulation, which the file would not name.
                'units': {
                    'length': length_unit,
                    'time': time_unit,
                    'diffusion_coefficient': diffusion_coefficient_label(length_unit, time_unit),
                },
            }
        }
        # With a noise range, noise_values holds each run's noise, in run order.
        if result.noise_range is not None:
            json_data['summary']['noise_range'] = dataclasses.asdict(result.noise_range)
        
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        written.append(json_file)
    
    # Write summary  
    summary_file = output_dir / f"{parameters.get('filename slug', 'dice_results')}_summary.txt"
    write_summary_file(
        result, str(summary_file), parameters, analysis,
        length_unit=length_unit, time_unit=time_unit, output_units=output_units,
    )
    written.append(summary_file)
    return written, analysis


def create_plots(result: MonteCarloOutput, parameters: Dict[str, Any],
                 output_dir: Path, args) -> List[Path]:
    """
    Create and save plots from simulation results.
    
    Parameters
    ----------
    result : dict
        Simulation results.
    parameters : dict
        Simulation parameters.
    output_dir : Path
        Output directory.
    args : Namespace
        Command-line arguments.

    Returns
    -------
    list of Path
        The plot written, or none.
    """
    # The results file and the summary are new; no plot of an earlier run
    # with this slug stays beside them, also when this run draws none.
    try:
        remove_plots_of_earlier_runs(output_dir, parameters.get('filename slug', 'dice'))
    except OSError as e:
        raise PlotError(
            "The results were saved, but a plot of an earlier run could not be removed: "
            f"{file_error_text(e)}") from e

    if args.no_plots:
        return []
    
    if not args.quiet:
        print("Creating plots...")
    use_file_backend()
    
    # Across a noise range the accuracy depends on the CNR, so it is plotted against it.
    if result.noise_range is not None:
        return create_accuracy_scatter_plot(result, parameters, output_dir, args)

    fig = None
    written = []
    try:
        # Prepare simulation result in legacy format for plotting
        # Extract diffusion coefficients (WLS estimates, slope / 2)
        d_estimates = result.d_estimates('wls')
        d_nominal = parameters['nominal diffusion coefficient']

        plot_file = output_dir / f"{parameters.get('filename slug', 'dice')}_accuracy_histogram.{parameters.get('image type', 'png')}"

        if len(d_estimates) > 0:
            d_ratios = d_estimates / d_nominal

            # Create legacy-format result for plotting
            legacy_result = {
                'collated results': {
                    'd_est_over_d_nom': d_ratios.tolist()
                }
            }

            # Create accuracy histogram
            fig = plot_accuracy_histogram(
                simulation_result=legacy_result,
                filename=str(plot_file),
                **histogram_options(parameters)
            )

            written.append(plot_file)
        else:
            report_missing_plot(result)

    except OSError as e:
        raise PlotError(
            f"The results were saved, but the plot could not be saved: {file_error_text(e)}") from e
    except Exception as e:
        raise PlotError(f"The results were saved, but the plot could not be created: {e}") from e
    finally:
        if fig is not None:
            import matplotlib.pyplot as plt
            plt.close(fig)
    return written


class PlotError(RuntimeError):
    """The plot of a finished simulation could not be created; its other output files are written."""


def create_accuracy_scatter_plot(result: MonteCarloOutput, parameters: Dict[str, Any],
                                 output_dir: Path, args) -> List[Path]:
    """
    Save the scatter plot of the WLS D_est/D_nom against initial CNR of a simulation with a noise range.

    The CNR axis is logarithmic for 'noise range, reciprocal log'.

    Parameters
    ----------
    result : MonteCarloOutput
        Simulation results with a noise range.
    parameters : dict
        Simulation parameters.
    output_dir : Path
        Output directory.
    args : Namespace
        Command-line arguments.
    """
    fig = None
    try:
        amplitude_0 = result.parameters.profile.amplitude_0
        fitted_runs = [run for run in result.run_results if run.slope_fit('wls') is not None]
        slug, image_type = parameters.get('filename slug', 'dice'), parameters.get('image type', 'png')
        plot_file = output_dir / f"{slug}_accuracy_vs_cnr.{image_type}"
        if fitted_runs:
            fig = plot_accuracy_vs_cnr(
                [nominal_cnr(amplitude_0, run.noise_sigma) for run in fitted_runs],
                [run.accuracy('wls') for run in fitted_runs],
                str(plot_file),
                log_cnr=result.noise_range.logarithmic,
                **accuracy_scatter_options(parameters),
            )

            return [plot_file]
        else:
            report_missing_plot(result)

    except OSError as e:
        raise PlotError(
            f"The results were saved, but the plot could not be saved: {file_error_text(e)}") from e
    except Exception as e:
        raise PlotError(f"The results were saved, but the plot could not be created: {e}") from e
    finally:
        if fig is not None:
            import matplotlib.pyplot as plt
            plt.close(fig)
    return []


def report_completion(result: MonteCarloOutput, analysis: Dict[str, Any], quiet: bool,
                      output_dir: Optional[Path] = None, written: Sequence[Path] = ()) -> None:
    """
    Print how the simulation ended: the folder of the output files, then the result.

    The result is the result block of the summary file in one line (result_sentence).
    A simulation without estimates is reported as a warning on the error
    stream, also with --quiet. It is printed directly: Python shows a
    warning from one place only once in a process, and a program can call
    main() several times.
    """
    if not quiet and output_dir is not None and written:
        print(f"Output files in {output_dir}: {', '.join(Path(path).name for path in written)}")
    sentence = result_sentence(result, analysis)
    if sentence is None:
        print(f"Warning: {completion_message(result)}", file=sys.stderr)
    elif not quiet:
        print(sentence)


def report_missing_plot(result: MonteCarloOutput) -> None:
    """Say that no plot was drawn because the weighted fit, which the plots show, gave no estimate."""
    if result.has_estimates():
        # Without any estimate, report_completion says so.
        print("Warning: No plot was drawn: the weighted fit gave no estimate.", file=sys.stderr)


def console_warning(message, category, filename, lineno, line=None) -> str:
    """A warning as one line of the console, in the form of the other messages of the command line."""
    return f"Warning: {message}\n"


def main(argv: Optional[list] = None) -> int:
    """
    Main entry point for the DICE CLI.

    Warnings are printed as "Warning: <message>", without the source file
    and line that Python adds by default.

    Parameters
    ----------
    argv : list, optional
        Command-line arguments. If None, uses sys.argv.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for error).
    """
    previous_format = warnings.formatwarning
    warnings.formatwarning = console_warning
    try:
        return run_command(argv)
    finally:
        warnings.formatwarning = previous_format


def run_command(argv: Optional[list] = None) -> int:
    """
    Load the parameters, run the simulation, and write its outputs.
    
    Parameters
    ----------
    argv : list, optional
        Command-line arguments. If None, uses sys.argv.
    
    Returns
    -------
    int
        Exit code (0 for success, non-zero for error).
    """
    try:
        # Parse arguments
        args = parse_arguments(argv)

        # Show parameter summary if verbose
        if args.verbose and not args.quiet:
            show_parameter_summary(args)
        
        # One root seed, split into independent streams for the noise draws of
        # the 'noise range' options, one for each run, and for the simulation runs.
        noise_draw_seed, simulation_seed = (
            np.random.SeedSequence(args.seed).spawn(2) if args.seed is not None else (None, None)
        )

        # Load parameters
        if not args.quiet:
            print(f"Loading parameters from: {args.parameters_file}")

        parameters = open_parameters(args.parameters_file, seed=noise_draw_seed)
        validate_parameters(parameters)
        
        if args.verbose and not args.quiet:
            print("Parameters loaded and validated")

        # A dry run ends here, after the same loading and validation as a real
        # run and before any output directory, simulation, or file.
        if handle_special_arguments(args):
            return 0

        # Setup output directory
        output_dir = setup_output_directory(args.output_dir, parameters)
        
        if not args.quiet:
            print(f"Output directory: {output_dir}")
        
        # Run simulation
        result = run_simulation(parameters, args, seed=simulation_seed)
        # --seed N reproduces the simulation from the command line; the
        # stream the runs drew from is a child of N.
        result.seed = args.seed
        
        # Save results
        written, analysis = save_results(result, parameters, output_dir, args)
        
        # Create plots
        written += create_plots(result, parameters, output_dir, args)
        
        report_completion(result, analysis, args.quiet, output_dir, written)
        
        return 0
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        return 1
    
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose if 'args' in locals() else False:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())