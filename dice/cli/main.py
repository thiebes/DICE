"""
Main command-line interface for DICE.

This module provides the main entry point for the DICE command-line interface,
orchestrating argument parsing, parameter loading, simulation execution, and output.
"""

import sys
import os
import time
import warnings
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

from .arguments import parse_arguments, show_parameter_summary, handle_special_arguments
from ..io.parameters import open_parameters, validate_parameters
from ..analysis.simulation import run_monte_carlo_simulation
from ..core.noise import Seed
from ..analysis.statistics import analyze_simulation_results
from ..io.results import export_collated_results, write_summary_file
from ..visualization.histograms import plot_accuracy_histogram
from ..utils.axes import make_x_axis, make_time_axis
from ..utils.units import global_length_unit, global_time_unit


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
        Update interval for progress messages.
    
    Returns
    -------
    callable or None
        Progress callback function or None if not needed.
    """
    if quiet or not verbose:
        return None
    
    def progress_callback(current: int, total: int):
        if current % interval == 0 or current == total:
            percent = (current / total) * 100
            print(f"Progress: {current}/{total} runs completed ({percent:.1f}%)")
    
    return progress_callback


def run_simulation(parameters: Dict[str, Any], args, seed: Seed = None) -> Dict[str, Any]:
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
    
    # Setup multiprocessing
    if args.multiprocessing is not None:
        use_parallel = args.multiprocessing != 0
        if args.multiprocessing == -1:
            n_jobs = -1  # Use all cores
        else:
            n_jobs = args.multiprocessing
    else:
        use_parallel = parameters.get('multiprocessing', 1) != 0
        n_jobs = -1
    
    # Create progress callback
    progress_callback = create_progress_callback(
        args.verbose, args.quiet, 
        getattr(args, 'progress_interval', 100)
    )
    
    # Convert legacy parameters format to modern structure
    from ..utils.legacy_compatibility import create_parameters_from_legacy
    
    # Create simulation parameters object
    sim_params = create_parameters_from_legacy(
        parameters_dict=parameters,
        diffusion_coefficient=parameters['nominal diffusion coefficient'],
        lifetime=parameters['nominal lifetime (tau)'],
        diffusion_length=parameters['nominal diffusion length']
    )
    
    # Run simulation
    start_time = time.time()
    
    if not args.quiet:
        print(f"Running {parameters['number of runs']} simulations...")
        if args.verbose:
            print(f"  Spatial points: {len(x_axis)}")
            print(f"  Time points: {len(time_axis)}")
            print(f"  Noise levels: {len(parameters['noise series'])}")
            print(f"  Multiprocessing: {use_parallel}")
    
    # Execute Monte Carlo simulation
    result = run_monte_carlo_simulation(
        parameters=sim_params,
        x_axis=x_axis,
        time_axis=time_axis,
        noise_values=parameters['noise series'],
        num_runs=parameters['number of runs'],
        multiprocessing=use_parallel,
        retain_profile_data=args.retain_profiles if hasattr(args, 'retain_profiles') else False,
        progress_callback=progress_callback,
        seed=seed,
    )
    
    end_time = time.time()
    
    if not args.quiet:
        print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    return result


def save_results(result: Dict[str, Any], parameters: Dict[str, Any], 
                output_dir: Path, args) -> None:
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
    """
    length_unit = global_length_unit(parameters)
    time_unit = global_time_unit(parameters)

    if not args.quiet:
        print("Saving results...")

    # Analyze results with standard proximity levels
    analysis = analyze_simulation_results(result)
    
    # Prepare collated results for export
    collated_data = {
        'analysis': analysis,
        'parameters': parameters,
        'run_results': result.run_results
    }
    
    # Export format
    export_format = getattr(args, 'export_format', 'csv')
    
    # Export results  
    if export_format in ['csv', 'both']:
        csv_file = output_dir / f"{parameters.get('filename slug', 'dice_results')}.csv"
        # The export function returns a DataFrame and writes CSV automatically
        df = export_collated_results(
            result, str(csv_file), length_unit=length_unit, time_unit=time_unit
        )
        if not args.quiet:
            print(f"Results saved to: {csv_file}")
    
    if export_format in ['json', 'both']:
        # For JSON, we'll create our own export since the function doesn't support it
        json_file = output_dir / f"{parameters.get('filename slug', 'dice_results')}.json"
        import json
        
        # Convert result to JSON-serializable format
        json_data = {
            'analysis': analysis,
            'parameters': {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in parameters.items()},
            'summary': {
                'total_runs': len(result.run_results),
                'noise_values': result.noise_values
            }
        }
        
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        if not args.quiet:
            print(f"Results saved to: {json_file}")
    
    # Write summary  
    summary_file = output_dir / f"{parameters.get('filename slug', 'dice_results')}_summary.txt"
    write_summary_file(
        result, str(summary_file), parameters, analysis,
        length_unit=length_unit, time_unit=time_unit,
    )
    if not args.quiet:
        print(f"Summary saved to: {summary_file}")


def create_plots(result: Dict[str, Any], parameters: Dict[str, Any], 
                output_dir: Path, args) -> None:
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
    """
    if args.no_plots:
        return
    
    if not args.quiet:
        print("Creating plots...")
    
    fig = None
    try:
        # Prepare simulation result in legacy format for plotting
        # Extract diffusion coefficients (WLS estimates)
        d_estimates = []
        d_nominal = parameters['nominal diffusion coefficient']

        for run_result in result.run_results:
            if hasattr(run_result, 'wls_slope') and run_result.wls_slope is not None:
                d_est = run_result.wls_slope / 2  # Convert slope to diffusion coefficient
                d_estimates.append(d_est)

        if d_estimates:
            d_ratios = np.array(d_estimates) / d_nominal

            # Create legacy-format result for plotting
            legacy_result = {
                'collated results': {
                    'd_est_over_d_nom': d_ratios.tolist()
                }
            }

            # Create accuracy histogram
            plot_file = output_dir / f"{parameters.get('filename slug', 'dice')}_accuracy_histogram.{parameters.get('image type', 'png')}"

            fig = plot_accuracy_histogram(
                simulation_result=legacy_result,
                filename=str(plot_file),
                image_type=parameters.get('image type', 'png'),
                width=parameters.get('image width', 10),
                height=parameters.get('image height', 6),
                dpi=parameters.get('image dpi', 100),
                font_size=12
            )

            if not args.quiet:
                print(f"Accuracy histogram saved to: {plot_file}")
        else:
            if not args.quiet:
                print("Warning: No valid diffusion estimates found for plotting")

    except Exception as e:
        warnings.warn(f"Error creating plots: {e}", stacklevel=2)
    finally:
        if fig is not None:
            import matplotlib.pyplot as plt
            plt.close(fig)


def main(argv: Optional[list] = None) -> int:
    """
    Main entry point for the DICE CLI.
    
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
        
        # Handle special arguments
        if handle_special_arguments(args):
            return 0
        
        # Show parameter summary if verbose
        if args.verbose and not args.quiet:
            show_parameter_summary(args)
        
        # One root seed, split into independent streams for the noise draw of
        # the 'noise range' options and for the simulation runs.
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
        
        # Setup output directory
        output_dir = setup_output_directory(args.output_dir, parameters)
        
        if not args.quiet:
            print(f"Output directory: {output_dir}")
        
        # Run simulation
        result = run_simulation(parameters, args, seed=simulation_seed)
        
        # Save results
        save_results(result, parameters, output_dir, args)
        
        # Create plots
        create_plots(result, parameters, output_dir, args)
        
        if not args.quiet:
            print("DICE simulation completed successfully!")
        
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