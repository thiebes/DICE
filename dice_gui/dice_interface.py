"""
Interface between GUI and DICE simulation engine.

This module handles parameter conversion and simulation execution.
"""

from typing import Dict, Any, Optional
import sys
import os


class DiceInterface:
    """Interface for running DICE simulations from GUI."""

    def __init__(self):
        """Initialize the interface."""
        self.last_result = None
        self.last_parameters = None

    def build_parameters_dict(self, gui_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a parameters dictionary from GUI inputs in the format expected by dice.py.

        Args:
            gui_params: Dictionary containing all GUI parameter values

        Returns:
            Parameters dictionary formatted for dice.dice_runner()
        """
        params = {}

        # Simulation control
        params['number of runs'] = gui_params['number_of_runs']
        params['filename slug'] = gui_params['filename_slug']

        # Units
        if 'length_unit' in gui_params and gui_params['length_unit']:
            params['length unit'] = gui_params['length_unit']
        if 'time_unit' in gui_params and gui_params['time_unit']:
            params['time unit'] = gui_params['time_unit']

        # Initial profile
        params['amplitude_0'] = gui_params['amplitude_0']
        params['mean_0'] = gui_params['mean_0']

        # Profile width (mutually exclusive)
        if gui_params['profile_width_type'] == 'fwhm':
            params['FWHM_0'] = gui_params['profile_width_value']
        else:  # sigma
            params['sigma_0'] = gui_params['profile_width_value']

        # Diffusion (mutually exclusive)
        if gui_params['diffusion_type'] == 'length':
            params['nominal diffusion length'] = gui_params['diffusion_length']
        else:  # coefficient + lifetime
            params['nominal diffusion coefficient'] = gui_params['diffusion_coefficient']
            params['nominal lifetime (tau)'] = gui_params['lifetime']

        # Noise (mutually exclusive)
        if gui_params['noise_type'] == 'fixed':
            params['noise value'] = gui_params['noise_value']
        else:  # estimate from data
            params['estimate noise from data'] = gui_params['noise_data_file']

        # Spatial axis
        params['spatial width'] = gui_params['spatial_width']
        params['pixel width'] = gui_params['pixel_width']

        # Temporal axis (mutually exclusive)
        if gui_params['time_type'] == 'range':
            params['time range'] = [
                gui_params['time_start'],
                gui_params['time_stop'],
                gui_params['time_steps']
            ]
        else:  # series
            # Convert comma-separated string to list of floats
            params['time series'] = [float(v.strip()) for v in gui_params['time_series'].split(',') if v.strip()]

        # Analysis
        params['proximity level'] = gui_params['proximity_level']

        # Optional advanced parameters
        if 'image_type' in gui_params and gui_params['image_type']:
            params['image type'] = gui_params['image_type']
        if 'image_width' in gui_params and gui_params['image_width']:
            params['image width'] = gui_params['image_width']
        if 'image_height' in gui_params and gui_params['image_height']:
            params['image height'] = gui_params['image_height']
        if 'image_dpi' in gui_params and gui_params['image_dpi']:
            params['image dpi'] = gui_params['image_dpi']
        if 'image_font_size' in gui_params and gui_params['image_font_size']:
            params['image font size'] = gui_params['image_font_size']
        if 'image_tick_length' in gui_params and gui_params['image_tick_length']:
            params['image tick length'] = gui_params['image_tick_length']
        if 'image_tick_width' in gui_params and gui_params['image_tick_width']:
            params['image tick width'] = gui_params['image_tick_width']
        if 'image_numbins' in gui_params and gui_params['image_numbins']:
            params['image numbins'] = gui_params['image_numbins']
        if 'retain_profile_data' in gui_params:
            params['retain profile data'] = gui_params['retain_profile_data']
        if 'multiprocessing' in gui_params:
            params['multiprocessing'] = gui_params['multiprocessing']

        return params

    def run_simulation(self, parameters: Dict[str, Any]) -> Optional[Any]:
        """
        Run DICE simulation with given parameters.

        Args:
            parameters: Dictionary of parameters

        Returns:
            Simulation results or None if error occurred
        """
        try:
            # Import required modules from dice
            from dice.io.parameters import parameter_parser, validate_parameters
            from dice.analysis.simulation import run_monte_carlo_simulation
            from dice.analysis.statistics import analyze_simulation_results
            from dice.io.results import export_collated_results, write_summary_file
            from dice.visualization.histograms import plot_accuracy_histogram
            from dice.utils.legacy_compatibility import create_parameters_from_legacy
            import numpy as np
            from pathlib import Path

            # Parse and process parameters (same as open_parameters does)
            processed_params = parameter_parser(parameters)

            # Store parameters
            self.last_parameters = processed_params

            # Validate parameters
            validate_parameters(processed_params)

            # Create axes
            x_axis = processed_params['x array']
            time_axis = processed_params['time series']

            # Create simulation parameters object from legacy format
            sim_params = create_parameters_from_legacy(
                parameters_dict=processed_params,
                diffusion_coefficient=processed_params['nominal diffusion coefficient'],
                lifetime=processed_params['nominal lifetime (tau)'],
                diffusion_length=processed_params['nominal diffusion length']
            )

            # Run Monte Carlo simulation
            result = run_monte_carlo_simulation(
                parameters=sim_params,
                x_axis=x_axis,
                time_axis=time_axis,
                noise_values=processed_params['noise series'],
                num_runs=processed_params['number of runs'],
                multiprocessing=processed_params.get('multiprocessing', 1) != 0,
                retain_profile_data=processed_params.get('retain profile data', 0) != 0
            )

            # Analyze results
            proximity_levels = [processed_params['proximity level']]
            analysis = analyze_simulation_results(result, proximity_levels)

            # Setup output directory
            slug = processed_params.get('filename slug', 'dice_output')
            output_dir = Path.cwd() / 'output' / slug
            output_dir.mkdir(parents=True, exist_ok=True)

            # Export results to CSV
            csv_file = output_dir / f"{slug}.csv"
            export_collated_results(result, str(csv_file))

            # Write summary file
            summary_file = output_dir / f"{slug}_summary.txt"
            write_summary_file(result, str(summary_file), processed_params, analysis)

            # Create accuracy histogram
            d_estimates = []
            d_nominal = processed_params['nominal diffusion coefficient']

            for run_result in result.run_results:
                if hasattr(run_result, 'wls_slope') and run_result.wls_slope is not None:
                    d_est = run_result.wls_slope / 2
                    d_estimates.append(d_est)

            if d_estimates:
                d_ratios = np.array(d_estimates) / d_nominal

                # Create legacy-format result for plotting
                legacy_result = {
                    'collated results': {
                        'd_wls_over_d_nom': d_ratios.tolist()
                    }
                }

                # Create plot
                plot_file = output_dir / f"{slug}_accuracy_histogram.{processed_params.get('image type', 'png')}"

                plot_accuracy_histogram(
                    simulation_result=legacy_result,
                    proximity=processed_params.get('proximity level', 0.1),
                    filename=str(plot_file),
                    image_type=processed_params.get('image type', 'png'),
                    width=processed_params.get('image width', 10),
                    height=processed_params.get('image height', 6),
                    dpi=processed_params.get('image dpi', 100),
                    font_size=12
                )

            # Store result
            self.last_result = result

            return result

        except ImportError as e:
            raise ImportError(f"Failed to import dice module: {e}")
        except Exception as e:
            raise RuntimeError(f"Simulation failed: {e}")

    def validate_parameters(self, parameters: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate parameters before running simulation.

        Args:
            parameters: Parameters dictionary

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required parameters
        required = [
            'number of runs',
            'spatial width',
            'pixel width',
            'mean_0',
            'amplitude_0',
        ]

        for param in required:
            if param not in parameters:
                return False, f"Missing required parameter: {param}"

        # Check mutually exclusive groups
        if 'FWHM_0' in parameters and 'sigma_0' in parameters:
            return False, "Cannot specify both FWHM_0 and sigma_0"

        if 'nominal diffusion length' in parameters and (
            'nominal diffusion coefficient' in parameters or 'nominal lifetime (tau)' in parameters
        ):
            return False, "Cannot specify both diffusion length and (coefficient + lifetime)"

        if 'noise value' in parameters and 'estimate noise from data' in parameters:
            return False, "Cannot specify both noise value and estimate from data"

        if 'time range' in parameters and 'time series' in parameters:
            return False, "Cannot specify both time range and time series"

        # Check that at least one option from each mutually exclusive group is present
        if 'FWHM_0' not in parameters and 'sigma_0' not in parameters:
            return False, "Must specify either FWHM_0 or sigma_0"

        if 'nominal diffusion length' not in parameters and (
            'nominal diffusion coefficient' not in parameters or 'nominal lifetime (tau)' not in parameters
        ):
            return False, "Must specify either diffusion length or (coefficient and lifetime)"

        if 'noise value' not in parameters and 'estimate noise from data' not in parameters:
            return False, "Must specify either noise value or estimate from data"

        if 'time range' not in parameters and 'time series' not in parameters:
            return False, "Must specify either time range or time series"

        return True, ""
