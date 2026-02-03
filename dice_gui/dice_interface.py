"""
Interface between GUI and DICE simulation engine.

This module handles parameter conversion and simulation execution.
"""

from typing import Dict, Any, Optional, Callable
import sys
import os

from dice.models.results import RunResult, MonteCarloOutput


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

    def run_simulation(self, parameters: Dict[str, Any],
                       progress_callback: Optional[Callable[[int, int], None]] = None) -> Optional[Any]:
        """
        Run DICE simulation with given parameters.

        Args:
            parameters: Dictionary of parameters
            progress_callback: Optional callback for progress updates (current, total)

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
                retain_profile_data=processed_params.get('retain profile data', 0) != 0,
                progress_callback=progress_callback
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

    def regenerate_plot_from_memory(
        self,
        filename: str,
        image_settings: Dict[str, Any],
        use_wls: bool = True
    ) -> None:
        """
        Regenerate plot from data in memory with specified image settings.

        Args:
            filename: Path where plot should be saved
            image_settings: Dictionary with image settings (type, width, height, dpi, etc.)
            use_wls: If True, use WLS slopes; if False, use OLS slopes

        Raises:
            ValueError: If no data is available in memory
            RuntimeError: If plot generation fails
        """
        try:
            import numpy as np
            from dice.visualization.histograms import plot_accuracy_histogram

            # Check if data exists
            if self.last_result is None:
                raise ValueError("No simulation data in memory")

            if self.last_parameters is None:
                raise ValueError("No parameters in memory")

            # Extract diffusion estimates based on method selection
            d_estimates = []
            d_nominal = self.last_parameters['nominal diffusion coefficient']

            for run_result in self.last_result.run_results:
                if use_wls:
                    # Use WLS slopes
                    if hasattr(run_result, 'wls_slope') and run_result.wls_slope is not None:
                        d_est = run_result.wls_slope / 2
                        d_estimates.append(d_est)
                else:
                    # Use OLS slopes
                    if hasattr(run_result, 'ols_slope') and run_result.ols_slope is not None:
                        d_est = run_result.ols_slope / 2
                        d_estimates.append(d_est)

            if not d_estimates:
                method_name = "WLS" if use_wls else "OLS"
                raise ValueError(f"No {method_name} diffusion estimates found in results")

            # Calculate ratios
            d_ratios = np.array(d_estimates) / d_nominal

            # Create legacy-format result for plotting
            legacy_result = {
                'collated results': {
                    'd_wls_over_d_nom': d_ratios.tolist()
                }
            }

            # Get proximity from stored parameters
            proximity = self.last_parameters.get('proximity level', 0.1)

            # Create plot with specified settings
            plot_accuracy_histogram(
                simulation_result=legacy_result,
                proximity=proximity,
                filename=filename,
                image_type=image_settings.get('image_type', 'png'),
                width=image_settings.get('image_width', 16.0),
                height=image_settings.get('image_height', 10.0),
                dpi=image_settings.get('image_dpi', 100),
                font_size=image_settings.get('image_font_size', 6),
                tick_length=image_settings.get('image_tick_length', 6),
                tick_width=image_settings.get('image_tick_width', 2),
                num_bins=image_settings.get('image_numbins', 35)
            )

        except ImportError as e:
            raise ImportError(f"Failed to import required modules: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to regenerate plot: {e}")

    def load_and_plot_from_csv(
        self,
        csv_file: str,
        filename: str,
        proximity: float,
        image_settings: Dict[str, Any],
        use_wls: bool = True
    ) -> None:
        """
        Load results from CSV file and generate plot.

        Args:
            csv_file: Path to CSV file with simulation results
            filename: Path where plot should be saved
            proximity: Proximity level for plot
            image_settings: Dictionary with image settings (type, width, height, dpi, etc.)
            use_wls: If True, use WLS slopes; if False, use OLS slopes

        Raises:
            FileNotFoundError: If CSV file does not exist
            ValueError: If CSV is missing required columns
            RuntimeError: If plot generation fails
        """
        try:
            import pandas as pd
            import numpy as np
            from pathlib import Path
            from dice.visualization.histograms import plot_accuracy_histogram

            # Check if file exists
            csv_path = Path(csv_file)
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {csv_file}")

            # Load CSV
            df = pd.read_csv(csv_file)

            # Determine which columns to use
            if use_wls:
                slope_col = 'weighted fit diffusion slope'
                method_name = 'WLS'
            else:
                slope_col = 'unweighted fit diffusion slope'
                method_name = 'OLS'

            # Validate required columns exist
            required_cols = [slope_col, 'nominal diffusion coeff']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"CSV missing required columns: {missing_cols}")

            # Extract slopes and nominal values
            slopes = df[slope_col].dropna().values

            if len(slopes) == 0:
                raise ValueError(f"No valid {method_name} slopes found in CSV")

            # Get nominal diffusion coefficient (should be same for all runs)
            d_nominal = df['nominal diffusion coeff'].iloc[0]

            # Calculate diffusion estimates (D = slope/2)
            d_estimates = slopes / 2

            # Calculate ratios
            d_ratios = d_estimates / d_nominal

            # Create legacy-format result for plotting
            legacy_result = {
                'collated results': {
                    'd_wls_over_d_nom': d_ratios.tolist()
                }
            }

            # Create plot with specified settings
            plot_accuracy_histogram(
                simulation_result=legacy_result,
                proximity=proximity,
                filename=filename,
                image_type=image_settings.get('image_type', 'png'),
                width=image_settings.get('image_width', 16.0),
                height=image_settings.get('image_height', 10.0),
                dpi=image_settings.get('image_dpi', 100),
                font_size=image_settings.get('image_font_size', 6),
                tick_length=image_settings.get('image_tick_length', 6),
                tick_width=image_settings.get('image_tick_width', 2),
                num_bins=image_settings.get('image_numbins', 35)
            )

            # Store loaded data in memory for regeneration
            # Get both WLS and OLS slopes if available for future regeneration
            wls_slopes = df['weighted fit diffusion slope'].dropna().values if 'weighted fit diffusion slope' in df.columns else None
            ols_slopes = df['unweighted fit diffusion slope'].dropna().values if 'unweighted fit diffusion slope' in df.columns else None

            # Create RunResult objects with both WLS and OLS slopes
            # Use placeholder values for required fields not available in CSV
            run_results = []
            num_runs = len(slopes)
            for i in range(num_runs):
                run_result = RunResult(
                    run_id=i,
                    nominal_diffusion_coefficient=d_nominal,
                    nominal_lifetime=0.0,
                    nominal_diffusion_length=0.0,
                    nominal_sigma2_0=0.0,
                    noise_sigma=0.0,
                    cnr_0_estimate=0.0,
                    wls_slope=wls_slopes[i] if wls_slopes is not None and i < len(wls_slopes) else None,
                    ols_slope=ols_slopes[i] if ols_slopes is not None and i < len(ols_slopes) else None
                )
                run_results.append(run_result)

            # Create MonteCarloOutput (parameters and noise_values unavailable from CSV)
            self.last_result = MonteCarloOutput(
                parameters=None,
                run_results=run_results,
                num_runs=num_runs,
                noise_values=[]
            )

            # Store parameters for regeneration
            self.last_parameters = {
                'nominal diffusion coefficient': d_nominal,
                'proximity level': proximity,
                'filename slug': csv_path.stem
            }

        except ImportError as e:
            raise ImportError(f"Failed to import required modules: {e}")
        except FileNotFoundError:
            raise
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to load and plot from CSV: {e}")
