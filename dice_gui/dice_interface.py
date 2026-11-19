"""
Interface between GUI and DICE simulation engine.

This module handles parameter conversion and simulation execution.
"""

from typing import Dict, Any, Optional, Callable
import sys
import os

from dice.models.estimators import ESTIMATORS, SlopeFit, get_estimator
from dice.models.results import RunResult, MonteCarloOutput
from dice_gui.output_location import output_root


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

        # Diffusion coefficient and lifetime
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
            try:
                time_values = [float(v.strip()) for v in gui_params['time_series'].split(',') if v.strip()]
            except ValueError:
                raise ValueError("Time series must be comma-separated numbers (e.g., '0, 1, 2, 3')")
            if len(time_values) == 0:
                raise ValueError("Time series must contain at least one value")
            if len(time_values) != len(set(time_values)):
                raise ValueError("Time series contains duplicate values")
            params['time series'] = sorted(time_values)

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

        # Per-parameter unit overrides (pass through for resolve_units)
        unit_keys = [
            'fwhm_0_unit', 'sigma_0_unit', 'mu_0_unit', 'spatial_width_unit',
            'diffusion_length_unit', 'diffusion_coefficient_length_unit',
            'diffusion_coefficient_time_unit', 'lifetime_unit',
            'time_start_unit', 'time_stop_unit',
        ]
        for key in unit_keys:
            if key in gui_params:
                params[key] = gui_params[key]

        return params

    def run_simulation(self, parameters: Dict[str, Any],
                       progress_callback: Optional[Callable[[int, int], None]] = None) -> Optional[tuple]:
        """
        Run DICE simulation with given parameters.

        Args:
            parameters: Dictionary of parameters
            progress_callback: Optional callback for progress updates (current, total)

        Returns:
            Tuple of (MonteCarloOutput, processed_params dict), or None if error occurred.
        """
        try:
            # Import required modules from dice
            from dice.io.parameters import parameter_parser, validate_parameters
            from dice.analysis.simulation import run_monte_carlo_simulation
            from dice.analysis.statistics import analyze_simulation_results
            from dice.io.results import export_collated_results, write_summary_file
            from dice.visualization.histograms import histogram_options, plot_accuracy_histogram
            from dice.utils.legacy_compatibility import create_parameters_from_legacy
            from pathlib import Path

            # Resolve per-parameter unit overrides before parsing,
            # so that values are in the global unit system when the
            # parser converts between representations (e.g., FWHM -> sigma^2).
            from dice.utils.units import resolve_units, global_length_unit, global_time_unit
            parameters = resolve_units(parameters)

            # Parse and process parameters (same as open_parameters does)
            processed_params = parameter_parser(parameters)

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

            # Analyze results with standard proximity levels
            analysis = analyze_simulation_results(result)

            # Setup output directory
            slug = processed_params.get('filename slug', 'dice_output')
            output_dir = output_root() / slug
            output_dir.mkdir(parents=True, exist_ok=True)

            # Results are in the global units resolve_units() converted to;
            # the exporters label and convert from these.
            length_unit = global_length_unit(processed_params)
            time_unit = global_time_unit(processed_params)

            # Export results to CSV
            csv_file = output_dir / f"{slug}.csv"
            export_collated_results(
                result, str(csv_file), length_unit=length_unit, time_unit=time_unit
            )

            # Write summary file
            summary_file = output_dir / f"{slug}_summary.txt"
            write_summary_file(
                result, str(summary_file), processed_params, analysis,
                length_unit=length_unit, time_unit=time_unit,
            )

            # Create accuracy histogram of the WLS estimates
            d_estimates = result.d_estimates('wls')
            d_nominal = processed_params['nominal diffusion coefficient']

            if len(d_estimates) > 0:
                d_ratios = d_estimates / d_nominal

                # Create legacy-format result for plotting
                legacy_result = {
                    'collated results': {
                        'd_est_over_d_nom': d_ratios.tolist()
                    }
                }

                # Create plot
                plot_file = output_dir / f"{slug}_accuracy_histogram.{processed_params.get('image type', 'png')}"

                plot_accuracy_histogram(
                    simulation_result=legacy_result,
                    filename=str(plot_file),
                    **histogram_options(processed_params)
                )

            return result, processed_params

        except InterruptedError:
            # A progress callback requested cancellation; not a failure.
            raise
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
        estimator: str = 'wls'
    ) -> None:
        """
        Regenerate plot from data in memory with specified image settings.

        Args:
            filename: Path where plot should be saved
            image_settings: Dictionary with image settings (type, width, height, dpi, etc.)
            estimator: Key of the estimator whose slopes to plot (dice.models.estimators)

        Raises:
            ValueError: If no data is available in memory
            RuntimeError: If plot generation fails
        """
        try:
            from dice.visualization.histograms import plot_accuracy_histogram

            # Check if data exists
            if self.last_result is None:
                raise ValueError("No simulation data in memory")

            if self.last_parameters is None:
                raise ValueError("No parameters in memory")

            # Extract diffusion estimates of the selected estimator
            d_nominal = self.last_parameters['nominal diffusion coefficient']
            d_estimates = self.last_result.d_estimates(estimator)

            if len(d_estimates) == 0:
                method_name = get_estimator(estimator).short_name
                raise ValueError(f"No {method_name} diffusion estimates found in results")

            # Calculate ratios
            d_ratios = d_estimates / d_nominal

            # Create legacy-format result for plotting
            legacy_result = {
                'collated results': {
                    'd_est_over_d_nom': d_ratios.tolist()
                }
            }

            # Create plot with specified settings
            plot_accuracy_histogram(
                simulation_result=legacy_result,
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
        image_settings: Dict[str, Any],
        estimator: str = 'wls'
    ) -> None:
        """
        Load results from CSV file and generate plot.

        Args:
            csv_file: Path to CSV file with simulation results
            filename: Path where plot should be saved
            image_settings: Dictionary with image settings (type, width, height, dpi, etc.)
            estimator: Key of the estimator whose slopes to plot (dice.models.estimators)

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

            def _find_csv_column(prefix):
                # Prefer exact match to avoid 'slope' matching 'slope stderr'
                if prefix in df.columns:
                    return prefix
                matches = [c for c in df.columns if c.startswith(prefix)]
                return matches[0] if matches else None

            # Determine which columns to use
            spec = get_estimator(estimator)
            slope_col = _find_csv_column(f'{spec.csv_label} diffusion slope')
            method_name = spec.short_name

            nom_col = _find_csv_column('nominal diffusion coeff')

            # Validate required columns exist
            if slope_col is None:
                raise ValueError(
                    f"CSV missing {method_name} slope column. "
                    f"Available columns: {list(df.columns)}"
                )
            if nom_col is None:
                raise ValueError(
                    f"CSV missing nominal diffusion coefficient column. "
                    f"Available columns: {list(df.columns)}"
                )

            # Extract slopes and nominal values
            slopes = df[slope_col].dropna().values

            if len(slopes) == 0:
                raise ValueError(f"No valid {method_name} slopes found in CSV")

            # Get nominal diffusion coefficient (should be same for all runs)
            d_nominal = df[nom_col].iloc[0]

            # Calculate diffusion estimates (D = slope/2)
            d_estimates = slopes / 2

            # Calculate ratios
            d_ratios = d_estimates / d_nominal

            # Create legacy-format result for plotting
            legacy_result = {
                'collated results': {
                    'd_est_over_d_nom': d_ratios.tolist()
                }
            }

            # Create plot with specified settings
            plot_accuracy_histogram(
                simulation_result=legacy_result,
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

            # Store loaded data in memory for regeneration: every estimator's
            # slopes whose column the CSV has, one entry per CSV row, so each
            # row stays one run even where only one estimator has no slope.
            stored_slopes = {
                spec.key: df[column].values
                for spec in ESTIMATORS
                if (column := f'{spec.csv_label} diffusion slope') in df.columns
            }

            # Create RunResult objects with those slopes
            # Use placeholder values for required fields not available in CSV
            run_results = []
            num_runs = len(df)
            for i in range(num_runs):
                run_result = RunResult(
                    run_id=i,
                    nominal_diffusion_coefficient=d_nominal,
                    nominal_lifetime=0.0,
                    nominal_diffusion_length=0.0,
                    nominal_sigma2_0=0.0,
                    noise_sigma=0.0,
                    cnr_0_estimate=0.0,
                )
                for key, key_slopes in stored_slopes.items():
                    if not pd.isna(key_slopes[i]):
                        run_result.set_slope_fit(key, SlopeFit(slope=key_slopes[i]))
                run_results.append(run_result)

            # Create MonteCarloOutput (parameters and noise_values unavailable from CSV)
            self.last_result = MonteCarloOutput(
                parameters=None,
                run_results=run_results,
                num_runs=num_runs,
                noise_values=[],
                estimator_keys=tuple(stored_slopes),
            )

            # Store parameters for regeneration
            self.last_parameters = {
                'nominal diffusion coefficient': d_nominal,
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

    def calculate_proximity_on_demand(
        self,
        proximity_level: float,
        estimator: str = 'wls'
    ) -> Optional[float]:
        """
        Calculate fraction of estimates within proximity of nominal, on demand.

        Args:
            proximity_level: Fractional tolerance (e.g., 0.1 for ±10%)
            estimator: Key of the estimator whose estimates to use (dice.models.estimators)

        Returns:
            Fraction within proximity (0-1), or None if no data available.
        """
        if self.last_result is None:
            return None

        from dice.analysis.statistics import calculate_precision

        d_estimates = self.last_result.d_estimates(estimator)

        if len(d_estimates) == 0:
            return None

        nominal_d = self.last_result.run_results[0].nominal_diffusion_coefficient
        if nominal_d == 0:
            return None

        return calculate_precision(d_estimates, nominal_d, proximity_level)
