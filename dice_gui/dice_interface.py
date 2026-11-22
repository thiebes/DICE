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

# Keys of last_parameters, after a results CSV is loaded, that hold the named
# unit of the statistics of its estimates and the factor to that unit.
CSV_D_UNIT_KEY = 'diffusion coefficient unit'
CSV_D_SCALE_KEY = 'diffusion coefficient scale'


# Keys of a parameter file whose value the form holds in a whole-number field.
WHOLE_NUMBER_KEYS = ('number of runs', 'pixel width', 'image dpi', 'image numbins')


def parameters_for_form(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare the dictionary of a parameter file for the fields of the GUI.

    The form holds some values in another shape than a file may give them:

    - Whole numbers written as floats (50.0) become ints.
    - 'time_range_unit' becomes the units of the start and stop fields, with
      the precedence resolve_units gives it over 'time_start_unit' and
      'time_stop_unit'.
    - A 'time series' with a unit key is converted to the global time unit,
      because the form has no unit selector for a series.
    - The space spelling 'diffusion length unit' becomes 'diffusion_length_unit'.

    Parameters
    ----------
    params : dict
        The dictionary of a parameter file, with the keys the files document.

    Returns
    -------
    dict
        A copy that populate_gui_from_parameters can show without loss.

    Raises
    ------
    TypeError
        If a whole-number field is given a number with a fraction (the
        command-line interface rejects it too; 50.0 stands for 50 in both).
    ValueError
        If a value cannot be shown in the form. Nothing has been changed then.
    """
    from dice.utils.units import convert_time, global_time_unit, validate_time_unit
    from dice.utils.validators import validate_integer

    prepared = dict(params)

    for key in WHOLE_NUMBER_KEYS:
        if key in prepared:
            prepared[key] = validate_integer(prepared[key], f"'{key}'")

    if 'time range' in prepared:
        time_range = prepared['time range']
        if not isinstance(time_range, (list, tuple)) or len(time_range) != 3:
            raise ValueError(f"'time range' must be [start, stop, steps], got {time_range!r}")
        steps = validate_integer(time_range[2], "The number of steps in 'time range'")
        prepared['time range'] = [time_range[0], time_range[1], steps]

    if 'time series' in prepared and not isinstance(prepared['time series'], (list, tuple)):
        raise ValueError(f"'time series' must be a list of times, got {prepared['time series']!r}")

    global_time = global_time_unit(prepared)
    range_unit = prepared.pop('time_range_unit', None)
    series_unit = prepared.pop('time_series_unit', None)
    range_unit_applies = range_unit is not None and range_unit != global_time
    if range_unit_applies:
        validate_time_unit(range_unit, 'time_range_unit')
        prepared['time_start_unit'] = prepared['time_stop_unit'] = range_unit
        series_unit = range_unit
    if series_unit is not None and series_unit != global_time and 'time series' in prepared:
        validate_time_unit(series_unit, 'time_series_unit')
        prepared['time series'] = [
            convert_time(time, series_unit, global_time) for time in prepared['time series']
        ]

    if 'diffusion length unit' in prepared:
        prepared.setdefault('diffusion_length_unit', prepared.pop('diffusion length unit'))

    return prepared


class DiceInterface:
    """Interface for running DICE simulations from GUI."""

    def __init__(self):
        """Initialize the interface."""
        self.last_result = None
        self.last_parameters = None
        # Why the output files of the last run could not be written, if they could not.
        self.last_output_error: Optional[str] = None

    def diffusion_coefficient_unit(self) -> tuple[float, Optional[str]]:
        """
        Factor from the stored D estimates to the unit they are reported in, and that unit's label.

        After a run, the estimates are in the global units of its parameters,
        and are reported in the unit of the summary file's diffusion
        coefficients. A loaded results CSV does not name the global units, so
        the unit is that of its labeled nominal column, or None when the CSV
        has no such column.
        """
        from dice.io.results import output_unit_preferences, summary_diffusion_units
        from dice.utils.units import (
            convert_diffusion_coefficient, diffusion_coefficient_label,
            global_length_unit, global_time_unit,
        )

        parameters = self.last_parameters or {}
        if CSV_D_UNIT_KEY in parameters:
            return parameters[CSV_D_SCALE_KEY], parameters[CSV_D_UNIT_KEY]
        length_unit, time_unit = global_length_unit(parameters), global_time_unit(parameters)
        d_length_unit, d_time_unit = summary_diffusion_units(
            length_unit, time_unit, output_unit_preferences(parameters))
        scale = convert_diffusion_coefficient(1.0, length_unit, time_unit, d_length_unit, d_time_unit)
        return scale, diffusion_coefficient_label(d_length_unit, d_time_unit)

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
                       progress_callback: Optional[Callable[[int, int], None]] = None) -> tuple:
        """
        Run DICE simulation with given parameters, then write its output files.

        A finished simulation is returned even when its output files cannot be
        written (a results file open in another program, for example).
        last_output_error then holds the reason, and retry_write_outputs writes
        the files again.

        Args:
            parameters: Dictionary of parameters
            progress_callback: Optional callback for progress updates (current, total)

        Returns:
            Tuple of (MonteCarloOutput, processed_params dict).

        Raises:
            InterruptedError: If the progress callback asked the run to stop.
            RuntimeError: If the parameters are invalid or the simulation fails.
        """
        try:
            # Import required modules from dice
            from dice.io.parameters import parameter_parser, validate_parameters
            from dice.analysis.simulation import run_monte_carlo_simulation
            from dice.utils.legacy_compatibility import create_parameters_from_legacy

            # Resolve per-parameter unit overrides before parsing,
            # so that values are in the global unit system when the
            # parser converts between representations (e.g., FWHM -> sigma^2).
            from dice.utils.units import resolve_units
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
        except InterruptedError:
            # A progress callback requested cancellation; not a failure.
            raise
        except ImportError as e:
            raise ImportError(f"Failed to import dice module: {e}") from e
        except Exception as e:
            raise RuntimeError(str(e)) from e

        # The result exists now, so a failed write must not discard it.
        self.last_output_error = None
        try:
            self.write_outputs(result, processed_params)
        except Exception as e:
            self.last_output_error = str(e)

        return result, processed_params

    def write_outputs(self, result: MonteCarloOutput, processed_params: Dict[str, Any]) -> None:
        """
        Write the results file, the summary and the accuracy histogram of a simulation.

        Args:
            result: The simulation results.
            processed_params: The parsed parameters the simulation ran with.

        Raises:
            OSError: If a file cannot be written.
        """
        from dice.analysis.statistics import analyze_simulation_results
        from dice.io.results import (
            export_collated_results, output_unit_preferences, write_summary_file,
        )
        from dice.visualization.histograms import histogram_options, plot_accuracy_histogram
        from dice.utils.units import global_length_unit, global_time_unit

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
        output_units = output_unit_preferences(processed_params)

        # Export results to CSV
        csv_file = output_dir / f"{slug}.csv"
        export_collated_results(
            result, str(csv_file), length_unit=length_unit, time_unit=time_unit,
            output_units=output_units,
        )

        # Write summary file
        summary_file = output_dir / f"{slug}_summary.txt"
        write_summary_file(
            result, str(summary_file), processed_params, analysis,
            length_unit=length_unit, time_unit=time_unit, output_units=output_units,
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

            figure = plot_accuracy_histogram(
                simulation_result=legacy_result,
                filename=str(plot_file),
                **histogram_options(processed_params)
            )
            # The GUI shows the saved file; close the figure so a long session does not keep one per plot.
            if figure is not None:
                import matplotlib.pyplot as plt
                plt.close(figure)

    def retry_write_outputs(self) -> Optional[str]:
        """
        Write the output files of the last simulation again.

        Returns:
            None when the files are written, or the reason they could not be.
        """
        try:
            self.write_outputs(self.last_result, self.last_parameters)
        except Exception as e:
            self.last_output_error = str(e)
        else:
            self.last_output_error = None
        return self.last_output_error

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
            figure = plot_accuracy_histogram(
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
            # The GUI shows the saved file; close the figure so a long session does not keep one per plot.
            if figure is not None:
                import matplotlib.pyplot as plt
                plt.close(figure)

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
            from dice.io.results import NOMINAL_D_COLUMN, column_unit, results_column

            df = pd.read_csv(csv_file)

            def _find_csv_column(name):
                try:
                    return results_column(df.columns, name)
                except KeyError:
                    return None

            # Determine which columns to use. The first column of the nominal
            # diffusion coefficient is in the units of the slopes.
            spec = get_estimator(estimator)
            slope_col = _find_csv_column(f'{spec.csv_label} diffusion slope')
            method_name = spec.short_name

            nom_col = _find_csv_column(NOMINAL_D_COLUMN)

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

            # The statistics of the estimates are reported in a named unit:
            # the output unit of the file when it has a second column of the
            # nominal diffusion coefficient (for example in cm²/s), otherwise
            # the unit of the slopes. A file of DICE 1.2.0 names no unit for
            # its slopes.
            output_nom_col = next(
                (c for c in df.columns
                 if c != nom_col and c.startswith(f'{NOMINAL_D_COLUMN} [') and c.endswith(']')),
                None,
            )
            if output_nom_col is not None and d_nominal != 0:
                d_unit = column_unit(output_nom_col)
                d_scale = float(df[output_nom_col].iloc[0] / d_nominal)
            else:
                d_unit, d_scale = column_unit(nom_col), 1.0

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
            figure = plot_accuracy_histogram(
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
            # The GUI shows the saved file; close the figure so a long session does not keep one per plot.
            if figure is not None:
                import matplotlib.pyplot as plt
                plt.close(figure)

            # Store loaded data in memory for regeneration: every estimator's
            # slopes whose column the CSV has, one entry per CSV row, so each
            # row stays one run even where only one estimator has no slope.
            stored_slopes = {
                spec.key: df[column].values
                for spec in ESTIMATORS
                if (column := _find_csv_column(f'{spec.csv_label} diffusion slope')) is not None
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

            # Store parameters for regeneration, and the unit for statistics
            self.last_parameters = {
                'nominal diffusion coefficient': d_nominal,
                'filename slug': csv_path.stem,
                CSV_D_UNIT_KEY: d_unit,
                CSV_D_SCALE_KEY: d_scale,
            }

        except ImportError as e:
            raise ImportError(f"Failed to import required modules: {e}")
        except FileNotFoundError:
            raise
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to load and plot from CSV: {e}")

    def estimate_ratios(self, estimator: str = 'wls'):
        """
        The estimates of the last results as ratios to the nominal diffusion coefficient.

        Args:
            estimator: Key of the estimator whose estimates to use (dice.models.estimators)

        Returns:
            Array of D estimate / nominal D, or None if no data available.
        """
        if self.last_result is None:
            return None

        d_estimates = self.last_result.d_estimates(estimator)
        if len(d_estimates) == 0:
            return None

        nominal_d = self.last_result.run_results[0].nominal_diffusion_coefficient
        if nominal_d == 0:
            return None

        return d_estimates / nominal_d

    def proximity_detail(self, proximity_level: float, estimator: str = 'wls') -> Optional[Dict[str, float]]:
        """
        The fraction of estimates within proximity of nominal, with its count and 95% interval.

        Args:
            proximity_level: Fractional tolerance (e.g., 0.1 for ±10%)
            estimator: Key of the estimator whose estimates to use (dice.models.estimators)

        Returns:
            'fraction', 'count', 'total', 'lower' and 'upper'
            (dice.analysis.statistics.precision_interval), or None if no data available.
        """
        fraction = self.calculate_proximity_on_demand(proximity_level, estimator)
        if fraction is None:
            return None

        from dice.analysis.statistics import precision_interval

        total = len(self.last_result.d_estimates(estimator))
        return {'fraction': fraction, **precision_interval(fraction, total)}

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
