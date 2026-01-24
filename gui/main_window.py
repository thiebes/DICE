import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QLineEdit, QSpinBox, QComboBox,
    QRadioButton, QGroupBox, QFormLayout, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QApplication, QFileDialog, QCheckBox,
    QProgressBar, QTextEdit, QMessageBox, QTableView, QTabWidget,
    QScrollArea, QSplitter, QFrame
)
from PyQt6.QtGui import QDoubleValidator, QPixmap, QIntValidator, QPalette, QValidator
from PyQt6.QtCore import QTimer, QThread, pyqtSignal, Qt # For delayed exit and threading
import traceback # For error logging
import pandas as pd # For table model and results handling
import json
import os # For directory memory in file dialogs
# Matplotlib imports for embedding plot
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import the simulation function
from dice.simulation import run_simulation
from dice.reporting import plot_accuracy_histogram, summarize_results # For displaying plot and summary
from dice.parameters import (
    parameter_parser,
    process_nominal_diffusion_length,
    process_initial_profile,
    process_noise_parameters,
    process_time_axis
)
from .pandas_model import PandasTableModel # For QTableView


class ValidatedLineEdit(QLineEdit):
    """
    Custom QLineEdit with visual validation feedback that only appears after user interaction.

    This widget extends QLineEdit to provide color-coded visual feedback based on validation
    state, but only after the user has interacted with the field (to avoid flashing validation
    styles when the widget is first created or programmatically populated).

    Visual feedback:
        - Green border: Valid input (QValidator.State.Acceptable or non-empty when no validator)
        - Orange border: Intermediate input (QValidator.State.Intermediate or empty when no validator)
        - Red border: Invalid input (QValidator.State.Invalid)

    The validation styling is triggered by:
        - textChanged signal (but only if user has interacted)
        - editingFinished signal (which also marks the field as interacted)
    """

    def __init__(self, validator=None, parent=None):
        """
        Initialize the validated line edit.

        Parameters:
            validator (QValidator, optional): Qt validator to apply to the input. Defaults to None.
            parent (QWidget, optional): Parent widget. Defaults to None.
        """
        super().__init__(parent)
        self._user_interacted = False
        if validator:
            self.setValidator(validator)
        self.textChanged.connect(self._on_text_changed)
        self.editingFinished.connect(self._mark_interacted)

    def _mark_interacted(self):
        """
        Mark the field as having been interacted with by the user.

        This slot is connected to the editingFinished signal and sets the internal
        flag to enable validation styling on subsequent text changes.
        """
        self._user_interacted = True

    def _on_text_changed(self):
        """
        Apply validation styling based on current input state.

        This slot is connected to the textChanged signal and applies color-coded borders
        based on validation state. Styling is only applied if the user has already
        interacted with the field to prevent visual flashing on initialization.

        Validation logic:
            - If a validator is set: Uses QValidator state (Acceptable/Intermediate/Invalid)
            - If no validator: Checks if text is non-empty (green) or empty (orange)
        """
        if not self._user_interacted:
            return

        if self.validator():
            state = self.validator().validate(self.text(), 0)[0]
            if state == QValidator.State.Acceptable:
                self.setStyleSheet("border: 2px solid green;")
            elif state == QValidator.State.Intermediate:
                self.setStyleSheet("border: 2px solid orange;")
            else:
                self.setStyleSheet("border: 2px solid red;")
        else:
            # No validator, check if empty
            if self.text().strip():
                self.setStyleSheet("border: 2px solid green;")
            else:
                self.setStyleSheet("border: 2px solid orange;")


# Simulation Thread
class SimulationThread(QThread):
    """
    Background thread for running DICE simulations without blocking the GUI.

    This thread runs the simulation in the background and emits signals to update
    the GUI with progress, messages, and final results. Error handling is implemented
    to catch exceptions and report them back to the main thread.

    Signals:
        progress_updated(int, int): Emitted with (current_step, total_steps) during simulation.
        message_logged(str): Emitted with log messages during simulation.
        simulation_finished(object, bool): Emitted with (result_or_error, success) when complete.
            - If successful: (result_dictionary, True)
            - If failed: (error_message, False)
    """
    progress_updated = pyqtSignal(int, int)
    message_logged = pyqtSignal(str)
    simulation_finished = pyqtSignal(object, bool) # (result_or_error, success)

    def __init__(self, parameters_dict, parent=None):
        """
        Initialize the simulation thread.

        Parameters:
            parameters_dict (dict): Dictionary of simulation parameters.
            parent (QObject, optional): Parent QObject. Defaults to None.
        """
        super().__init__(parent)
        self.parameters_dict = parameters_dict

    def run(self):
        """
        Execute the simulation in the background thread.

        This method is called automatically when the thread starts. It runs the
        simulation with the provided parameters and emits appropriate signals
        for progress updates, messages, and final results. Exceptions are caught
        and reported via the simulation_finished signal.
        """
        try:
            # run_simulation now takes progress_callback and message_callback
            result_dictionary = run_simulation(
                self.parameters_dict,
                progress_callback=self.emit_progress,
                message_callback=self.emit_message
            )
            self.simulation_finished.emit(result_dictionary, True)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            import traceback
            tb_str = traceback.format_exc()
            self.emit_message(f"ERROR: {error_msg}\n\nFull traceback:\n{tb_str}")
            self.simulation_finished.emit(error_msg, False)

    def emit_progress(self, current_step, total_steps):
        """
        Emit progress update signal.

        Parameters:
            current_step (int): Current simulation step.
            total_steps (int): Total number of steps.
        """
        self.progress_updated.emit(current_step, total_steps)

    def emit_message(self, message):
        """
        Emit message log signal.

        Parameters:
            message (str): Log message to display in GUI.
        """
        self.message_logged.emit(message)


class MainWindow(QMainWindow):
    """
    Main window for the DICE simulation GUI application.

    This window provides a comprehensive interface for configuring and running DICE
    (Diffusion Insight Computation Engine) simulations. It includes parameter input
    widgets, validation, simulation execution with progress tracking, and results
    visualization.

    Key features:
        - Parameter input with real-time validation and visual feedback
        - Background simulation execution using QThread
        - Progress bar and log output for monitoring simulation status
        - Results visualization with matplotlib integration
        - Export capabilities for plots, summaries, and CSV data
        - Directory memory for file dialogs

    Attributes:
        simulation_results (dict): Most recent simulation results, or None if no simulation run.
        results_canvas (FigureCanvas): Matplotlib canvas for displaying results plots.
        _last_directory (str): Last directory used in file dialogs for improved UX.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DICE Simulation")
        self.setGeometry(100, 100, 800, 600) # x, y, width, height

        # Directory memory for file dialogs
        self._last_directory = os.path.expanduser("~")

        # Central Widget and Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create tab widget
        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        # Tab 1: Basic Setup
        basic_tab = QWidget()
        basic_layout = QVBoxLayout(basic_tab)
        tabs.addTab(basic_tab, "Basic Setup")

        # Filename and Runs Group
        run_settings_group = QGroupBox("Run Settings")
        run_settings_layout = QFormLayout(run_settings_group)
        self.filename_slug_edit = QLineEdit("example")
        self.num_runs_spinbox = QSpinBox()
        self.num_runs_spinbox.setRange(1, 1_000_000)
        self.num_runs_spinbox.setValue(1000)
        run_settings_layout.addRow("Filename Slug:", self.filename_slug_edit)
        run_settings_layout.addRow("Number of Runs:", self.num_runs_spinbox)
        basic_layout.addWidget(run_settings_group)

        # Units Group
        units_group = QGroupBox("Units")
        units_layout = QFormLayout(units_group)
        self.length_unit_combo = QComboBox()
        self.length_unit_combo.addItems(['micrometer', 'nanometer', 'angstrom', 'centimeter', 'millimeter', 'meter', 'picometer'])
        self.length_unit_combo.setCurrentText('micrometer')
        self.time_unit_combo = QComboBox()
        self.time_unit_combo.addItems(['nanosecond', 'picosecond', 'femtosecond', 'attosecond', 'microsecond', 'millisecond', 'second'])
        self.time_unit_combo.setCurrentText('nanosecond')
        units_layout.addRow("Length Unit:", self.length_unit_combo)
        units_layout.addRow("Time Unit:", self.time_unit_combo)
        basic_layout.addWidget(units_group)
        basic_layout.addStretch()

        # Tab 2: Physical Parameters
        physical_tab = QWidget()
        physical_layout = QVBoxLayout(physical_tab)
        tabs.addTab(physical_tab, "Physical Parameters")

        # Diffusion and Lifetime Group
        diffusion_lifetime_group = QGroupBox("Diffusion and Lifetime")
        diffusion_lifetime_layout = QVBoxLayout(diffusion_lifetime_group)
        
        self.radio_diffusion_length = QRadioButton("Nominal Diffusion Length")
        self.radio_diffusion_coeff_lifetime = QRadioButton("Nominal Diffusion Coefficient & Lifetime")
        
        diffusion_lifetime_inputs_layout = QFormLayout()
        self.nominal_diffusion_length_edit = QLineEdit("0.1")
        self.nominal_diffusion_length_edit.setValidator(QDoubleValidator())
        self.nominal_diffusion_coeff_edit = QLineEdit("0.01")
        self.nominal_diffusion_coeff_edit.setValidator(QDoubleValidator())
        self.nominal_lifetime_edit = QLineEdit("1")
        self.nominal_lifetime_edit.setValidator(QDoubleValidator())

        diffusion_lifetime_inputs_layout.addRow("Nominal Diffusion Length (Ld):", self.nominal_diffusion_length_edit)
        diffusion_lifetime_inputs_layout.addRow("Nominal Diffusion Coefficient (D):", self.nominal_diffusion_coeff_edit)
        diffusion_lifetime_inputs_layout.addRow("Nominal Lifetime (τ):", self.nominal_lifetime_edit)

        diffusion_lifetime_layout.addWidget(self.radio_diffusion_length)
        diffusion_lifetime_layout.addWidget(self.radio_diffusion_coeff_lifetime)
        diffusion_lifetime_layout.addLayout(diffusion_lifetime_inputs_layout)
        physical_layout.addWidget(diffusion_lifetime_group)

        # Radio button logic
        self.radio_diffusion_length.setChecked(True)
        self.nominal_diffusion_coeff_edit.setEnabled(False)
        self.nominal_lifetime_edit.setEnabled(False)

        self.radio_diffusion_length.toggled.connect(self.toggle_diffusion_inputs)
        self.radio_diffusion_coeff_lifetime.toggled.connect(self.toggle_diffusion_inputs)

        # Dynamic synchronization: Ld = sqrt(D * τ)
        self._updating_diffusion_fields = False
        self.nominal_diffusion_length_edit.editingFinished.connect(self._on_diffusion_length_changed)
        self.nominal_diffusion_coeff_edit.editingFinished.connect(self._on_diffusion_coeff_changed)
        self.nominal_lifetime_edit.editingFinished.connect(self._on_lifetime_changed)

        # Initial Profile Parameters Group
        initial_profile_group = QGroupBox("Initial Profile")
        initial_profile_main_layout = QVBoxLayout(initial_profile_group)
        
        self.radio_fwhm0 = QRadioButton("FWHM_0")
        self.radio_sigma0 = QRadioButton("sigma_0")
        
        initial_profile_inputs_layout = QFormLayout()
        self.fwhm0_edit = QLineEdit("1.0")
        self.fwhm0_edit.setValidator(QDoubleValidator())
        self.sigma0_edit = QLineEdit("0.4032")
        self.sigma0_edit.setValidator(QDoubleValidator())
        self.amplitude0_edit = QLineEdit("1.0")
        self.amplitude0_edit.setValidator(QDoubleValidator())
        self.mean0_edit = QLineEdit("0.0")
        self.mean0_edit.setValidator(QDoubleValidator())

        initial_profile_inputs_layout.addRow("FWHM_0:", self.fwhm0_edit)
        initial_profile_inputs_layout.addRow("sigma_0:", self.sigma0_edit)
        initial_profile_inputs_layout.addRow("Amplitude_0:", self.amplitude0_edit)
        initial_profile_inputs_layout.addRow("Mean_0:", self.mean0_edit)
        
        initial_profile_main_layout.addWidget(self.radio_fwhm0)
        initial_profile_main_layout.addWidget(self.radio_sigma0)
        initial_profile_main_layout.addLayout(initial_profile_inputs_layout)
        physical_layout.addWidget(initial_profile_group)

        self.radio_fwhm0.setChecked(True)
        self.sigma0_edit.setEnabled(False)
        self.radio_fwhm0.toggled.connect(self.toggle_initial_profile_inputs)
        self.radio_sigma0.toggled.connect(self.toggle_initial_profile_inputs)

        # Dynamic synchronization: FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.355 * sigma
        self._updating_profile_width_fields = False
        self.fwhm0_edit.editingFinished.connect(self._on_fwhm_changed)
        self.sigma0_edit.editingFinished.connect(self._on_sigma_changed)

        # Noise Parameter Group
        noise_group = QGroupBox("Noise")
        noise_main_layout = QVBoxLayout(noise_group)

        self.radio_noise_value = QRadioButton("Noise Value")
        self.radio_estimate_noise = QRadioButton("Estimate Noise from Data")

        noise_inputs_layout = QFormLayout()
        self.noise_value_edit = QLineEdit("0.02")
        self.noise_value_edit.setValidator(QDoubleValidator())
        self.estimate_noise_file_edit = QLineEdit("example_profile.csv")
        self.browse_noise_file_button = QPushButton("Browse...")
        
        noise_inputs_layout.addRow("Noise Value:", self.noise_value_edit)
        noise_inputs_layout.addRow("Estimate Noise File:", self.estimate_noise_file_edit)
        noise_inputs_layout.addRow(self.browse_noise_file_button)

        noise_main_layout.addWidget(self.radio_noise_value)
        noise_main_layout.addWidget(self.radio_estimate_noise)
        noise_main_layout.addLayout(noise_inputs_layout)
        physical_layout.addWidget(noise_group)
        physical_layout.addStretch()

        # Tab 3: Spatial & Time
        spatial_time_tab = QWidget()
        spatial_time_layout = QVBoxLayout(spatial_time_tab)
        tabs.addTab(spatial_time_tab, "Spatial & Time")

        self.radio_noise_value.setChecked(True)
        self.estimate_noise_file_edit.setEnabled(False)
        self.browse_noise_file_button.setEnabled(False)
        self.radio_noise_value.toggled.connect(self.toggle_noise_inputs)
        self.radio_estimate_noise.toggled.connect(self.toggle_noise_inputs)
        self.browse_noise_file_button.clicked.connect(self._browse_noise_file)

        # Spatial Axis Parameters Group
        spatial_axis_group = QGroupBox("Spatial Axis")
        spatial_axis_layout = QFormLayout(spatial_axis_group)
        self.spatial_width_edit = QLineEdit("5")
        self.spatial_width_edit.setValidator(QDoubleValidator())
        self.pixel_width_spinbox = QSpinBox()
        self.pixel_width_spinbox.setRange(1, 10000)
        self.pixel_width_spinbox.setValue(100)
        spatial_axis_layout.addRow("Spatial Width:", self.spatial_width_edit)
        spatial_axis_layout.addRow("Pixel Width:", self.pixel_width_spinbox)
        spatial_time_layout.addWidget(spatial_axis_group)

        # Time Axis Parameters Group
        time_axis_group = QGroupBox("Time Axis")
        time_axis_main_layout = QVBoxLayout(time_axis_group)

        self.radio_time_range = QRadioButton("Time Range")
        self.radio_time_series = QRadioButton("Time Series")

        time_range_layout = QHBoxLayout()
        self.time_range_start_edit = QLineEdit("0")
        self.time_range_start_edit.setValidator(QDoubleValidator())
        self.time_range_stop_edit = QLineEdit("1")
        self.time_range_stop_edit.setValidator(QDoubleValidator())
        self.time_range_steps_edit = QLineEdit("10")
        self.time_range_steps_edit.setValidator(QIntValidator())
        time_range_layout.addWidget(QLabel("Start:"))
        time_range_layout.addWidget(self.time_range_start_edit)
        time_range_layout.addWidget(QLabel("Stop:"))
        time_range_layout.addWidget(self.time_range_stop_edit)
        time_range_layout.addWidget(QLabel("Steps:"))
        time_range_layout.addWidget(self.time_range_steps_edit)
        
        self.time_series_edit = QLineEdit("1,3,5,10,15,25,60,100")

        time_axis_main_layout.addWidget(self.radio_time_range)
        time_axis_main_layout.addLayout(time_range_layout)
        time_axis_main_layout.addWidget(self.radio_time_series)
        time_axis_main_layout.addWidget(self.time_series_edit)
        spatial_time_layout.addWidget(time_axis_group)

        self.radio_time_range.setChecked(True)
        self.time_series_edit.setEnabled(False)
        self.radio_time_range.toggled.connect(self.toggle_time_axis_inputs)
        self.radio_time_series.toggled.connect(self.toggle_time_axis_inputs)

        # Analysis Group (Proximity Level)
        analysis_group = QGroupBox("Analysis")
        analysis_layout = QFormLayout(analysis_group)
        self.proximity_level_edit = QLineEdit("0.5")
        proximity_validator = QDoubleValidator(0.0, 1.0, 2) # Min, Max, Decimals
        self.proximity_level_edit.setValidator(proximity_validator)
        analysis_layout.addRow("Diffusion Coefficient Proximity Threshold (0.0-1.0):", self.proximity_level_edit)
        spatial_time_layout.addWidget(analysis_group)
        spatial_time_layout.addStretch()

        # Tab 4: Plot Settings
        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)
        tabs.addTab(plot_tab, "Plot Settings")

        # Plot Image Settings Group
        plot_image_group = QGroupBox("Plot Image Settings")
        plot_image_layout = QFormLayout(plot_image_group)

        self.image_type_combo = QComboBox()
        self.image_type_combo.addItems(['png', 'jpg', 'svg', 'tif'])
        self.image_type_combo.setCurrentText('png')
        self.image_width_edit = QLineEdit("8.5")
        self.image_width_edit.setValidator(QDoubleValidator(0.1, 100.0, 2)) # Min, Max, Decimals
        self.image_height_edit = QLineEdit("5.0")
        self.image_height_edit.setValidator(QDoubleValidator(0.1, 100.0, 2))
        self.image_dpi_spinbox = QSpinBox()
        self.image_dpi_spinbox.setRange(72, 1200)
        self.image_dpi_spinbox.setValue(300)
        self.image_font_size_spinbox = QSpinBox()
        self.image_font_size_spinbox.setRange(5, 72)
        self.image_font_size_spinbox.setValue(8)
        self.image_tick_length_spinbox = QSpinBox()
        self.image_tick_length_spinbox.setRange(1, 50)
        self.image_tick_length_spinbox.setValue(6)
        self.image_tick_width_spinbox = QSpinBox()
        self.image_tick_width_spinbox.setRange(1, 50)
        self.image_tick_width_spinbox.setValue(2)
        self.image_numbins_spinbox = QSpinBox()
        self.image_numbins_spinbox.setRange(5, 200)
        self.image_numbins_spinbox.setValue(35)
        self.image_xlim_edit = QLineEdit("None")

        plot_image_layout.addRow("Image Type:", self.image_type_combo)
        plot_image_layout.addRow("Image Width (inches):", self.image_width_edit)
        plot_image_layout.addRow("Image Height (inches):", self.image_height_edit)
        plot_image_layout.addRow("Image DPI:", self.image_dpi_spinbox)
        plot_image_layout.addRow("Image Font Size:", self.image_font_size_spinbox)
        plot_image_layout.addRow("Image Tick Length:", self.image_tick_length_spinbox)
        plot_image_layout.addRow("Image Tick Width:", self.image_tick_width_spinbox)
        plot_image_layout.addRow("Image Num Bins (Histogram):", self.image_numbins_spinbox)
        plot_image_layout.addRow("Image X-Limits (e.g., 0.0, 2.0 or None):", self.image_xlim_edit)
        plot_layout.addWidget(plot_image_group)

        # Performance Settings Group
        performance_group = QGroupBox("Performance")
        performance_layout = QFormLayout(performance_group)
        self.retain_profile_data_checkbox = QCheckBox()
        self.retain_profile_data_checkbox.setChecked(False)
        self.multiprocessing_checkbox = QCheckBox()
        self.multiprocessing_checkbox.setChecked(True)
        performance_layout.addRow("Retain Profile Data:", self.retain_profile_data_checkbox)
        performance_layout.addRow("Enable Multiprocessing:", self.multiprocessing_checkbox)
        plot_layout.addWidget(performance_group)
        plot_layout.addStretch()

        # Tab 5: Run & Results
        run_results_tab = QWidget()
        run_results_layout = QVBoxLayout(run_results_tab)
        tabs.addTab(run_results_tab, "Run & Results")

        # Simulation Control
        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self._start_simulation)
        run_results_layout.addWidget(self.run_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        run_results_layout.addWidget(self.progress_bar)

        self.log_output_area = QTextEdit()
        self.log_output_area.setReadOnly(True)
        run_results_layout.addWidget(self.log_output_area)

        # Results Display Area
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group) # Main layout for this group

        # Matplotlib Canvas for Plot
        self.results_canvas = FigureCanvas(Figure(figsize=(5, 3))) # Initial empty figure
        results_layout.addWidget(self.results_canvas)

        # Table View for Data
        self.results_table_view = QTableView()
        self.table_model = PandasTableModel() # Create instance of PandasTableModel
        self.results_table_view.setModel(self.table_model) # Set the model
        results_layout.addWidget(self.results_table_view)

        # Save Buttons
        save_buttons_layout = QHBoxLayout()
        self.save_plot_button = QPushButton("Save Plot")
        self.save_plot_button.clicked.connect(self._save_plot)
        self.save_plot_button.setEnabled(False)
        save_buttons_layout.addWidget(self.save_plot_button)

        self.save_summary_button = QPushButton("Save Summary")
        self.save_summary_button.clicked.connect(self._save_summary)
        self.save_summary_button.setEnabled(False)
        save_buttons_layout.addWidget(self.save_summary_button)

        self.save_csv_button = QPushButton("Save CSV Data")
        self.save_csv_button.clicked.connect(self._save_csv_data)
        self.save_csv_button.setEnabled(False)
        save_buttons_layout.addWidget(self.save_csv_button)
        
        results_layout.addLayout(save_buttons_layout) # Add button layout to the results group

        run_results_layout.addWidget(results_group)

        self.simulation_thread = None # To hold the thread object
        self.simulation_results = None # To hold results or exception

    def _start_simulation(self):
        """
        Initiate the simulation workflow after validation and parameter parsing.

        This slot is connected to the "Run Simulation" button and orchestrates the full
        simulation startup process:
            1. Clear log output and reset progress bar
            2. Disable save buttons (no results yet)
            3. Validate parameters using validate_parameters()
            4. Build parameter dictionary from GUI widgets
            5. Parse parameters with parameter_parser() to add derived fields
            6. Create and start SimulationThread with parsed parameters
            7. Connect thread signals to GUI update slots

        If validation or parsing fails, appropriate error messages are shown to the user
        and the simulation is not started.

        This method handles:
            - ValueError: Parameter validation errors
            - KeyError: Missing required parameters
            - TypeError: Parameter type mismatches

        Side effects:
            - Disables run button during simulation
            - Updates log output area with status messages
            - Shows QMessageBox for validation/parsing errors
            - Creates and starts self.simulation_thread
        """
        self.log_output_area.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setRange(0, 100) # Initial range, will be updated

        # Disable save buttons when starting a new simulation
        self.save_plot_button.setEnabled(False)
        self.save_summary_button.setEnabled(False)
        self.save_csv_button.setEnabled(False)

        # Validate parameters first
        is_valid, errors = self.validate_parameters()
        if not is_valid:
            error_msg = "Parameter validation failed:\n\n" + "\n".join(f"- {e}" for e in errors)
            self.log_output_area.append(error_msg)
            QMessageBox.critical(self, "Validation Error", error_msg)
            return

        params = self.get_parameters_dict()
        if params is None:
            # get_parameters_dict already logs to log_output_area and can show a QMessageBox
            # if it returns None, so we just ensure the button is re-enabled.
            self.run_button.setEnabled(True) 
            return

        self.run_button.setEnabled(False)
        self.log_output_area.append("Attempting to fully parse parameters...")

        try:
            parsed_params = parameter_parser(params.copy()) # Use a copy to avoid modifying the original dict
            self.log_output_area.append("Parameters successfully parsed. Starting simulation...")
        except (ValueError, KeyError, TypeError) as e: # Catch specific errors from parameter_parser
            error_msg = f"Parameter Error from parser: {e}"
            self.log_output_area.append(error_msg)
            QMessageBox.critical(self, "Parameter Validation Error", error_msg)
            self.run_button.setEnabled(True) # Re-enable button
            return # Do not start simulation

        self.simulation_thread = SimulationThread(parsed_params)
        self.simulation_thread.progress_updated.connect(self._on_simulation_progress)
        self.simulation_thread.message_logged.connect(self._on_simulation_message)
        self.simulation_thread.simulation_finished.connect(self._on_simulation_finished)
        self.simulation_thread.start()

    def _on_simulation_progress(self, current_step, total_steps):
        if total_steps > 0:
            self.progress_bar.setMaximum(total_steps)
            self.progress_bar.setValue(current_step)
        else: # Should ideally not happen if run_simulation provides valid totals
            self.progress_bar.setRange(0,0) # Indeterminate

    def _on_simulation_message(self, message):
        self.log_output_area.append(message)

    def _on_simulation_finished(self, result, success):
        """
        Handle simulation completion signal from SimulationThread.

        This slot is connected to the simulation_finished signal and processes the
        simulation results or error based on the success flag.

        Parameters:
            result (dict or str): Simulation results dictionary if success=True,
                                 or error message string if success=False.
            success (bool): True if simulation completed successfully, False if error occurred.

        Behavior on success:
            - Sets progress bar to 100%
            - Stores results in self.simulation_results
            - Displays results via _display_results()
            - Shows success message dialog
            - Re-enables run button

        Behavior on failure:
            - Resets progress bar to 0
            - Disables save buttons
            - Shows error message dialog
            - Sets self.simulation_results to None
            - Re-enables run button
        """
        self.run_button.setEnabled(True)
        if not success:
            # result is error message string
            self.progress_bar.setValue(0)
            self.save_plot_button.setEnabled(False)
            self.save_summary_button.setEnabled(False)
            self.save_csv_button.setEnabled(False)
            # Message already logged by thread
            QMessageBox.critical(self, "Simulation Error", result)
            self.simulation_results = None
        else:
            # result is dictionary
            self.progress_bar.setValue(self.progress_bar.maximum())
            self.log_output_area.append("Simulation complete.")
            self.simulation_results = result
            QMessageBox.information(self, "Simulation Complete", "Simulation finished successfully.")
            self._display_results(result)


    def _display_results(self, result_dictionary):
        """
        Display simulation results in the GUI results area.

        This method generates the accuracy histogram plot using plot_accuracy_histogram()
        and displays it in the matplotlib canvas widget. It also enables the save buttons
        for exporting plots, summaries, and CSV data.

        Parameters:
            result_dictionary (dict): Simulation results from run_simulation().
                Required keys: 'parameters', 'collated results'

        Side effects:
            - Replaces existing matplotlib canvas with new plot
            - Enables save plot/summary/CSV buttons
            - Logs plot generation status to log output area
            - Properly disposes of old canvas to prevent memory leaks

        Error handling:
            - Validates result_dictionary structure
            - Catches and displays plot generation errors
            - Logs errors to log output area
        """
        if not result_dictionary or 'parameters' not in result_dictionary or 'collated results' not in result_dictionary:
            self.log_output_area.append("Error: Invalid result dictionary for display.")
            return

        # Display Plot
        try:
            self.log_output_area.append("Generating plot...")
            # Prepare parameters for plot_accuracy_histogram
            plot_params = {
                'simulation_result': result_dictionary,
                'proximity': result_dictionary['parameters'].get('proximity level', 0.1),
                'filename': "gui_temp_plot.png",  # Dummy, as we are returning the figure
                'image_type': result_dictionary['parameters'].get('image type', 'png'),
                'width': result_dictionary['parameters'].get('image width', 8.5),
                'height': result_dictionary['parameters'].get('image height', 5.0),
                'dpi': result_dictionary['parameters'].get('image dpi', 300),
                'font_size': result_dictionary['parameters'].get('image font size', 8),
                'tick_length': result_dictionary['parameters'].get('image tick length', 6),
                'tick_width': result_dictionary['parameters'].get('image tick width', 2),
                'num_bins': result_dictionary['parameters'].get('image numbins', 35),
                'x_lim': result_dictionary['parameters'].get('image x_lim', None),
                'return_figure': True
            }
            
            # plot_accuracy_histogram should not save file if return_figure is True,
            # and should not call plt.close()
            returned_fig = plot_accuracy_histogram(**plot_params)

            if returned_fig:
                old_canvas = self.results_canvas
                new_canvas = FigureCanvas(returned_fig)

                layout = self.results_canvas.parent().layout()
                layout.replaceWidget(old_canvas, new_canvas)

                old_canvas.deleteLater()

                self.results_canvas = new_canvas
                self.results_canvas.draw()
                self.log_output_area.append("Plot displayed.")
            else:
                self.log_output_area.append("Plot generation failed or returned no figure.")

        except Exception as e:
            self.log_output_area.append(f"Error displaying plot: {e}")
            traceback.print_exc()

        # Display Table Data
        try:
            collated_df = result_dictionary.get('collated results')
            if isinstance(collated_df, pd.DataFrame):
                self.table_model.setData(collated_df)
                self.log_output_area.append("Collated results displayed in table.")
                # Enable save buttons as results are available
                self.save_plot_button.setEnabled(True)
                self.save_summary_button.setEnabled(True)
                self.save_csv_button.setEnabled(True)
            else:
                self.log_output_area.append("No collated results DataFrame found to display in table.")
                # Keep save buttons disabled if no valid data
                self.save_plot_button.setEnabled(False)
                self.save_summary_button.setEnabled(False)
                self.save_csv_button.setEnabled(False)
        except Exception as e:
            self.log_output_area.append(f"Error displaying table data: {e}")
            traceback.print_exc()
            self.save_plot_button.setEnabled(False)
            self.save_summary_button.setEnabled(False)
            self.save_csv_button.setEnabled(False)

    def _save_plot(self):
        if not (hasattr(self.results_canvas.figure, 'axes') and self.results_canvas.figure.axes):
            QMessageBox.warning(self, "Save Error", "No plot to save.")
            return

        fileName, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", self._last_directory,
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;SVG (*.svg);;PDF (*.pdf);;All Files (*)"
        )
        if fileName:
            try:
                self.results_canvas.figure.savefig(fileName)
                self._last_directory = os.path.dirname(fileName)
                QMessageBox.information(self, "Success", f"Plot saved to {fileName}")
            except Exception as e:
                QMessageBox.warning(self, "Save Error", f"Could not save plot: {e}")
                traceback.print_exc()
    
    def _save_summary(self):
        if self.simulation_results is None or isinstance(self.simulation_results, Exception):
            QMessageBox.warning(self, "Save Error", "No simulation summary available to save.")
            return

        fileName, _ = QFileDialog.getSaveFileName(
            self, "Save Summary", self._last_directory, "Text Files (*.txt);;All Files (*)"
        )
        if fileName:
            try:
                summary_lines = summarize_results(self.simulation_results)
                with open(fileName, 'w') as f:
                    for line in summary_lines:
                        f.write(line + '\n')
                self._last_directory = os.path.dirname(fileName)
                QMessageBox.information(self, "Success", f"Summary saved to {fileName}")
            except Exception as e:
                QMessageBox.warning(self, "Save Error", f"Could not save summary: {e}")
                traceback.print_exc()

    def _save_csv_data(self):
        if (self.simulation_results is None or
            isinstance(self.simulation_results, Exception) or
            'collated results' not in self.simulation_results or
            not isinstance(self.simulation_results['collated results'], pd.DataFrame)):
            QMessageBox.warning(self, "Save Error", "No CSV data available to save.")
            return

        fileName, _ = QFileDialog.getSaveFileName(
            self, "Save CSV Data", self._last_directory, "CSV Files (*.csv);;All Files (*)"
        )
        if fileName:
            try:
                df = self.simulation_results['collated results']
                df.to_csv(fileName, index=False)
                self._last_directory = os.path.dirname(fileName)
                QMessageBox.information(self, "Success", f"CSV data saved to {fileName}")
            except Exception as e:
                QMessageBox.warning(self, "Save Error", f"Could not save CSV data: {e}")
                traceback.print_exc()

    def _browse_noise_file(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Noise File", self._last_directory, "CSV Files (*.csv);;All Files (*)")
        if fileName:
            self.estimate_noise_file_edit.setText(fileName)
            self._last_directory = os.path.dirname(fileName)

    def toggle_initial_profile_inputs(self):
        is_fwhm0_selected = self.radio_fwhm0.isChecked()
        self.fwhm0_edit.setEnabled(is_fwhm0_selected)
        self.sigma0_edit.setEnabled(not is_fwhm0_selected)

    def _on_fwhm_changed(self):
        """When FWHM changes: sigma = FWHM / (2 * sqrt(2 * ln(2)))"""
        if self._updating_profile_width_fields:
            return
        try:
            import math
            fwhm = float(self.fwhm0_edit.text())
            if fwhm <= 0:
                return
            self._updating_profile_width_fields = True
            # FWHM = 2 * sqrt(2 * ln(2)) * sigma
            fwhm_to_sigma = 2 * math.sqrt(2 * math.log(2))
            sigma = fwhm / fwhm_to_sigma
            self.sigma0_edit.setText(str(sigma))
        except ValueError:
            pass
        finally:
            self._updating_profile_width_fields = False

    def _on_sigma_changed(self):
        """When sigma changes: FWHM = 2 * sqrt(2 * ln(2)) * sigma"""
        if self._updating_profile_width_fields:
            return
        try:
            import math
            sigma = float(self.sigma0_edit.text())
            if sigma <= 0:
                return
            self._updating_profile_width_fields = True
            # FWHM = 2 * sqrt(2 * ln(2)) * sigma
            fwhm_to_sigma = 2 * math.sqrt(2 * math.log(2))
            fwhm = sigma * fwhm_to_sigma
            self.fwhm0_edit.setText(str(fwhm))
        except ValueError:
            pass
        finally:
            self._updating_profile_width_fields = False

    def toggle_noise_inputs(self):
        is_noise_value_selected = self.radio_noise_value.isChecked()
        self.noise_value_edit.setEnabled(is_noise_value_selected)
        self.estimate_noise_file_edit.setEnabled(not is_noise_value_selected)
        self.browse_noise_file_button.setEnabled(not is_noise_value_selected)

    def toggle_time_axis_inputs(self):
        is_time_range_selected = self.radio_time_range.isChecked()
        self.time_range_start_edit.setEnabled(is_time_range_selected)
        self.time_range_stop_edit.setEnabled(is_time_range_selected)
        self.time_range_steps_edit.setEnabled(is_time_range_selected)
        self.time_series_edit.setEnabled(not is_time_range_selected)

    def take_screenshot(self):
        # Ensure the window is shown and sized correctly before taking a screenshot
        self.show() 
        QApplication.processEvents() # Process pending events to ensure window is drawn
        
        screenshot = self.grab() # Grab the widget's content
        screenshot.save("gui_screenshot_final.png", "png")
        print("Screenshot saved as gui_screenshot_final.png")

    def toggle_diffusion_inputs(self):
        is_diffusion_length_selected = self.radio_diffusion_length.isChecked()
        self.nominal_diffusion_length_edit.setEnabled(is_diffusion_length_selected)
        self.nominal_diffusion_coeff_edit.setEnabled(not is_diffusion_length_selected)
        self.nominal_lifetime_edit.setEnabled(not is_diffusion_length_selected)

    def _on_diffusion_length_changed(self):
        """When Ld changes: τ = 1.0, D = Ld²"""
        if self._updating_diffusion_fields:
            return
        try:
            ld = float(self.nominal_diffusion_length_edit.text())
            if ld <= 0:
                return
            self._updating_diffusion_fields = True
            tau = 1.0
            d = ld ** 2
            self.nominal_lifetime_edit.setText(str(tau))
            self.nominal_diffusion_coeff_edit.setText(str(d))
        except ValueError:
            pass
        finally:
            self._updating_diffusion_fields = False

    def _on_diffusion_coeff_changed(self):
        """When D changes: τ unchanged, Ld = sqrt(D * τ)"""
        if self._updating_diffusion_fields:
            return
        try:
            d = float(self.nominal_diffusion_coeff_edit.text())
            tau = float(self.nominal_lifetime_edit.text())
            if d <= 0 or tau <= 0:
                return
            self._updating_diffusion_fields = True
            ld = (d * tau) ** 0.5
            self.nominal_diffusion_length_edit.setText(str(ld))
        except ValueError:
            pass
        finally:
            self._updating_diffusion_fields = False

    def _on_lifetime_changed(self):
        """When τ changes: D unchanged, Ld = sqrt(D * τ)"""
        if self._updating_diffusion_fields:
            return
        try:
            d = float(self.nominal_diffusion_coeff_edit.text())
            tau = float(self.nominal_lifetime_edit.text())
            if d <= 0 or tau <= 0:
                return
            self._updating_diffusion_fields = True
            ld = (d * tau) ** 0.5
            self.nominal_diffusion_length_edit.setText(str(ld))
        except ValueError:
            pass
        finally:
            self._updating_diffusion_fields = False

    def validate_parameters(self):
        """
        Validate GUI parameters before starting simulation.

        Returns:
            tuple: (is_valid: bool, error_messages: list[str])
        """
        errors = []

        # Spatial width > 0
        try:
            spatial_width = float(self.spatial_width_edit.text())
            if spatial_width <= 0:
                errors.append("Spatial width must be greater than 0")
        except ValueError:
            errors.append("Spatial width must be a valid number")

        # Pixel width > 0
        pixel_width = self.pixel_width_spinbox.value()
        if pixel_width <= 0:
            errors.append("Pixel width must be greater than 0")

        # Proximity level 0-1
        try:
            proximity = float(self.proximity_level_edit.text())
            if not (0.0 <= proximity <= 1.0):
                errors.append("Proximity level must be between 0.0 and 1.0")
        except ValueError:
            errors.append("Proximity level must be a valid number between 0.0 and 1.0")

        # Time range/series validation
        has_time_range = self.radio_time_range.isChecked()
        has_time_series = self.radio_time_series.isChecked()

        if has_time_range:
            try:
                start = float(self.time_range_start_edit.text())
                stop = float(self.time_range_stop_edit.text())
                steps = int(self.time_range_steps_edit.text())

                if start >= stop:
                    errors.append("Time range start must be less than stop")
            except ValueError:
                errors.append("Time range values must be valid numbers")
        elif has_time_series:
            try:
                series_text = self.time_series_edit.text().strip()
                if not series_text:
                    errors.append("Time series cannot be empty")
                else:
                    values = [float(t.strip()) for t in series_text.split(',') if t.strip()]
                    if len(values) == 0:
                        errors.append("Time series must contain at least one value")
            except ValueError:
                errors.append("Time series must contain valid comma-separated numbers")
        else:
            errors.append("Either time range or time series must be selected")

        return (len(errors) == 0, errors)

    def get_parameters_dict(self):
        """
        Build parameter dictionary from current GUI widget values.

        This method extracts all simulation parameters from the GUI widgets and constructs
        a dictionary suitable for passing to run_simulation(). It includes fallback default
        values for cases where user input cannot be parsed as the expected type.

        Parameter groups collected:
            - Run settings: filename slug, number of runs, units
            - Diffusion: nominal diffusion length OR (coefficient + lifetime)
            - Initial profile: amplitude, mean, FWHM/sigma
            - Noise: noise value OR estimate from data file
            - Spatial axis: spatial width, pixel width
            - Time axis: time range (start, stop, steps) OR time series (list)
            - Analysis: proximity level
            - Plot settings: image type, dimensions, DPI, font, ticks, bins, x limits
            - Processing: retain profile data, multiprocessing

        Returns:
            dict: Complete parameter dictionary with all required keys for simulation.

        Notes:
            - Uses try/except blocks with fallback defaults for robust error handling
            - Prints warnings to console for invalid inputs
            - Does not validate parameter logical constraints (use validate_parameters() first)
        """
        params = {}

        # Basic validation flags
        valid_params = True
        error_messages = []

        # Helper for float conversion
        def get_float(widget, name, default_val):
            nonlocal valid_params
            try:
                return float(widget.text())
            except ValueError:
                error_messages.append(f"Invalid float value for '{name}'. Using default: {default_val}.")
                # self._on_simulation_message(f"Error: Invalid float value for '{name}'. Check input.") # Log to GUI
                # valid_params = False # This would stop processing immediately
                return default_val # Or raise error / return None to stop

        # Helper for int conversion
        def get_int(widget, name, default_val):
            nonlocal valid_params
            try:
                return int(widget.text())
            except ValueError:
                error_messages.append(f"Invalid integer value for '{name}'. Using default: {default_val}.")
                return default_val
        
        params['filename slug'] = self.filename_slug_edit.text()
        params['number of runs'] = self.num_runs_spinbox.value()
        params['length unit'] = self.length_unit_combo.currentText()
        params['time unit'] = self.time_unit_combo.currentText()

        # Diffusion and Lifetime - wrapped by process_nominal_diffusion_length
        nominal_diffusion_length = None
        nominal_diffusion_coeff = None
        nominal_lifetime_tau = None

        if self.radio_diffusion_length.isChecked():
            nominal_diffusion_length = get_float(self.nominal_diffusion_length_edit, 'nominal diffusion length', 0.1)
        else:
            nominal_diffusion_coeff = get_float(self.nominal_diffusion_coeff_edit, 'nominal diffusion coefficient', 0.01)
            nominal_lifetime_tau = get_float(self.nominal_lifetime_edit, 'nominal lifetime (tau)', 1.0)
        
        try:
            diff_params = process_nominal_diffusion_length(
                params['length unit'], params['time unit'],
                nominal_diffusion_length, nominal_diffusion_coeff, nominal_lifetime_tau
            )
            params.update(diff_params)
        except ValueError as e:
            error_messages.append(f"Diffusion/Lifetime processing error: {e}")
            valid_params = False


        # Initial Profile - wrapped by process_initial_profile
        amplitude_0 = get_float(self.amplitude0_edit, 'amplitude_0', 1.0)
        mean_0 = get_float(self.mean0_edit, 'mean_0', 0.0)
        fwhm_0 = None
        sigma_0 = None
        if self.radio_fwhm0.isChecked():
            fwhm_0 = get_float(self.fwhm0_edit, 'FWHM_0', 1.0)
        else:
            sigma_0 = get_float(self.sigma0_edit, 'sigma_0', 0.4032)

        try:
            profile_params = process_initial_profile(params['length unit'], fwhm_0, sigma_0, amplitude_0, mean_0)
            params.update(profile_params)
        except ValueError as e:
            error_messages.append(f"Initial Profile processing error: {e}")
            valid_params = False
        
        # Noise - wrapped by process_noise_parameters
        noise_value_txt = None
        estimate_noise_file = None
        if self.radio_noise_value.isChecked():
            noise_value_txt = self.noise_value_edit.text()
        else:
            estimate_noise_file = self.estimate_noise_file_edit.text()
        
        try:
            noise_params = process_noise_parameters(
                noise_value_txt, estimate_noise_file, params['number of runs']
            )
            params.update(noise_params)
        except (ValueError, FileNotFoundError) as e: # FileNotFoundError for estimate_noise_file
            error_messages.append(f"Noise processing error: {e}")
            valid_params = False

        # Spatial Axis
        params['spatial width'] = get_float(self.spatial_width_edit, 'spatial width', 5.0)
        params['pixel width'] = self.pixel_width_spinbox.value()
        # 'x array' is derived in open_parameters, needs to be done here too or passed differently
        # For now, let dice.simulation.run_simulation handle x_array creation if not present or do it in a pre-processing step
        # For DICE GUI, it might be better to compute it here if open_parameters is not called directly by GUI.
        # Let's assume run_simulation/open_parameters logic will create it if needed.
        # from dice.utils import make_x_axis
        # params['x array'] = make_x_axis(params['spatial width'], params['pixel width'], params['t0 Gaussian sigma^2, amplitude, mean'][2]) # mu_0
        # This creates a dependency on mu_0 being available. Let's simplify:
        # Parameters like 'x array', 't0 Gaussian sigma^2, amplitude, mean' are expected by run_simulation
        # but are derived by open_parameters. The GUI needs to replicate this derivation.
        # For now, 'x array' will be missing and 't0 Gaussian sigma^2, amplitude, mean' will be missing.
        # This indicates a need to refactor parameter processing to be shared.
        # Let's assume these are handled by a later stage or added by run_simulation based on primary inputs.


        # Time Axis - wrapped by process_time_axis
        time_range_tuple = None
        time_series_str = None
        if self.radio_time_range.isChecked():
            start = get_float(self.time_range_start_edit, 'time range start', 0.0)
            stop = get_float(self.time_range_stop_edit, 'time range stop', 1.0)
            steps = get_int(self.time_range_steps_edit, 'time range steps', 10)
            time_range_tuple = (start, stop, steps)
        else:
            time_series_str = self.time_series_edit.text()
        
        try:
            time_axis_params = process_time_axis(time_range_tuple, time_series_str)
            params.update(time_axis_params) # provides 'time series'
        except ValueError as e:
            error_messages.append(f"Time Axis processing error: {e}")
            valid_params = False
        
        # Analysis (Proximity Level)
        params['proximity level'] = get_float(self.proximity_level_edit, 'proximity level', 0.5)

        # Plot Image Settings
        params['image type'] = self.image_type_combo.currentText()
        params['image width'] = get_float(self.image_width_edit, 'image width', 8.5)
        params['image height'] = get_float(self.image_height_edit, 'image height', 5.0)
        params['image dpi'] = self.image_dpi_spinbox.value()
        params['image font size'] = self.image_font_size_spinbox.value()
        params['image tick length'] = self.image_tick_length_spinbox.value()
        params['image tick width'] = self.image_tick_width_spinbox.value()
        params['image numbins'] = self.image_numbins_spinbox.value()
        
        xlim_text = self.image_xlim_edit.text().strip()
        if xlim_text.lower() == 'none' or not xlim_text:
            params['image x_lim'] = None
        else:
            try:
                parts = [float(p.strip()) for p in xlim_text.split(',')]
                if len(parts) == 2:
                    params['image x_lim'] = parts
                else:
                    error_messages.append("Invalid format for 'image x_lim'. Expected 'None' or 'float1, float2'. Using None.")
                    params['image x_lim'] = None # Default on format error
            except ValueError:
                error_messages.append("Invalid float value in 'image x_lim'. Using None.")
                params['image x_lim'] = None # Default on value error

        # Performance Settings
        params['retain profile data'] = self.retain_profile_data_checkbox.isChecked()
        params['multiprocessing'] = self.multiprocessing_checkbox.isChecked()
        
        if error_messages:
            # Log all collected errors to the GUI log area
            for msg in error_messages:
                if hasattr(self, 'log_output_area') and self.log_output_area is not None: # Check if log area exists
                     self.log_output_area.append(f"Parameter Warning: {msg}")
                else: # Fallback if log area not ready (e.g. during early init)
                    print(f"Parameter Warning: {msg}")
        
        if not valid_params:
            if hasattr(self, 'log_output_area') and self.log_output_area is not None:
                self.log_output_area.append("Critical parameter errors found. Simulation cannot start.")
            else:
                print("Critical parameter errors found. Simulation cannot start.")
            return None # Signal critical error

        # The following parameters are derived by dice.parameters.open_parameters
        # and need to be replicated or passed to a shared processing function.
        # 'x array', 't0 Gaussian sigma^2, amplitude, mean'
        # For now, run_simulation will use the primary values and derive these if its internal logic supports it
        # based on what open_parameters usually does.
        # This is a known gap from the current structure.
        # print(params) # For verification
        return params

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    # # Automatically click the button and take a screenshot then exit
    # def auto_actions():
    #     window.get_params_button.click() # Click the button
    #     window.take_screenshot() # Take screenshot
    #     app.quit() # Quit the application

    # QTimer.singleShot(1000, auto_actions) # Wait 1 second for GUI to render, then perform actions
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
