import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QLineEdit, QSpinBox, QComboBox,
    QRadioButton, QGroupBox, QFormLayout, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QApplication, QFileDialog, QCheckBox,
    QProgressBar, QTextEdit, QMessageBox, QTableView
)
from PyQt6.QtGui import QDoubleValidator, QPixmap, QIntValidator
from PyQt6.QtCore import QTimer, QThread, pyqtSignal, Qt # For delayed exit and threading
import traceback # For error logging
import pandas as pd # For table model and results handling

# Import the simulation function
from dice.simulation import run_simulation
from dice.reporting import plot_accuracy_histogram, summarize_results # For displaying plot and summary
from dice.parameters import ( # For parameter processing
    process_nominal_diffusion_length, process_time_axis, 
    process_initial_profile, process_noise_parameters, parameter_parser # Added parameter_parser
)
from .pandas_model import PandasTableModel # For QTableView

# Matplotlib imports for embedding plot
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# Simulation Thread
class SimulationThread(QThread):
    progress_updated = pyqtSignal(int, int)
    message_logged = pyqtSignal(str)
    simulation_finished = pyqtSignal(object) # Can be result_dict or exception

    def __init__(self, parameters_dict, parent=None):
        super().__init__(parent)
        self.parameters_dict = parameters_dict

    def run(self):
        try:
            # run_simulation now takes progress_callback and message_callback
            result_dictionary = run_simulation(
                self.parameters_dict,
                progress_callback=self.emit_progress,
                message_callback=self.emit_message
            )
            self.simulation_finished.emit(result_dictionary)
        except Exception as e:
            self.simulation_finished.emit(e)

    def emit_progress(self, current_step, total_steps):
        self.progress_updated.emit(current_step, total_steps)

    def emit_message(self, message):
        self.message_logged.emit(message)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DICE Simulation")
        self.setGeometry(100, 100, 800, 600) # x, y, width, height

        # Central Widget and Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Filename and Runs Group
        run_settings_group = QGroupBox("Run Settings")
        run_settings_layout = QFormLayout(run_settings_group)
        self.filename_slug_edit = QLineEdit("example")
        self.num_runs_spinbox = QSpinBox()
        self.num_runs_spinbox.setRange(1, 1_000_000)
        self.num_runs_spinbox.setValue(1000)
        run_settings_layout.addRow("Filename Slug:", self.filename_slug_edit)
        run_settings_layout.addRow("Number of Runs:", self.num_runs_spinbox)
        main_layout.addWidget(run_settings_group)

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
        main_layout.addWidget(units_group)

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
        main_layout.addWidget(diffusion_lifetime_group)

        # Radio button logic
        self.radio_diffusion_length.setChecked(True)
        self.nominal_diffusion_coeff_edit.setEnabled(False)
        self.nominal_lifetime_edit.setEnabled(False)

        self.radio_diffusion_length.toggled.connect(self.toggle_diffusion_inputs)
        self.radio_diffusion_coeff_lifetime.toggled.connect(self.toggle_diffusion_inputs)

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
        main_layout.addWidget(initial_profile_group)

        self.radio_fwhm0.setChecked(True)
        self.sigma0_edit.setEnabled(False)
        self.radio_fwhm0.toggled.connect(self.toggle_initial_profile_inputs)
        self.radio_sigma0.toggled.connect(self.toggle_initial_profile_inputs)

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
        main_layout.addWidget(noise_group)

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
        main_layout.addWidget(spatial_axis_group)

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
        main_layout.addWidget(time_axis_group)

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
        main_layout.addWidget(analysis_group)

        # Temporary button to get parameters
        self.get_params_button = QPushButton("Get Parameters (Print to Console)")
        self.get_params_button.clicked.connect(self.get_parameters_dict)
        main_layout.addWidget(self.get_params_button)


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
        main_layout.addWidget(plot_image_group)

        # Performance Settings Group
        performance_group = QGroupBox("Performance")
        performance_layout = QFormLayout(performance_group)
        self.retain_profile_data_checkbox = QCheckBox()
        self.retain_profile_data_checkbox.setChecked(False)
        self.multiprocessing_checkbox = QCheckBox()
        self.multiprocessing_checkbox.setChecked(True)
        performance_layout.addRow("Retain Profile Data:", self.retain_profile_data_checkbox)
        performance_layout.addRow("Enable Multiprocessing:", self.multiprocessing_checkbox)
        main_layout.addWidget(performance_group)

        # Simulation Control
        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self._start_simulation)
        main_layout.addWidget(self.run_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        self.log_output_area = QTextEdit()
        self.log_output_area.setReadOnly(True)
        main_layout.addWidget(self.log_output_area)

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
        
        main_layout.addWidget(results_group)
        
        self.simulation_thread = None # To hold the thread object
        self.simulation_results = None # To hold results or exception

    def _start_simulation(self):
        self.log_output_area.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setRange(0, 100) # Initial range, will be updated
        
        # Disable save buttons when starting a new simulation
        self.save_plot_button.setEnabled(False)
        self.save_summary_button.setEnabled(False)
        self.save_csv_button.setEnabled(False)

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

    def _on_simulation_finished(self, result):
        self.run_button.setEnabled(True)
        if isinstance(result, Exception):
            self.progress_bar.setValue(0) # Or some indication of error
            # Ensure save buttons remain disabled on error
            self.save_plot_button.setEnabled(False)
            self.save_summary_button.setEnabled(False)
            self.save_csv_button.setEnabled(False)
            error_message = f"Simulation Error: {type(result).__name__}: {result}"
            self.log_output_area.append(error_message)
            QMessageBox.critical(self, "Simulation Error", error_message)
            self.simulation_results = result # Store the exception
            # Print full traceback to console for development/debugging
            print("--- Simulation Thread Exception ---")
            traceback.print_exception(type(result), result, result.__traceback__)
            print("---------------------------------")
        else:
            self.progress_bar.setValue(self.progress_bar.maximum())
            self.log_output_area.append("Simulation complete.")
            self.simulation_results = result # Store the result dictionary
            QMessageBox.information(self, "Simulation Complete", "The simulation has finished successfully.")
            self._display_results(result)


    def _display_results(self, result_dictionary):
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
                # Clear previous figure from canvas
                self.results_canvas.figure.clear() 
                # Assign the new figure to the canvas
                self.results_canvas.figure = returned_fig
                # Redraw the canvas
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
            self, "Save Plot", "", 
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;SVG (*.svg);;PDF (*.pdf);;All Files (*)"
        )
        if fileName:
            try:
                self.results_canvas.figure.savefig(fileName)
                QMessageBox.information(self, "Success", f"Plot saved to {fileName}")
            except Exception as e:
                QMessageBox.warning(self, "Save Error", f"Could not save plot: {e}")
                traceback.print_exc()
    
    def _save_summary(self):
        if self.simulation_results is None or isinstance(self.simulation_results, Exception):
            QMessageBox.warning(self, "Save Error", "No simulation summary available to save.")
            return

        fileName, _ = QFileDialog.getSaveFileName(
            self, "Save Summary", "", "Text Files (*.txt);;All Files (*)"
        )
        if fileName:
            try:
                summary_lines = summarize_results(self.simulation_results)
                with open(fileName, 'w') as f:
                    for line in summary_lines:
                        f.write(line + '\n')
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
            self, "Save CSV Data", "", "CSV Files (*.csv);;All Files (*)"
        )
        if fileName:
            try:
                df = self.simulation_results['collated results']
                df.to_csv(fileName, index=False)
                QMessageBox.information(self, "Success", f"CSV data saved to {fileName}")
            except Exception as e:
                QMessageBox.warning(self, "Save Error", f"Could not save CSV data: {e}")
                traceback.print_exc()

    def _browse_noise_file(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Noise File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if fileName:
            self.estimate_noise_file_edit.setText(fileName)

    def toggle_initial_profile_inputs(self):
        is_fwhm0_selected = self.radio_fwhm0.isChecked()
        self.fwhm0_edit.setEnabled(is_fwhm0_selected)
        self.sigma0_edit.setEnabled(not is_fwhm0_selected)

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

    def get_parameters_dict(self):
        params = {}
        params['filename slug'] = self.filename_slug_edit.text()
        params['number of runs'] = self.num_runs_spinbox.value()
        params['length unit'] = self.length_unit_combo.currentText()
        params['time unit'] = self.time_unit_combo.currentText()

        # Diffusion and Lifetime
        if self.radio_diffusion_length.isChecked():
            try:
                params['nominal diffusion length'] = float(self.nominal_diffusion_length_edit.text())
            except ValueError:
                params['nominal diffusion length'] = 0.1 
        else:
            try:
                params['nominal diffusion coefficient'] = float(self.nominal_diffusion_coeff_edit.text())
                params['nominal lifetime (tau)'] = float(self.nominal_lifetime_edit.text())
            except ValueError:
                params['nominal diffusion coefficient'] = 0.01 
                params['nominal lifetime (tau)'] = 1.0
        
        # Initial Profile
        try:
            params['amplitude_0'] = float(self.amplitude0_edit.text())
            params['mean_0'] = float(self.mean0_edit.text())
            if self.radio_fwhm0.isChecked():
                params['FWHM_0'] = float(self.fwhm0_edit.text())
            else:
                params['sigma_0'] = float(self.sigma0_edit.text())
        except ValueError:
            print("Warning: Invalid float value in Initial Profile.")
            params['amplitude_0'] = 1.0
            params['mean_0'] = 0.0
            if self.radio_fwhm0.isChecked():
                params['FWHM_0'] = 1.0
            else:
                params['sigma_0'] = 0.4032

        # Noise
        if self.radio_noise_value.isChecked():
            try:
                params['noise value'] = float(self.noise_value_edit.text())
            except ValueError:
                params['noise value'] = 0.02
        else:
            params['estimate noise from data'] = self.estimate_noise_file_edit.text()

        # Spatial Axis
        try:
            params['spatial width'] = float(self.spatial_width_edit.text())
        except ValueError:
            params['spatial width'] = 5.0
        params['pixel width'] = self.pixel_width_spinbox.value()

        # Time Axis
        if self.radio_time_range.isChecked():
            try:
                start = float(self.time_range_start_edit.text())
                stop = float(self.time_range_stop_edit.text())
                steps = int(self.time_range_steps_edit.text())
                params['time range'] = [start, stop, steps]
            except ValueError:
                print("Warning: Invalid float/int value in Time Range.")
                params['time range'] = [0.0, 1.0, 10]
        else:
            try:
                params['time series'] = [float(t.strip()) for t in self.time_series_edit.text().split(',') if t.strip()]
            except ValueError:
                print("Warning: Invalid float value in Time Series.")
                params['time series'] = [1.0,3.0,5.0,10.0,15.0,25.0,60.0,100.0]
        
        # Analysis (Proximity Level)
        try:
            params['proximity level'] = float(self.proximity_level_edit.text())
        except ValueError:
            params['proximity level'] = 0.5

        # Plot Image Settings
        params['image type'] = self.image_type_combo.currentText()
        try:
            params['image width'] = float(self.image_width_edit.text())
        except ValueError:
            params['image width'] = 8.5
        try:
            params['image height'] = float(self.image_height_edit.text())
        except ValueError:
            params['image height'] = 5.0
        params['image dpi'] = self.image_dpi_spinbox.value()
        params['image font size'] = self.image_font_size_spinbox.value()
        params['image tick length'] = self.image_tick_length_spinbox.value()
        params['image tick width'] = self.image_tick_width_spinbox.value()
        params['image numbins'] = self.image_numbins_spinbox.value()
        
        xlim_text = self.image_xlim_edit.text().strip()
        if xlim_text.lower() == 'none':
            params['image x_lim'] = None
        else:
            try:
                parts = [float(p.strip()) for p in xlim_text.split(',')]
                if len(parts) == 2:
                    params['image x_lim'] = parts
                else:
                    print("Warning: Invalid format for 'image x_lim'. Expected 'None' or 'float1, float2'. Using None.")
                    params['image x_lim'] = None
            except ValueError:
                print("Warning: Invalid float value in 'image x_lim'. Using None.")
                params['image x_lim'] = None

        # Performance Settings
        params['retain profile data'] = self.retain_profile_data_checkbox.isChecked()
        params['multiprocessing'] = self.multiprocessing_checkbox.isChecked()
        
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
