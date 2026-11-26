"""
DICE GUI - Main Application

Graphical user interface for the Diffusion Insight Computation Engine.
"""

import math
import sys
import webbrowser
from pathlib import Path
from PyQt6.QtWidgets import (
    QScrollArea,
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QComboBox, QFileDialog,
    QProgressBar, QMessageBox, QStatusBar, QMenu,
    QAbstractButton, QAbstractSpinBox, QDoubleSpinBox, QLineEdit, QPlainTextEdit, QSpinBox,
    QDialog, QDialogButtonBox,
)
from PyQt6.QtCore import QLocale, Qt, QSettings
from PyQt6.QtGui import QAction, QKeySequence

from dice import __version__
from dice.utils.runtime import bundled_license_notice, examples_directory
from dice_gui.validators import (
    validate_positive_integer, validate_positive_float, validate_float,
    validate_time_range, validate_time_series,
    validate_file_path, validate_filename_slug,
    calculate_pixel_size
)
from dice_gui.dice_interface import DiceInterface, parameters_for_form
from dice_gui.tabs.base import set_spin_value
from dice_gui.output_location import output_root
from dice_gui.styles import DiceTheme, apply_theme
from dice_gui.accessibility import (
    name_inputs_for_screen_readers,
    add_accessible_name_and_description
)
from dice_gui.validation_manager import ValidationManager, apply_validation_style, clear_validation_style
from dice_gui.simulation_thread import SimulationThread
from dice_gui.example_parameters import EXAMPLE_PARAMETERS
from dice_gui.tabs import (
    create_tab_simulation_setup,
    create_tab_physical_parameters,
    create_tab_experimental_conditions,
    create_tab_analysis_settings,
    create_tab_output_settings,
    toggle_noise_inputs,
    toggle_time_inputs,
    update_diffusion_fields,
    update_noise_cnr_display,
    update_width_conversion,
    update_pixel_size,
    combo_value,
    set_combo_value,
    selected_estimator_key,
)



class DiceGUI(QMainWindow):
    """Main GUI window for DICE."""

    def __init__(self):
        super().__init__()
        # Spin boxes follow the widget locale. The C locale gives them the
        # decimal point that the text fields and parameter files use; child
        # widgets and dialogs inherit it.
        self.setLocale(QLocale.c())
        self.interface = DiceInterface()
        self.simulation_thread = None
        # True from the moment the user closes the window during a run until it has closed.
        self._close_requested = False
        # True while a stop is pending whose Stop button had the keyboard focus.
        self._stop_had_focus = False
        self.theme = DiceTheme()

        # File management
        self.current_parameter_file = None
        self.parameters_modified = False
        self.settings = QSettings("DICE", "DICE_GUI")
        self.loaded_data_file = None  # Track loaded CSV file for status bar
        self._populating = False      # Prevent modification marking during load
        self._syncing_units = False   # Prevent _user_modified marking during sync
        self.validation_manager = None  # Will be initialized in init_ui
        self._error_labels = {}  # field_id -> inline error QLabel

        # Map tabs to their validated field IDs for tab completion indicators
        self._tab_fields = {
            0: ["filename_slug"],  # Tab 1: Simulation Setup
            1: ["diffusion_length", "diffusion_coeff", "lifetime",
                "amplitude", "mean", "width"],  # Tab 2: Physical Parameters
            2: ["noise_value", "noise_file", "spatial_width",
                "time_start", "time_stop", "time_series"],  # Tab 3: Experimental Conditions
            3: [],  # Tab 4: Analysis (no validated fields)
            4: [],  # Tab 5: Output (no validated fields)
        }

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("DICE - Diffusion Insight Computation Engine")
        self.setGeometry(100, 100, 900, 750)

        # Create menu bar
        self.create_menu_bar()

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Content area. Each tab scrolls its own content, so the tab bar
        # stays in view in a small window.
        content_area = QWidget()
        content_layout = QVBoxLayout(content_area)
        # No bottom margin: the control panel below has a margin of its own.
        content_layout.setContentsMargins(10, 10, 10, 0)

        # Initialize hidden global unit combos (not in visible layout, but
        # other code reads/writes them for per-parameter unit syncing).
        self._init_global_unit_combos()

        # Create tab widget (using modular tab functions from dice_gui.tabs)
        self.tabs = QTabWidget()
        self.tab1 = create_tab_simulation_setup(self)
        self.tab2 = create_tab_physical_parameters(self)
        self.tab3 = create_tab_experimental_conditions(self)
        self.tab4 = create_tab_analysis_settings(self)
        self.tab5 = create_tab_output_settings(self)

        self.tabs.addTab(self.tab1, "Simulation Setup")
        self.tabs.addTab(self.tab2, "Physical Parameters")
        self.tabs.addTab(self.tab3, "Experimental Conditions")
        self.tabs.addTab(self.tab4, "Analysis")
        self.tabs.addTab(self.tab5, "Output")

        content_layout.addWidget(self.tabs)
        main_layout.addWidget(content_area, stretch=1)

        # Add sticky bottom control panel (below the tabs)
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)

        # Create status bar
        self.create_status_bar()

        # Initialize default values
        self.set_default_values()

        # Connect modification tracking signals
        self.connect_modification_signals()

        # Setup validation system
        self.setup_validation()
        name_inputs_for_screen_readers(self)

        # Initialize output path preview
        self._update_output_path_preview()

    def create_menu_bar(self):
        """Create the menu bar with File and Help menus."""
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu("&File")

        # New Parameters
        new_action = QAction("&New Parameters", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.setStatusTip("Reset all parameters to default values")
        new_action.triggered.connect(self.new_parameters)
        file_menu.addAction(new_action)

        # Open Parameters
        open_action = QAction("&Open Parameters...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.setStatusTip("Load parameters from file")
        open_action.triggered.connect(self.load_parameters)
        file_menu.addAction(open_action)

        # Save Parameters
        save_action = QAction("&Save Parameters", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.setStatusTip("Save parameters to file")
        save_action.triggered.connect(self.save_parameters)
        file_menu.addAction(save_action)

        # Save Parameters As
        save_as_action = QAction("Save Parameters &As...", self)
        save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_as_action.setStatusTip("Save parameters to a new file")
        save_as_action.triggered.connect(self.save_parameters_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        # Recent Parameters (submenu)
        self.recent_menu = file_menu.addMenu("Recent &Parameters")
        self.update_recent_menu()

        file_menu.addSeparator()

        # Load Results
        load_results_action = QAction("Load &Results...", self)
        load_results_action.setShortcut(QKeySequence("Ctrl+L"))
        load_results_action.setStatusTip("Load simulation results from CSV file")
        load_results_action.triggered.connect(self.load_and_plot_results)
        file_menu.addAction(load_results_action)

        # Enabled only while a finished simulation has no output files.
        self.write_outputs_action = QAction("&Write Output Files", self)
        self.write_outputs_action.setStatusTip(
            "Write the results file, the summary and the plot of the last simulation, which could not be written")
        self.write_outputs_action.triggered.connect(self.write_output_files)
        self.write_outputs_action.setEnabled(False)
        file_menu.addAction(self.write_outputs_action)

        file_menu.addSeparator()

        # Exit
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.setStatusTip("Exit application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Plots Menu
        plots_menu = menubar.addMenu("&Plots")

        # Accuracy Histogram (active)
        histogram_action = QAction("&Accuracy Histogram", self)
        histogram_action.setShortcut(QKeySequence("Ctrl+H"))
        histogram_action.setStatusTip("Generate accuracy histogram from loaded data")
        histogram_action.triggered.connect(self.regenerate_plot)
        plots_menu.addAction(histogram_action)

        plots_menu.addSeparator()

        # Future plot types (disabled)
        profile_action = QAction("&Profile Evolution", self)
        profile_action.setEnabled(False)
        profile_action.setStatusTip("Plot profile evolution over time (coming soon)")
        plots_menu.addAction(profile_action)

        msd_action = QAction("&MSD Analysis", self)
        msd_action.setEnabled(False)
        msd_action.setStatusTip("Plot mean squared displacement analysis (coming soon)")
        plots_menu.addAction(msd_action)

        cnr_action = QAction("&CNR Dependence", self)
        cnr_action.setEnabled(False)
        cnr_action.setStatusTip("Plot CNR dependence analysis (coming soon)")
        plots_menu.addAction(cnr_action)

        plots_menu.addSeparator()

        batch_action = QAction("&Batch Generate All", self)
        batch_action.setEnabled(False)
        batch_action.setStatusTip("Generate all plot types (coming soon)")
        plots_menu.addAction(batch_action)

        # Settings Menu
        settings_menu = menubar.addMenu("&Settings")

        global_units_action = QAction("&Global Units...", self)
        global_units_action.setStatusTip("Change global length and time units")
        global_units_action.triggered.connect(self._open_global_units_dialog)
        settings_menu.addAction(global_units_action)

        # Help Menu
        help_menu = menubar.addMenu("&Help")

        # Documentation
        doc_action = QAction("&Documentation", self)
        doc_action.setStatusTip("Open DICE documentation on GitHub")
        doc_action.triggered.connect(self.open_documentation)
        help_menu.addAction(doc_action)

        # Keyboard Shortcuts
        shortcuts_action = QAction("&Keyboard Shortcuts", self)
        shortcuts_action.setStatusTip("View keyboard shortcuts for all tabs")
        shortcuts_action.triggered.connect(self.show_keyboard_shortcuts)
        help_menu.addAction(shortcuts_action)

        help_menu.addSeparator()

        # Licenses
        licenses_action = QAction("&Licenses", self)
        licenses_action.setStatusTip("Licenses of DICE and of the packages it uses")
        licenses_action.triggered.connect(self.show_licenses_dialog)
        help_menu.addAction(licenses_action)

        # About
        about_action = QAction("&About DICE", self)
        about_action.setStatusTip("About DICE")
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    def create_header(self) -> QWidget:
        """Create the header section."""
        header = QWidget()
        layout = QVBoxLayout(header)

        title = QLabel("DICE")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        add_accessible_name_and_description(
            title,
            "DICE Title",
            "Diffusion Insight Computation Engine"
        )

        subtitle = QLabel("Diffusion Insight Computation Engine")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        add_accessible_name_and_description(
            subtitle,
            "Application subtitle",
            "Full name of the DICE application"
        )

        layout.addWidget(title)
        layout.addWidget(subtitle)

        return header

    def _init_global_unit_combos(self):
        """Create hidden global unit combos used for per-parameter syncing."""
        from dice_gui.tabs.base import LENGTH_UNITS_DISPLAY, TIME_UNITS_DISPLAY

        self.length_unit_combo = QComboBox()
        self.length_unit_combo.addItems(LENGTH_UNITS_DISPLAY)
        self.length_unit_combo.setCurrentText("micrometer")
        self.length_unit_combo.currentTextChanged.connect(self.sync_unit_combos)

        self.time_unit_combo = QComboBox()
        self.time_unit_combo.addItems(TIME_UNITS_DISPLAY)
        self.time_unit_combo.setCurrentText("nanosecond")
        self.time_unit_combo.currentTextChanged.connect(self.sync_unit_combos)

    def _open_global_units_dialog(self):
        """Open the global units dialog."""
        from dice_gui.global_units_dialog import GlobalUnitsDialog

        dialog = GlobalUnitsDialog(
            parent=self,
            current_length=self.length_unit_combo.currentText(),
            current_time=self.time_unit_combo.currentText(),
        )
        if dialog.exec() == GlobalUnitsDialog.DialogCode.Accepted:
            self.length_unit_combo.setCurrentText(dialog.selected_length_unit())
            self.time_unit_combo.setCurrentText(dialog.selected_time_unit())

    def create_control_panel(self) -> QWidget:
        """Create the sticky bottom control panel."""
        from PyQt6.QtCore import QElapsedTimer

        panel = QWidget()
        panel.setObjectName("sticky-action-bar")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)

        # Progress row
        progress_layout = QHBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar, stretch=1)

        self.elapsed_label = QLabel("Elapsed: 00:00")
        self.elapsed_label.setVisible(False)
        self.elapsed_label.setMinimumWidth(100)
        progress_layout.addWidget(self.elapsed_label)

        self.remaining_label = QLabel("")
        self.remaining_label.setVisible(False)
        self.remaining_label.setMinimumWidth(120)
        progress_layout.addWidget(self.remaining_label)

        layout.addLayout(progress_layout)

        # Status label
        self.status_label = QLabel("Ready to run")
        self.status_label.setProperty("class", "status-info")
        layout.addWidget(self.status_label)

        # Buttons row
        button_layout = QHBoxLayout()

        # Load Example dropdown
        self.example_button = QPushButton("Load Example...")
        self.example_button.setToolTip("Load pre-configured example parameters")
        self.example_menu = QMenu()
        self.example_menu.addAction("Quick Test (100 runs)",
                                    lambda: self.load_example("quick_test"))
        self.example_menu.addAction("High Precision (10,000 runs)",
                                    lambda: self.load_example("high_precision"))
        self.example_menu.addAction("Publication Quality",
                                    lambda: self.load_example("publication"))
        self.example_button.setMenu(self.example_menu)

        self.run_button = QPushButton("Run Simulation (Ctrl+R)")
        self.run_button.setObjectName("run-button")
        self.run_button.setShortcut(QKeySequence("Ctrl+R"))
        self.run_button.clicked.connect(self.run_simulation)

        self.stop_button = QPushButton("Stop (Ctrl+.)")
        self.stop_button.setObjectName("stop-button")
        self.stop_button.setShortcut(QKeySequence("Ctrl+."))
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_simulation)

        button_layout.addWidget(self.example_button)
        button_layout.addStretch()
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.stop_button)

        layout.addLayout(button_layout)

        # Initialize elapsed time timer
        self.elapsed_timer = QElapsedTimer()
        from PyQt6.QtCore import QTimer
        self.elapsed_display_timer = QTimer()
        self.elapsed_display_timer.timeout.connect(self._update_elapsed_time)

        return panel

    def _update_elapsed_time(self):
        """Update elapsed time display."""
        elapsed_ms = self.elapsed_timer.elapsed()
        seconds = elapsed_ms // 1000
        minutes = seconds // 60
        secs = seconds % 60
        self.elapsed_label.setText(f"Elapsed: {minutes:02d}:{secs:02d}")

    def set_default_values(self):
        """Set default values for all inputs."""
        from dice_gui.tabs.tab_output import _reset_all_plot_settings

        default_params = {
            'number of runs': 1000,
            'filename slug': 'DICE_results',
            'length unit': 'micrometer',
            'time unit': 'nanosecond',
            'amplitude_0': 1.0,
            'mean_0': 0.0,
            'FWHM_0': 1.0,
            'nominal diffusion coefficient': 0.01,
            'nominal lifetime (tau)': 2.0,
            'noise value': 0.05,
            'spatial width': 10.0,
            'pixel width': 101,
            'time range': [0, 2, 10],
            'multiprocessing': True,
            'retain profile data': False,
        }

        self._populating = True
        try:
            self.populate_gui_from_parameters(default_params)
        finally:
            self._populating = False

        # Reset per-parameter units to follow global
        self.reset_per_param_units()

        # Reset output/image settings to defaults
        _reset_all_plot_settings(self)

        # Apply toggle states based on current radio button selections
        toggle_noise_inputs(self, self.noise_fixed_radio.isChecked())
        toggle_time_inputs(self, self.time_range_radio.isChecked())

    def sync_unit_combos(self):
        """Sync per-parameter unit combos with the global unit selection.

        For each per-parameter combo that the user has not explicitly changed,
        update it to match the current global unit. Then refresh all calculated
        value displays.
        """
        global_length = self.length_unit_combo.currentText()
        global_time = self.time_unit_combo.currentText()

        self._syncing_units = True
        try:
            # Length-dimension combos (regular text-based)
            for combo in [
                self.diffusion_length_unit_combo,
                self.mean_unit_combo,
                self.width_unit_combo,
                self.spatial_width_unit_combo,
            ]:
                if not combo.property("_user_modified"):
                    combo.setCurrentText(global_length)

            # Length-dimension squared combo (uses data-based lookup)
            if not self.diffusion_coeff_length_unit_combo.property("_user_modified"):
                set_combo_value(self.diffusion_coeff_length_unit_combo, global_length)

            # Time-dimension combos
            for combo in [
                self.lifetime_unit_combo,
                self.time_start_unit_combo,
                self.time_stop_unit_combo,
                self.diffusion_coeff_time_unit_combo,
            ]:
                if not combo.property("_user_modified"):
                    combo.setCurrentText(global_time)
        finally:
            self._syncing_units = False

        # Update calculated values (signals from combo changes handle
        # diffusion field linking, but width/pixel need explicit refresh)
        update_width_conversion(self)
        update_pixel_size(self)

    def _on_per_param_unit_changed(self, combo: QComboBox):
        """Mark a per-parameter unit combo as user-modified.

        Skipped during programmatic updates (sync_unit_combos, populate).
        """
        if not self._syncing_units and not self._populating:
            combo.setProperty("_user_modified", True)

    def reset_per_param_units(self):
        """Reset all per-parameter unit combos to follow the global units."""
        for combo in self._get_all_per_param_unit_combos():
            combo.setProperty("_user_modified", False)
        self.sync_unit_combos()

    def _get_all_per_param_unit_combos(self):
        """Return a list of all per-parameter unit combo boxes."""
        return [
            self.diffusion_length_unit_combo,
            self.diffusion_coeff_length_unit_combo,
            self.diffusion_coeff_time_unit_combo,
            self.lifetime_unit_combo,
            self.mean_unit_combo,
            self.width_unit_combo,
            self.spatial_width_unit_combo,
            self.time_start_unit_combo,
            self.time_stop_unit_combo,
        ]

    def _update_output_path_preview(self):
        """Update the output path preview based on current slug."""
        slug = self.filename_slug_input.text().strip()
        if not slug:
            self.output_path_preview.setText("Enter a filename slug to see output paths")
            self.output_path_preview.setToolTip("")
            return

        output_dir = output_root() / slug
        image_type = getattr(self, 'image_type_combo', None)
        ext = image_type.currentText() if image_type else 'png'

        self.output_path_preview.setText(
            f"<b>Directory:</b> {output_dir}<br>"
            f"<b>Files:</b> {slug}.csv, {slug}_summary.txt, {slug}_accuracy_histogram.{ext}"
        )
        # The label cuts off a path that is wider than the window.
        self.output_path_preview.setToolTip(str(output_dir))

    def collect_image_settings(self) -> dict:
        """Collect current image settings and convert to standard units."""
        # Convert image dimensions to cm (standard unit)
        width_value = self.image_width_spin.value()
        width_unit = self.image_width_unit_combo.currentText()
        if width_unit == "in":
            width_cm = width_value * 2.54
        elif width_unit == "mm":
            width_cm = width_value / 10.0
        else:  # cm
            width_cm = width_value

        height_value = self.image_height_spin.value()
        height_unit = self.image_height_unit_combo.currentText()
        if height_unit == "in":
            height_cm = height_value * 2.54
        elif height_unit == "mm":
            height_cm = height_value / 10.0
        else:  # cm
            height_cm = height_value

        # Resolution in DPI
        dpi = self.image_dpi_spin.value()

        # Convert font size to points
        font_size = self.image_font_size_spin.value()
        font_unit = self.image_font_unit_combo.currentText()
        if font_unit == "px":
            font_size_pt = font_size * 0.75
        else:  # pt
            font_size_pt = font_size

        # Convert tick length to points
        tick_length = self.image_tick_length_spin.value()
        tick_length_unit = self.image_tick_length_unit_combo.currentText()
        if tick_length_unit == "px":
            tick_length_pt = tick_length * 0.75
        else:  # pt
            tick_length_pt = tick_length

        # Convert tick width to points
        tick_width = self.image_tick_width_spin.value()
        tick_width_unit = self.image_tick_width_unit_combo.currentText()
        if tick_width_unit == "px":
            tick_width_pt = tick_width * 0.75
        else:  # pt
            tick_width_pt = tick_width

        return {
            'image_type': self.image_type_combo.currentText(),
            'image_width': width_cm,
            'image_height': height_cm,
            'image_dpi': dpi,
            'image_font_size': font_size_pt,
            'image_tick_length': tick_length_pt,
            'image_tick_width': tick_width_pt,
            'image_numbins': self.image_numbins_spin.value()
        }

    def get_plot_filename(self, output_dir, slug: str, image_type: str) -> str:
        """Get plot filename, handling overwrites with user confirmation."""
        from pathlib import Path

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"{slug}_accuracy_histogram.{image_type}"

        if filename.exists():
            reply = QMessageBox.question(
                self, "File Exists",
                f"File already exists:\n{filename}\n\n"
                "Do you want to overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.No:
                # Open save dialog
                new_file, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save Plot As",
                    str(filename),
                    f"{image_type.upper()} Files (*.{image_type});;All Files (*)"
                )
                if new_file:
                    filename = Path(new_file)
                else:
                    raise ValueError("Save cancelled by user")

        return str(filename)

    def regenerate_plot(self):
        """Regenerate plot from data in memory with current image settings."""
        # Check if data exists in memory
        if self.interface.last_result is None:
            reply = QMessageBox.question(
                self, "No Data in Memory",
                "No data in memory to plot.\n\n"
                "Would you like to load data from a CSV file?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.load_and_plot_results()
            return

        try:
            # Collect current image settings
            image_settings = self.collect_image_settings()

            # Get plot method (WLS or OLS)
            estimator = selected_estimator_key(self)

            # Get output location (original directory)
            slug = self.interface.filename_slug(self.interface.last_parameters)
            output_dir = self.interface.output_directory(self.interface.last_parameters)

            # Get plot filename with overwrite handling
            filename = self.get_plot_filename(output_dir, slug, image_settings['image_type'])

            # Call interface method to regenerate
            self.interface.regenerate_plot_from_memory(
                filename=filename,
                image_settings=image_settings,
                estimator=estimator
            )

            # Show success message
            QMessageBox.information(self, "Plot Generated",
                                  f"Plot saved to:\n{filename}")

        except ValueError as e:
            if "cancelled" in str(e).lower():
                return  # User cancelled save dialog
            QMessageBox.critical(self, "Error", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Error Generating Plot",
                               f"Failed to generate plot:\n\n{str(e)}")

    def load_and_plot_results(self):
        """Load results from CSV and generate plot."""
        from pathlib import Path

        if not self._confirm_discard_unwritten_results("Loading a results file"):
            return

        # Get default directory from filename slug output path
        root = output_root()
        default_dir = root / self.filename_slug_input.text()
        if not default_dir.exists():
            default_dir = root
        if not default_dir.exists():
            default_dir = root.parent

        # Open file dialog
        csv_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Results CSV File",
            str(default_dir),
            "CSV Files (*.csv);;All Files (*)"
        )

        if not csv_file:
            return

        try:
            # Collect current image settings
            image_settings = self.collect_image_settings()

            # Get plot method
            estimator = selected_estimator_key(self)

            # Determine output location
            csv_path = Path(csv_file)
            output_dir = csv_path.parent
            slug = csv_path.stem  # Use CSV filename without extension

            # Get plot filename with overwrite handling
            filename = self.get_plot_filename(output_dir, slug, image_settings['image_type'])

            # Call interface method to load and plot
            self.interface.load_and_plot_from_csv(
                csv_file=csv_file,
                filename=filename,
                image_settings=image_settings,
                estimator=estimator
            )
            self.write_outputs_action.setEnabled(self.interface.last_output_error is not None)
            # The status line may still speak of the simulation these results replace.
            self.status_label.setText("Results loaded")

            # Update status bar
            self.loaded_data_file = csv_file
            self.update_status_bar()

            # Populate precision summary from loaded data
            self._populate_precision_from_results()

            # Show success message with info
            QMessageBox.information(
                self, "Data Loaded and Plot Generated",
                f"Loaded results from:\n{csv_file}\n\n"
                f"Plot saved to:\n{filename}\n\n"
                "You can adjust image settings and click 'Regenerate Plot' "
                "to create a new version with different formatting."
            )

        except ValueError as e:
            if "cancelled" in str(e).lower():
                return  # User cancelled save dialog
            QMessageBox.critical(self, "Error", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Results",
                               f"Failed to load and plot results:\n\n{str(e)}")

    def _in_global_time_unit(self, text: str, unit_combo) -> str:
        """The time in a field as text in the global time unit; a text that is no number is returned as it is."""
        from dice.utils.units import convert_time
        try:
            return str(convert_time(float(text), combo_value(unit_combo), self.time_unit_combo.currentText()))
        except ValueError:
            return text  # validate_time_range reports it

    def validate_all_inputs(self) -> tuple[bool, str]:
        """Validate all inputs before running simulation."""
        # Validate based on current radio button selections

        # Number of runs
        result = validate_positive_integer(str(self.num_runs_spin.value()), "Number of runs")
        if not result:
            return False, result.error_message

        # Filename slug
        result = validate_filename_slug(self.filename_slug_input.text())
        if not result:
            return False, result.error_message

        # Amplitude and mean
        result = validate_positive_float(self.amplitude_input.text(), "Amplitude", allow_zero=True)
        if not result:
            return False, result.error_message

        result = validate_float(self.mean_input.text(), "Mean position")
        if not result:
            return False, result.error_message

        # Width
        result = validate_positive_float(self.width_input.text(), "Profile width")
        if not result:
            return False, result.error_message

        # Diffusion (all three fields linked; any two compute the third).
        # L is infinite when tau is 0 (no decay); D must be positive because
        # the accuracy analysis divides estimates by the nominal D.
        result = validate_positive_float(
            self.diffusion_length_input.text(), "Diffusion length", allow_zero=True, allow_inf=True)
        if not result:
            return False, result.error_message
        result = validate_positive_float(self.diffusion_coeff_input.text(), "Diffusion coefficient")
        if not result:
            return False, result.error_message
        result = validate_positive_float(self.lifetime_input.text(), "Lifetime", allow_zero=True)
        if not result:
            return False, result.error_message

        # Noise
        if self.noise_fixed_radio.isChecked():
            result = validate_positive_float(self.noise_value_input.text(), "Noise value", allow_zero=True)
            if not result:
                return False, result.error_message
        else:
            result = validate_file_path(self.noise_file_input.text())
            if not result:
                return False, result.error_message

        # Spatial
        result = validate_positive_float(self.spatial_width_input.text(), "Spatial width")
        if not result:
            return False, result.error_message

        # Time
        if self.time_range_radio.isChecked():
            # Start and stop each have a unit selector, so they are compared
            # in the global time unit: 500 ps is before 1 ns.
            result = validate_time_range(
                self._in_global_time_unit(self.time_start_input.text(), self.time_start_unit_combo),
                self._in_global_time_unit(self.time_stop_input.text(), self.time_stop_unit_combo),
                str(self.time_steps_input.value())
            )
            if not result:
                return False, result.error_message
        else:
            result = validate_time_series(self.time_series_input.toPlainText())
            if not result:
                return False, result.error_message

        return True, ""

    def collect_parameters(self) -> dict:
        """Collect all parameters from GUI into a dictionary."""
        # Convert image dimensions to cm (standard unit)
        width_value = self.image_width_spin.value()
        width_unit = self.image_width_unit_combo.currentText()
        if width_unit == "in":
            width_cm = width_value * 2.54
        elif width_unit == "mm":
            width_cm = width_value / 10.0
        else:  # cm
            width_cm = width_value

        height_value = self.image_height_spin.value()
        height_unit = self.image_height_unit_combo.currentText()
        if height_unit == "in":
            height_cm = height_value * 2.54
        elif height_unit == "mm":
            height_cm = height_value / 10.0
        else:  # cm
            height_cm = height_value

        # Resolution in DPI
        dpi = self.image_dpi_spin.value()

        # Font size, tick length, and tick width units
        # For now, we'll pass both value and unit, but matplotlib expects points
        # If px is selected, we may need conversion (1 pt = 1.333 px at 96 DPI)
        font_size = self.image_font_size_spin.value()
        font_unit = self.image_font_unit_combo.currentText()
        if font_unit == "px":
            font_size_pt = font_size * 0.75  # Convert px to pt
        else:  # pt
            font_size_pt = font_size

        tick_length = self.image_tick_length_spin.value()
        tick_length_unit = self.image_tick_length_unit_combo.currentText()
        if tick_length_unit == "px":
            tick_length_pt = tick_length * 0.75
        else:  # pt
            tick_length_pt = tick_length

        tick_width = self.image_tick_width_spin.value()
        tick_width_unit = self.image_tick_width_unit_combo.currentText()
        if tick_width_unit == "px":
            tick_width_pt = tick_width * 0.75
        else:  # pt
            tick_width_pt = tick_width

        params = {
            'number_of_runs': self.num_runs_spin.value(),
            'filename_slug': self.filename_slug_input.text(),
            'length_unit': self.length_unit_combo.currentText(),
            'time_unit': self.time_unit_combo.currentText(),
            'amplitude_0': float(self.amplitude_input.text()),
            'mean_0': float(self.mean_input.text()),
            'profile_width_type': 'fwhm' if self.fwhm_radio.isChecked() else 'sigma',
            'profile_width_value': float(self.width_input.text()),
            'spatial_width': float(self.spatial_width_input.text()),
            'pixel_width': self.pixel_width_input.value(),
            'multiprocessing': self.multiprocessing_check.isChecked(),
            'retain_profile_data': self.retain_profile_check.isChecked(),
            'image_type': self.image_type_combo.currentText(),
            'image_width': width_cm,
            'image_height': height_cm,
            'image_dpi': dpi,
            'image_font_size': font_size_pt,
            'image_tick_length': tick_length_pt,
            'image_tick_width': tick_width_pt,
            'image_numbins': self.image_numbins_spin.value(),
        }

        # Diffusion (always send D and tau; L is derived)
        params['diffusion_coefficient'] = float(self.diffusion_coeff_input.text())
        params['lifetime'] = float(self.lifetime_input.text())

        # Noise
        if self.noise_fixed_radio.isChecked():
            params['noise_type'] = 'fixed'
            params['noise_value'] = float(self.noise_value_input.text())
        else:
            params['noise_type'] = 'estimate'
            params['noise_data_file'] = self.noise_file_input.text()

        # Time
        if self.time_range_radio.isChecked():
            params['time_type'] = 'range'
            params['time_start'] = float(self.time_start_input.text())
            params['time_stop'] = float(self.time_stop_input.text())
            params['time_steps'] = self.time_steps_input.value()
        else:
            params['time_type'] = 'series'
            params['time_series'] = self.time_series_input.toPlainText()

        # Per-parameter unit overrides (only when different from global)
        global_length = self.length_unit_combo.currentText()
        global_time = self.time_unit_combo.currentText()

        # FWHM / sigma width unit
        w_unit = combo_value(self.width_unit_combo)
        if w_unit != global_length:
            if self.fwhm_radio.isChecked():
                params['fwhm_0_unit'] = w_unit
            else:
                params['sigma_0_unit'] = w_unit

        # Mean position unit
        mu_unit = combo_value(self.mean_unit_combo)
        if mu_unit != global_length:
            params['mu_0_unit'] = mu_unit

        # Diffusion coefficient units (compound: length^2/time)
        dcl_unit = combo_value(self.diffusion_coeff_length_unit_combo)
        dct_unit = combo_value(self.diffusion_coeff_time_unit_combo)
        if dcl_unit != global_length:
            params['diffusion_coefficient_length_unit'] = dcl_unit
        if dct_unit != global_time:
            params['diffusion_coefficient_time_unit'] = dct_unit

        # Lifetime unit
        lt_unit = combo_value(self.lifetime_unit_combo)
        if lt_unit != global_time:
            params['lifetime_unit'] = lt_unit

        # Spatial width unit
        sw_unit = combo_value(self.spatial_width_unit_combo)
        if sw_unit != global_length:
            params['spatial_width_unit'] = sw_unit

        # Time start/stop units
        ts_unit = combo_value(self.time_start_unit_combo)
        if ts_unit != global_time:
            params['time_start_unit'] = ts_unit
        tp_unit = combo_value(self.time_stop_unit_combo)
        if tp_unit != global_time:
            params['time_stop_unit'] = tp_unit

        return params

    def _check_parameter_consistency(self, gui_params: dict) -> list[str]:
        """Check parameter combinations for conditions that may produce unreliable results.

        Returns a list of warning/informational strings. Empty list means no issues.
        """
        from dice.utils.units import (
            convert_diffusion_coefficient, convert_time, convert_length, time_abbreviation, length_abbreviation,
        )
        from dice.utils.converters import fwhm_to_sigma
        from dice.core.profiles import diffusion_sigma2_t

        messages = []
        global_length = gui_params['length_unit']
        global_time = gui_params['time_unit']

        # Resolve lifetime to global time unit
        lifetime_unit = gui_params.get('lifetime_unit', global_time)
        lifetime = convert_time(gui_params['lifetime'], lifetime_unit, global_time)

        # Resolve t_stop and num_time_points
        t_stop = None
        num_time_points = 0

        if gui_params['time_type'] == 'range':
            t_stop_unit = gui_params.get('time_stop_unit', global_time)
            t_stop = convert_time(gui_params['time_stop'], t_stop_unit, global_time)
            num_time_points = gui_params['time_steps']
        else:
            try:
                time_values = [float(v.strip()) for v in gui_params['time_series'].split(',') if v.strip()]
            except ValueError:
                time_values = []
            if time_values:
                t_stop = max(time_values)
                num_time_points = len(time_values)

        # The initial width, in the global length unit
        if gui_params['profile_width_type'] == 'fwhm':
            width_unit = gui_params.get('fwhm_0_unit', global_length)
            fwhm_global = convert_length(gui_params['profile_width_value'], width_unit, global_length)
            sigma_0 = fwhm_to_sigma(fwhm_global)
        else:
            width_unit = gui_params.get('sigma_0_unit', global_length)
            sigma_0 = convert_length(gui_params['profile_width_value'], width_unit, global_length)

        # Check 1: Final CNR. The profile keeps its area while it spreads, so
        # its peak falls by the decay and by sigma_0 / sigma_t, with
        # sigma_t^2 = sigma_0^2 + 2 D t (dice.core.profiles).
        if gui_params['noise_type'] == 'fixed' and t_stop is not None and t_stop >= 0:
            amplitude = gui_params['amplitude_0']
            noise_sigma = gui_params['noise_value']
            diffusion_coefficient = convert_diffusion_coefficient(
                gui_params['diffusion_coefficient'],
                gui_params.get('diffusion_coefficient_length_unit', global_length),
                gui_params.get('diffusion_coefficient_time_unit', global_time),
                global_length, global_time,
            )

            decay = math.exp(-t_stop / lifetime) if lifetime > 0 else 1.0
            final_variance = float(diffusion_sigma2_t(diffusion_coefficient, sigma_0 ** 2, t_stop))
            spreading = sigma_0 / math.sqrt(final_variance) if final_variance > 0 else 1.0
            final_amplitude = amplitude * decay * spreading

            if noise_sigma > 0:
                final_cnr = final_amplitude / noise_sigma
                if final_cnr < 5:
                    t_abbr = time_abbreviation(global_time)
                    messages.append(
                        f"\u26a0 Low final CNR: At t = {t_stop:.4g} {t_abbr}, "
                        f"decay and spreading lower the peak to {final_amplitude:.4g}, "
                        f"giving CNR = {final_cnr:.2f}. "
                        f"Below ~5, Gaussian fitting becomes unreliable."
                    )

        # Check 2: Profile width undersampled relative to pixel grid
        spatial_unit = gui_params.get('spatial_width_unit', global_length)
        spatial_width = convert_length(gui_params['spatial_width'], spatial_unit, global_length)
        pixel_size_result = calculate_pixel_size(str(spatial_width), gui_params['pixel_width'])

        if pixel_size_result and sigma_0 < 2 * pixel_size_result.value:
            pixel_size = pixel_size_result.value
            l_abbr = length_abbreviation(global_length)
            messages.append(
                f"\u26a0 Undersampled profile: \u03c3\u2080 = {sigma_0:.4g} {l_abbr} "
                f"is narrower than 2\u00d7 the pixel size ({pixel_size:.4g} {l_abbr}). "
                f"The initial Gaussian may be too narrow for the pixel grid "
                f"to resolve, which prevents reliable fitting."
            )

        # Check 3: Few time points (informational)
        if 0 < num_time_points < 5:
            messages.append(
                f"\u2139 Few time points: Only {num_time_points} time point(s) will be used "
                f"for the MSD linear fit. Fewer points increase sensitivity to noise."
            )

        # Check 4: Measured peak too narrow for fft-cnr to estimate its CNR well.
        # The parser warns too, but a Python warning is not visible in the GUI.
        if gui_params['noise_type'] == 'estimate':
            import numpy as np
            from dice.io.parameters import narrow_peak_warning
            noise_file = gui_params['noise_data_file']
            try:
                # The encoding of the parser, which accepts a byte-order mark.
                message = narrow_peak_warning(
                    np.loadtxt(noise_file, delimiter=',', encoding='utf-8-sig'), noise_file)
            except (OSError, ValueError):
                message = None  # the run itself reports a file it cannot read
            if message:
                messages.append(f"\u26a0 Narrow measured peak: {message}")

        return messages

    def _show_field(self, widget) -> None:
        """Switch to the tab that holds an input and give the input the focus."""
        for index in range(self.tabs.count()):
            if self.tabs.widget(index).isAncestorOf(widget):
                self.tabs.setCurrentIndex(index)
                break
        widget.setFocus(Qt.FocusReason.OtherFocusReason)
        # A scroll area follows the Tab key, not a focus set by the program.
        parent = widget.parentWidget()
        while parent is not None and not isinstance(parent, QScrollArea):
            parent = parent.parentWidget()
        if parent is not None:
            parent.ensureWidgetVisible(widget)

    def run_simulation(self):
        """Run the DICE simulation."""
        if self.simulation_thread is not None:
            return  # a simulation is running; it ends before another starts

        # Validate all active fields (this triggers visual updates)
        is_valid, error_message = self.validation_manager.validate_all()
        if not is_valid:
            # Name the first field that blocks the run and take the user to it.
            field = self.validation_manager.first_invalid_field()
            self._show_field(field.widget)
            name = field.widget.accessibleName() or "A field"
            QMessageBox.warning(
                self,
                "Missing or Invalid Parameters",
                f"{name}: {error_message}\n\n"
                "The field has the focus now. Every field with an error is marked "
                "with a red border and a message below it."
            )
            return

        # Additional validation for inputs not managed by validation manager
        is_valid, error_message = self.validate_all_inputs()
        if not is_valid:
            QMessageBox.warning(self, "Validation Error", error_message)
            return

        # Collect parameters
        gui_params = self.collect_parameters()

        # Build DICE parameters
        try:
            dice_params = self.interface.build_parameters_dict(gui_params)
        except Exception as e:
            QMessageBox.critical(self, "Parameter Error", f"Failed to build parameters: {str(e)}")
            return

        # Validate parameters
        is_valid, error_message = self.interface.validate_parameters(dice_params)
        if not is_valid:
            QMessageBox.critical(self, "Parameter Validation Error", error_message)
            return

        # Check parameter consistency (non-blocking warnings)
        warnings = self._check_parameter_consistency(gui_params)
        if warnings:
            warning_text = "\n\n".join(warnings)
            reply = QMessageBox.question(
                self, "Parameter Warnings",
                warning_text + "\n\nProceed with simulation?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            if reply == QMessageBox.StandardButton.No:
                return

        if not self._confirm_discard_unwritten_results("A new simulation"):
            return
        self.interface.last_output_error = None
        self.write_outputs_action.setEnabled(False)

        # Update UI for running state
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        # Run is disabled now; the focus goes to Stop, the one control of a run.
        self.stop_button.setFocus(Qt.FocusReason.OtherFocusReason)
        self.progress_bar.setVisible(True)
        self.elapsed_label.setVisible(True)

        # A parallel run reports once its first runs have finished. Until then the bar
        # shows activity without a value; _update_iteration_progress gives it
        # its range with the first report.
        if dice_params.get('multiprocessing', True):
            self.progress_bar.setRange(0, 0)  # Indeterminate
        else:
            num_runs = dice_params.get('number of runs', 1000)
            self.progress_bar.setRange(0, num_runs)
            self.progress_bar.setValue(0)

        self.status_label.setText("Running simulation...")

        # Start elapsed time tracking
        self.elapsed_timer.start()
        self.elapsed_display_timer.start(1000)

        # Create and start simulation thread
        self.simulation_thread = SimulationThread(self.interface, dice_params)
        self.simulation_thread.progress.connect(self.update_progress)
        self.simulation_thread.iteration_progress.connect(self._update_iteration_progress)
        self.simulation_thread.finished.connect(self.simulation_finished)
        self.simulation_thread.error.connect(self.simulation_error)
        self.simulation_thread.cancelled.connect(self.simulation_cancelled)
        self.simulation_thread.start()

    def stop_simulation(self):
        """Ask the running simulation to stop; simulation_cancelled() resets the window when it has.

        A serial simulation stops after its current run and a parallel one
        within a few seconds: joblib stops its workers, and the process pool
        of a standalone build ends its current batch. The thread is not
        terminated, because a terminated parallel run leaves its worker
        processes behind.
        """
        if self.simulation_thread and self.simulation_thread.isRunning():
            self.simulation_thread.request_stop()
            # Qt moves the focus away from a button that is disabled;
            # reset_ui_after_simulation gives it to Run when the run has ended.
            self._stop_had_focus = self.focusWidget() is self.stop_button
            self.stop_button.setEnabled(False)
            self.status_label.setText("Stopping...")

    def update_progress(self, message: str):
        """Update progress display."""
        # A message queued before a stop request must not replace "Stopping...".
        if self.simulation_thread is not None and self.simulation_thread.stop_requested():
            return
        self.status_label.setText(message)

    def _update_iteration_progress(self, current: int, total: int):
        """Update progress bar with iteration count and estimated time remaining."""
        # An update queued before a stop request must not replace "Stopping...".
        if self.simulation_thread is not None and self.simulation_thread.stop_requested():
            return
        if self.progress_bar.maximum() != total:
            self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
        self.status_label.setText(f"Running {current:,}/{total:,}")

        # Estimate the remaining time once a tenth of the runs is done
        if current > 0:
            elapsed_ms = self.elapsed_timer.elapsed()
            fraction_complete = current / total

            if fraction_complete >= 0.1:  # Only estimate after first 10%
                estimated_total_ms = elapsed_ms / fraction_complete
                remaining_ms = estimated_total_ms - elapsed_ms

                remaining_secs = max(0, int(remaining_ms / 1000))
                minutes = remaining_secs // 60
                secs = remaining_secs % 60

                self.remaining_label.setText(f"Remaining: ~{minutes:02d}:{secs:02d}")
                self.remaining_label.setVisible(True)

    def simulation_finished(self, result):
        """Handle simulation completion."""
        # Unpack result and processed parameters from the interface
        if isinstance(result, tuple):
            mc_result, processed_params = result
            self.interface.last_result = mc_result
            self.interface.last_parameters = processed_params
        else:
            self.interface.last_result = result

        self.reset_ui_after_simulation()
        self.status_label.setText("Simulation completed")

        self._release_simulation_thread()

        # Clear loaded data file (new simulation overwrites)
        self.loaded_data_file = None
        self.update_status_bar()

        # Populate precision summary from results
        self._populate_precision_from_results()

        written = self._offer_to_write_output_files()
        if self._close_requested:
            self._exit_if_requested()
            return
        if not written:
            return

        from dice.io.results import completion_message
        message = completion_message(self.interface.last_result)
        self.status_label.setText(message)
        QMessageBox.information(
            self,
            "Simulation Complete",
            f"{message}\n\n"
            "The output files have been written to:\n\n"
            f"{self.interface.output_directory(self.interface.last_parameters)}"
        )

    def _confirm_discard_unwritten_results(self, action: str) -> bool:
        """
        Ask before an action replaces the results of a simulation whose output files are not written.

        Args:
            action: What discards the results, as the start of a sentence, for example "A new simulation".

        Returns:
            True when no results wait for their files, or the user agrees to lose them.
        """
        if self.interface.last_output_error is None:
            return True
        reply = QMessageBox.question(
            self, "Output Files Not Written",
            f"{action} discards the results of the last simulation, whose output files "
            "have not been written. Discard them?\n\n"
            "To keep them, choose No, then File > Write Output Files.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        return reply == QMessageBox.StandardButton.Yes

    def _offer_to_write_output_files(self) -> bool:
        """
        Ask to write the output files again until they are written or the user closes the dialog.

        Returns:
            True when the output files of the last simulation are written.
        """
        output_error = self.interface.last_output_error
        while output_error is not None:
            answer = QMessageBox.warning(
                self,
                "Output Files Not Written",
                "The simulation finished, but its output files could not be written:\n\n"
                f"{output_error}\n\n"
                "The results are kept, and the Analysis tab shows them. If another program "
                "has an output file open, close it there, then choose Retry to write the files. "
                "After Close, File > Write Output Files writes them later.",
                QMessageBox.StandardButton.Retry | QMessageBox.StandardButton.Close,
                QMessageBox.StandardButton.Retry,
            )
            if answer != QMessageBox.StandardButton.Retry:
                self.status_label.setText("Simulation completed; output files not written")
                break
            output_error = self.interface.retry_write_outputs()
        self.write_outputs_action.setEnabled(output_error is not None)
        return output_error is None

    def write_output_files(self):
        """File > Write Output Files: write the files of a simulation whose files could not be written."""
        if self.interface.last_output_error is None:
            return  # no simulation waits for its files
        # Write first: the dialog is for a write that fails again.
        self.interface.retry_write_outputs()
        if self._offer_to_write_output_files():
            self.status_label.setText("Output files written")
            # The status line is easy to miss, so a dialog names the folder,
            # as the dialog of a run that writes its files at once does.
            QMessageBox.information(
                self,
                "Output Files Written",
                "The output files of the last simulation have been written to:\n\n"
                f"{self.interface.output_directory(self.interface.last_parameters)}"
            )

    def _populate_precision_from_results(self):
        """Extract precision metrics from last results and populate the analysis tab."""
        from dice_gui.tabs.tab_analysis import populate_precision_summary, reset_precision_summary

        result = self.interface.last_result
        if result is None:
            return

        # Extract D estimates of the selected fit method
        d_arr = result.d_estimates(selected_estimator_key(self))
        nominal_d = result.run_results[0].nominal_diffusion_coefficient if result.run_results else 0

        # Without estimates, or without a nominal value to compare them with,
        # the tab must not keep the numbers of an earlier fit or simulation.
        if len(d_arr) == 0 or nominal_d == 0:
            reset_precision_summary(self, "The selected fit has no estimates in these results.")
            return

        # The statistics are the engine's, so the tab and the summary file agree.
        # The standard deviation and bias are reported in a named unit; the ratios are unitless.
        from dice.analysis.statistics import calculate_accuracy_metrics
        accuracy = calculate_accuracy_metrics(d_arr, nominal_d)
        d_scale, d_unit = self.interface.diffusion_coefficient_unit()
        metrics = {
            'std': accuracy['std'] * d_scale,
            'relative_std_dev': accuracy['std'] / nominal_d,
            'bias': accuracy['bias'] * d_scale,
            'bias_standard_error': accuracy['standard_error'] * d_scale,
            'mean_relative_error': accuracy['mean'] / nominal_d,
        }

        populate_precision_summary(self, metrics, d_unit)

    def simulation_error(self, error_message: str):
        """Handle simulation error."""
        self.reset_ui_after_simulation()
        self._release_simulation_thread()
        self.status_label.setText("Simulation failed")
        QMessageBox.critical(self, "Simulation Error", f"Simulation failed:\n\n{error_message}")
        self._exit_if_requested()

    def simulation_cancelled(self):
        """Handle a run that stopped at the user's request."""
        self.reset_ui_after_simulation()
        self._release_simulation_thread()
        # Replaces the "Stopping..." that stop_simulation() showed.
        self.status_label.setText("Simulation stopped by user")
        self._exit_if_requested()

    def _release_simulation_thread(self) -> None:
        """
        Drop the simulation thread once its run() has returned.

        The finished, error and cancelled signals are emitted at the end of
        run() and reach these slots through the event loop, so the thread can
        still be running when they execute. Qt aborts the process when a
        QThread is destroyed while it runs, so wait for run() to return before
        releasing the thread; after the signal that takes only moments.
        """
        thread, self.simulation_thread = self.simulation_thread, None
        if thread is not None:
            thread.wait()
            thread.deleteLater()

    def reset_ui_after_simulation(self):
        """Reset UI elements after simulation completes or stops."""
        # The form may have become invalid while the simulation ran.
        self.run_button.setEnabled(self.validation_manager.is_valid())
        stop_had_focus = self._stop_had_focus or self.focusWidget() is self.stop_button
        self._stop_had_focus = False
        self.stop_button.setEnabled(False)
        if stop_had_focus and self.run_button.isEnabled():
            self.run_button.setFocus(Qt.FocusReason.OtherFocusReason)
        self.progress_bar.setVisible(False)
        self.elapsed_label.setVisible(False)
        self.remaining_label.setVisible(False)
        self.remaining_label.setText("")
        self.elapsed_display_timer.stop()

    # ============ Menu Actions ============

    def new_parameters(self):
        """Reset all parameters to default values."""
        # Check if there are unsaved changes
        if self.parameters_modified:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Do you want to discard them?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return

        # Reset to default values
        self.set_default_values()
        self.current_parameter_file = None
        self.parameters_modified = False
        self.update_window_title()

    def load_parameters(self):
        """Load parameters from a file."""
        if not self._confirm_discard_changes():
            return

        # Open file dialog in the folder that holds the output root
        default_dir = str(output_root().parent)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Parameters File",
            default_dir,
            "Parameter Files (*.txt);;All Files (*)"
        )

        if not file_path:
            return

        self._load_parameters_reporting_errors(file_path)

    def open_recent_parameters(self, file_path: str):
        """Load a file of the Recent Parameters menu, with the checks of File > Open."""
        if not self._confirm_discard_changes():
            return
        self._load_parameters_reporting_errors(file_path)
        # A file that has been moved or deleted leaves the menu.
        self.update_recent_menu()

    def _confirm_discard_changes(self) -> bool:
        """Ask before unsaved changes are discarded; True when there are none or the user agrees."""
        if not self.parameters_modified:
            return True
        reply = QMessageBox.question(
            self, "Unsaved Changes",
            "You have unsaved changes. Do you want to discard them?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        return reply == QMessageBox.StandardButton.Yes

    def _load_parameters_reporting_errors(self, file_path: str) -> None:
        """Load a parameters file; a file that cannot be loaded gives a message and leaves the form as it was."""
        try:
            self.load_parameters_from_file(file_path)
        except Exception as e:
            QMessageBox.critical(
                self, "Error Loading Parameters",
                f"Failed to load parameters:\n\n{str(e)}"
            )

    def load_parameters_from_file(self, file_path: str):
        """Load parameters from a specific file path."""
        params = self._read_parameter_file(file_path)

        # Populate GUI from parameters
        self.populate_gui_from_parameters(params)

        # Update file tracking
        self.current_parameter_file = file_path
        self.parameters_modified = False
        self.add_to_recent_files(file_path)
        self.update_window_title()

        from dice.io.parameters import NOISE_RANGE_KEYS

        message = f"Parameters loaded from:\n{file_path}"
        range_key = next((key for key in NOISE_RANGE_KEYS if key in params), None)
        if range_key is None:
            QMessageBox.information(self, "Parameters Loaded", message)
        else:
            # The GUI has no noise-range control, so say that it runs with its
            # own noise setting instead.
            QMessageBox.warning(
                self, "Noise Range Not Editable",
                f"{message}\n\n"
                f"This file sets '{range_key}': {params[range_key]!r}, which gives each run "
                "its own noise level, drawn at random from the range. The GUI cannot show, "
                "edit, or run a noise range. A simulation run from the GUI uses the noise "
                "setting on the Experimental Conditions tab instead, and saving the parameters "
                "from the GUI replaces the range with that setting.\n\n"
                "To simulate the noise range, run the file with the command-line interface:\n"
                f'dice "{file_path}"'
            )

    def _read_parameter_file(self, file_path: str) -> dict:
        """Parse a parameters.txt-format file into its dictionary.

        A relative 'estimate noise from data' path is resolved against the
        file's folder, as the command-line interface resolves it.
        """
        import ast
        from dice.io.parameters import read_parameters_text, resolve_noise_data_path
        from dice.models.parameter_keys import check_one_spelling_per_key

        params = ast.literal_eval(read_parameters_text(file_path))
        if not isinstance(params, dict):
            raise ValueError("The file does not hold a parameters dictionary.")
        # The command-line interface rejects such a file too, and the form
        # would show one of the two values without saying so. The form reads
        # only the spellings of the parameter reference, so the message says
        # which one to keep.
        try:
            check_one_spelling_per_key(params)
        except ValueError as error:
            raise ValueError(
                f"{error} The GUI reads the spelling of the parameter reference (docs/parameters.md), "
                "for example 'time range' and 'mean_0', and ignores the others."
            ) from error
        return resolve_noise_data_path(params, Path(file_path).parent)

    def load_example(self, example_name: str):
        """Load example parameters without making the bundled file the save target."""
        if self.parameters_modified:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Load example anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return

        # Prefer the bundled file; fall back to the embedded copy
        params = EXAMPLE_PARAMETERS.get(example_name)
        example_file = examples_directory() / f"{example_name}.txt"
        if example_file.exists():
            try:
                params = self._read_parameter_file(str(example_file))
            except (OSError, SyntaxError, ValueError):
                pass  # Fall back to embedded

        if params is not None:
            self.populate_gui_from_parameters(params)
            self.current_parameter_file = None
            self.parameters_modified = False
            self.update_window_title()

            QMessageBox.information(
                self, "Example Loaded",
                f"Loaded '{example_name}' example parameters."
            )

    def save_parameters(self):
        """Save parameters to the current file or prompt for location."""
        if self.current_parameter_file:
            try:
                self.save_parameters_to_file(self.current_parameter_file)
            except Exception as e:
                QMessageBox.critical(
                    self, "Error Saving Parameters",
                    f"Failed to save parameters:\n\n{str(e)}"
                )
        else:
            self.save_parameters_as()

    def save_parameters_as(self):
        """Save parameters to a new file."""
        default_dir = str(output_root().parent)
        default_name = "parameters.txt"
        if self.current_parameter_file:
            default_name = Path(self.current_parameter_file).name

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Parameters As",
            str(Path(default_dir) / default_name),
            "Parameter Files (*.txt);;All Files (*)"
        )

        if not file_path:
            return

        try:
            self.save_parameters_to_file(file_path)
        except Exception as e:
            QMessageBox.critical(
                self, "Error Saving Parameters",
                f"Failed to save parameters:\n\n{str(e)}"
            )

    def save_parameters_to_file(self, file_path: str):
        """Save current GUI parameters to a file."""
        # Collect parameters from GUI
        gui_params = self.collect_parameters()

        # Convert to file format
        file_content = self.parameters_to_file_format(gui_params)

        # UTF-8 holds every character a path or a slug can have; the encoding
        # of the locale does not, and a failed write would empty the file.
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(file_content)

        # Update file tracking
        self.current_parameter_file = file_path
        self.parameters_modified = False
        self.add_to_recent_files(file_path)
        self.update_window_title()

        QMessageBox.information(
            self, "Parameters Saved",
            f"Parameters saved to:\n{file_path}"
        )

    def parameters_to_file_format(self, gui_params: dict) -> str:
        """Convert GUI parameters to parameters.txt file format."""
        lines = [
            "##################################################################",
            "# DICE Parameter Configuration",
            "# Generated by DICE GUI",
            "##################################################################",
            "",
            "{"
        ]

        # Filename slug
        lines.append(f"    'filename slug': {gui_params['filename_slug']!r},")
        lines.append("")

        # Number of runs
        lines.append(f"    'number of runs': {gui_params['number_of_runs']},")
        lines.append("")

        # Units
        lines.append("    ### Units ###")
        lines.append(f"    'length unit': {gui_params['length_unit']!r},")
        lines.append(f"    'time unit': {gui_params['time_unit']!r},")
        lines.append("")

        # Diffusion parameters
        lines.append("    ### Nominal diffusion and lifetime parameters ###")
        lines.append(f"    'nominal diffusion coefficient': {gui_params['diffusion_coefficient']},")
        if 'diffusion_coefficient_length_unit' in gui_params:
            lines.append(f"    'diffusion_coefficient_length_unit': {gui_params['diffusion_coefficient_length_unit']!r},")
        if 'diffusion_coefficient_time_unit' in gui_params:
            lines.append(f"    'diffusion_coefficient_time_unit': {gui_params['diffusion_coefficient_time_unit']!r},")
        lines.append(f"    'nominal lifetime (tau)': {gui_params['lifetime']},")
        if 'lifetime_unit' in gui_params:
            lines.append(f"    'lifetime_unit': {gui_params['lifetime_unit']!r},")
        lines.append("")

        # Initial profile
        lines.append("    ### Initial profile parameters ###")
        if gui_params['profile_width_type'] == 'fwhm':
            lines.append(f"    'FWHM_0': {gui_params['profile_width_value']},")
            if 'fwhm_0_unit' in gui_params:
                lines.append(f"    'FWHM_0_unit': {gui_params['fwhm_0_unit']!r},")
        else:
            lines.append(f"    'sigma_0': {gui_params['profile_width_value']},")
            if 'sigma_0_unit' in gui_params:
                lines.append(f"    'sigma_0_unit': {gui_params['sigma_0_unit']!r},")
        lines.append(f"    'amplitude_0': {gui_params['amplitude_0']},")
        lines.append(f"    'mean_0': {gui_params['mean_0']},")
        if 'mu_0_unit' in gui_params:
            lines.append(f"    'mu_0_unit': {gui_params['mu_0_unit']!r},")
        lines.append("")

        # Noise
        lines.append("    ### Noise parameter ###")
        if gui_params['noise_type'] == 'fixed':
            lines.append(f"    'noise value': {gui_params['noise_value']},")
        else:
            lines.append(f"    'estimate noise from data': {gui_params['noise_data_file']!r},")
        lines.append("")

        # Spatial axis
        lines.append("    ### Spatial axis parameters ###")
        lines.append(f"    'spatial width': {gui_params['spatial_width']},")
        if 'spatial_width_unit' in gui_params:
            lines.append(f"    'spatial_width_unit': {gui_params['spatial_width_unit']!r},")
        lines.append(f"    'pixel width': {gui_params['pixel_width']},")
        lines.append("")

        # Time axis
        lines.append("    ### Time axis parameters ###")
        if gui_params['time_type'] == 'range':
            lines.append(f"    'time range': [{gui_params['time_start']}, {gui_params['time_stop']}, {gui_params['time_steps']}],")
            if 'time_start_unit' in gui_params:
                lines.append(f"    'time_start_unit': {gui_params['time_start_unit']!r},")
            if 'time_stop_unit' in gui_params:
                lines.append(f"    'time_stop_unit': {gui_params['time_stop_unit']!r},")
        else:
            try:
                ts_values = [float(v.strip()) for v in gui_params['time_series'].split(',') if v.strip()]
                lines.append(f"    'time series': {ts_values},")
            except ValueError:
                lines.append(f"    'time series': [{gui_params['time_series']}],")
        lines.append("")

        # Plot parameters
        lines.append("    ### Plot image parameters ###")
        lines.append(f"    'image type': {gui_params['image_type']!r},")
        lines.append(f"    'image width': {gui_params['image_width']},")
        lines.append(f"    'image height': {gui_params['image_height']},")
        lines.append(f"    'image dpi': {gui_params['image_dpi']},")
        lines.append(f"    'image font size': {gui_params['image_font_size']},")
        lines.append(f"    'image tick length': {gui_params['image_tick_length']},")
        lines.append(f"    'image tick width': {gui_params['image_tick_width']},")
        lines.append(f"    'image numbins': {gui_params['image_numbins']},")
        lines.append("")

        # Performance settings
        lines.append("    ### Performance settings ###")
        lines.append(f"    'retain profile data': {gui_params['retain_profile_data']},")
        lines.append(f"    'multiprocessing': {gui_params['multiprocessing']},")

        lines.append("}")
        lines.append("")

        return "\n".join(lines)

    def populate_gui_from_parameters(self, params: dict):
        """Populate GUI fields from a parameters dictionary.

        The form changes only when every value can be shown. A value it cannot
        show raises ValueError, and the form is left as it was.
        """
        from dice.utils.units import global_length_unit, global_time_unit

        params = parameters_for_form(params)
        snapshot = self._snapshot_inputs()
        self._populating = True  # Prevent modification marking

        try:
            # Basic settings
            if 'number of runs' in params:
                set_spin_value(self.num_runs_spin, params['number of runs'], "'number of runs'",
                               minimum_is_a_rule=True)
            if 'filename slug' in params:
                self.filename_slug_input.setText(params['filename slug'])

            # Units. A file without these keys is in micrometers and
            # nanoseconds, not in the units the form showed before.
            self.length_unit_combo.setCurrentText(global_length_unit(params))
            self.time_unit_combo.setCurrentText(global_time_unit(params))

            # Initial profile
            if 'amplitude_0' in params:
                self.amplitude_input.setText(str(params['amplitude_0']))
            if 'mean_0' in params:
                self.mean_input.setText(str(params['mean_0']))

            # Profile width (mutually exclusive)
            if 'FWHM_0' in params:
                self.fwhm_radio.setChecked(True)
                self.width_input.setText(str(params['FWHM_0']))
            elif 'sigma_0' in params:
                self.sigma_radio.setChecked(True)
                self.width_input.setText(str(params['sigma_0']))

            # Diffusion parameters (clear all first, then set provided values)
            self.diffusion_length_input.clear()
            self.diffusion_coeff_input.clear()
            self.lifetime_input.clear()
            if 'nominal diffusion coefficient' in params:
                self.diffusion_coeff_input.setText(str(params['nominal diffusion coefficient']))
            if 'nominal lifetime (tau)' in params:
                self.lifetime_input.setText(str(params['nominal lifetime (tau)']))
            if 'nominal diffusion length' in params:
                self.diffusion_length_input.setText(str(params['nominal diffusion length']))

            # Noise (mutually exclusive). A noise range has no control here and
            # leaves the noise fields as they are; load_parameters_from_file says so.
            # The inputs of the other mode are emptied, so that none keeps a
            # value from before the load.
            if 'noise value' in params:
                self.noise_fixed_radio.setChecked(True)
                self.noise_value_input.setText(str(params['noise value']))
                self.noise_file_input.clear()
            elif 'estimate noise from data' in params:
                self.noise_estimate_radio.setChecked(True)
                self.noise_file_input.setText(params['estimate noise from data'])
                self.noise_value_input.clear()
                self.noise_cnr_input.clear()

            # Spatial axis
            if 'spatial width' in params:
                self.spatial_width_input.setText(str(params['spatial width']))
            if 'pixel width' in params:
                set_spin_value(self.pixel_width_input, params['pixel width'], "'pixel width'",
                               minimum_is_a_rule=True)

            # Time axis (mutually exclusive)
            if 'time range' in params:
                self.time_range_radio.setChecked(True)
                time_range = params['time range']
                self.time_start_input.setText(str(time_range[0]))
                self.time_stop_input.setText(str(time_range[1]))
                set_spin_value(self.time_steps_input, time_range[2], "The number of steps in 'time range'",
                               minimum_is_a_rule=True)
                self.time_series_input.clear()
            elif 'time series' in params:
                self.time_series_radio.setChecked(True)
                # Convert list to comma-separated string
                time_series_str = ", ".join(str(t) for t in params['time series'])
                self.time_series_input.setPlainText(time_series_str)
                self.time_start_input.clear()
                self.time_stop_input.clear()

            # Performance
            if 'multiprocessing' in params:
                self.multiprocessing_check.setChecked(params['multiprocessing'])
            if 'retain profile data' in params:
                self.retain_profile_check.setChecked(params['retain profile data'])

            # Per-parameter unit overrides
            self._load_per_param_units(params)

            # Image settings (with defaults if not present)
            if 'image type' in params:
                self.image_type_combo.setCurrentText(params['image type'])
            # A parameter file gives image sizes in centimeters and points, so
            # each value sets its unit selector too.
            for key, spin, unit_combo, unit in (
                ('image width', self.image_width_spin, self.image_width_unit_combo, 'cm'),
                ('image height', self.image_height_spin, self.image_height_unit_combo, 'cm'),
                ('image font size', self.image_font_size_spin, self.image_font_unit_combo, 'pt'),
                ('image tick length', self.image_tick_length_spin, self.image_tick_length_unit_combo, 'pt'),
                ('image tick width', self.image_tick_width_spin, self.image_tick_width_unit_combo, 'pt'),
            ):
                if key in params:
                    unit_combo.setCurrentText(unit)
                    set_spin_value(spin, float(params[key]), f"'{key}'")
            if 'image dpi' in params:
                set_spin_value(self.image_dpi_spin, params['image dpi'], "'image dpi'")
            if 'image numbins' in params:
                set_spin_value(self.image_numbins_spin, params['image numbins'], "'image numbins'")
            # When only the diffusion length is provided, use the engine's
            # convention (tau = 1, so the time axis is in units of the
            # lifetime), then let the linked-field logic compute D = L^2.
            if ('nominal diffusion length' in params
                    and 'nominal diffusion coefficient' not in params
                    and 'nominal lifetime (tau)' not in params):
                self.lifetime_input.setText("1")

        except Exception:
            self._restore_inputs(snapshot)
            raise
        finally:
            self._populating = False  # Re-enable modification marking

        # Set up diffusion field tracking based on populated values and
        # trigger computation of the third field.
        self._diffusion_last_edited = []
        if self.diffusion_coeff_input.text().strip():
            self._diffusion_last_edited.append('D')
        if self.lifetime_input.text().strip():
            self._diffusion_last_edited.append('tau')
        if self.diffusion_length_input.text().strip():
            self._diffusion_last_edited.append('L')
        self._diffusion_last_edited = self._diffusion_last_edited[-2:]
        if len(self._diffusion_last_edited) >= 2:
            update_diffusion_fields(self, self._diffusion_last_edited[-1])

        # Trigger noise sigma/CNR linking
        self._noise_last_edited = 'sigma'
        if self.noise_value_input.text().strip():
            update_noise_cnr_display(self, 'sigma')

    def _snapshot_inputs(self) -> list:
        """The value of every input widget, as (setter, value) pairs for _restore_inputs."""
        snapshot = []
        for widget in self.findChildren(QWidget):
            if isinstance(widget.parent(), (QAbstractSpinBox, QComboBox)):
                continue  # the line edit inside a spin box or a combo box
            if isinstance(widget, QLineEdit):
                snapshot.append((widget.setText, widget.text()))
            elif isinstance(widget, QPlainTextEdit):
                snapshot.append((widget.setPlainText, widget.toPlainText()))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                snapshot.append((widget.setValue, widget.value()))
            elif isinstance(widget, QComboBox):
                snapshot.append((widget.setCurrentIndex, widget.currentIndex()))
                snapshot.append((
                    lambda value, combo=widget: combo.setProperty("_user_modified", value),
                    widget.property("_user_modified"),
                ))
            elif isinstance(widget, QAbstractButton) and widget.isCheckable():
                snapshot.append((widget.setChecked, widget.isChecked()))
        return snapshot

    def _restore_inputs(self, snapshot: list) -> None:
        """Put back the input values that _snapshot_inputs recorded."""
        for setter, value in snapshot:
            setter(value)

    def _load_per_param_units(self, params: dict):
        """Set per-parameter unit combos from parameter dict _unit keys.

        If a _unit key is present, the corresponding combo is set and marked
        as user-modified so that subsequent global unit changes do not
        override it. Combos without a key follow the global unit again, so
        overrides from a previously loaded file do not carry over.
        """
        for combo in self._get_all_per_param_unit_combos():
            combo.setProperty("_user_modified", False)
        self.sync_unit_combos()

        def _set_combo(combo, *keys):
            for key in keys:
                if key in params:
                    set_combo_value(combo, params[key])
                    combo.setProperty("_user_modified", True)
                    return

        _set_combo(self.width_unit_combo, 'FWHM_0_unit', 'fwhm_0_unit', 'sigma_0_unit')
        _set_combo(self.mean_unit_combo, 'mu_0_unit')
        _set_combo(self.diffusion_length_unit_combo, 'diffusion_length_unit')
        _set_combo(self.diffusion_coeff_length_unit_combo, 'diffusion_coefficient_length_unit')
        _set_combo(self.diffusion_coeff_time_unit_combo, 'diffusion_coefficient_time_unit')
        _set_combo(self.lifetime_unit_combo, 'lifetime_unit')
        _set_combo(self.spatial_width_unit_combo, 'spatial_width_unit', 'spatial width unit')
        _set_combo(self.time_start_unit_combo, 'time_start_unit')
        _set_combo(self.time_stop_unit_combo, 'time_stop_unit')

    def add_to_recent_files(self, file_path: str):
        """Add a file to the recent files list."""
        # Get existing recent files from settings
        recent = self.settings.value("recent_parameters", [])
        if not isinstance(recent, list):
            recent = []

        # Remove if already in list
        if file_path in recent:
            recent.remove(file_path)

        # Add to front
        recent.insert(0, file_path)

        # Limit to 5
        recent = recent[:5]

        # Save to settings
        self.settings.setValue("recent_parameters", recent)

        # Update menu
        self.update_recent_menu()

    def update_recent_menu(self):
        """Update the Recent Parameters submenu."""
        self.recent_menu.clear()

        recent = self.settings.value("recent_parameters", [])
        if not isinstance(recent, list):
            recent = []
        # A file that has been moved or deleted is not offered.
        recent = [file_path for file_path in recent if Path(file_path).exists()]

        if not recent:
            no_recent = QAction("No recent files", self)
            no_recent.setEnabled(False)
            self.recent_menu.addAction(no_recent)
            return

        for file_path in recent:
            action = QAction(Path(file_path).name, self)
            action.setStatusTip(file_path)
            action.triggered.connect(lambda checked, f=file_path: self.open_recent_parameters(f))
            self.recent_menu.addAction(action)

    def update_window_title(self):
        """Update window title with current file and modified state."""
        title = "DICE - Diffusion Insight Computation Engine"

        if self.current_parameter_file:
            filename = Path(self.current_parameter_file).name
            title = f"DICE - [{filename}]"

        if self.parameters_modified:
            title += "*"

        self.setWindowTitle(title)

    def create_status_bar(self):
        """Create status bar showing version and loaded data."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Validation status (shows error count when present)
        self.validation_status_label = QLabel()
        self.status_bar.addPermanentWidget(self.validation_status_label)

        # Permanent version label on right
        version_label = QLabel(f"DICE v{__version__}")
        self.status_bar.addPermanentWidget(version_label)

        # The left side names the results in use. A label, not a status
        # message: the status tip of a menu entry replaces a message for good.
        self.results_status_label = QLabel()
        # It shows the name of a loaded file, which is not markup.
        self.results_status_label.setTextFormat(Qt.TextFormat.PlainText)
        self.status_bar.addWidget(self.results_status_label)
        self.update_status_bar()

    def update_status_bar(self):
        """Show which results the Analysis and Output tabs work on.

        The results are those of a loaded CSV file, those of the last
        simulation in this session, or none.
        """
        result = self.interface.last_result
        if self.loaded_data_file:
            self.results_status_label.setText(f"Loaded: {Path(self.loaded_data_file).name}")
        elif result is not None:
            slug = (self.interface.last_parameters or {}).get('filename slug', '')
            name = f" {slug}" if slug else ""
            noun = "run" if result.num_runs == 1 else "runs"
            self.results_status_label.setText(f"Simulation results:{name} ({result.num_runs:,} {noun})")
        else:
            self.results_status_label.setText("No data loaded")

    def connect_modification_signals(self):
        """Connect all parameter widgets to modification tracking."""
        # Simulation control
        self.num_runs_spin.valueChanged.connect(self.mark_modified)
        self.filename_slug_input.textChanged.connect(self.mark_modified)

        # Units
        self.length_unit_combo.currentTextChanged.connect(self.mark_modified)
        self.time_unit_combo.currentTextChanged.connect(self.mark_modified)

        # Per-parameter unit combos: mark modified and track user changes
        for combo in self._get_all_per_param_unit_combos():
            combo.currentTextChanged.connect(self.mark_modified)
            combo.currentTextChanged.connect(
                lambda _text, c=combo: self._on_per_param_unit_changed(c)
            )

        # Initial profile parameters (amplitude also drives CNR display on noise tab)
        self.amplitude_input.textChanged.connect(self.mark_modified)
        self.amplitude_input.textChanged.connect(
            lambda: update_noise_cnr_display(self, 'amplitude')
        )
        self.mean_input.textChanged.connect(self.mark_modified)
        self.width_input.textChanged.connect(self.mark_modified)
        self.fwhm_radio.toggled.connect(self.mark_modified)
        self.sigma_radio.toggled.connect(self.mark_modified)

        # Diffusion parameters
        self.diffusion_length_input.textChanged.connect(self.mark_modified)
        self.diffusion_coeff_input.textChanged.connect(self.mark_modified)
        self.lifetime_input.textChanged.connect(self.mark_modified)

        # Noise parameters
        self.noise_fixed_radio.toggled.connect(self.mark_modified)
        self.noise_estimate_radio.toggled.connect(self.mark_modified)
        self.noise_value_input.textChanged.connect(self.mark_modified)
        self.noise_cnr_input.textChanged.connect(self.mark_modified)
        self.noise_file_input.textChanged.connect(self.mark_modified)

        # Spatial axis
        self.spatial_width_input.textChanged.connect(self.mark_modified)
        self.pixel_width_input.valueChanged.connect(self.mark_modified)

        # Time axis
        self.time_range_radio.toggled.connect(self.mark_modified)
        self.time_series_radio.toggled.connect(self.mark_modified)
        self.time_start_input.textChanged.connect(self.mark_modified)
        self.time_stop_input.textChanged.connect(self.mark_modified)
        self.time_steps_input.valueChanged.connect(self.mark_modified)
        self.time_series_input.textChanged.connect(self.mark_modified)

        # Advanced parameters (Output tab)
        self.image_type_combo.currentTextChanged.connect(self.mark_modified)
        self.image_width_spin.valueChanged.connect(self.mark_modified)
        self.image_height_spin.valueChanged.connect(self.mark_modified)
        self.image_dpi_spin.valueChanged.connect(self.mark_modified)
        self.image_font_size_spin.valueChanged.connect(self.mark_modified)
        self.image_tick_length_spin.valueChanged.connect(self.mark_modified)
        self.image_tick_width_spin.valueChanged.connect(self.mark_modified)
        self.image_numbins_spin.valueChanged.connect(self.mark_modified)
        self.retain_profile_check.stateChanged.connect(self.mark_modified)
        self.multiprocessing_check.stateChanged.connect(self.mark_modified)

    def mark_modified(self):
        """Mark parameters as modified."""
        if hasattr(self, '_populating') and self._populating:
            return  # Don't mark modified during initial population
        self.parameters_modified = True
        self.update_window_title()

    def setup_validation(self):
        """Configure real-time validation for all input fields."""
        self.validation_manager = ValidationManager(self)

        # Connect validation manager signals
        self.validation_manager.validity_changed.connect(self._on_validity_changed)
        self.validation_manager.field_validated.connect(self._on_field_validated)

        # Register always-required fields
        self.validation_manager.register_field(
            "filename_slug", self.filename_slug_input,
            lambda v: validate_filename_slug(v)
        )
        self.validation_manager.register_field(
            "amplitude", self.amplitude_input,
            lambda v: validate_positive_float(v, "Amplitude", allow_zero=True)
        )
        self.validation_manager.register_field(
            "mean", self.mean_input,
            lambda v: validate_float(v, "Mean position")
        )
        self.validation_manager.register_field(
            "width", self.width_input,
            lambda v: validate_positive_float(v, "Profile width")
        )
        self.validation_manager.register_field(
            "spatial_width", self.spatial_width_input,
            lambda v: validate_positive_float(v, "Spatial width")
        )

        # Register diffusion fields (all three always required; any two compute the third).
        # L is infinite when tau is 0 (no decay); D must be positive because
        # the accuracy analysis divides estimates by the nominal D.
        self.validation_manager.register_field(
            "diffusion_length", self.diffusion_length_input,
            lambda v: validate_positive_float(v, "Diffusion length", allow_zero=True, allow_inf=True),
        )
        self.validation_manager.register_field(
            "diffusion_coeff", self.diffusion_coeff_input,
            lambda v: validate_positive_float(v, "Diffusion coefficient"),
        )
        self.validation_manager.register_field(
            "lifetime", self.lifetime_input,
            lambda v: validate_positive_float(v, "Lifetime", allow_zero=True),
        )

        # Register conditional noise fields (fixed sigma or estimate from data)
        self.validation_manager.register_field(
            "noise_value", self.noise_value_input,
            lambda v: validate_positive_float(v, "Noise value", allow_zero=True),
            condition_group="noise_fixed", condition_category="noise"
        )
        self.validation_manager.register_field(
            "noise_file", self.noise_file_input,
            lambda v: validate_file_path(v),
            condition_group="noise_estimate", condition_category="noise"
        )

        # Register conditional time fields
        self.validation_manager.register_field(
            "time_start", self.time_start_input,
            lambda v: validate_float(v, "Start time"),
            condition_group="time_range", condition_category="time"
        )
        self.validation_manager.register_field(
            "time_stop", self.time_stop_input,
            lambda v: validate_float(v, "Stop time"),
            condition_group="time_range", condition_category="time"
        )
        self.validation_manager.register_field(
            "time_series", self.time_series_input,
            lambda v: validate_time_series(v),
            condition_group="time_series", condition_category="time"
        )

        # Connect radio buttons to condition manager
        self.noise_fixed_radio.toggled.connect(
            lambda checked: self._update_noise_validation_state()
        )
        self.time_range_radio.toggled.connect(
            lambda checked: self.validation_manager.set_condition_active(
                "time", "time_range" if checked else "time_series"
            )
        )

        # Set initial condition states
        self.validation_manager.set_condition_active("noise", "noise_fixed")
        self.validation_manager.set_condition_active("time", "time_range")

    def _update_noise_validation_state(self):
        """Update the noise validation condition based on fixed vs estimate."""
        if self.noise_fixed_radio.isChecked():
            self.validation_manager.set_condition_active("noise", "noise_fixed")
        else:
            self.validation_manager.set_condition_active("noise", "noise_estimate")

    def _on_validity_changed(self, is_valid: bool) -> None:
        """Handle overall form validity change."""
        if self.simulation_thread is not None:
            # A simulation is running: Run stays disabled and the status line
            # keeps the progress. reset_ui_after_simulation applies the validity.
            return
        self.run_button.setEnabled(is_valid)
        if is_valid:
            self.status_label.setText("Ready to run")
            self.status_label.setProperty("class", "status-info")
        else:
            self.status_label.setText("Please fix validation errors before running")
            self.status_label.setProperty("class", "status-error")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def _on_field_validated(self, field_id: str, is_valid: bool, error_message: str) -> None:
        """Handle individual field validation result."""
        field = self.validation_manager._fields.get(field_id)
        if not field:
            return

        # Only apply validation styling and errors to enabled fields
        if not field.widget.isEnabled():
            clear_validation_style(field.widget)
            error_label = self._error_labels.get(field_id)
            if error_label:
                error_label.setText("")
                error_label.setVisible(False)
            self._update_tab_indicators()
            self._update_validation_status_bar()
            return

        apply_validation_style(field.widget, is_valid)

        # Update inline error label
        error_label = self._error_labels.get(field_id)
        if error_label:
            if is_valid:
                error_label.setText("")
                error_label.setVisible(False)
            else:
                error_label.setText(error_message)
                error_label.setVisible(True)

        # Update tooltip with error message
        if not is_valid:
            original_tooltip = field.widget.property("original_tooltip")
            if original_tooltip is None:
                original_tooltip = field.widget.toolTip()
                field.widget.setProperty("original_tooltip", original_tooltip)
            field.widget.setToolTip(f"⚠ {error_message}\n\n{original_tooltip}")
        else:
            original_tooltip = field.widget.property("original_tooltip")
            if original_tooltip:
                field.widget.setToolTip(original_tooltip)

        # Update tab completion indicators
        self._update_tab_indicators()

        # Update status bar validation count
        self._update_validation_status_bar()

    def _update_tab_indicators(self):
        """Update tab labels with validation status indicators."""
        tab_names = [
            "Simulation Setup",
            "Physical Parameters",
            "Experimental Conditions",
            "Analysis",
            "Output"
        ]

        for tab_idx, field_ids in self._tab_fields.items():
            if not field_ids:
                self.tabs.setTabText(tab_idx, tab_names[tab_idx])
                continue

            has_invalid = False
            for fid in field_ids:
                field = self.validation_manager._fields.get(fid)
                if field and field.is_active and not field.last_result.is_valid:
                    has_invalid = True
                    break

            indicator = " ⚠" if has_invalid else ""
            self.tabs.setTabText(tab_idx, f"{tab_names[tab_idx]}{indicator}")

    def _update_validation_status_bar(self):
        """Update status bar with validation error count."""
        error_count = sum(
            1 for f in self.validation_manager._fields.values()
            if f.is_active and not f.last_result.is_valid
        )
        if error_count:
            self.validation_status_label.setText(
                f"{error_count} validation error{'s' if error_count != 1 else ''}"
            )
            self.validation_status_label.setProperty("class", "status-error")
        else:
            self.validation_status_label.setText("")
        self.validation_status_label.style().unpolish(self.validation_status_label)
        self.validation_status_label.style().polish(self.validation_status_label)

    def show_keyboard_shortcuts(self):
        """Show keyboard shortcuts reference dialog."""
        shortcuts_text = """
<h2>Keyboard Shortcuts</h2>
<p><b>Ctrl+R</b> runs the simulation and <b>Ctrl+.</b> stops it.</p>
<p>Press <b>Alt</b> + the underlined letter to jump to a field.
Shortcuts are scoped to the active tab.</p>

<h3>Tab 1: Simulation Setup</h3>
<table>
<tr><td><b>Alt+N</b></td><td>Number of Runs</td></tr>
<tr><td><b>Alt+U</b></td><td>Filename Slug</td></tr>
<tr><td><b>Alt+M</b></td><td>Multiprocessing</td></tr>
<tr><td><b>Alt+R</b></td><td>Data Retention</td></tr>
</table>

<h3>Tab 2: Physical Parameters</h3>
<table>
<tr><td><b>Alt+L</b></td><td>Diffusion Length</td></tr>
<tr><td><b>Alt+C</b></td><td>Diffusion Coefficient</td></tr>
<tr><td><b>Alt+T</b></td><td>Lifetime</td></tr>
<tr><td><b>Alt+A</b></td><td>Amplitude</td></tr>
<tr><td><b>Alt+M</b></td><td>Mean Position</td></tr>
<tr><td><b>Alt+Y</b></td><td>Width Type</td></tr>
<tr><td><b>Alt+W</b></td><td>Width Value</td></tr>
</table>

<h3>Tab 3: Experimental Conditions</h3>
<table>
<tr><td><b>Alt+N</b></td><td>Noise \u03c3</td></tr>
<tr><td><b>Alt+C</b></td><td>CNR</td></tr>
<tr><td><b>Alt+L</b></td><td>Noise File</td></tr>
<tr><td><b>Alt+W</b></td><td>Spatial Width</td></tr>
<tr><td><b>Alt+X</b></td><td>Number of Pixels</td></tr>
<tr><td><b>Alt+A</b></td><td>Start</td></tr>
<tr><td><b>Alt+O</b></td><td>Stop</td></tr>
<tr><td><b>Alt+E</b></td><td>Steps</td></tr>
</table>

<h3>Tab 4: Analysis</h3>
<table>
<tr><td><b>Alt+W</b></td><td>Weighted Least Squares</td></tr>
<tr><td><b>Alt+O</b></td><td>Ordinary Least Squares</td></tr>
<tr><td><b>Alt+T</b></td><td>Accuracy Threshold</td></tr>
</table>

<h3>Tab 5: Output</h3>
<table>
<tr><td><b>Alt+Q</b></td><td>Quick Setup</td></tr>
<tr><td><b>Alt+T</b></td><td>File Type</td></tr>
<tr><td><b>Alt+W</b></td><td>Width</td></tr>
<tr><td><b>Alt+E</b></td><td>Height</td></tr>
<tr><td><b>Alt+D</b></td><td>DPI</td></tr>
<tr><td><b>Alt+B</b></td><td>Histogram Bins</td></tr>
<tr><td><b>Alt+N</b></td><td>Font Size</td></tr>
<tr><td><b>Alt+K</b></td><td>Tick Length</td></tr>
<tr><td><b>Alt+I</b></td><td>Tick Width</td></tr>
</table>
"""
        QMessageBox.information(self, "Keyboard Shortcuts", shortcuts_text)

    def show_about_dialog(self):
        """Show the About DICE dialog."""
        about_text = f"""
<h2>DICE - Diffusion Insight Computation Engine</h2>

<p><b>Version:</b> {__version__}</p>

<p>DICE is a Python-based scientific computing tool for quantifying noise effects
in optical measures of excited state transport in optoelectronic semiconducting materials.</p>

<p><b>Citation:</b><br>
If you use DICE in your research, please cite:<br>
Thiebes, J. J. (2023). <i>Diffusion Insight Computation Engine (DICE)</i> <br>
[Software]. Zenodo. https://doi.org/10.5281/zenodo.10258191</p>

<p><b>GitHub:</b> <a href="https://github.com/thiebes/DICE">https://github.com/thiebes/DICE</a></p>

<p><b>License:</b> MIT for the source code of DICE. Help &gt; Licenses gives the
licenses of the packages DICE uses.</p>
"""
        QMessageBox.about(self, "About DICE", about_text)

    def show_licenses_dialog(self):
        """Show the licenses of DICE and of the packages it uses.

        A standalone application shows the license notice it carries, with
        the license text of every bundled package. An installed copy has no
        such file, so it names the licenses that matter most.
        """
        notice = bundled_license_notice()
        if notice is None:
            QMessageBox.information(
                self, "Licenses",
                "The source code of DICE is licensed under the MIT License (LICENSE.md).\n\n"
                "The GUI uses PyQt6, which is licensed under the GNU General Public License "
                "version 3, and the Qt libraries, under the GNU Lesser General Public License "
                "version 3. Every installed package carries its own license.\n\n"
                "The standalone DICE applications are distributed under the GNU General Public "
                "License version 3 and carry the license text of every package they contain."
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Licenses")
        layout = QVBoxLayout(dialog)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText(notice.read_text(encoding='utf-8'))
        text.setAccessibleName("Licenses of DICE and of the packages in this application")
        layout.addWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.resize(760, 560)
        dialog.exec()

    def open_documentation(self):
        """Open DICE documentation on GitHub."""
        url = "https://github.com/thiebes/DICE#readme"
        webbrowser.open(url)

    def closeEvent(self, event):
        """Handle window close event to check for unsaved changes."""
        if self._close_requested and self.simulation_thread is not None:
            # Already stopping: the window closes when the run has ended.
            event.ignore()
            return

        if self.parameters_modified and not self._close_requested:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Do you want to save before exiting?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save
            )

            if reply == QMessageBox.StandardButton.Save:
                self.save_parameters()
                # If save was cancelled, don't exit
                if self.parameters_modified:
                    event.ignore()
                    return
            elif reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return

        if self.simulation_thread is not None and self.simulation_thread.isRunning():
            reply = QMessageBox.question(
                self, "Simulation Running",
                "A simulation is running. Stop it and exit?\n\n"
                "DICE stays open until the simulation has stopped, which can take a few "
                "seconds, and then exits.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

            # The question runs an event loop, so the run can have ended and
            # its thread been released while the question was open. Then
            # there is nothing to stop, and the checks below decide.
            if self.simulation_thread is not None and self.simulation_thread.isRunning():
                # The thread is not terminated, because a terminated parallel
                # run leaves its worker processes behind. The window stays,
                # with its inputs disabled and a message, and closes when
                # the thread reports its end.
                self.simulation_thread.request_stop()
                self._close_requested = True
                self._set_inputs_enabled(False)
                self.stop_button.setEnabled(False)
                self.status_label.setText("Stopping. DICE exits when the simulation has stopped...")
                event.ignore()
                return

        if self.interface.last_output_error is not None and not self._close_requested:
            reply = QMessageBox.question(
                self, "Output Files Not Written",
                "The output files of the last simulation have not been written, and its "
                "results are lost when DICE exits. Exit anyway?\n\n"
                "To keep them, choose No, then File > Write Output Files.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

        event.accept()

    def _set_inputs_enabled(self, enabled: bool) -> None:
        """Enable or disable the tabs, the example list and the menus; the status line stays readable."""
        for widget in (self.tabs, self.example_button, self.menuBar()):
            widget.setEnabled(enabled)

    def _exit_if_requested(self) -> None:
        """Close the window after a run that the user stopped by closing the window."""
        if not self._close_requested:
            return
        self._set_inputs_enabled(True)
        if self.interface.last_output_error is not None:
            # The run finished before the stop took effect and its files are
            # not written: the user decides, so the window stays.
            self._close_requested = False
            return
        # closeEvent skips its questions now: they were answered before the wait.
        self.close()


def main():
    """Main entry point for the GUI application."""
    app = QApplication(sys.argv)

    # Apply theme with accessibility support
    apply_theme(app)

    window = DiceGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
