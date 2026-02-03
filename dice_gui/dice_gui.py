"""
DICE GUI - Main Application

Graphical user interface for the Diffusion Insight Computation Engine.
"""

import sys
import webbrowser
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox,
    QComboBox, QRadioButton, QButtonGroup, QGroupBox, QFileDialog,
    QProgressBar, QMessageBox, QTextEdit, QSlider, QFormLayout, QScrollArea,
    QCheckBox, QStatusBar, QMenu, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt6.QtGui import QFont, QDoubleValidator, QIntValidator, QAction, QKeySequence

from dice import __version__
from dice_gui.validators import (
    validate_positive_integer, validate_positive_float, validate_float,
    validate_proximity_level, validate_time_range, validate_time_series,
    validate_file_path, validate_filename_slug,
    convert_fwhm_to_sigma, convert_sigma_to_fwhm,
    calculate_diffusion_length, calculate_pixel_size
)
from dice_gui.dice_interface import DiceInterface
from dice_gui.styles import DiceTheme, apply_theme
from dice_gui.accessibility import (
    add_accessible_name_and_description,
    add_tooltip_with_accessible_description,
    set_tab_order
)
from dice_gui.proximity_widget import ProximityTargetWidget
from dice_gui.validation_manager import ValidationManager, apply_validation_style, clear_validation_style

# Example parameter configurations
EXAMPLE_PARAMETERS = {
    "quick_test": {
        'filename slug': 'quick_test',
        'number of runs': 100,
        'nominal diffusion length': 0.1,
        'FWHM_0': 1,
        'amplitude_0': 1,
        'mean_0': 0,
        'noise value': 0.02,
        'spatial width': 5,
        'pixel width': 50,
        'time range': [0, 1, 5],
        'proximity level': 0.1,
    },
    "high_precision": {
        'filename slug': 'high_precision',
        'number of runs': 10000,
        'nominal diffusion length': 0.1,
        'FWHM_0': 1,
        'amplitude_0': 1,
        'mean_0': 0,
        'noise value': 0.02,
        'spatial width': 10,
        'pixel width': 200,
        'time range': [0, 2, 20],
        'proximity level': 0.1,
    },
    "publication": {
        'filename slug': 'publication_example',
        'number of runs': 1000,
        'nominal diffusion length': 0.1,
        'FWHM_0': 1,
        'amplitude_0': 1,
        'mean_0': 0,
        'noise value': 0.02,
        'spatial width': 5,
        'pixel width': 100,
        'time range': [0, 1, 10],
        'proximity level': 0.5,
    }
}


class SimulationThread(QThread):
    """Thread for running simulations without blocking the GUI."""

    progress = pyqtSignal(str)
    iteration_progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, interface: DiceInterface, parameters: dict):
        super().__init__()
        self.interface = interface
        self.parameters = parameters
        self._stop_requested = False

    def run(self):
        """Run the simulation in a separate thread."""
        try:
            num_runs = self.parameters.get('number of runs', 1000)
            is_parallel = self.parameters.get('multiprocessing', True)

            if is_parallel:
                import os
                cpu_count = os.cpu_count() or 1
                self.progress.emit(f"Running {num_runs:,} iterations ({cpu_count} CPU cores)...")
            else:
                self.progress.emit("Starting simulation...")

            def progress_callback(current: int, total: int):
                self.iteration_progress.emit(current, total)

            result = self.interface.run_simulation(
                self.parameters,
                progress_callback=progress_callback
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

    def request_stop(self):
        """Request the simulation to stop."""
        self._stop_requested = True


class DiceGUI(QMainWindow):
    """Main GUI window for DICE."""

    def __init__(self):
        super().__init__()
        self.interface = DiceInterface()
        self.simulation_thread = None
        self.theme = DiceTheme()

        # File management
        self.current_parameter_file = None
        self.parameters_modified = False
        self.settings = QSettings("DICE", "DICE_GUI")
        self.loaded_data_file = None  # Track loaded CSV file for status bar
        self._populating = False      # Prevent modification marking during load
        self.validation_manager = None  # Will be initialized in init_ui
        self._error_labels = {}  # field_id -> inline error QLabel

        # Map tabs to their validated field IDs for tab completion indicators
        self._tab_fields = {
            0: ["filename_slug"],  # Tab 1: Simulation Setup
            1: ["diffusion_length", "diffusion_coeff", "lifetime",
                "amplitude", "mean", "width"],  # Tab 2: Physical Parameters
            2: ["noise_value", "noise_file", "spatial_width",
                "time_start", "time_stop", "time_series"],  # Tab 3: Experimental Conditions
            3: [],  # Tab 4: Analysis Settings (no validated fields)
            4: [],  # Tab 5: Output Settings (no validated fields)
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

        # Create scrollable content area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(10, 10, 10, 10)

        # Add header
        header = self.create_header()
        scroll_layout.addWidget(header)

        # Add unit selection panel
        unit_panel = self.create_unit_panel()
        scroll_layout.addWidget(unit_panel)

        # Create tab widget
        self.tabs = QTabWidget()
        self.tab1 = self.create_tab1_simulation_setup()
        self.tab2 = self.create_tab2_physical_parameters()
        self.tab3 = self.create_tab3_experimental_conditions()
        self.tab4 = self.create_tab4_analysis_settings()
        self.tab5 = self.create_tab5_output_settings()

        self.tabs.addTab(self.tab1, "Simulation Setup")
        self.tabs.addTab(self.tab2, "Physical Parameters")
        self.tabs.addTab(self.tab3, "Experimental Conditions")
        self.tabs.addTab(self.tab4, "Analysis Settings")
        self.tabs.addTab(self.tab5, "Output Settings")

        scroll_layout.addWidget(self.tabs)

        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area, stretch=1)

        # Add sticky bottom control panel (outside scroll area)
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

        # Help Menu
        help_menu = menubar.addMenu("&Help")

        # Documentation
        doc_action = QAction("&Documentation", self)
        doc_action.setStatusTip("Open DICE documentation on GitHub")
        doc_action.triggered.connect(self.open_documentation)
        help_menu.addAction(doc_action)

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

    def create_unit_panel(self) -> QWidget:
        """Create the unit selection panel."""
        panel = QGroupBox("Units")
        layout = QHBoxLayout(panel)

        # Length unit
        length_label = QLabel("Length Unit:")
        self.length_unit_combo = QComboBox()
        self.length_unit_combo.addItems([
            "angstrom", "picometer", "nanometer", "micrometer",
            "millimeter", "centimeter", "meter"
        ])
        self.length_unit_combo.setCurrentText("micrometer")
        self.length_unit_combo.currentTextChanged.connect(self.update_unit_labels)

        # Time unit
        time_label = QLabel("Time Unit:")
        self.time_unit_combo = QComboBox()
        self.time_unit_combo.addItems([
            "attosecond", "femtosecond", "picosecond", "nanosecond",
            "microsecond", "millisecond", "second"
        ])
        self.time_unit_combo.setCurrentText("nanosecond")
        self.time_unit_combo.currentTextChanged.connect(self.update_unit_labels)

        layout.addWidget(length_label)
        layout.addWidget(self.length_unit_combo)
        layout.addStretch()
        layout.addWidget(time_label)
        layout.addWidget(self.time_unit_combo)

        return panel

    def create_tab1_simulation_setup(self) -> QWidget:
        """Create Tab 1: Simulation Setup."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Basic settings group
        basic_group = QGroupBox("Basic Settings")
        basic_layout = QFormLayout(basic_group)

        # Number of runs
        self.num_runs_spin = QSpinBox()
        self.num_runs_spin.setMinimum(1)
        self.num_runs_spin.setMaximum(1000000)
        self.num_runs_spin.setValue(1000)
        self.num_runs_spin.setToolTip(
            "Number of Monte Carlo simulation iterations to run.\n\n"
            "Higher values provide better statistical precision but take longer to compute.\n"
            "Typical values: 100-1000 for testing, 1000-10000 for publication-quality results.\n\n"
            "Each run generates a noisy profile, fits it, and estimates the diffusion coefficient."
        )
        basic_layout.addRow("Number of Runs:", self.num_runs_spin)

        # Filename slug
        self.filename_slug_input = QLineEdit()
        self.filename_slug_input.setText("dice_simulation")
        self.filename_slug_input.setToolTip(
            "Prefix for all output filenames.\n\n"
            "Output files will be saved as: output/<slug>/<slug>_results.csv, <slug>_accuracy_histogram.png, etc.\n\n"
            "Use descriptive names to organize multiple simulations (e.g., 'high_SNR_test' or 'sample_A_analysis')."
        )
        basic_layout.addRow("Filename Slug:", self.filename_slug_input)

        # Filename slug validation error label
        self._error_labels["filename_slug"] = self._create_error_label()
        basic_layout.addRow("", self._error_labels["filename_slug"])

        # Output path preview
        self.output_path_preview = QLabel()
        self.output_path_preview.setProperty("class", "output-path-preview")
        self.output_path_preview.setWordWrap(True)
        self.output_path_preview.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        basic_layout.addRow("Output Location:", self.output_path_preview)
        self.filename_slug_input.textChanged.connect(self._update_output_path_preview)

        # Performance settings group
        performance_group = QGroupBox("Performance Settings")
        performance_layout = QFormLayout(performance_group)

        # Multiprocessing checkbox
        self.multiprocessing_check = QCheckBox("Enable parallel processing")
        self.multiprocessing_check.setChecked(True)
        self.multiprocessing_check.setToolTip("Use multiple CPU cores to speed up simulation")
        performance_layout.addRow("Multiprocessing:", self.multiprocessing_check)

        # Retain profile data checkbox
        self.retain_profile_check = QCheckBox("Retain profile data")
        self.retain_profile_check.setChecked(False)
        self.retain_profile_check.setToolTip("Keep raw profile data (memory intensive)")
        performance_layout.addRow("Data Retention:", self.retain_profile_check)

        # Two-column layout for groups
        columns = QHBoxLayout()
        columns.addWidget(basic_group)
        columns.addWidget(performance_group)
        layout.addLayout(columns)

        # Info label
        info_label = QLabel(
            "Number of Runs: Number of Monte Carlo simulation iterations.\n\n"
            "Filename Slug: Prefix for output files.\n\n"
            "Multiprocessing: Enable to use multiple CPU cores for faster execution.\n\n"
            "Retain Profile Data: Keep raw profile data in memory. Only enable if you "
            "need the data for analysis, as it can be memory intensive for large simulations."
        )
        info_label.setWordWrap(True)
        info_label.setProperty("class", "info-text")
        layout.addWidget(info_label)

        layout.addStretch()

        return tab

    def create_tab2_physical_parameters(self) -> QWidget:
        """Create Tab 2: Physical Parameters."""
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)

        # Diffusion parameters group
        diffusion_group = QGroupBox("Diffusion Parameters")
        diffusion_layout = QVBoxLayout(diffusion_group)

        # Radio buttons for diffusion type
        self.diffusion_button_group = QButtonGroup()
        self.diffusion_length_radio = QRadioButton("Diffusion Length")
        self.diffusion_length_radio.setToolTip(
            "Specify diffusion as a single length parameter.\n\n"
            "Diffusion length L = sqrt(D*tau) is the characteristic distance a carrier diffuses during its lifetime.\n"
            "Use this when you know the overall transport distance but not individual D and tau values.\n\n"
            "Typical values: 10-1000 nm for organic semiconductors, 100-10000 nm for inorganic materials."
        )
        self.diffusion_coeff_radio = QRadioButton("Diffusion Coefficient + Lifetime")
        self.diffusion_coeff_radio.setToolTip(
            "Specify diffusion coefficient D and lifetime tau separately.\n\n"
            "Use this when you know both parameters independently from experiments.\n"
            "D controls spatial spreading rate, tau controls temporal decay.\n\n"
            "Typical D: 0.001-1 cm²/s (organics), 1-100 cm²/s (inorganics)\n"
            "Typical tau: 1-1000 ns"
        )
        self.diffusion_button_group.addButton(self.diffusion_length_radio, 0)
        self.diffusion_button_group.addButton(self.diffusion_coeff_radio, 1)
        self.diffusion_length_radio.setChecked(True)

        diffusion_layout.addWidget(self.diffusion_length_radio)

        # Diffusion length input
        length_container, length_layout = self._create_option_card()

        length_widget = QWidget()
        length_widget_layout = QHBoxLayout(length_widget)
        length_widget_layout.setContentsMargins(0, 0, 0, 0)
        self.diffusion_length_input = QLineEdit()
        self.diffusion_length_input.setPlaceholderText("e.g., 1.0")
        self.diffusion_length_input.setToolTip(
            "Characteristic diffusion length during carrier lifetime.\n\n"
            "This is the nominal value used to generate synthetic data.\n"
            "The simulation assesses how accurately this value can be recovered from noisy measurements.\n\n"
            "Must be positive. Units set by length unit selector above."
        )
        self.diffusion_length_label = QLabel("μm")
        length_widget_layout.addWidget(self.diffusion_length_input)
        length_widget_layout.addWidget(self.diffusion_length_label)

        length_layout.addRow("Diffusion Length:", length_widget)
        self._error_labels["diffusion_length"] = self._create_error_label()
        length_layout.addRow("", self._error_labels["diffusion_length"])
        diffusion_layout.addWidget(length_container)

        diffusion_layout.addWidget(self.diffusion_coeff_radio)

        # Diffusion coefficient + lifetime inputs
        coeff_container, coeff_layout = self._create_option_card()

        d_widget = QWidget()
        d_widget_layout = QHBoxLayout(d_widget)
        d_widget_layout.setContentsMargins(0, 0, 0, 0)
        self.diffusion_coeff_input = QLineEdit()
        self.diffusion_coeff_input.setPlaceholderText("e.g., 0.5")
        self.diffusion_coeff_input.setToolTip(
            "Diffusion coefficient describing spatial spreading rate.\n\n"
            "In 1D Fickian diffusion, variance grows as: sigma²(t) = sigma²(0) + 2*D*t\n\n"
            "Typical values:\n"
            "- Organic semiconductors: 0.001-0.1 cm²/s (0.01-10 μm²/ns)\n"
            "- Inorganic semiconductors: 0.1-100 cm²/s (10-10000 μm²/ns)\n\n"
            "Must be non-negative. Zero means no diffusion (only decay)."
        )
        self.diffusion_coeff_label = QLabel("μm²/ns")
        d_widget_layout.addWidget(self.diffusion_coeff_input)
        d_widget_layout.addWidget(self.diffusion_coeff_label)

        tau_widget = QWidget()
        tau_widget_layout = QHBoxLayout(tau_widget)
        tau_widget_layout.setContentsMargins(0, 0, 0, 0)
        self.lifetime_input = QLineEdit()
        self.lifetime_input.setPlaceholderText("e.g., 2.0")
        self.lifetime_input.setToolTip(
            "Excited state lifetime (tau) for exponential decay.\n\n"
            "Intensity decays as: I(t) = I(0) * exp(-t/tau)\n\n"
            "Typical values:\n"
            "- Fluorescence: 0.1-10 ns\n"
            "- Phosphorescence: 10-1000 ns\n"
            "- Triplet excitons: 1-1000 ns\n\n"
            "Must be non-negative. Zero means no decay (infinite lifetime)."
        )
        self.lifetime_label = QLabel("ns")
        tau_widget_layout.addWidget(self.lifetime_input)
        tau_widget_layout.addWidget(self.lifetime_label)

        coeff_layout.addRow("Diffusion Coefficient (D):", d_widget)
        self._error_labels["diffusion_coeff"] = self._create_error_label()
        coeff_layout.addRow("", self._error_labels["diffusion_coeff"])

        coeff_layout.addRow("Lifetime (τ):", tau_widget)
        self._error_labels["lifetime"] = self._create_error_label()
        coeff_layout.addRow("", self._error_labels["lifetime"])

        # Calculated diffusion length display
        self.calc_length_label = QLabel("Diffusion Length: ---")
        self.calc_length_label.setProperty("class", "calculated-value")
        coeff_layout.addRow("", self.calc_length_label)

        diffusion_layout.addWidget(coeff_container)

        # Connect radio buttons to enable/disable fields
        self.diffusion_length_radio.toggled.connect(self.toggle_diffusion_inputs)
        self.diffusion_coeff_input.textChanged.connect(self.update_calculated_length)
        self.lifetime_input.textChanged.connect(self.update_calculated_length)

        # Initial Profile group
        profile_group = QGroupBox("Initial Profile")
        profile_layout = QFormLayout(profile_group)

        # Amplitude
        self.amplitude_input = QLineEdit()
        self.amplitude_input.setText("1.0")
        self.amplitude_input.setToolTip(
            "Initial peak intensity of the Gaussian profile at t=0.\n\n"
            "Typically normalized to 1.0 for convenience.\n"
            "The noise level is specified relative to this amplitude.\n\n"
            "Can be zero or positive. Zero amplitude means no signal (only noise)."
        )
        profile_layout.addRow("Amplitude₀:", self.amplitude_input)
        self._error_labels["amplitude"] = self._create_error_label()
        profile_layout.addRow("", self._error_labels["amplitude"])

        # Mean position
        mean_widget = QWidget()
        mean_layout = QHBoxLayout(mean_widget)
        mean_layout.setContentsMargins(0, 0, 0, 0)
        self.mean_input = QLineEdit()
        self.mean_input.setText("0.0")
        self.mean_input.setToolTip(
            "Center position of the initial Gaussian profile along the spatial axis.\n\n"
            "Typically set to 0.0 (centered on the spatial window).\n"
            "The profile center does not move during diffusion (only spreads and decays).\n\n"
            "Should be within the spatial width defined in Experimental Conditions."
        )
        self.mean_label = QLabel("μm")
        mean_layout.addWidget(self.mean_input)
        mean_layout.addWidget(self.mean_label)
        profile_layout.addRow("Mean Position (μ₀):", mean_widget)
        self._error_labels["mean"] = self._create_error_label()
        profile_layout.addRow("", self._error_labels["mean"])

        # Profile width radio buttons
        self.width_button_group = QButtonGroup()
        self.fwhm_radio = QRadioButton("FWHM")
        self.fwhm_radio.setToolTip(
            "Full Width at Half Maximum of the Gaussian profile.\n\n"
            "FWHM is the width measured at 50% of peak intensity.\n"
            "Common in microscopy and spectroscopy (easier to measure experimentally).\n\n"
            "Relationship: FWHM = 2*sqrt(2*ln(2))*sigma ≈ 2.355*sigma"
        )
        self.sigma_radio = QRadioButton("Sigma (σ)")
        self.sigma_radio.setToolTip(
            "Standard deviation of the Gaussian profile.\n\n"
            "Sigma is the mathematical parameter in the Gaussian function: exp(-(x-μ)²/(2*sigma²))\n"
            "Preferred for theoretical analysis and diffusion calculations.\n\n"
            "Relationship: sigma = FWHM / 2.355"
        )
        self.width_button_group.addButton(self.fwhm_radio, 0)
        self.width_button_group.addButton(self.sigma_radio, 1)
        self.fwhm_radio.setChecked(True)

        width_radio_widget = QWidget()
        width_radio_layout = QHBoxLayout(width_radio_widget)
        width_radio_layout.setContentsMargins(0, 0, 0, 0)
        width_radio_layout.addWidget(self.fwhm_radio)
        width_radio_layout.addWidget(self.sigma_radio)
        width_radio_layout.addStretch()
        profile_layout.addRow("Width Type:", width_radio_widget)

        # Width input
        width_widget = QWidget()
        width_layout = QHBoxLayout(width_widget)
        width_layout.setContentsMargins(0, 0, 0, 0)
        self.width_input = QLineEdit()
        self.width_input.setPlaceholderText("e.g., 1.0")
        self.width_input.setToolTip(
            "Initial width of the Gaussian profile (FWHM or sigma, depending on selection above).\n\n"
            "This represents the spatial extent of the initial excitation (e.g., laser spot size).\n"
            "During diffusion, this width increases over time.\n\n"
            "Typical values: 0.1-10 μm for confocal microscopy\n"
            "Should be smaller than the spatial window to avoid edge effects.\n\n"
            "Must be positive."
        )
        self.width_unit_label = QLabel("μm")
        width_layout.addWidget(self.width_input)
        width_layout.addWidget(self.width_unit_label)
        profile_layout.addRow("Width Value:", width_widget)
        self._error_labels["width"] = self._create_error_label()
        profile_layout.addRow("", self._error_labels["width"])

        # Conversion display
        self.width_conversion_label = QLabel("Equivalent: ---")
        self.width_conversion_label.setProperty("class", "calculated-value")
        profile_layout.addRow("", self.width_conversion_label)

        # Connect width inputs
        self.fwhm_radio.toggled.connect(self.update_width_conversion)
        self.sigma_radio.toggled.connect(self.update_width_conversion)
        self.width_input.textChanged.connect(self.update_width_conversion)

        # Two-column layout for groups
        columns = QHBoxLayout()
        columns.addWidget(diffusion_group)
        columns.addWidget(profile_group)
        layout.addLayout(columns)
        layout.addStretch()

        scroll.setWidget(scroll_content)
        tab_layout = QVBoxLayout(tab)
        tab_layout.addWidget(scroll)

        return tab

    def create_tab3_experimental_conditions(self) -> QWidget:
        """Create Tab 3: Experimental Conditions."""
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)

        # Noise group
        noise_group = QGroupBox("Noise Parameters")
        noise_layout = QVBoxLayout(noise_group)
        noise_layout.setSpacing(4)

        # Radio buttons for noise type
        self.noise_button_group = QButtonGroup()
        self.noise_fixed_radio = QRadioButton("Fixed Noise Value")
        self.noise_fixed_radio.setToolTip(
            "Specify noise level directly as standard deviation.\n\n"
            "Use this when you know the noise level from calibration or previous measurements.\n"
            "Noise is added as Gaussian white noise with the specified standard deviation.\n\n"
            "For normalized amplitude of 1.0, CNR = 1/noise_value"
        )
        self.noise_estimate_radio = QRadioButton("Estimate from Data")
        self.noise_estimate_radio.setToolTip(
            "Estimate noise level from experimental profile data using FFT method.\n\n"
            "Load a CSV file containing an experimental profile.\n"
            "DICE will analyze high-frequency components to estimate background noise.\n\n"
            "Useful when noise level is unknown but experimental data is available."
        )
        self.noise_button_group.addButton(self.noise_fixed_radio, 0)
        self.noise_button_group.addButton(self.noise_estimate_radio, 1)
        self.noise_fixed_radio.setChecked(True)

        noise_layout.addWidget(self.noise_fixed_radio)

        # Fixed noise input
        fixed_container, fixed_layout = self._create_option_card()
        self.noise_value_input = QLineEdit()
        self.noise_value_input.setPlaceholderText("e.g., 0.01")
        self.noise_value_input.setToolTip(
            "Standard deviation of Gaussian white noise added to profiles.\n\n"
            "This value is constant across all pixels and time points.\n"
            "For amplitude=1.0, a noise value of 0.01 gives CNR=100, 0.1 gives CNR=10.\n\n"
            "Typical values: 0.001-0.1 (0.1%-10% of signal amplitude)\n"
            "Higher noise makes diffusion coefficient estimation more difficult.\n\n"
            "Must be non-negative. Zero means no noise (perfect measurements)."
        )
        fixed_layout.addRow("Noise σ:", self.noise_value_input)
        self._error_labels["noise_value"] = self._create_error_label()
        fixed_layout.addRow("", self._error_labels["noise_value"])
        noise_layout.addWidget(fixed_container)

        noise_layout.addWidget(self.noise_estimate_radio)

        # Estimate from data
        estimate_container, estimate_layout = self._create_option_card()

        file_widget = QWidget()
        file_widget_layout = QHBoxLayout(file_widget)
        file_widget_layout.setContentsMargins(0, 0, 0, 0)
        self.noise_file_input = QLineEdit()
        self.noise_file_input.setPlaceholderText("Path to CSV file...")
        self.noise_file_input.setToolTip(
            "Path to CSV file containing experimental profile data for noise estimation.\n\n"
            "File should contain spatial profile data with background noise.\n"
            "DICE will use FFT analysis to separate signal from noise components.\n\n"
            "Click Browse to select file."
        )
        self.noise_browse_button = QPushButton("Browse...")
        self.noise_browse_button.setToolTip("Select CSV file containing experimental profile data")
        self.noise_browse_button.clicked.connect(self.browse_noise_file)
        file_widget_layout.addWidget(self.noise_file_input)
        file_widget_layout.addWidget(self.noise_browse_button)
        estimate_layout.addRow("File:", file_widget)
        self._error_labels["noise_file"] = self._create_error_label()
        estimate_layout.addRow("", self._error_labels["noise_file"])

        self.noise_cnr_label = QLabel("Estimated CNR: ---")
        self.noise_cnr_label.setProperty("class", "calculated-value")
        estimate_layout.addRow("", self.noise_cnr_label)
        noise_layout.addWidget(estimate_container)

        # Connect radio buttons
        self.noise_fixed_radio.toggled.connect(self.toggle_noise_inputs)

        # Spatial domain group
        spatial_group = QGroupBox("Spatial Domain")
        spatial_layout = QFormLayout(spatial_group)

        # Spatial width
        spatial_width_widget = QWidget()
        spatial_width_layout = QHBoxLayout(spatial_width_widget)
        spatial_width_layout.setContentsMargins(0, 0, 0, 0)
        self.spatial_width_input = QLineEdit()
        self.spatial_width_input.setPlaceholderText("e.g., 10.0")
        self.spatial_width_input.setToolTip(
            "Total spatial width of the observation window.\n\n"
            "This defines the x-axis range from -width/2 to +width/2.\n"
            "Should be large enough to contain the spreading profile without edge truncation.\n\n"
            "Rule of thumb: Make this 3-5 times the final profile width.\n"
            "For diffusion length L and max time t_max: width ≈ 5*sqrt(sigma_0² + 2*D*t_max)\n\n"
            "Typical values: 5-50 μm for microscopy experiments"
        )
        self.spatial_width_label = QLabel("μm")
        spatial_width_layout.addWidget(self.spatial_width_input)
        spatial_width_layout.addWidget(self.spatial_width_label)
        spatial_layout.addRow("Spatial Width:", spatial_width_widget)
        self._error_labels["spatial_width"] = self._create_error_label()
        spatial_layout.addRow("", self._error_labels["spatial_width"])

        # Pixel width
        self.pixel_width_input = QSpinBox()
        self.pixel_width_input.setMinimum(1)
        self.pixel_width_input.setMaximum(100000)
        self.pixel_width_input.setValue(100)
        self.pixel_width_input.setToolTip(
            "Number of pixels (spatial sampling points) across the profile.\n\n"
            "Higher values provide better spatial resolution but increase computation time.\n\n"
            "Rule of thumb: At least 10-20 pixels per profile FWHM for accurate Gaussian fitting.\n"
            "For initial FWHM=1 μm and width=10 μm: 100 pixels gives 0.1 μm/pixel resolution.\n\n"
            "Typical values: 50-500 pixels\n"
            "Minimum practical: ~20-30 pixels"
        )
        spatial_layout.addRow("Number of Pixels:", self.pixel_width_input)

        # Calculated pixel size
        self.pixel_size_label = QLabel("Pixel Size: ---")
        self.pixel_size_label.setProperty("class", "calculated-value")
        spatial_layout.addRow("", self.pixel_size_label)

        # Connect for calculation
        self.spatial_width_input.textChanged.connect(self.update_pixel_size)
        self.pixel_width_input.valueChanged.connect(self.update_pixel_size)

        # Temporal domain group
        temporal_group = QGroupBox("Temporal Domain")
        temporal_layout = QVBoxLayout(temporal_group)

        # Radio buttons for time type
        self.time_button_group = QButtonGroup()
        self.time_range_radio = QRadioButton("Time Range")
        self.time_range_radio.setToolTip(
            "Define time points as evenly-spaced range.\n\n"
            "Generates linear time series: linspace(start, stop, steps)\n"
            "Convenient for uniform temporal sampling.\n\n"
            "Example: start=0, stop=10, steps=11 gives [0, 1, 2, ..., 10]"
        )
        self.time_series_radio = QRadioButton("Time Series")
        self.time_series_radio.setToolTip(
            "Specify arbitrary time points as comma-separated list.\n\n"
            "Allows non-uniform sampling (e.g., logarithmic spacing).\n"
            "Useful for matching experimental time delays.\n\n"
            "Example: 0.1, 0.5, 1, 2, 5, 10, 20, 50"
        )
        self.time_button_group.addButton(self.time_range_radio, 0)
        self.time_button_group.addButton(self.time_series_radio, 1)
        self.time_range_radio.setChecked(True)

        temporal_layout.addWidget(self.time_range_radio)

        # Time range inputs
        range_container, range_layout = self._create_option_card()

        start_widget = QWidget()
        start_widget_layout = QHBoxLayout(start_widget)
        start_widget_layout.setContentsMargins(0, 0, 0, 0)
        self.time_start_input = QLineEdit()
        self.time_start_input.setPlaceholderText("e.g., 0.0")
        self.time_start_input.setToolTip(
            "First time point for profile measurements.\n\n"
            "Often set to 0 (initial excitation) but can be non-zero.\n"
            "For non-zero start, initial profile still has width specified in Physical Parameters.\n\n"
            "Must be less than stop time."
        )
        self.time_start_label = QLabel("ns")
        start_widget_layout.addWidget(self.time_start_input)
        start_widget_layout.addWidget(self.time_start_label)

        stop_widget = QWidget()
        stop_widget_layout = QHBoxLayout(stop_widget)
        stop_widget_layout.setContentsMargins(0, 0, 0, 0)
        self.time_stop_input = QLineEdit()
        self.time_stop_input.setPlaceholderText("e.g., 10.0")
        self.time_stop_input.setToolTip(
            "Final time point for profile measurements.\n\n"
            "Should be long enough to observe significant diffusion but not so long that signal decays to noise.\n\n"
            "Rule of thumb: For lifetime tau, useful range is ~0.1*tau to ~2*tau\n"
            "For diffusion, need enough time for measurable width increase (delta_sigma² > noise sensitivity)\n\n"
            "Must be greater than start time."
        )
        self.time_stop_label = QLabel("ns")
        stop_widget_layout.addWidget(self.time_stop_input)
        stop_widget_layout.addWidget(self.time_stop_label)

        self.time_steps_input = QSpinBox()
        self.time_steps_input.setMinimum(2)
        self.time_steps_input.setMaximum(10000)
        self.time_steps_input.setValue(10)
        self.time_steps_input.setToolTip(
            "Number of time points in the range (including start and stop).\n\n"
            "More time points improve linear regression fit but increase computation time.\n\n"
            "Rule of thumb: At least 5-10 points for reliable linear fit.\n"
            "Typical values: 10-50 time points\n\n"
            "Minimum: 2 (though 3+ strongly recommended for meaningful statistics)"
        )

        range_layout.addRow("Start:", start_widget)
        self._error_labels["time_start"] = self._create_error_label()
        range_layout.addRow("", self._error_labels["time_start"])

        range_layout.addRow("Stop:", stop_widget)
        self._error_labels["time_stop"] = self._create_error_label()
        range_layout.addRow("", self._error_labels["time_stop"])

        range_layout.addRow("Steps:", self.time_steps_input)
        temporal_layout.addWidget(range_container)

        temporal_layout.addWidget(self.time_series_radio)

        # Time series input
        series_container, series_layout = self._create_option_card("vbox")
        series_label = QLabel("Comma-separated time values:")
        self.time_series_input = QTextEdit()
        self.time_series_input.setPlaceholderText("e.g., 0.1, 0.5, 1.0, 2.0, 5.0")
        self.time_series_input.setMaximumHeight(80)
        self.time_series_input.setToolTip(
            "Arbitrary time points as comma-separated values.\n\n"
            "Allows custom temporal sampling to match experimental conditions.\n"
            "Useful for logarithmic spacing or irregular time delays.\n\n"
            "Example: 0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50\n\n"
            "Must have at least 2 time points for linear regression.\n"
            "Time points should be in ascending order (though not strictly required)."
        )
        series_layout.addWidget(series_label)
        series_layout.addWidget(self.time_series_input)
        self._error_labels["time_series"] = self._create_error_label()
        series_layout.addWidget(self._error_labels["time_series"])
        temporal_layout.addWidget(series_container)

        # Connect radio buttons
        self.time_range_radio.toggled.connect(self.toggle_time_inputs)

        # Three-column layout: noise | spatial | temporal
        columns = QHBoxLayout()
        columns.addWidget(noise_group)
        columns.addWidget(spatial_group)
        columns.addWidget(temporal_group)
        layout.addLayout(columns)
        layout.addStretch()

        scroll.setWidget(scroll_content)
        tab_layout = QVBoxLayout(tab)
        tab_layout.addWidget(scroll)

        return tab

    def create_tab4_analysis_settings(self) -> QWidget:
        """Create Tab 4: Analysis Settings."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Proximity level group
        proximity_group = QGroupBox("Accuracy Threshold")
        proximity_layout = QVBoxLayout(proximity_group)

        proximity_label = QLabel("Proximity Level:")
        self.proximity_spin = QDoubleSpinBox()
        self.proximity_spin.setMinimum(0.001)
        self.proximity_spin.setMaximum(1000.0)
        self.proximity_spin.setValue(0.10)
        self.proximity_spin.setDecimals(3)
        self.proximity_spin.setSingleStep(0.01)
        self.proximity_spin.setToolTip(
            "Threshold for accuracy analysis as fractional tolerance.\n\n"
            "Defines the acceptable range for diffusion coefficient estimates:\n"
            "- 0.10 means ±10% (estimates between 0.9*D_nominal and 1.1*D_nominal are 'accurate')\n"
            "- 0.20 means ±20%\n"
            "- 0.05 means ±5%\n\n"
            "The simulation reports what fraction of Monte Carlo runs fall within this range.\n"
            "This helps assess whether your experimental conditions provide reliable measurements.\n\n"
            "Typical values: 0.10-0.20 for most applications\n"
            "Stricter: 0.05 for high-precision requirements"
        )

        proximity_input_layout = QHBoxLayout()
        proximity_input_layout.addWidget(proximity_label)
        proximity_input_layout.addWidget(self.proximity_spin)
        proximity_input_layout.addStretch()

        proximity_layout.addLayout(proximity_input_layout)

        description = QLabel(
            "The proximity level determines the threshold for accuracy analysis.\n"
            "For example, 0.10 means estimates within ±10% of the nominal value\n"
            "are considered accurate."
        )
        description.setWordWrap(True)
        description.setProperty("class", "info-text")
        proximity_layout.addWidget(description)

        # Target visualization below description (left-aligned)
        self.proximity_target = ProximityTargetWidget(proximity=0.10)
        self.proximity_spin.valueChanged.connect(self.update_proximity_target)
        proximity_layout.addWidget(self.proximity_target)

        # Fit method group
        fit_method_group = QGroupBox("Fit Method")
        fit_method_layout = QVBoxLayout(fit_method_group)

        self.plot_method_wls_radio = QRadioButton("Weighted Least Squares (WLS)")
        self.plot_method_ols_radio = QRadioButton("Ordinary Least Squares (OLS)")
        self.plot_method_wls_radio.setChecked(True)
        self.plot_method_wls_radio.setToolTip(
            "Weighted Least Squares regression.\n\n"
            "Weights each data point by the inverse of its variance.\n"
            "Recommended for most applications as it accounts for\n"
            "heteroscedasticity in MSD measurements."
        )
        self.plot_method_ols_radio.setToolTip(
            "Ordinary Least Squares regression.\n\n"
            "Treats all data points equally regardless of variance.\n"
            "May be preferred when measurement uncertainties are uniform."
        )
        fit_method_layout.addWidget(self.plot_method_wls_radio)
        fit_method_layout.addWidget(self.plot_method_ols_radio)
        fit_method_layout.addStretch()

        # Two-column layout
        columns = QHBoxLayout()
        columns.addWidget(proximity_group)
        columns.addWidget(fit_method_group)
        layout.addLayout(columns)
        layout.addStretch()

        return tab

    def create_tab5_output_settings(self) -> QWidget:
        """Create Tab 5: Output Settings."""
        from dice_gui.presets import PRESETS, OUTPUT_DEFAULTS, get_preset

        tab = QWidget()
        layout = QVBoxLayout(tab)

        # === Preset Section ===
        preset_layout = QHBoxLayout()
        preset_label = QLabel("Quick Setup:")
        self.preset_combo = QComboBox()
        self.preset_combo.setProperty("class", "preset-selector")
        self.preset_combo.addItem("Custom", None)
        self.preset_combo.addItem("Publication (journal-ready)", "publication")
        self.preset_combo.addItem("Presentation (large fonts)", "presentation")
        self.preset_combo.addItem("Draft (quick preview)", "draft")
        self.preset_combo.currentIndexChanged.connect(self._apply_output_preset)

        preset_layout.addWidget(preset_label)
        preset_layout.addWidget(self.preset_combo)
        preset_layout.addStretch()
        layout.addLayout(preset_layout)

        # === Column 1: Image Format ===
        format_group = QGroupBox("Image Format")
        format_layout = QFormLayout(format_group)

        # Image type
        self.image_type_combo = QComboBox()
        self.image_type_combo.addItems(["png", "jpg", "svg", "tif"])
        self.image_type_combo.setCurrentText("png")
        self.image_type_combo.setToolTip(
            "File format for saved plots.\n\n"
            "- PNG: Best for general use, lossless compression (recommended)\n"
            "- JPG: Smaller files but lossy compression\n"
            "- SVG: Vector format, scalable, ideal for publications\n"
            "- TIF: Uncompressed, maximum quality"
        )
        format_layout.addRow("File Type:", self.image_type_combo)
        self.image_type_combo.currentTextChanged.connect(self._update_output_path_preview)

        # Image width
        width_widget = QWidget()
        width_layout = QHBoxLayout(width_widget)
        width_layout.setContentsMargins(0, 0, 0, 0)
        self.image_width_spin = QDoubleSpinBox()
        self.image_width_spin.setMinimum(0.1)
        self.image_width_spin.setMaximum(100.0)
        self.image_width_spin.setValue(16.0)
        self.image_width_spin.setDecimals(2)
        self.image_width_unit_combo = QComboBox()
        self.image_width_unit_combo.addItems(["cm", "in", "mm"])
        width_layout.addWidget(self.image_width_spin)
        width_layout.addWidget(self.image_width_unit_combo)
        format_layout.addRow("Width:", width_widget)

        # Image height
        height_widget = QWidget()
        height_layout = QHBoxLayout(height_widget)
        height_layout.setContentsMargins(0, 0, 0, 0)
        self.image_height_spin = QDoubleSpinBox()
        self.image_height_spin.setMinimum(0.1)
        self.image_height_spin.setMaximum(100.0)
        self.image_height_spin.setValue(10.0)
        self.image_height_spin.setDecimals(2)
        self.image_height_unit_combo = QComboBox()
        self.image_height_unit_combo.addItems(["cm", "in", "mm"])
        height_layout.addWidget(self.image_height_spin)
        height_layout.addWidget(self.image_height_unit_combo)
        format_layout.addRow("Height:", height_widget)

        # === Column 2: Resolution & Histogram ===
        resolution_group = QGroupBox("Resolution & Histogram")
        resolution_layout = QFormLayout(resolution_group)

        # DPI
        self.image_dpi_spin = QSpinBox()
        self.image_dpi_spin.setMinimum(50)
        self.image_dpi_spin.setMaximum(1200)
        self.image_dpi_spin.setValue(300)
        self.image_dpi_spin.setToolTip("Resolution: 300 DPI for publications, 96 for screen")
        resolution_layout.addRow("DPI:", self.image_dpi_spin)

        # Number of bins
        self.image_numbins_spin = QSpinBox()
        self.image_numbins_spin.setMinimum(5)
        self.image_numbins_spin.setMaximum(200)
        self.image_numbins_spin.setValue(35)
        self.image_numbins_spin.setToolTip("Number of bins for accuracy histogram")
        resolution_layout.addRow("Histogram Bins:", self.image_numbins_spin)

        # === Column 3: Typography ===
        typography_group = QGroupBox("Typography")
        typography_layout = QFormLayout(typography_group)

        # Font size
        font_widget = QWidget()
        font_layout = QHBoxLayout(font_widget)
        font_layout.setContentsMargins(0, 0, 0, 0)
        self.image_font_size_spin = QSpinBox()
        self.image_font_size_spin.setMinimum(4)
        self.image_font_size_spin.setMaximum(72)
        self.image_font_size_spin.setValue(6)
        self.image_font_unit_combo = QComboBox()
        self.image_font_unit_combo.addItems(["pt", "px"])
        font_layout.addWidget(self.image_font_size_spin)
        font_layout.addWidget(self.image_font_unit_combo)
        typography_layout.addRow("Font Size:", font_widget)

        # Tick length
        tick_length_widget = QWidget()
        tick_length_layout = QHBoxLayout(tick_length_widget)
        tick_length_layout.setContentsMargins(0, 0, 0, 0)
        self.image_tick_length_spin = QSpinBox()
        self.image_tick_length_spin.setMinimum(1)
        self.image_tick_length_spin.setMaximum(50)
        self.image_tick_length_spin.setValue(6)
        self.image_tick_length_unit_combo = QComboBox()
        self.image_tick_length_unit_combo.addItems(["pt", "px"])
        tick_length_layout.addWidget(self.image_tick_length_spin)
        tick_length_layout.addWidget(self.image_tick_length_unit_combo)
        typography_layout.addRow("Tick Length:", tick_length_widget)

        # Tick width
        tick_width_widget = QWidget()
        tick_width_layout = QHBoxLayout(tick_width_widget)
        tick_width_layout.setContentsMargins(0, 0, 0, 0)
        self.image_tick_width_spin = QSpinBox()
        self.image_tick_width_spin.setMinimum(1)
        self.image_tick_width_spin.setMaximum(20)
        self.image_tick_width_spin.setValue(2)
        self.image_tick_width_unit_combo = QComboBox()
        self.image_tick_width_unit_combo.addItems(["pt", "px"])
        tick_width_layout.addWidget(self.image_tick_width_spin)
        tick_width_layout.addWidget(self.image_tick_width_unit_combo)
        typography_layout.addRow("Tick Width:", tick_width_widget)

        # Three-column layout
        columns = QHBoxLayout()
        columns.addWidget(format_group)
        columns.addWidget(resolution_group)
        columns.addWidget(typography_group)
        layout.addLayout(columns)

        # === Plot Actions ===
        actions_layout = QHBoxLayout()

        self.regenerate_plot_button = QPushButton("Regenerate Plot")
        self.regenerate_plot_button.setToolTip("Regenerate plot with current settings")
        self.regenerate_plot_button.clicked.connect(self.regenerate_plot)

        self.load_results_button = QPushButton("Load Results")
        self.load_results_button.setToolTip("Load results from CSV file")
        self.load_results_button.clicked.connect(self.load_and_plot_results)

        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.setProperty("class", "reset-button")
        reset_btn.clicked.connect(self._reset_all_plot_settings)

        actions_layout.addWidget(self.regenerate_plot_button)
        actions_layout.addWidget(self.load_results_button)
        actions_layout.addStretch()
        actions_layout.addWidget(reset_btn)

        layout.addLayout(actions_layout)
        layout.addStretch()

        return tab

    def _apply_output_preset(self, index: int):
        """Apply selected output preset."""
        from dice_gui.presets import get_preset

        preset_key = self.preset_combo.currentData()
        if preset_key is None:
            return

        preset = get_preset(preset_key)
        if preset is None:
            return

        settings = preset.settings

        self.image_type_combo.setCurrentText(settings.get("image_type", "png"))
        self.image_width_spin.setValue(settings.get("image_width", 16.0))
        self.image_width_unit_combo.setCurrentText(settings.get("image_width_unit", "cm"))
        self.image_height_spin.setValue(settings.get("image_height", 10.0))
        self.image_height_unit_combo.setCurrentText(settings.get("image_height_unit", "cm"))
        self.image_dpi_spin.setValue(settings.get("image_dpi", 300))
        self.image_numbins_spin.setValue(settings.get("image_numbins", 35))
        self.image_font_size_spin.setValue(settings.get("image_font_size", 6))
        self.image_font_unit_combo.setCurrentText(settings.get("image_font_unit", "pt"))
        self.image_tick_length_spin.setValue(settings.get("image_tick_length", 6))
        self.image_tick_width_spin.setValue(settings.get("image_tick_width", 2))

    def _reset_all_plot_settings(self):
        """Reset all plot settings to defaults."""
        from dice_gui.presets import OUTPUT_DEFAULTS

        # Image format
        self.image_type_combo.setCurrentText(OUTPUT_DEFAULTS["image_type"])
        self.image_width_spin.setValue(OUTPUT_DEFAULTS["image_width"])
        self.image_width_unit_combo.setCurrentText(OUTPUT_DEFAULTS["image_width_unit"])
        self.image_height_spin.setValue(OUTPUT_DEFAULTS["image_height"])
        self.image_height_unit_combo.setCurrentText(OUTPUT_DEFAULTS["image_height_unit"])

        # Resolution & histogram
        self.image_dpi_spin.setValue(OUTPUT_DEFAULTS["image_dpi"])
        self.image_numbins_spin.setValue(OUTPUT_DEFAULTS["image_numbins"])

        # Typography
        self.image_font_size_spin.setValue(OUTPUT_DEFAULTS["image_font_size"])
        self.image_font_unit_combo.setCurrentText(OUTPUT_DEFAULTS["image_font_unit"])
        self.image_tick_length_spin.setValue(OUTPUT_DEFAULTS["image_tick_length"])
        self.image_tick_length_unit_combo.setCurrentText(OUTPUT_DEFAULTS["image_tick_length_unit"])
        self.image_tick_width_spin.setValue(OUTPUT_DEFAULTS["image_tick_width"])
        self.image_tick_width_unit_combo.setCurrentText(OUTPUT_DEFAULTS["image_tick_width_unit"])

        self.preset_combo.setCurrentIndex(0)

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

        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("stop-button")
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
        # Already set in create methods, but can add more here if needed
        self.toggle_diffusion_inputs()
        self.toggle_noise_inputs()
        self.toggle_time_inputs()

    def update_unit_labels(self):
        """Update all unit labels based on selected units."""
        length_unit = self.length_unit_combo.currentText()
        time_unit = self.time_unit_combo.currentText()

        # Map full names to abbreviations
        length_abbrev = {
            "meter": "m", "centimeter": "cm", "millimeter": "mm",
            "micrometer": "μm", "nanometer": "nm", "angstrom": "Å",
            "picometer": "pm"
        }.get(length_unit, length_unit)

        time_abbrev = {
            "second": "s", "millisecond": "ms", "microsecond": "μs",
            "nanosecond": "ns", "picosecond": "ps", "femtosecond": "fs",
            "attosecond": "as"
        }.get(time_unit, time_unit)

        # Update all labels
        self.diffusion_length_label.setText(length_abbrev)
        self.diffusion_coeff_label.setText(f"{length_abbrev}²/{time_abbrev}")
        self.lifetime_label.setText(time_abbrev)
        self.mean_label.setText(length_abbrev)
        self.width_unit_label.setText(length_abbrev)
        self.spatial_width_label.setText(length_abbrev)
        self.time_start_label.setText(time_abbrev)
        self.time_stop_label.setText(time_abbrev)

        # Update calculated values
        self.update_calculated_length()
        self.update_width_conversion()
        self.update_pixel_size()

    def toggle_diffusion_inputs(self):
        """Enable/disable diffusion inputs based on radio selection."""
        if self.diffusion_length_radio.isChecked():
            self.diffusion_length_input.setEnabled(True)
            self.diffusion_coeff_input.setEnabled(False)
            self.lifetime_input.setEnabled(False)
            self.diffusion_coeff_input.clear()
            self.lifetime_input.clear()
            self.calc_length_label.setText("Diffusion Length: ---")
            # Clear validation styling on disabled fields
            clear_validation_style(self.diffusion_coeff_input)
            clear_validation_style(self.lifetime_input)
            self._error_labels.get("diffusion_coeff", QLabel()).setVisible(False)
            self._error_labels.get("lifetime", QLabel()).setVisible(False)
        else:
            self.diffusion_length_input.setEnabled(False)
            self.diffusion_coeff_input.setEnabled(True)
            self.lifetime_input.setEnabled(True)
            self.diffusion_length_input.clear()
            # Clear validation styling on disabled field
            clear_validation_style(self.diffusion_length_input)
            self._error_labels.get("diffusion_length", QLabel()).setVisible(False)

    def toggle_noise_inputs(self):
        """Enable/disable noise inputs based on radio selection."""
        if self.noise_fixed_radio.isChecked():
            self.noise_value_input.setEnabled(True)
            self.noise_file_input.setEnabled(False)
            self.noise_browse_button.setEnabled(False)
            self.noise_file_input.clear()
            self.noise_cnr_label.setText("Estimated CNR: ---")
            # Clear validation styling on disabled field
            clear_validation_style(self.noise_file_input)
            self._error_labels.get("noise_file", QLabel()).setVisible(False)
        else:
            self.noise_value_input.setEnabled(False)
            self.noise_file_input.setEnabled(True)
            self.noise_browse_button.setEnabled(True)
            self.noise_value_input.clear()
            # Clear validation styling on disabled field
            clear_validation_style(self.noise_value_input)
            self._error_labels.get("noise_value", QLabel()).setVisible(False)

    def toggle_time_inputs(self):
        """Enable/disable time inputs based on radio selection."""
        if self.time_range_radio.isChecked():
            self.time_start_input.setEnabled(True)
            self.time_stop_input.setEnabled(True)
            self.time_steps_input.setEnabled(True)
            self.time_series_input.setEnabled(False)
            self.time_series_input.clear()
            # Clear validation styling on disabled field
            clear_validation_style(self.time_series_input)
            self._error_labels.get("time_series", QLabel()).setVisible(False)
        else:
            self.time_start_input.setEnabled(False)
            self.time_stop_input.setEnabled(False)
            self.time_steps_input.setEnabled(False)
            self.time_series_input.setEnabled(True)
            self.time_start_input.clear()
            self.time_stop_input.clear()
            # Clear validation styling on disabled fields
            clear_validation_style(self.time_start_input)
            clear_validation_style(self.time_stop_input)
            self._error_labels.get("time_start", QLabel()).setVisible(False)
            self._error_labels.get("time_stop", QLabel()).setVisible(False)

    def update_calculated_length(self):
        """Update calculated diffusion length from D and tau."""
        if not self.diffusion_coeff_radio.isChecked():
            return

        try:
            D = float(self.diffusion_coeff_input.text())
            tau = float(self.lifetime_input.text())
            if D >= 0 and tau >= 0:
                length = calculate_diffusion_length(D, tau)
                length_unit = self.length_unit_combo.currentText()
                length_abbrev = {
                    "meter": "m", "centimeter": "cm", "millimeter": "mm",
                    "micrometer": "μm", "nanometer": "nm", "angstrom": "Å",
                    "picometer": "pm"
                }.get(length_unit, length_unit)
                self.calc_length_label.setText(f"Diffusion Length: {length:.4f} {length_abbrev}")
            else:
                self.calc_length_label.setText("Diffusion Length: ---")
        except (ValueError, ZeroDivisionError):
            self.calc_length_label.setText("Diffusion Length: ---")

    def update_width_conversion(self):
        """Update width conversion display."""
        try:
            value = float(self.width_input.text())
            if value > 0:
                if self.fwhm_radio.isChecked():
                    sigma = convert_fwhm_to_sigma(value)
                    self.width_conversion_label.setText(f"Equivalent σ: {sigma:.4f}")
                else:
                    fwhm = convert_sigma_to_fwhm(value)
                    self.width_conversion_label.setText(f"Equivalent FWHM: {fwhm:.4f}")
            else:
                self.width_conversion_label.setText("Equivalent: ---")
        except ValueError:
            self.width_conversion_label.setText("Equivalent: ---")

    def update_pixel_size(self):
        """Update calculated pixel size."""
        try:
            spatial_width = float(self.spatial_width_input.text())
            pixel_width = self.pixel_width_input.value()
            if spatial_width > 0 and pixel_width > 0:
                pixel_size = calculate_pixel_size(spatial_width, pixel_width)
                length_unit = self.length_unit_combo.currentText()
                length_abbrev = {
                    "meter": "m", "centimeter": "cm", "millimeter": "mm",
                    "micrometer": "μm", "nanometer": "nm", "angstrom": "Å",
                    "picometer": "pm"
                }.get(length_unit, length_unit)
                self.pixel_size_label.setText(f"Pixel Size: {pixel_size:.4f} {length_abbrev}/pixel")
            else:
                self.pixel_size_label.setText("Pixel Size: ---")
        except ValueError:
            self.pixel_size_label.setText("Pixel Size: ---")

    def update_proximity_target(self):
        """Update the proximity target visualization."""
        proximity_value = self.proximity_spin.value()
        self.proximity_target.set_proximity(proximity_value)

    def _update_output_path_preview(self):
        """Update the output path preview based on current slug."""
        from pathlib import Path

        slug = self.filename_slug_input.text().strip()
        if not slug:
            self.output_path_preview.setText("Enter a filename slug to see output paths")
            return

        output_dir = Path.cwd() / 'output' / slug
        image_type = getattr(self, 'image_type_combo', None)
        ext = image_type.currentText() if image_type else 'png'

        self.output_path_preview.setText(
            f"<b>Directory:</b> {output_dir}<br>"
            f"<b>Files:</b> {slug}.csv, {slug}_summary.txt, {slug}_accuracy_histogram.{ext}"
        )

    def browse_noise_file(self):
        """Open file dialog to select noise data file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Noise Data File",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self.noise_file_input.setText(file_path)
            # TODO: Estimate CNR from file using FFT method
            self.noise_cnr_label.setText("Estimated CNR: (calculation not yet implemented)")

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
            use_wls = self.plot_method_wls_radio.isChecked()

            # Get output location (original directory)
            slug = self.interface.last_parameters.get('filename slug', 'plot')
            from pathlib import Path
            output_dir = Path.cwd() / 'output' / slug

            # Get plot filename with overwrite handling
            filename = self.get_plot_filename(output_dir, slug, image_settings['image_type'])

            # Call interface method to regenerate
            self.interface.regenerate_plot_from_memory(
                filename=filename,
                image_settings=image_settings,
                use_wls=use_wls
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

        # Get default directory from filename slug output path
        default_dir = Path.cwd() / 'output' / self.filename_slug_input.text()
        if not default_dir.exists():
            default_dir = Path.cwd() / 'output'
        if not default_dir.exists():
            default_dir = Path.cwd()

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

            # Get proximity from Analysis Settings tab
            proximity = self.proximity_spin.value()

            # Get plot method
            use_wls = self.plot_method_wls_radio.isChecked()

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
                proximity=proximity,
                image_settings=image_settings,
                use_wls=use_wls
            )

            # Update status bar
            self.loaded_data_file = csv_file
            self.update_status_bar()

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

        # Diffusion
        if self.diffusion_length_radio.isChecked():
            result = validate_positive_float(self.diffusion_length_input.text(), "Diffusion length")
            if not result:
                return False, result.error_message
        else:
            result = validate_positive_float(self.diffusion_coeff_input.text(), "Diffusion coefficient", allow_zero=True)
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
            result = validate_time_range(
                self.time_start_input.text(),
                self.time_stop_input.text(),
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
            'proximity_level': self.proximity_spin.value(),
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

        # Diffusion
        if self.diffusion_length_radio.isChecked():
            params['diffusion_type'] = 'length'
            params['diffusion_length'] = float(self.diffusion_length_input.text())
        else:
            params['diffusion_type'] = 'coefficient'
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

        return params

    def run_simulation(self):
        """Run the DICE simulation."""
        # Validate all active fields (this triggers visual updates)
        is_valid, error_message = self.validation_manager.validate_all()
        if not is_valid:
            QMessageBox.warning(
                self,
                "Missing or Invalid Parameters",
                "Please fill in all required fields.\n\n"
                "Fields with errors are highlighted in red."
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

        # Update UI for running state
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.elapsed_label.setVisible(True)

        # Set progress bar mode based on multiprocessing
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
        self.simulation_thread.start()

    def stop_simulation(self):
        """Stop the running simulation."""
        if self.simulation_thread and self.simulation_thread.isRunning():
            self.simulation_thread.terminate()
            self.simulation_thread.wait()
            self.reset_ui_after_simulation()
            self.status_label.setText("Simulation stopped by user")

    def update_progress(self, message: str):
        """Update progress display."""
        self.status_label.setText(message)

    def _update_iteration_progress(self, current: int, total: int):
        """Update progress bar with iteration count and estimated time remaining."""
        self.progress_bar.setValue(current)
        self.status_label.setText(f"Running {current:,}/{total:,}")

        # Calculate estimated time remaining after first batch (10%)
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
        self.reset_ui_after_simulation()
        self.status_label.setText("Simulation completed successfully!")

        # Clear loaded data file (new simulation overwrites)
        self.loaded_data_file = None
        self.update_status_bar()

        QMessageBox.information(
            self,
            "Simulation Complete",
            "DICE simulation completed successfully!\n\n"
            "Results have been saved to output files."
        )

    def simulation_error(self, error_message: str):
        """Handle simulation error."""
        self.reset_ui_after_simulation()
        self.status_label.setText("Simulation failed")
        QMessageBox.critical(self, "Simulation Error", f"Simulation failed:\n\n{error_message}")

    def reset_ui_after_simulation(self):
        """Reset UI elements after simulation completes or stops."""
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
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

        # Open file dialog
        default_dir = str(Path.cwd())
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Parameters File",
            default_dir,
            "Parameter Files (*.txt);;All Files (*)"
        )

        if not file_path:
            return

        try:
            self.load_parameters_from_file(file_path)
        except Exception as e:
            QMessageBox.critical(
                self, "Error Loading Parameters",
                f"Failed to load parameters:\n\n{str(e)}"
            )

    def load_parameters_from_file(self, file_path: str):
        """Load parameters from a specific file path."""
        import ast

        # Read and parse the parameter file
        with open(file_path, 'r') as f:
            content = f.read()

        # Parse the dictionary
        params = ast.literal_eval(content)

        # Populate GUI from parameters
        self.populate_gui_from_parameters(params)

        # Update file tracking
        self.current_parameter_file = file_path
        self.parameters_modified = False
        self.add_to_recent_files(file_path)
        self.update_window_title()

        QMessageBox.information(
            self, "Parameters Loaded",
            f"Parameters loaded from:\n{file_path}"
        )

    def load_example(self, example_name: str):
        """Load example parameters."""
        if self.parameters_modified:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Load example anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return

        # Try to load from file first
        example_file = Path(__file__).parent.parent / "data" / "examples" / f"{example_name}.txt"
        if example_file.exists():
            try:
                self.load_parameters_from_file(str(example_file))
                return
            except Exception:
                pass  # Fall back to embedded

        # Use embedded parameters
        if example_name in EXAMPLE_PARAMETERS:
            self._populating = True
            try:
                self.populate_gui_from_parameters(EXAMPLE_PARAMETERS[example_name])
            finally:
                self._populating = False
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
        default_dir = str(Path.cwd())
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

        # Write to file
        with open(file_path, 'w') as f:
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
        lines.append(f"    'filename slug': '{gui_params['filename_slug']}',")
        lines.append("")

        # Number of runs
        lines.append(f"    'number of runs': {gui_params['number_of_runs']},")
        lines.append("")

        # Units
        lines.append("    ### Units ###")
        lines.append(f"    'length unit': '{gui_params['length_unit']}',")
        lines.append(f"    'time unit': '{gui_params['time_unit']}',")
        lines.append("")

        # Diffusion parameters
        lines.append("    ### Nominal diffusion and lifetime parameters ###")
        if gui_params['diffusion_type'] == 'length':
            lines.append(f"    'nominal diffusion length': {gui_params['diffusion_length']},")
        else:
            lines.append(f"    'nominal diffusion coefficient': {gui_params['diffusion_coefficient']},")
            lines.append(f"    'nominal lifetime (tau)': {gui_params['lifetime']},")
        lines.append("")

        # Initial profile
        lines.append("    ### Initial profile parameters ###")
        if gui_params['profile_width_type'] == 'fwhm':
            lines.append(f"    'FWHM_0': {gui_params['profile_width_value']},")
        else:
            lines.append(f"    'sigma_0': {gui_params['profile_width_value']},")
        lines.append(f"    'amplitude_0': {gui_params['amplitude_0']},")
        lines.append(f"    'mean_0': {gui_params['mean_0']},")
        lines.append("")

        # Noise
        lines.append("    ### Noise parameter ###")
        if gui_params['noise_type'] == 'fixed':
            lines.append(f"    'noise value': {gui_params['noise_value']},")
        else:
            lines.append(f"    'estimate noise from data': '{gui_params['noise_data_file']}',")
        lines.append("")

        # Spatial axis
        lines.append("    ### Spatial axis parameters ###")
        lines.append(f"    'spatial width': {gui_params['spatial_width']},")
        lines.append(f"    'pixel width': {gui_params['pixel_width']},")
        lines.append("")

        # Time axis
        lines.append("    ### Time axis parameters ###")
        if gui_params['time_type'] == 'range':
            lines.append(f"    'time range': [{gui_params['time_start']}, {gui_params['time_stop']}, {gui_params['time_steps']}],")
        else:
            lines.append(f"    'time series': [{gui_params['time_series']}],")
        lines.append("")

        # Proximity
        lines.append("    ### Diffusion coefficient proximity threshold ###")
        lines.append(f"    'proximity level': {gui_params['proximity_level']},")
        lines.append("")

        # Plot parameters
        lines.append("    ### Plot image parameters ###")
        lines.append(f"    'image type': '{gui_params['image_type']}',")
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
        """Populate GUI fields from a parameters dictionary."""
        self._populating = True  # Prevent modification marking

        try:
            # Basic settings
            if 'number of runs' in params:
                self.num_runs_spin.setValue(params['number of runs'])
            if 'filename slug' in params:
                self.filename_slug_input.setText(params['filename slug'])

            # Units
            if 'length unit' in params:
                self.length_unit_combo.setCurrentText(params['length unit'])
            if 'time unit' in params:
                self.time_unit_combo.setCurrentText(params['time unit'])

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

            # Diffusion (mutually exclusive)
            if 'nominal diffusion length' in params:
                self.diffusion_length_radio.setChecked(True)
                self.diffusion_length_input.setText(str(params['nominal diffusion length']))
            elif 'nominal diffusion coefficient' in params and 'nominal lifetime (tau)' in params:
                self.diffusion_coefficient_radio.setChecked(True)
                self.diffusion_coefficient_input.setText(str(params['nominal diffusion coefficient']))
                self.lifetime_input.setText(str(params['nominal lifetime (tau)']))

            # Noise (mutually exclusive)
            if 'noise value' in params:
                self.noise_fixed_radio.setChecked(True)
                self.noise_value_input.setText(str(params['noise value']))
            elif 'estimate noise from data' in params:
                self.noise_estimate_radio.setChecked(True)
                self.noise_file_input.setText(params['estimate noise from data'])

            # Spatial axis
            if 'spatial width' in params:
                self.spatial_width_input.setText(str(params['spatial width']))
            if 'pixel width' in params:
                self.pixel_width_input.setValue(params['pixel width'])

            # Time axis (mutually exclusive)
            if 'time range' in params:
                self.time_range_radio.setChecked(True)
                time_range = params['time range']
                self.time_start_input.setText(str(time_range[0]))
                self.time_stop_input.setText(str(time_range[1]))
                self.time_steps_input.setValue(time_range[2])
            elif 'time series' in params:
                self.time_series_radio.setChecked(True)
                # Convert list to comma-separated string
                time_series_str = ", ".join(str(t) for t in params['time series'])
                self.time_series_input.setPlainText(time_series_str)

            # Proximity
            if 'proximity level' in params:
                self.proximity_spin.setValue(params['proximity level'])

            # Performance
            if 'multiprocessing' in params:
                self.multiprocessing_check.setChecked(params['multiprocessing'])
            if 'retain profile data' in params:
                self.retain_profile_check.setChecked(params['retain profile data'])

            # Image settings (with defaults if not present)
            if 'image type' in params:
                self.image_type_combo.setCurrentText(params['image type'])
            if 'image width' in params:
                self.image_width_spin.setValue(params['image width'])
            if 'image height' in params:
                self.image_height_spin.setValue(params['image height'])
            if 'image dpi' in params:
                self.image_dpi_spin.setValue(int(params['image dpi']))
            if 'image font size' in params:
                self.image_font_size_spin.setValue(int(params['image font size']))
            if 'image tick length' in params:
                self.image_tick_length_spin.setValue(int(params['image tick length']))
            if 'image tick width' in params:
                self.image_tick_width_spin.setValue(int(params['image tick width']))
            if 'image numbins' in params:
                self.image_numbins_spin.setValue(params['image numbins'])
        finally:
            self._populating = False  # Re-enable modification marking

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

        if not recent:
            no_recent = QAction("No recent files", self)
            no_recent.setEnabled(False)
            self.recent_menu.addAction(no_recent)
            return

        for file_path in recent:
            if Path(file_path).exists():
                action = QAction(Path(file_path).name, self)
                action.setStatusTip(file_path)
                action.triggered.connect(lambda checked, f=file_path: self.load_parameters_from_file(f))
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

        # Left side shows loaded data file (if any)
        self.update_status_bar()

    def update_status_bar(self):
        """Update status bar with current loaded data filename."""
        if self.loaded_data_file:
            self.status_bar.showMessage(f"Loaded: {Path(self.loaded_data_file).name}")
        else:
            self.status_bar.showMessage("No data loaded")

    def connect_modification_signals(self):
        """Connect all parameter widgets to modification tracking."""
        # Simulation control
        self.num_runs_spin.valueChanged.connect(self.mark_modified)
        self.filename_slug_input.textChanged.connect(self.mark_modified)

        # Units
        self.length_unit_combo.currentTextChanged.connect(self.mark_modified)
        self.time_unit_combo.currentTextChanged.connect(self.mark_modified)

        # Initial profile parameters
        self.amplitude_input.textChanged.connect(self.mark_modified)
        self.mean_input.textChanged.connect(self.mark_modified)
        self.width_input.textChanged.connect(self.mark_modified)
        self.fwhm_radio.toggled.connect(self.mark_modified)
        self.sigma_radio.toggled.connect(self.mark_modified)

        # Diffusion parameters
        self.diffusion_length_radio.toggled.connect(self.mark_modified)
        self.diffusion_coeff_radio.toggled.connect(self.mark_modified)
        self.diffusion_length_input.textChanged.connect(self.mark_modified)
        self.diffusion_coeff_input.textChanged.connect(self.mark_modified)
        self.lifetime_input.textChanged.connect(self.mark_modified)

        # Noise parameters
        self.noise_fixed_radio.toggled.connect(self.mark_modified)
        self.noise_estimate_radio.toggled.connect(self.mark_modified)
        self.noise_value_input.textChanged.connect(self.mark_modified)
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

        # Analysis
        self.proximity_spin.valueChanged.connect(self.mark_modified)

        # Advanced parameters (Output Settings tab)
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

        # Register conditional diffusion fields
        self.validation_manager.register_field(
            "diffusion_length", self.diffusion_length_input,
            lambda v: validate_positive_float(v, "Diffusion length"),
            condition_group="diffusion_length", condition_category="diffusion"
        )
        self.validation_manager.register_field(
            "diffusion_coeff", self.diffusion_coeff_input,
            lambda v: validate_positive_float(v, "Diffusion coefficient", allow_zero=True),
            condition_group="diffusion_coeff", condition_category="diffusion"
        )
        self.validation_manager.register_field(
            "lifetime", self.lifetime_input,
            lambda v: validate_positive_float(v, "Lifetime", allow_zero=True),
            condition_group="diffusion_coeff", condition_category="diffusion"
        )

        # Register conditional noise fields
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
        self.diffusion_length_radio.toggled.connect(
            lambda checked: self.validation_manager.set_condition_active(
                "diffusion", "diffusion_length" if checked else "diffusion_coeff"
            )
        )
        self.noise_fixed_radio.toggled.connect(
            lambda checked: self.validation_manager.set_condition_active(
                "noise", "noise_fixed" if checked else "noise_estimate"
            )
        )
        self.time_range_radio.toggled.connect(
            lambda checked: self.validation_manager.set_condition_active(
                "time", "time_range" if checked else "time_series"
            )
        )

        # Set initial condition states
        self.validation_manager.set_condition_active("diffusion", "diffusion_length")
        self.validation_manager.set_condition_active("noise", "noise_fixed")
        self.validation_manager.set_condition_active("time", "time_range")

    def _create_option_card(self, layout_type: str = "form") -> tuple:
        """Create a styled option card with specified layout type.

        Args:
            layout_type: "form" for QFormLayout (default), "vbox" for QVBoxLayout

        Returns:
            Tuple of (frame, layout) for adding content.
        """
        frame = QFrame()
        frame.setProperty("class", "option-card")
        if layout_type == "vbox":
            layout = QVBoxLayout(frame)
        else:
            layout = QFormLayout(frame)
        layout.setContentsMargins(12, 8, 12, 8)
        return frame, layout

    def _create_field_with_unit(self, unit_label: str) -> tuple[QWidget, QLineEdit, QLabel]:
        """Create an input field with unit label suffix.

        Returns:
            Tuple of (container_widget, line_edit, unit_label).
        """
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        line_edit = QLineEdit()
        label = QLabel(unit_label)
        layout.addWidget(line_edit)
        layout.addWidget(label)
        return widget, line_edit, label

    def _create_error_label(self) -> QLabel:
        """Create an inline error label for a validated field."""
        error_label = QLabel("")
        error_label.setProperty("class", "validation-error")
        error_label.setWordWrap(True)
        error_label.setMinimumHeight(20)
        error_label.setVisible(False)
        return error_label

    def _on_validity_changed(self, is_valid: bool) -> None:
        """Handle overall form validity change."""
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
            "Analysis Settings",
            "Output Settings"
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

<p><b>License:</b> MIT</p>
"""
        QMessageBox.about(self, "About DICE", about_text)

    def open_documentation(self):
        """Open DICE documentation on GitHub."""
        url = "https://github.com/thiebes/DICE#readme"
        webbrowser.open(url)

    def closeEvent(self, event):
        """Handle window close event to check for unsaved changes."""
        if self.parameters_modified:
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

        event.accept()


def main():
    """Main entry point for the GUI application."""
    app = QApplication(sys.argv)

    # Apply theme with accessibility support
    theme = apply_theme(app)

    window = DiceGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
