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
    QCheckBox, QStatusBar
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


class SimulationThread(QThread):
    """Thread for running simulations without blocking the GUI."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, interface: DiceInterface, parameters: dict):
        super().__init__()
        self.interface = interface
        self.parameters = parameters

    def run(self):
        """Run the simulation in a separate thread."""
        try:
            self.progress.emit("Starting simulation...")
            result = self.interface.run_simulation(self.parameters)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


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

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("DICE - Diffusion Insight Computation Engine")
        self.setGeometry(100, 100, 900, 1050)

        # Create menu bar
        self.create_menu_bar()

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Add header
        header = self.create_header()
        main_layout.addWidget(header)

        # Add unit selection panel
        unit_panel = self.create_unit_panel()
        main_layout.addWidget(unit_panel)

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

        main_layout.addWidget(self.tabs)

        # Add bottom control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)

        # Create status bar
        self.create_status_bar()

        # Initialize default values
        self.set_default_values()

        # Connect modification tracking signals
        self.connect_modification_signals()

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

        layout.addWidget(basic_group)

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

        layout.addWidget(performance_group)

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
        length_container = QWidget()
        length_layout = QHBoxLayout(length_container)
        length_layout.setContentsMargins(30, 0, 0, 0)
        self.diffusion_length_input = QLineEdit()
        self.diffusion_length_input.setPlaceholderText("e.g., 1.0")
        self.diffusion_length_input.setToolTip(
            "Characteristic diffusion length during carrier lifetime.\n\n"
            "This is the nominal value used to generate synthetic data.\n"
            "The simulation assesses how accurately this value can be recovered from noisy measurements.\n\n"
            "Must be positive. Units set by length unit selector above."
        )
        self.diffusion_length_label = QLabel("μm")
        length_layout.addWidget(QLabel("Diffusion Length:"))
        length_layout.addWidget(self.diffusion_length_input)
        length_layout.addWidget(self.diffusion_length_label)
        length_layout.addStretch()
        diffusion_layout.addWidget(length_container)

        diffusion_layout.addWidget(self.diffusion_coeff_radio)

        # Diffusion coefficient + lifetime inputs
        coeff_container = QWidget()
        coeff_layout = QFormLayout(coeff_container)
        coeff_layout.setContentsMargins(30, 0, 0, 0)

        d_widget = QWidget()
        d_layout = QHBoxLayout(d_widget)
        d_layout.setContentsMargins(0, 0, 0, 0)
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
        d_layout.addWidget(self.diffusion_coeff_input)
        d_layout.addWidget(self.diffusion_coeff_label)

        tau_widget = QWidget()
        tau_layout = QHBoxLayout(tau_widget)
        tau_layout.setContentsMargins(0, 0, 0, 0)
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
        tau_layout.addWidget(self.lifetime_input)
        tau_layout.addWidget(self.lifetime_label)

        coeff_layout.addRow("Diffusion Coefficient (D):", d_widget)
        coeff_layout.addRow("Lifetime (τ):", tau_widget)

        # Calculated diffusion length display
        self.calc_length_label = QLabel("Diffusion Length: ---")
        self.calc_length_label.setProperty("class", "calculated-value")
        coeff_layout.addRow("", self.calc_length_label)

        diffusion_layout.addWidget(coeff_container)

        # Connect radio buttons to enable/disable fields
        self.diffusion_length_radio.toggled.connect(self.toggle_diffusion_inputs)
        self.diffusion_coeff_input.textChanged.connect(self.update_calculated_length)
        self.lifetime_input.textChanged.connect(self.update_calculated_length)

        layout.addWidget(diffusion_group)

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

        # Conversion display
        self.width_conversion_label = QLabel("Equivalent: ---")
        self.width_conversion_label.setProperty("class", "calculated-value")
        profile_layout.addRow("", self.width_conversion_label)

        # Connect width inputs
        self.fwhm_radio.toggled.connect(self.update_width_conversion)
        self.sigma_radio.toggled.connect(self.update_width_conversion)
        self.width_input.textChanged.connect(self.update_width_conversion)

        layout.addWidget(profile_group)
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
        fixed_container = QWidget()
        fixed_layout = QHBoxLayout(fixed_container)
        fixed_layout.setContentsMargins(30, 0, 0, 0)
        fixed_layout.addWidget(QLabel("Noise σ:"))
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
        fixed_layout.addWidget(self.noise_value_input)
        fixed_layout.addStretch()
        noise_layout.addWidget(fixed_container)

        noise_layout.addWidget(self.noise_estimate_radio)

        # Estimate from data
        estimate_container = QWidget()
        estimate_layout = QHBoxLayout(estimate_container)
        estimate_layout.setContentsMargins(30, 0, 0, 0)
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
        self.noise_cnr_label = QLabel("Estimated CNR: ---")
        self.noise_cnr_label.setProperty("class", "calculated-value")
        estimate_layout.addWidget(self.noise_file_input)
        estimate_layout.addWidget(self.noise_browse_button)
        noise_layout.addWidget(estimate_container)
        noise_layout.addWidget(self.noise_cnr_label)

        # Connect radio buttons
        self.noise_fixed_radio.toggled.connect(self.toggle_noise_inputs)

        layout.addWidget(noise_group)

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

        layout.addWidget(spatial_group)

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
        range_container = QWidget()
        range_layout = QFormLayout(range_container)
        range_layout.setContentsMargins(30, 0, 0, 0)

        start_widget = QWidget()
        start_layout = QHBoxLayout(start_widget)
        start_layout.setContentsMargins(0, 0, 0, 0)
        self.time_start_input = QLineEdit()
        self.time_start_input.setPlaceholderText("e.g., 0.0")
        self.time_start_input.setToolTip(
            "First time point for profile measurements.\n\n"
            "Often set to 0 (initial excitation) but can be non-zero.\n"
            "For non-zero start, initial profile still has width specified in Physical Parameters.\n\n"
            "Must be less than stop time."
        )
        self.time_start_label = QLabel("ns")
        start_layout.addWidget(self.time_start_input)
        start_layout.addWidget(self.time_start_label)

        stop_widget = QWidget()
        stop_layout = QHBoxLayout(stop_widget)
        stop_layout.setContentsMargins(0, 0, 0, 0)
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
        stop_layout.addWidget(self.time_stop_input)
        stop_layout.addWidget(self.time_stop_label)

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
        range_layout.addRow("Stop:", stop_widget)
        range_layout.addRow("Steps:", self.time_steps_input)
        temporal_layout.addWidget(range_container)

        temporal_layout.addWidget(self.time_series_radio)

        # Time series input
        series_container = QWidget()
        series_layout = QVBoxLayout(series_container)
        series_layout.setContentsMargins(30, 0, 0, 0)
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
        temporal_layout.addWidget(series_container)

        # Connect radio buttons
        self.time_range_radio.toggled.connect(self.toggle_time_inputs)

        layout.addWidget(temporal_group)
        layout.addStretch()

        scroll.setWidget(scroll_content)
        tab_layout = QVBoxLayout(tab)
        tab_layout.addWidget(scroll)

        return tab

    def create_tab4_analysis_settings(self) -> QWidget:
        """Create Tab 4: Analysis Settings."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Proximity level
        proximity_group = QGroupBox("Analysis Parameters")
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

        layout.addWidget(proximity_group)
        layout.addStretch()

        return tab

    def create_tab5_output_settings(self) -> QWidget:
        """Create Tab 5: Output Settings."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Plot/Image settings group
        plot_group = QGroupBox("Plot Settings")
        plot_layout = QFormLayout(plot_group)

        # Image type
        self.image_type_combo = QComboBox()
        self.image_type_combo.addItems(["png", "jpg", "svg", "tif"])
        self.image_type_combo.setCurrentText("png")
        self.image_type_combo.setToolTip(
            "File format for saved plots.\n\n"
            "- PNG: Best for general use, lossless compression, good quality (recommended)\n"
            "- JPG: Smaller files but lossy compression, may show artifacts\n"
            "- SVG: Vector format, scalable without quality loss, ideal for publications\n"
            "- TIF: Uncompressed, largest files, maximum quality\n\n"
            "For publications: PNG or SVG\n"
            "For presentations: PNG\n"
            "For archiving: TIF or SVG"
        )
        plot_layout.addRow("Image Type:", self.image_type_combo)

        # Image width (value + unit)
        width_widget = QWidget()
        width_layout = QHBoxLayout(width_widget)
        width_layout.setContentsMargins(0, 0, 0, 0)
        self.image_width_spin = QDoubleSpinBox()
        self.image_width_spin.setMinimum(0.1)
        self.image_width_spin.setMaximum(100.0)
        self.image_width_spin.setValue(16.0)
        self.image_width_spin.setDecimals(2)
        self.image_width_spin.setToolTip(
            "Physical width of output plot.\n\n"
            "For publications:\n"
            "- Single column: 8-9 cm (3-3.5 in)\n"
            "- Double column: 16-18 cm (6-7 in)\n\n"
            "For presentations: 10-12 in (25-30 cm)\n\n"
            "Larger sizes provide more detail but may not fit journal requirements."
        )
        self.image_width_unit_combo = QComboBox()
        self.image_width_unit_combo.addItems(["cm", "in", "mm"])
        self.image_width_unit_combo.setCurrentText("cm")
        width_layout.addWidget(self.image_width_spin)
        width_layout.addWidget(self.image_width_unit_combo)
        plot_layout.addRow("Image Width:", width_widget)

        # Image height (value + unit)
        height_widget = QWidget()
        height_layout = QHBoxLayout(height_widget)
        height_layout.setContentsMargins(0, 0, 0, 0)
        self.image_height_spin = QDoubleSpinBox()
        self.image_height_spin.setMinimum(0.1)
        self.image_height_spin.setMaximum(100.0)
        self.image_height_spin.setValue(10.0)
        self.image_height_spin.setDecimals(2)
        self.image_height_spin.setToolTip(
            "Physical height of output plot.\n\n"
            "Common aspect ratios:\n"
            "- 16:10 (wide): Good for histograms\n"
            "- 4:3 (standard): Balanced appearance\n"
            "- 1:1 (square): Compact\n\n"
            "Height is typically 60-70% of width for most plots."
        )
        self.image_height_unit_combo = QComboBox()
        self.image_height_unit_combo.addItems(["cm", "in", "mm"])
        self.image_height_unit_combo.setCurrentText("cm")
        height_layout.addWidget(self.image_height_spin)
        height_layout.addWidget(self.image_height_unit_combo)
        plot_layout.addRow("Image Height:", height_widget)

        # Image resolution (value + unit)
        resolution_widget = QWidget()
        resolution_layout = QHBoxLayout(resolution_widget)
        resolution_layout.setContentsMargins(0, 0, 0, 0)
        self.image_dpi_spin = QSpinBox()
        self.image_dpi_spin.setMinimum(50)
        self.image_dpi_spin.setMaximum(1200)
        self.image_dpi_spin.setValue(300)
        self.image_dpi_spin.setToolTip(
            "Resolution in dots per inch (DPI).\n\n"
            "Common standards:\n"
            "- 72-96 DPI: Screen display, presentations\n"
            "- 150 DPI: Draft prints\n"
            "- 300 DPI: Publication quality (most journals require this)\n"
            "- 600 DPI: High-quality prints, posters\n\n"
            "Higher DPI increases file size and generation time.\n"
            "For PNG/JPG/TIF; SVG is resolution-independent."
        )
        self.image_dpi_unit_combo = QComboBox()
        self.image_dpi_unit_combo.addItems(["dpi", "dpcm"])
        self.image_dpi_unit_combo.setCurrentText("dpi")
        resolution_layout.addWidget(self.image_dpi_spin)
        resolution_layout.addWidget(self.image_dpi_unit_combo)
        plot_layout.addRow("Resolution:", resolution_widget)

        # Font size (value + unit)
        font_widget = QWidget()
        font_layout = QHBoxLayout(font_widget)
        font_layout.setContentsMargins(0, 0, 0, 0)
        self.image_font_size_spin = QSpinBox()
        self.image_font_size_spin.setMinimum(4)
        self.image_font_size_spin.setMaximum(72)
        self.image_font_size_spin.setValue(6)
        self.image_font_size_spin.setToolTip(
            "Font size for axis labels, titles, and annotations.\n\n"
            "Publication guidelines:\n"
            "- Minimum: 6-8 pt (must be readable when printed)\n"
            "- Standard: 8-10 pt\n"
            "- Larger: 12-14 pt (for presentations)\n\n"
            "Font size should scale with image dimensions.\n"
            "Points (pt) are standard for print; pixels (px) for screen."
        )
        self.image_font_unit_combo = QComboBox()
        self.image_font_unit_combo.addItems(["pt", "px"])
        self.image_font_unit_combo.setCurrentText("pt")
        font_layout.addWidget(self.image_font_size_spin)
        font_layout.addWidget(self.image_font_unit_combo)
        plot_layout.addRow("Font Size:", font_widget)

        # Tick length (value + unit)
        tick_length_widget = QWidget()
        tick_length_layout = QHBoxLayout(tick_length_widget)
        tick_length_layout.setContentsMargins(0, 0, 0, 0)
        self.image_tick_length_spin = QSpinBox()
        self.image_tick_length_spin.setMinimum(1)
        self.image_tick_length_spin.setMaximum(50)
        self.image_tick_length_spin.setValue(6)
        self.image_tick_length_spin.setToolTip(
            "Length of axis tick marks in points or pixels.\n\n"
            "Typical values:\n"
            "- Short: 3-4 pt (subtle)\n"
            "- Standard: 5-7 pt (recommended)\n"
            "- Long: 8-12 pt (emphasis)\n\n"
            "Should be proportional to plot size and line widths."
        )
        self.image_tick_length_unit_combo = QComboBox()
        self.image_tick_length_unit_combo.addItems(["pt", "px"])
        self.image_tick_length_unit_combo.setCurrentText("pt")
        tick_length_layout.addWidget(self.image_tick_length_spin)
        tick_length_layout.addWidget(self.image_tick_length_unit_combo)
        plot_layout.addRow("Tick Length:", tick_length_widget)

        # Tick width (value + unit)
        tick_width_widget = QWidget()
        tick_width_layout = QHBoxLayout(tick_width_widget)
        tick_width_layout.setContentsMargins(0, 0, 0, 0)
        self.image_tick_width_spin = QSpinBox()
        self.image_tick_width_spin.setMinimum(1)
        self.image_tick_width_spin.setMaximum(20)
        self.image_tick_width_spin.setValue(2)
        self.image_tick_width_spin.setToolTip(
            "Thickness of axis tick marks and plot borders.\n\n"
            "Typical values:\n"
            "- Thin: 0.5-1 pt (delicate)\n"
            "- Standard: 1-2 pt (recommended)\n"
            "- Thick: 2-4 pt (bold)\n\n"
            "Should match axis line width for consistency."
        )
        self.image_tick_width_unit_combo = QComboBox()
        self.image_tick_width_unit_combo.addItems(["pt", "px"])
        self.image_tick_width_unit_combo.setCurrentText("pt")
        tick_width_layout.addWidget(self.image_tick_width_spin)
        tick_width_layout.addWidget(self.image_tick_width_unit_combo)
        plot_layout.addRow("Tick Width:", tick_width_widget)

        # Number of bins (no unit)
        self.image_numbins_spin = QSpinBox()
        self.image_numbins_spin.setMinimum(5)
        self.image_numbins_spin.setMaximum(200)
        self.image_numbins_spin.setValue(35)
        self.image_numbins_spin.setToolTip(
            "Number of bins for accuracy histogram.\n\n"
            "More bins show finer distribution detail but may appear noisy with few data points.\n"
            "Fewer bins smooth the distribution but may hide features.\n\n"
            "Rule of thumb: sqrt(N) to N/10 bins, where N is number of Monte Carlo runs.\n"
            "For 1000 runs: 30-100 bins is reasonable.\n\n"
            "Typical values: 20-50 bins"
        )
        plot_layout.addRow("Histogram Bins:", self.image_numbins_spin)

        # Plot method selection (WLS vs OLS)
        method_widget = QWidget()
        method_layout = QHBoxLayout(method_widget)
        method_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_method_wls_radio = QRadioButton("Weighted Least Squares (WLS)")
        self.plot_method_ols_radio = QRadioButton("Ordinary Least Squares (OLS)")
        self.plot_method_wls_radio.setChecked(True)
        self.plot_method_wls_radio.setToolTip(
            "Use Weighted Least Squares for diffusion coefficient estimation.\n\n"
            "WLS accounts for heteroscedasticity (varying uncertainty across time points).\n"
            "Weights are calculated from Gaussian fit parameter uncertainties.\n\n"
            "Recommended for most cases as it provides more accurate estimates when\n"
            "measurement precision varies with time (e.g., due to intensity decay).\n\n"
            "This affects which data column is plotted in histograms."
        )
        self.plot_method_ols_radio.setToolTip(
            "Use Ordinary Least Squares for diffusion coefficient estimation.\n\n"
            "OLS treats all time points equally regardless of measurement uncertainty.\n"
            "Simpler method, appropriate when all points have similar precision.\n\n"
            "May be less accurate than WLS when signal quality varies with time,\n"
"but easier to interpret and faster to compute.\n\n"
            "This affects which data column is plotted in histograms."
        )
        method_layout.addWidget(self.plot_method_wls_radio)
        method_layout.addWidget(self.plot_method_ols_radio)
        method_layout.addStretch()
        plot_layout.addRow("Fit Method:", method_widget)

        layout.addWidget(plot_group)

        # Plot Actions group
        actions_group = QGroupBox("Plot Actions")
        actions_layout = QHBoxLayout(actions_group)

        self.regenerate_plot_button = QPushButton("Regenerate Plot")
        self.regenerate_plot_button.setToolTip(
            "Regenerate plot using data currently in memory with updated settings.\n\n"
            "Use this to adjust plot appearance (size, resolution, fonts, bins) without\n"
            "re-running the entire simulation.\n\n"
            "Requires data from a completed simulation or loaded CSV file.\n"
            "All plot settings above will be applied to the regenerated plot."
        )
        self.regenerate_plot_button.clicked.connect(self.regenerate_plot)

        self.load_results_button = QPushButton("Load Results")
        self.load_results_button.setToolTip(
            "Load simulation results from a previously saved CSV file.\n\n"
            "Opens a file dialog to select a results CSV file.\n"
            "Generates accuracy histogram plot with current image settings.\n\n"
            "Useful for creating plots with different formatting or proximity levels\n"
            "without re-running time-consuming simulations."
        )
        self.load_results_button.clicked.connect(self.load_and_plot_results)

        actions_layout.addWidget(self.regenerate_plot_button)
        actions_layout.addWidget(self.load_results_button)
        actions_layout.addStretch()

        layout.addWidget(actions_group)

        # Info label
        info_label = QLabel(
            "These settings control the appearance and format of output plots.\n\n"
            "Image Type: File format for saved plots (PNG recommended for most uses).\n\n"
            "Image Width/Height: Physical dimensions of the output image.\n\n"
            "Resolution: Higher DPI values produce sharper images but larger file sizes.\n\n"
            "Font/Tick sizes: Typography units (pt = points, px = pixels)."
        )
        info_label.setWordWrap(True)
        info_label.setProperty("class", "info-text")
        layout.addWidget(info_label)

        layout.addStretch()

        return tab

    def create_control_panel(self) -> QWidget:
        """Create the bottom control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setProperty("class", "status-info")
        layout.addWidget(self.status_label)

        # Buttons
        button_layout = QHBoxLayout()

        self.run_button = QPushButton("Run Simulation")
        self.run_button.setObjectName("run-button")
        self.run_button.clicked.connect(self.run_simulation)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("stop-button")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_simulation)

        button_layout.addStretch()
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addStretch()

        layout.addLayout(button_layout)

        return panel

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
        else:
            self.diffusion_length_input.setEnabled(False)
            self.diffusion_coeff_input.setEnabled(True)
            self.lifetime_input.setEnabled(True)
            self.diffusion_length_input.clear()

    def toggle_noise_inputs(self):
        """Enable/disable noise inputs based on radio selection."""
        if self.noise_fixed_radio.isChecked():
            self.noise_value_input.setEnabled(True)
            self.noise_file_input.setEnabled(False)
            self.noise_browse_button.setEnabled(False)
            self.noise_file_input.clear()
            self.noise_cnr_label.setText("Estimated CNR: ---")
        else:
            self.noise_value_input.setEnabled(False)
            self.noise_file_input.setEnabled(True)
            self.noise_browse_button.setEnabled(True)
            self.noise_value_input.clear()

    def toggle_time_inputs(self):
        """Enable/disable time inputs based on radio selection."""
        if self.time_range_radio.isChecked():
            self.time_start_input.setEnabled(True)
            self.time_stop_input.setEnabled(True)
            self.time_steps_input.setEnabled(True)
            self.time_series_input.setEnabled(False)
            self.time_series_input.clear()
        else:
            self.time_start_input.setEnabled(False)
            self.time_stop_input.setEnabled(False)
            self.time_steps_input.setEnabled(False)
            self.time_series_input.setEnabled(True)
            self.time_start_input.clear()
            self.time_stop_input.clear()

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

        # Convert resolution to dpi (standard unit)
        dpi_value = self.image_dpi_spin.value()
        dpi_unit = self.image_dpi_unit_combo.currentText()
        if dpi_unit == "dpcm":
            dpi = dpi_value * 2.54
        else:  # dpi
            dpi = dpi_value

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

        # Convert resolution to dpi (standard unit)
        dpi_value = self.image_dpi_spin.value()
        dpi_unit = self.image_dpi_unit_combo.currentText()
        if dpi_unit == "dpcm":
            dpi = dpi_value * 2.54
        else:  # dpi
            dpi = dpi_value

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
        # Validate all inputs
        is_valid, error_message = self.validate_all_inputs()
        if not is_valid:
            QMessageBox.critical(self, "Validation Error", error_message)
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

        # Update UI
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.status_label.setText("Running simulation...")

        # Create and start simulation thread
        self.simulation_thread = SimulationThread(self.interface, dice_params)
        self.simulation_thread.progress.connect(self.update_progress)
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
