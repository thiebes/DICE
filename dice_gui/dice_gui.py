"""
DICE GUI - Main Application

Graphical user interface for the Diffusion Insight Computation Engine.
"""

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox,
    QComboBox, QRadioButton, QButtonGroup, QGroupBox, QFileDialog,
    QProgressBar, QMessageBox, QTextEdit, QSlider, QFormLayout, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QDoubleValidator, QIntValidator

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
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("DICE - Diffusion Insight Computation Engine")
        self.setGeometry(100, 100, 900, 700)

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

        self.tabs.addTab(self.tab1, "Simulation Setup")
        self.tabs.addTab(self.tab2, "Physical Parameters")
        self.tabs.addTab(self.tab3, "Experimental Conditions")
        self.tabs.addTab(self.tab4, "Analysis Settings")

        main_layout.addWidget(self.tabs)

        # Add bottom control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)

        # Initialize default values
        self.set_default_values()

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
            "micrometer", "nanometer", "millimeter", "centimeter",
            "meter", "angstrom", "picometer"
        ])
        self.length_unit_combo.setCurrentText("micrometer")
        self.length_unit_combo.currentTextChanged.connect(self.update_unit_labels)

        # Time unit
        time_label = QLabel("Time Unit:")
        self.time_unit_combo = QComboBox()
        self.time_unit_combo.addItems([
            "nanosecond", "picosecond", "microsecond", "millisecond",
            "second", "femtosecond", "attosecond"
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
        layout = QFormLayout(tab)
        layout.setSpacing(15)

        # Number of runs
        self.num_runs_spin = QSpinBox()
        self.num_runs_spin.setMinimum(1)
        self.num_runs_spin.setMaximum(1000000)
        self.num_runs_spin.setValue(1000)
        layout.addRow("Number of Runs:", self.num_runs_spin)

        # Filename slug
        self.filename_slug_input = QLineEdit()
        self.filename_slug_input.setText("dice_simulation")
        layout.addRow("Filename Slug:", self.filename_slug_input)

        layout.addRow(QLabel(""))  # Spacer
        info_label = QLabel(
            "Number of Runs: Number of Monte Carlo simulation iterations.\n\n"
            "Filename Slug: Prefix for output files."
        )
        info_label.setWordWrap(True)
        info_label.setProperty("class", "info-text")
        layout.addRow(info_label)

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
        self.diffusion_coeff_radio = QRadioButton("Diffusion Coefficient + Lifetime")
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
        self.diffusion_coeff_label = QLabel("μm²/ns")
        d_layout.addWidget(self.diffusion_coeff_input)
        d_layout.addWidget(self.diffusion_coeff_label)

        tau_widget = QWidget()
        tau_layout = QHBoxLayout(tau_widget)
        tau_layout.setContentsMargins(0, 0, 0, 0)
        self.lifetime_input = QLineEdit()
        self.lifetime_input.setPlaceholderText("e.g., 2.0")
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
        profile_layout.addRow("Amplitude₀:", self.amplitude_input)

        # Mean position
        mean_widget = QWidget()
        mean_layout = QHBoxLayout(mean_widget)
        mean_layout.setContentsMargins(0, 0, 0, 0)
        self.mean_input = QLineEdit()
        self.mean_input.setText("0.0")
        self.mean_label = QLabel("μm")
        mean_layout.addWidget(self.mean_input)
        mean_layout.addWidget(self.mean_label)
        profile_layout.addRow("Mean Position (μ₀):", mean_widget)

        # Profile width radio buttons
        self.width_button_group = QButtonGroup()
        self.fwhm_radio = QRadioButton("FWHM")
        self.sigma_radio = QRadioButton("Sigma (σ)")
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
        self.noise_estimate_radio = QRadioButton("Estimate from Data")
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
        self.noise_browse_button = QPushButton("Browse...")
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
        self.spatial_width_label = QLabel("μm")
        spatial_width_layout.addWidget(self.spatial_width_input)
        spatial_width_layout.addWidget(self.spatial_width_label)
        spatial_layout.addRow("Spatial Width:", spatial_width_widget)

        # Pixel width
        self.pixel_width_input = QSpinBox()
        self.pixel_width_input.setMinimum(1)
        self.pixel_width_input.setMaximum(100000)
        self.pixel_width_input.setValue(100)
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
        self.time_series_radio = QRadioButton("Time Series")
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
        self.time_start_label = QLabel("ns")
        start_layout.addWidget(self.time_start_input)
        start_layout.addWidget(self.time_start_label)

        stop_widget = QWidget()
        stop_layout = QHBoxLayout(stop_widget)
        stop_layout.setContentsMargins(0, 0, 0, 0)
        self.time_stop_input = QLineEdit()
        self.time_stop_input.setPlaceholderText("e.g., 10.0")
        self.time_stop_label = QLabel("ns")
        stop_layout.addWidget(self.time_stop_input)
        stop_layout.addWidget(self.time_stop_label)

        self.time_steps_input = QSpinBox()
        self.time_steps_input.setMinimum(2)
        self.time_steps_input.setMaximum(10000)
        self.time_steps_input.setValue(10)

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

        label_widget = QWidget()
        label_layout = QHBoxLayout(label_widget)
        label_layout.setContentsMargins(0, 0, 0, 0)
        proximity_label = QLabel("Proximity Level:")
        self.proximity_display = QLabel("0.10 (±10%)")
        self.proximity_display.setObjectName("proximity-display")
        label_layout.addWidget(proximity_label)
        label_layout.addWidget(self.proximity_display)
        label_layout.addStretch()

        self.proximity_slider = QSlider(Qt.Orientation.Horizontal)
        self.proximity_slider.setMinimum(1)
        self.proximity_slider.setMaximum(50)
        self.proximity_slider.setValue(10)
        self.proximity_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.proximity_slider.setTickInterval(5)
        self.proximity_slider.valueChanged.connect(self.update_proximity_display)

        proximity_layout.addWidget(label_widget)
        proximity_layout.addWidget(self.proximity_slider)

        description = QLabel(
            "The proximity level determines the threshold for accuracy analysis.\n"
            "For example, 0.10 means estimates within ±10% of the nominal value\n"
            "are considered accurate."
        )
        description.setWordWrap(True)
        description.setProperty("class", "info-text")
        proximity_layout.addWidget(description)

        layout.addWidget(proximity_group)
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

    def update_proximity_display(self):
        """Update proximity level display."""
        value = self.proximity_slider.value() / 100.0
        percentage = int(value * 100)
        self.proximity_display.setText(f"{value:.2f} (±{percentage}%)")

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
            'proximity_level': self.proximity_slider.value() / 100.0,
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
