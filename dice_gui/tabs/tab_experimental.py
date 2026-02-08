"""
Tab 3: Experimental Conditions

Configuration for noise, spatial domain, and temporal domain parameters.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLineEdit, QLabel, QRadioButton, QButtonGroup,
    QScrollArea, QPushButton, QSpinBox, QTextEdit, QFileDialog
)

from dice_gui.tabs.base import create_option_card, create_error_label, create_unit_combo
from dice_gui.validators import calculate_pixel_size

if TYPE_CHECKING:
    from dice_gui.dice_gui import DiceGUI


def create_tab_experimental_conditions(main_window: "DiceGUI") -> QWidget:
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
    main_window.noise_button_group = QButtonGroup()
    main_window.noise_fixed_radio = QRadioButton("Fixed Noise Value")
    main_window.noise_fixed_radio.setToolTip(
        "Specify noise level directly as standard deviation.\n\n"
        "Use this when you know the noise level from calibration or previous measurements.\n"
        "Noise is added as Gaussian white noise with the specified standard deviation.\n\n"
        "For normalized amplitude of 1.0, CNR = 1/noise_value"
    )
    main_window.noise_estimate_radio = QRadioButton("Estimate from Data")
    main_window.noise_estimate_radio.setToolTip(
        "Estimate noise level from experimental profile data using FFT method.\n\n"
        "Load a CSV file containing an experimental profile.\n"
        "DICE will analyze high-frequency components to estimate background noise.\n\n"
        "Useful when noise level is unknown but experimental data is available."
    )
    main_window.noise_button_group.addButton(main_window.noise_fixed_radio, 0)
    main_window.noise_button_group.addButton(main_window.noise_estimate_radio, 1)
    main_window.noise_fixed_radio.setChecked(True)

    noise_layout.addWidget(main_window.noise_fixed_radio)

    # Fixed noise input
    fixed_container, fixed_layout = create_option_card()
    main_window.noise_value_input = QLineEdit()
    main_window.noise_value_input.setPlaceholderText("e.g., 0.01")
    main_window.noise_value_input.setToolTip(
        "Standard deviation of Gaussian white noise added to profiles.\n\n"
        "This value is constant across all pixels and time points.\n"
        "For amplitude=1.0, a noise value of 0.01 gives CNR=100, 0.1 gives CNR=10.\n\n"
        "Typical values: 0.001-0.1 (0.1%-10% of signal amplitude)\n"
        "Higher noise makes diffusion coefficient estimation more difficult.\n\n"
        "Must be non-negative. Zero means no noise (perfect measurements)."
    )
    fixed_layout.addRow("Noise σ:", main_window.noise_value_input)
    main_window._error_labels["noise_value"] = create_error_label()
    fixed_layout.addRow("", main_window._error_labels["noise_value"])
    noise_layout.addWidget(fixed_container)

    noise_layout.addWidget(main_window.noise_estimate_radio)

    # Estimate from data
    estimate_container, estimate_layout = create_option_card()

    file_widget = QWidget()
    file_widget_layout = QHBoxLayout(file_widget)
    file_widget_layout.setContentsMargins(0, 0, 0, 0)
    main_window.noise_file_input = QLineEdit()
    main_window.noise_file_input.setPlaceholderText("Path to CSV file...")
    main_window.noise_file_input.setToolTip(
        "Path to CSV file containing experimental profile data for noise estimation.\n\n"
        "File should contain spatial profile data with background noise.\n"
        "DICE will use FFT analysis to separate signal from noise components.\n\n"
        "Click Browse to select file."
    )
    main_window.noise_browse_button = QPushButton("Browse...")
    main_window.noise_browse_button.setToolTip("Select CSV file containing experimental profile data")
    main_window.noise_browse_button.clicked.connect(lambda: browse_noise_file(main_window))
    file_widget_layout.addWidget(main_window.noise_file_input)
    file_widget_layout.addWidget(main_window.noise_browse_button)
    estimate_layout.addRow("File:", file_widget)
    main_window._error_labels["noise_file"] = create_error_label()
    estimate_layout.addRow("", main_window._error_labels["noise_file"])

    main_window.noise_cnr_label = QLabel("Estimated CNR: ---")
    main_window.noise_cnr_label.setProperty("class", "calculated-value")
    estimate_layout.addRow("", main_window.noise_cnr_label)
    noise_layout.addWidget(estimate_container)

    # Connect radio buttons
    main_window.noise_fixed_radio.toggled.connect(lambda checked: toggle_noise_inputs(main_window, checked))

    # Spatial domain group
    spatial_group = QGroupBox("Spatial Domain")
    spatial_layout = QFormLayout(spatial_group)

    # Spatial width
    spatial_width_widget = QWidget()
    spatial_width_layout = QHBoxLayout(spatial_width_widget)
    spatial_width_layout.setContentsMargins(0, 0, 0, 0)
    main_window.spatial_width_input = QLineEdit()
    main_window.spatial_width_input.setPlaceholderText("e.g., 10.0")
    main_window.spatial_width_input.setToolTip(
        "Total spatial width of the observation window.\n\n"
        "This defines the x-axis range from -width/2 to +width/2.\n"
        "Should be large enough to contain the spreading profile without edge truncation.\n\n"
        "Rule of thumb: Make this 3-5 times the final profile width.\n"
        "For diffusion length L and max time t_max: width ≈ 5*sqrt(sigma_0² + 2*D*t_max)\n\n"
        "Typical values: 5-50 μm for microscopy experiments"
    )
    main_window.spatial_width_unit_combo = create_unit_combo('length')
    spatial_width_layout.addWidget(main_window.spatial_width_input)
    spatial_width_layout.addWidget(main_window.spatial_width_unit_combo)
    spatial_layout.addRow("Spatial Width:", spatial_width_widget)
    main_window._error_labels["spatial_width"] = create_error_label()
    spatial_layout.addRow("", main_window._error_labels["spatial_width"])

    # Pixel width
    main_window.pixel_width_input = QSpinBox()
    main_window.pixel_width_input.setMinimum(1)
    main_window.pixel_width_input.setMaximum(100000)
    main_window.pixel_width_input.setValue(100)
    main_window.pixel_width_input.setToolTip(
        "Number of pixels (spatial sampling points) across the profile.\n\n"
        "Higher values provide better spatial resolution but increase computation time.\n\n"
        "Rule of thumb: At least 10-20 pixels per profile FWHM for accurate Gaussian fitting.\n"
        "For initial FWHM=1 μm and width=10 μm: 100 pixels gives 0.1 μm/pixel resolution.\n\n"
        "Typical values: 50-500 pixels\n"
        "Minimum practical: ~20-30 pixels"
    )
    spatial_layout.addRow("Number of Pixels:", main_window.pixel_width_input)

    # Calculated pixel size
    main_window.pixel_size_label = QLabel("Pixel Size: ---")
    main_window.pixel_size_label.setProperty("class", "calculated-value")
    spatial_layout.addRow("", main_window.pixel_size_label)

    # Connect for calculation
    main_window.spatial_width_input.textChanged.connect(lambda: update_pixel_size(main_window))
    main_window.pixel_width_input.valueChanged.connect(lambda: update_pixel_size(main_window))
    main_window.spatial_width_unit_combo.currentTextChanged.connect(lambda: update_pixel_size(main_window))

    # Temporal domain group
    temporal_group = QGroupBox("Temporal Domain")
    temporal_layout = QVBoxLayout(temporal_group)

    # Radio buttons for time type
    main_window.time_button_group = QButtonGroup()
    main_window.time_range_radio = QRadioButton("Time Range")
    main_window.time_range_radio.setToolTip(
        "Define time points as evenly-spaced range.\n\n"
        "Generates linear time series: linspace(start, stop, steps)\n"
        "Convenient for uniform temporal sampling.\n\n"
        "Example: start=0, stop=10, steps=11 gives [0, 1, 2, ..., 10]"
    )
    main_window.time_series_radio = QRadioButton("Time Series")
    main_window.time_series_radio.setToolTip(
        "Specify arbitrary time points as comma-separated list.\n\n"
        "Allows non-uniform sampling (e.g., logarithmic spacing).\n"
        "Useful for matching experimental time delays.\n\n"
        "Example: 0.1, 0.5, 1, 2, 5, 10, 20, 50"
    )
    main_window.time_button_group.addButton(main_window.time_range_radio, 0)
    main_window.time_button_group.addButton(main_window.time_series_radio, 1)
    main_window.time_range_radio.setChecked(True)

    temporal_layout.addWidget(main_window.time_range_radio)

    # Time range inputs
    range_container, range_layout = create_option_card()

    start_widget = QWidget()
    start_widget_layout = QHBoxLayout(start_widget)
    start_widget_layout.setContentsMargins(0, 0, 0, 0)
    main_window.time_start_input = QLineEdit()
    main_window.time_start_input.setPlaceholderText("e.g., 0.0")
    main_window.time_start_input.setToolTip(
        "First time point for profile measurements.\n\n"
        "Often set to 0 (initial excitation) but can be non-zero.\n"
        "For non-zero start, initial profile still has width specified in Physical Parameters.\n\n"
        "Must be less than stop time."
    )
    main_window.time_start_unit_combo = create_unit_combo('time')
    start_widget_layout.addWidget(main_window.time_start_input)
    start_widget_layout.addWidget(main_window.time_start_unit_combo)

    stop_widget = QWidget()
    stop_widget_layout = QHBoxLayout(stop_widget)
    stop_widget_layout.setContentsMargins(0, 0, 0, 0)
    main_window.time_stop_input = QLineEdit()
    main_window.time_stop_input.setPlaceholderText("e.g., 10.0")
    main_window.time_stop_input.setToolTip(
        "Final time point for profile measurements.\n\n"
        "Should be long enough to observe significant diffusion but not so long that signal decays to noise.\n\n"
        "Rule of thumb: For lifetime tau, useful range is ~0.1*tau to ~2*tau\n"
        "For diffusion, need enough time for measurable width increase (delta_sigma² > noise sensitivity)\n\n"
        "Must be greater than start time."
    )
    main_window.time_stop_unit_combo = create_unit_combo('time')
    stop_widget_layout.addWidget(main_window.time_stop_input)
    stop_widget_layout.addWidget(main_window.time_stop_unit_combo)

    main_window.time_steps_input = QSpinBox()
    main_window.time_steps_input.setMinimum(2)
    main_window.time_steps_input.setMaximum(10000)
    main_window.time_steps_input.setValue(10)
    main_window.time_steps_input.setToolTip(
        "Number of time points in the range (including start and stop).\n\n"
        "More time points improve linear regression fit but increase computation time.\n\n"
        "Rule of thumb: At least 5-10 points for reliable linear fit.\n"
        "Typical values: 10-50 time points\n\n"
        "Minimum: 2 (though 3+ strongly recommended for meaningful statistics)"
    )

    range_layout.addRow("Start:", start_widget)
    main_window._error_labels["time_start"] = create_error_label()
    range_layout.addRow("", main_window._error_labels["time_start"])

    range_layout.addRow("Stop:", stop_widget)
    main_window._error_labels["time_stop"] = create_error_label()
    range_layout.addRow("", main_window._error_labels["time_stop"])

    range_layout.addRow("Steps:", main_window.time_steps_input)
    temporal_layout.addWidget(range_container)

    temporal_layout.addWidget(main_window.time_series_radio)

    # Time series input
    series_container, series_layout = create_option_card("vbox")
    series_label = QLabel("Comma-separated time values:")
    main_window.time_series_input = QTextEdit()
    main_window.time_series_input.setPlaceholderText("e.g., 0.1, 0.5, 1.0, 2.0, 5.0")
    main_window.time_series_input.setMaximumHeight(80)
    main_window.time_series_input.setToolTip(
        "Arbitrary time points as comma-separated values.\n\n"
        "Allows custom temporal sampling to match experimental conditions.\n"
        "Useful for logarithmic spacing or irregular time delays.\n\n"
        "Example: 0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50\n\n"
        "Must have at least 2 time points for linear regression.\n"
        "Time points should be in ascending order (though not strictly required)."
    )
    series_layout.addWidget(series_label)
    series_layout.addWidget(main_window.time_series_input)
    main_window._error_labels["time_series"] = create_error_label()
    series_layout.addWidget(main_window._error_labels["time_series"])
    temporal_layout.addWidget(series_container)

    # Connect radio buttons
    main_window.time_range_radio.toggled.connect(lambda checked: toggle_time_inputs(main_window, checked))

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


def toggle_noise_inputs(main_window: "DiceGUI", checked: bool) -> None:
    """Enable/disable noise input fields based on radio selection."""
    use_fixed = main_window.noise_fixed_radio.isChecked()

    main_window.noise_value_input.setEnabled(use_fixed)
    main_window.noise_file_input.setEnabled(not use_fixed)
    main_window.noise_browse_button.setEnabled(not use_fixed)

    # Clear values when disabled
    if use_fixed:
        main_window.noise_file_input.clear()
        from dice_gui.validation_manager import clear_validation_style
        clear_validation_style(main_window.noise_file_input)
    else:
        main_window.noise_value_input.clear()
        from dice_gui.validation_manager import clear_validation_style
        clear_validation_style(main_window.noise_value_input)


def toggle_time_inputs(main_window: "DiceGUI", checked: bool) -> None:
    """Enable/disable time input fields based on radio selection."""
    use_range = main_window.time_range_radio.isChecked()

    main_window.time_start_input.setEnabled(use_range)
    main_window.time_stop_input.setEnabled(use_range)
    main_window.time_steps_input.setEnabled(use_range)
    main_window.time_series_input.setEnabled(not use_range)

    # Clear values when disabled
    if use_range:
        main_window.time_series_input.clear()
        from dice_gui.validation_manager import clear_validation_style
        clear_validation_style(main_window.time_series_input)
    else:
        main_window.time_start_input.clear()
        main_window.time_stop_input.clear()
        from dice_gui.validation_manager import clear_validation_style
        clear_validation_style(main_window.time_start_input)
        clear_validation_style(main_window.time_stop_input)


def update_pixel_size(main_window: "DiceGUI") -> None:
    """Update the calculated pixel size display."""
    from dice.utils.units import length_abbreviation

    spatial_text = main_window.spatial_width_input.text().strip()
    pixel_count = main_window.pixel_width_input.value()

    result = calculate_pixel_size(spatial_text, pixel_count)
    if result.is_valid:
        width_unit = main_window.spatial_width_unit_combo.currentText()
        unit_abbrev = length_abbreviation(width_unit)
        main_window.pixel_size_label.setText(f"Pixel Size: {result.value:.4g} {unit_abbrev}/pixel")
    else:
        main_window.pixel_size_label.setText("Pixel Size: ---")


def browse_noise_file(main_window: "DiceGUI") -> None:
    """Open file dialog to select noise data file."""
    file_path, _ = QFileDialog.getOpenFileName(
        main_window,
        "Select Noise Data File",
        "",
        "CSV Files (*.csv);;All Files (*)"
    )
    if file_path:
        main_window.noise_file_input.setText(file_path)
