"""
Tab 2: Physical Parameters

Configuration for diffusion parameters and initial profile settings.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLineEdit, QLabel, QRadioButton, QButtonGroup,
    QScrollArea
)

from dice_gui.tabs.base import create_option_card, create_error_label
from dice_gui.validators import (
    validate_positive_float, validate_float,
    convert_fwhm_to_sigma, convert_sigma_to_fwhm,
    calculate_diffusion_length
)

if TYPE_CHECKING:
    from dice_gui.dice_gui import DiceGUI


def create_tab_physical_parameters(main_window: "DiceGUI") -> QWidget:
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
    main_window.diffusion_button_group = QButtonGroup()
    main_window.diffusion_length_radio = QRadioButton("Diffusion Length")
    main_window.diffusion_length_radio.setToolTip(
        "Specify diffusion as a single length parameter.\n\n"
        "Diffusion length L = sqrt(D*tau) is the characteristic distance a carrier diffuses during its lifetime.\n"
        "Use this when you know the overall transport distance but not individual D and tau values.\n\n"
        "Typical values: 10-1000 nm for organic semiconductors, 100-10000 nm for inorganic materials."
    )
    main_window.diffusion_coeff_radio = QRadioButton("Diffusion Coefficient + Lifetime")
    main_window.diffusion_coeff_radio.setToolTip(
        "Specify diffusion coefficient D and lifetime tau separately.\n\n"
        "Use this when you know both parameters independently from experiments.\n"
        "D controls spatial spreading rate, tau controls temporal decay.\n\n"
        "Typical D: 0.001-1 cm²/s (organics), 1-100 cm²/s (inorganics)\n"
        "Typical tau: 1-1000 ns"
    )
    main_window.diffusion_button_group.addButton(main_window.diffusion_length_radio, 0)
    main_window.diffusion_button_group.addButton(main_window.diffusion_coeff_radio, 1)
    main_window.diffusion_length_radio.setChecked(True)

    diffusion_layout.addWidget(main_window.diffusion_length_radio)

    # Diffusion length input
    length_container, length_layout = create_option_card()

    length_widget = QWidget()
    length_widget_layout = QHBoxLayout(length_widget)
    length_widget_layout.setContentsMargins(0, 0, 0, 0)
    main_window.diffusion_length_input = QLineEdit()
    main_window.diffusion_length_input.setPlaceholderText("e.g., 1.0")
    main_window.diffusion_length_input.setToolTip(
        "Characteristic diffusion length during carrier lifetime.\n\n"
        "This is the nominal value used to generate synthetic data.\n"
        "The simulation assesses how accurately this value can be recovered from noisy measurements.\n\n"
        "Must be positive. Units set by length unit selector above."
    )
    main_window.diffusion_length_label = QLabel("μm")
    length_widget_layout.addWidget(main_window.diffusion_length_input)
    length_widget_layout.addWidget(main_window.diffusion_length_label)

    length_layout.addRow("Diffusion Length:", length_widget)
    main_window._error_labels["diffusion_length"] = create_error_label()
    length_layout.addRow("", main_window._error_labels["diffusion_length"])
    diffusion_layout.addWidget(length_container)

    diffusion_layout.addWidget(main_window.diffusion_coeff_radio)

    # Diffusion coefficient + lifetime inputs
    coeff_container, coeff_layout = create_option_card()

    d_widget = QWidget()
    d_widget_layout = QHBoxLayout(d_widget)
    d_widget_layout.setContentsMargins(0, 0, 0, 0)
    main_window.diffusion_coeff_input = QLineEdit()
    main_window.diffusion_coeff_input.setPlaceholderText("e.g., 0.5")
    main_window.diffusion_coeff_input.setToolTip(
        "Diffusion coefficient describing spatial spreading rate.\n\n"
        "In 1D Fickian diffusion, variance grows as: sigma²(t) = sigma²(0) + 2*D*t\n\n"
        "Typical values:\n"
        "- Organic semiconductors: 0.001-0.1 cm²/s (0.01-10 μm²/ns)\n"
        "- Inorganic semiconductors: 0.1-100 cm²/s (10-10000 μm²/ns)\n\n"
        "Must be non-negative. Zero means no diffusion (only decay)."
    )
    main_window.diffusion_coeff_label = QLabel("μm²/ns")
    d_widget_layout.addWidget(main_window.diffusion_coeff_input)
    d_widget_layout.addWidget(main_window.diffusion_coeff_label)

    tau_widget = QWidget()
    tau_widget_layout = QHBoxLayout(tau_widget)
    tau_widget_layout.setContentsMargins(0, 0, 0, 0)
    main_window.lifetime_input = QLineEdit()
    main_window.lifetime_input.setPlaceholderText("e.g., 2.0")
    main_window.lifetime_input.setToolTip(
        "Excited state lifetime (tau) for exponential decay.\n\n"
        "Intensity decays as: I(t) = I(0) * exp(-t/tau)\n\n"
        "Typical values:\n"
        "- Fluorescence: 0.1-10 ns\n"
        "- Phosphorescence: 10-1000 ns\n"
        "- Triplet excitons: 1-1000 ns\n\n"
        "Must be non-negative. Zero means no decay (infinite lifetime)."
    )
    main_window.lifetime_label = QLabel("ns")
    tau_widget_layout.addWidget(main_window.lifetime_input)
    tau_widget_layout.addWidget(main_window.lifetime_label)

    coeff_layout.addRow("Diffusion Coefficient (D):", d_widget)
    main_window._error_labels["diffusion_coeff"] = create_error_label()
    coeff_layout.addRow("", main_window._error_labels["diffusion_coeff"])

    coeff_layout.addRow("Lifetime (τ):", tau_widget)
    main_window._error_labels["lifetime"] = create_error_label()
    coeff_layout.addRow("", main_window._error_labels["lifetime"])

    # Calculated diffusion length display
    main_window.calc_length_label = QLabel("Diffusion Length: ---")
    main_window.calc_length_label.setProperty("class", "calculated-value")
    coeff_layout.addRow("", main_window.calc_length_label)

    diffusion_layout.addWidget(coeff_container)

    # Connect radio buttons to enable/disable fields
    main_window.diffusion_length_radio.toggled.connect(lambda checked: toggle_diffusion_inputs(main_window, checked))
    main_window.diffusion_coeff_input.textChanged.connect(lambda: update_calculated_length(main_window))
    main_window.lifetime_input.textChanged.connect(lambda: update_calculated_length(main_window))

    # Initial Profile group
    profile_group = QGroupBox("Initial Profile")
    profile_layout = QFormLayout(profile_group)

    # Amplitude
    main_window.amplitude_input = QLineEdit()
    main_window.amplitude_input.setText("1.0")
    main_window.amplitude_input.setToolTip(
        "Initial peak intensity of the Gaussian profile at t=0.\n\n"
        "Typically normalized to 1.0 for convenience.\n"
        "The noise level is specified relative to this amplitude.\n\n"
        "Can be zero or positive. Zero amplitude means no signal (only noise)."
    )
    profile_layout.addRow("Amplitude₀:", main_window.amplitude_input)
    main_window._error_labels["amplitude"] = create_error_label()
    profile_layout.addRow("", main_window._error_labels["amplitude"])

    # Mean position
    mean_widget = QWidget()
    mean_layout = QHBoxLayout(mean_widget)
    mean_layout.setContentsMargins(0, 0, 0, 0)
    main_window.mean_input = QLineEdit()
    main_window.mean_input.setText("0.0")
    main_window.mean_input.setToolTip(
        "Center position of the initial Gaussian profile along the spatial axis.\n\n"
        "Typically set to 0.0 (centered on the spatial window).\n"
        "The profile center does not move during diffusion (only spreads and decays).\n\n"
        "Should be within the spatial width defined in Experimental Conditions."
    )
    main_window.mean_label = QLabel("μm")
    mean_layout.addWidget(main_window.mean_input)
    mean_layout.addWidget(main_window.mean_label)
    profile_layout.addRow("Mean Position (μ₀):", mean_widget)
    main_window._error_labels["mean"] = create_error_label()
    profile_layout.addRow("", main_window._error_labels["mean"])

    # Profile width radio buttons
    main_window.width_button_group = QButtonGroup()
    main_window.fwhm_radio = QRadioButton("FWHM")
    main_window.fwhm_radio.setToolTip(
        "Full Width at Half Maximum of the Gaussian profile.\n\n"
        "FWHM is the width measured at 50% of peak intensity.\n"
        "Common in microscopy and spectroscopy (easier to measure experimentally).\n\n"
        "Relationship: FWHM = 2*sqrt(2*ln(2))*sigma ≈ 2.355*sigma"
    )
    main_window.sigma_radio = QRadioButton("Sigma (σ)")
    main_window.sigma_radio.setToolTip(
        "Standard deviation of the Gaussian profile.\n\n"
        "Sigma is the mathematical parameter in the Gaussian function: exp(-(x-μ)²/(2*sigma²))\n"
        "Preferred for theoretical analysis and diffusion calculations.\n\n"
        "Relationship: sigma = FWHM / 2.355"
    )
    main_window.width_button_group.addButton(main_window.fwhm_radio, 0)
    main_window.width_button_group.addButton(main_window.sigma_radio, 1)
    main_window.fwhm_radio.setChecked(True)

    width_radio_widget = QWidget()
    width_radio_layout = QHBoxLayout(width_radio_widget)
    width_radio_layout.setContentsMargins(0, 0, 0, 0)
    width_radio_layout.addWidget(main_window.fwhm_radio)
    width_radio_layout.addWidget(main_window.sigma_radio)
    width_radio_layout.addStretch()
    profile_layout.addRow("Width Type:", width_radio_widget)

    # Width input
    width_widget = QWidget()
    width_layout_widget = QHBoxLayout(width_widget)
    width_layout_widget.setContentsMargins(0, 0, 0, 0)
    main_window.width_input = QLineEdit()
    main_window.width_input.setPlaceholderText("e.g., 1.0")
    main_window.width_input.setToolTip(
        "Initial width of the Gaussian profile (FWHM or sigma, depending on selection above).\n\n"
        "This represents the spatial extent of the initial excitation (e.g., laser spot size).\n"
        "During diffusion, this width increases over time.\n\n"
        "Typical values: 0.1-10 μm for confocal microscopy\n"
        "Should be smaller than the spatial window to avoid edge effects.\n\n"
        "Must be positive."
    )
    main_window.width_unit_label = QLabel("μm")
    width_layout_widget.addWidget(main_window.width_input)
    width_layout_widget.addWidget(main_window.width_unit_label)
    profile_layout.addRow("Width Value:", width_widget)
    main_window._error_labels["width"] = create_error_label()
    profile_layout.addRow("", main_window._error_labels["width"])

    # Conversion display
    main_window.width_conversion_label = QLabel("Equivalent: ---")
    main_window.width_conversion_label.setProperty("class", "calculated-value")
    profile_layout.addRow("", main_window.width_conversion_label)

    # Connect width inputs
    main_window.fwhm_radio.toggled.connect(lambda: update_width_conversion(main_window))
    main_window.sigma_radio.toggled.connect(lambda: update_width_conversion(main_window))
    main_window.width_input.textChanged.connect(lambda: update_width_conversion(main_window))

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


def toggle_diffusion_inputs(main_window: "DiceGUI", checked: bool) -> None:
    """Enable/disable diffusion input fields based on radio selection."""
    use_length = main_window.diffusion_length_radio.isChecked()

    main_window.diffusion_length_input.setEnabled(use_length)
    main_window.diffusion_coeff_input.setEnabled(not use_length)
    main_window.lifetime_input.setEnabled(not use_length)
    main_window.calc_length_label.setVisible(not use_length)

    # Clear values and validation styling when disabled
    if use_length:
        main_window.diffusion_coeff_input.clear()
        main_window.lifetime_input.clear()
        from dice_gui.validation_manager import clear_validation_style
        clear_validation_style(main_window.diffusion_coeff_input)
        clear_validation_style(main_window.lifetime_input)
    else:
        main_window.diffusion_length_input.clear()
        from dice_gui.validation_manager import clear_validation_style
        clear_validation_style(main_window.diffusion_length_input)


def update_calculated_length(main_window: "DiceGUI") -> None:
    """Update the calculated diffusion length display."""
    d_text = main_window.diffusion_coeff_input.text().strip()
    tau_text = main_window.lifetime_input.text().strip()

    result = calculate_diffusion_length(d_text, tau_text)
    if result.is_valid:
        length_unit = main_window.length_unit_combo.currentText()
        main_window.calc_length_label.setText(f"Diffusion Length: {result.value:.4g} {length_unit}")
    else:
        main_window.calc_length_label.setText("Diffusion Length: ---")


def update_width_conversion(main_window: "DiceGUI") -> None:
    """Update the FWHM/sigma conversion display."""
    width_text = main_window.width_input.text().strip()
    length_unit = main_window.length_unit_combo.currentText()

    if main_window.fwhm_radio.isChecked():
        result = convert_fwhm_to_sigma(width_text)
        if result.is_valid:
            main_window.width_conversion_label.setText(f"Equivalent: σ = {result.value:.4g} {length_unit}")
        else:
            main_window.width_conversion_label.setText("Equivalent: ---")
    else:
        result = convert_sigma_to_fwhm(width_text)
        if result.is_valid:
            main_window.width_conversion_label.setText(f"Equivalent: FWHM = {result.value:.4g} {length_unit}")
        else:
            main_window.width_conversion_label.setText("Equivalent: ---")
