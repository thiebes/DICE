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

from dice_gui.tabs.base import (
    create_option_card, create_error_label, create_unit_combo, combo_value,
)
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
    diffusion_layout = QFormLayout(diffusion_group)

    # Formula label
    formula_label = QLabel("L\u00b2 = D \u00d7 \u03c4")
    formula_label.setProperty("class", "calculated-value")
    formula_label.setToolTip(
        "Diffusion length L is related to diffusion coefficient D and "
        "lifetime \u03c4 by L = \u221a(D \u00d7 \u03c4).\n\n"
        "Enter any two of the three parameters and the third will be computed."
    )
    diffusion_layout.addRow("", formula_label)

    # Diffusion length (L)
    l_widget = QWidget()
    l_widget_layout = QHBoxLayout(l_widget)
    l_widget_layout.setContentsMargins(0, 0, 0, 0)
    main_window.diffusion_length_input = QLineEdit()
    main_window.diffusion_length_input.setPlaceholderText("e.g., 1.0")
    main_window.diffusion_length_input.setToolTip(
        "Characteristic diffusion length during carrier lifetime.\n\n"
        "L = \u221a(D \u00d7 \u03c4)\n\n"
        "This is the nominal value used to generate synthetic data.\n"
        "The simulation assesses how accurately this value can be recovered "
        "from noisy measurements.\n\n"
        "Typical values: 10-1000 nm for organic semiconductors, "
        "100-10000 nm for inorganic materials.\n\n"
        "Must be positive."
    )
    main_window.diffusion_length_unit_combo = create_unit_combo('length')
    l_widget_layout.addWidget(main_window.diffusion_length_input)
    l_widget_layout.addWidget(main_window.diffusion_length_unit_combo)
    diffusion_layout.addRow("Diffusion Length (L):", l_widget)
    main_window._error_labels["diffusion_length"] = create_error_label()
    diffusion_layout.addRow("", main_window._error_labels["diffusion_length"])

    # Diffusion coefficient (D)
    d_widget = QWidget()
    d_widget_layout = QHBoxLayout(d_widget)
    d_widget_layout.setContentsMargins(0, 0, 0, 0)
    main_window.diffusion_coeff_input = QLineEdit()
    main_window.diffusion_coeff_input.setPlaceholderText("e.g., 0.5")
    main_window.diffusion_coeff_input.setToolTip(
        "Diffusion coefficient describing spatial spreading rate.\n\n"
        "In 1D Fickian diffusion, variance grows as: \u03c3\u00b2(t) = "
        "\u03c3\u00b2(0) + 2\u00b7D\u00b7t\n\n"
        "Typical values:\n"
        "- Organic semiconductors: 0.001-0.1 cm\u00b2/s\n"
        "- Inorganic semiconductors: 0.1-100 cm\u00b2/s\n\n"
        "Must be non-negative. Zero means no diffusion (only decay)."
    )
    main_window.diffusion_coeff_length_unit_combo = create_unit_combo(
        'length', squared=True
    )
    dc_unit_separator = QLabel(" per ")
    main_window.diffusion_coeff_time_unit_combo = create_unit_combo('time')
    d_widget_layout.addWidget(main_window.diffusion_coeff_input)
    d_widget_layout.addWidget(main_window.diffusion_coeff_length_unit_combo)
    d_widget_layout.addWidget(dc_unit_separator)
    d_widget_layout.addWidget(main_window.diffusion_coeff_time_unit_combo)
    diffusion_layout.addRow("Diffusion Coefficient (D):", d_widget)
    main_window._error_labels["diffusion_coeff"] = create_error_label()
    diffusion_layout.addRow("", main_window._error_labels["diffusion_coeff"])

    # Lifetime (tau)
    tau_widget = QWidget()
    tau_widget_layout = QHBoxLayout(tau_widget)
    tau_widget_layout.setContentsMargins(0, 0, 0, 0)
    main_window.lifetime_input = QLineEdit()
    main_window.lifetime_input.setPlaceholderText("e.g., 2.0")
    main_window.lifetime_input.setToolTip(
        "Excited state lifetime (\u03c4) for exponential decay.\n\n"
        "Intensity decays as: I(t) = I(0) \u00d7 exp(-t/\u03c4)\n\n"
        "Typical values:\n"
        "- Fluorescence: 0.1-10 ns\n"
        "- Phosphorescence: 10-1000 ns\n"
        "- Triplet excitons: 1-1000 ns\n\n"
        "Must be non-negative. Zero means no decay (infinite lifetime)."
    )
    main_window.lifetime_unit_combo = create_unit_combo('time')
    tau_widget_layout.addWidget(main_window.lifetime_input)
    tau_widget_layout.addWidget(main_window.lifetime_unit_combo)
    diffusion_layout.addRow("Lifetime (\u03c4):", tau_widget)
    main_window._error_labels["lifetime"] = create_error_label()
    diffusion_layout.addRow("", main_window._error_labels["lifetime"])

    # Diffusion field linking state
    main_window._updating_diffusion = False
    main_window._diffusion_last_edited = []

    # Connect diffusion field signals
    main_window.diffusion_length_input.textChanged.connect(
        lambda: update_diffusion_fields(main_window, 'L')
    )
    main_window.diffusion_length_unit_combo.currentTextChanged.connect(
        lambda: update_diffusion_fields(main_window, 'L')
    )
    main_window.diffusion_coeff_input.textChanged.connect(
        lambda: update_diffusion_fields(main_window, 'D')
    )
    main_window.diffusion_coeff_length_unit_combo.currentTextChanged.connect(
        lambda: update_diffusion_fields(main_window, 'D')
    )
    main_window.diffusion_coeff_time_unit_combo.currentTextChanged.connect(
        lambda: update_diffusion_fields(main_window, 'D')
    )
    main_window.lifetime_input.textChanged.connect(
        lambda: update_diffusion_fields(main_window, 'tau')
    )
    main_window.lifetime_unit_combo.currentTextChanged.connect(
        lambda: update_diffusion_fields(main_window, 'tau')
    )

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
    profile_layout.addRow("Amplitude\u2080:", main_window.amplitude_input)
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
    main_window.mean_unit_combo = create_unit_combo('length')
    mean_layout.addWidget(main_window.mean_input)
    mean_layout.addWidget(main_window.mean_unit_combo)
    profile_layout.addRow("Mean Position (\u03bc\u2080):", mean_widget)
    main_window._error_labels["mean"] = create_error_label()
    profile_layout.addRow("", main_window._error_labels["mean"])

    # Profile width radio buttons
    main_window.width_button_group = QButtonGroup()
    main_window.fwhm_radio = QRadioButton("FWHM")
    main_window.fwhm_radio.setToolTip(
        "Full Width at Half Maximum of the Gaussian profile.\n\n"
        "FWHM is the width measured at 50% of peak intensity.\n"
        "Common in microscopy and spectroscopy (easier to measure experimentally).\n\n"
        "Relationship: FWHM = 2\u00d7\u221a(2\u00d7ln(2))\u00d7\u03c3 \u2248 2.355\u00d7\u03c3"
    )
    main_window.sigma_radio = QRadioButton("Sigma (\u03c3)")
    main_window.sigma_radio.setToolTip(
        "Standard deviation of the Gaussian profile.\n\n"
        "Sigma is the mathematical parameter in the Gaussian function: "
        "exp(-(x-\u03bc)\u00b2/(2\u00d7\u03c3\u00b2))\n"
        "Preferred for theoretical analysis and diffusion calculations.\n\n"
        "Relationship: \u03c3 = FWHM / 2.355"
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
        "Initial width of the Gaussian profile (FWHM or sigma, depending on "
        "selection above).\n\n"
        "This represents the spatial extent of the initial excitation "
        "(e.g., laser spot size).\n"
        "During diffusion, this width increases over time.\n\n"
        "Typical values: 0.1-10 \u03bcm for confocal microscopy\n"
        "Should be smaller than the spatial window to avoid edge effects.\n\n"
        "Must be positive."
    )
    main_window.width_unit_combo = create_unit_combo('length')
    width_layout_widget.addWidget(main_window.width_input)
    width_layout_widget.addWidget(main_window.width_unit_combo)
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
    main_window.width_unit_combo.currentTextChanged.connect(lambda: update_width_conversion(main_window))

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


def update_diffusion_fields(main_window: "DiceGUI", source: str) -> None:
    """Update the linked diffusion fields (L, D, tau).

    Maintains the constraint L^2 = D * tau. Tracks the last two fields
    edited by the user and computes the third.

    Args:
        source: Which field triggered the update ('L', 'D', or 'tau').
    """
    if main_window._updating_diffusion:
        return
    if getattr(main_window, '_populating', False):
        return

    import math
    from dice.utils.units import (
        convert_diffusion_coefficient, convert_time, convert_length,
    )

    main_window._updating_diffusion = True
    try:
        # Only update tracking when user is directly editing
        # (not during unit sync which shouldn't alter which field is computed)
        if not getattr(main_window, '_syncing_units', False):
            le = main_window._diffusion_last_edited
            if source in le:
                le.remove(source)
            le.append(source)
            if len(le) > 2:
                le.pop(0)

        le = main_window._diffusion_last_edited
        if len(le) < 2:
            return  # need two fields to compute the third

        all_fields = {'L', 'D', 'tau'}
        computed = (all_fields - set(le)).pop()

        # Read L
        l_text = main_window.diffusion_length_input.text().strip()
        l_unit = combo_value(main_window.diffusion_length_unit_combo)
        # Read D
        d_text = main_window.diffusion_coeff_input.text().strip()
        d_l_unit = combo_value(main_window.diffusion_coeff_length_unit_combo)
        d_t_unit = combo_value(main_window.diffusion_coeff_time_unit_combo)
        # Read tau
        tau_text = main_window.lifetime_input.text().strip()
        tau_unit = combo_value(main_window.lifetime_unit_combo)

        if computed == 'tau':
            # Compute tau from L and D
            if not l_text or not d_text:
                main_window.lifetime_input.clear()
                return
            try:
                l_val = float(l_text)
                d_val = float(d_text)
                if l_val <= 0 or d_val <= 0:
                    main_window.lifetime_input.clear()
                    return
            except ValueError:
                main_window.lifetime_input.clear()
                return

            l_si = convert_length(l_val, l_unit, 'meter')
            d_si = convert_diffusion_coefficient(
                d_val, d_l_unit, d_t_unit, 'meter', 'second'
            )
            if d_si <= 0:
                main_window.lifetime_input.clear()
                return

            tau_si = (l_si ** 2) / d_si
            tau_display = convert_time(tau_si, 'second', tau_unit)
            main_window.lifetime_input.setText(f"{tau_display:.4g}")

        elif computed == 'D':
            # Compute D from L and tau
            if not l_text or not tau_text:
                main_window.diffusion_coeff_input.clear()
                return
            try:
                l_val = float(l_text)
                tau_val = float(tau_text)
                if l_val <= 0 or tau_val <= 0:
                    main_window.diffusion_coeff_input.clear()
                    return
            except ValueError:
                main_window.diffusion_coeff_input.clear()
                return

            l_si = convert_length(l_val, l_unit, 'meter')
            tau_si = convert_time(tau_val, tau_unit, 'second')
            if tau_si <= 0:
                main_window.diffusion_coeff_input.clear()
                return

            d_si = (l_si ** 2) / tau_si
            d_display = convert_diffusion_coefficient(
                d_si, 'meter', 'second', d_l_unit, d_t_unit
            )
            main_window.diffusion_coeff_input.setText(f"{d_display:.4g}")

        else:  # computed == 'L'
            # Compute L from D and tau
            if not d_text or not tau_text:
                main_window.diffusion_length_input.clear()
                return
            try:
                d_val = float(d_text)
                tau_val = float(tau_text)
                if d_val < 0 or tau_val < 0:
                    main_window.diffusion_length_input.clear()
                    return
            except ValueError:
                main_window.diffusion_length_input.clear()
                return

            d_si = convert_diffusion_coefficient(
                d_val, d_l_unit, d_t_unit, 'meter', 'second'
            )
            tau_si = convert_time(tau_val, tau_unit, 'second')
            if d_si < 0 or tau_si < 0:
                main_window.diffusion_length_input.clear()
                return

            l_si = math.sqrt(d_si * tau_si)
            l_display = convert_length(l_si, 'meter', l_unit)
            main_window.diffusion_length_input.setText(f"{l_display:.4g}")
    finally:
        main_window._updating_diffusion = False


def update_width_conversion(main_window: "DiceGUI") -> None:
    """Update the FWHM/sigma conversion display."""
    from dice.utils.units import length_abbreviation

    width_text = main_window.width_input.text().strip()
    width_unit = combo_value(main_window.width_unit_combo)
    unit_abbrev = length_abbreviation(width_unit)

    if main_window.fwhm_radio.isChecked():
        result = convert_fwhm_to_sigma(width_text)
        if result.is_valid:
            main_window.width_conversion_label.setText(
                f"Equivalent: \u03c3 = {result.value:.4g} {unit_abbrev}"
            )
        else:
            main_window.width_conversion_label.setText("Equivalent: ---")
    else:
        result = convert_sigma_to_fwhm(width_text)
        if result.is_valid:
            main_window.width_conversion_label.setText(
                f"Equivalent: FWHM = {result.value:.4g} {unit_abbrev}"
            )
        else:
            main_window.width_conversion_label.setText("Equivalent: ---")
