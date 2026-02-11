"""
Tab 4: Analysis Settings

Configuration for accuracy threshold and fit method.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QDoubleSpinBox, QRadioButton
)

from dice_gui.proximity_widget import ProximityTargetWidget

if TYPE_CHECKING:
    from dice_gui.dice_gui import DiceGUI


def create_tab_analysis_settings(main_window: "DiceGUI") -> QWidget:
    """Create Tab 4: Analysis Settings."""
    tab = QWidget()
    layout = QVBoxLayout(tab)

    # Proximity level group
    proximity_group = QGroupBox("Accuracy Threshold")
    proximity_layout = QVBoxLayout(proximity_group)

    proximity_label = QLabel("Proximity Level:")
    main_window.proximity_spin = QDoubleSpinBox()
    main_window.proximity_spin.setMinimum(0.001)
    main_window.proximity_spin.setMaximum(1.0)
    main_window.proximity_spin.setValue(0.10)
    main_window.proximity_spin.setDecimals(3)
    main_window.proximity_spin.setSingleStep(0.01)
    main_window.proximity_spin.setToolTip(
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
    proximity_input_layout.addWidget(main_window.proximity_spin)
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
    main_window.proximity_target = ProximityTargetWidget(proximity=0.10)
    main_window.proximity_spin.valueChanged.connect(lambda: update_proximity_target(main_window))
    proximity_layout.addWidget(main_window.proximity_target)

    # Fit method group
    fit_method_group = QGroupBox("Fit Method")
    fit_method_layout = QVBoxLayout(fit_method_group)

    main_window.plot_method_wls_radio = QRadioButton("Weighted Least Squares (WLS)")
    main_window.plot_method_ols_radio = QRadioButton("Ordinary Least Squares (OLS)")
    main_window.plot_method_wls_radio.setChecked(True)
    main_window.plot_method_wls_radio.setToolTip(
        "Weighted Least Squares regression.\n\n"
        "Weights each data point by the inverse of its variance.\n"
        "Recommended for most applications as it accounts for\n"
        "heteroscedasticity in MSD measurements."
    )
    main_window.plot_method_ols_radio.setToolTip(
        "Ordinary Least Squares regression.\n\n"
        "Treats all data points equally regardless of variance.\n"
        "May be preferred when measurement uncertainties are uniform."
    )
    fit_method_layout.addWidget(main_window.plot_method_wls_radio)
    fit_method_layout.addWidget(main_window.plot_method_ols_radio)
    fit_method_layout.addStretch()

    # Two-column layout
    columns = QHBoxLayout()
    columns.addWidget(proximity_group)
    columns.addWidget(fit_method_group)
    layout.addLayout(columns)
    layout.addStretch()

    return tab


def update_proximity_target(main_window: "DiceGUI") -> None:
    """Update the proximity target widget visualization."""
    main_window.proximity_target.set_proximity(main_window.proximity_spin.value())
