"""
Tab 4: Analysis Settings

Fit method selection and post-hoc precision analysis.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QDoubleSpinBox, QRadioButton
)

from dice_gui.tabs.base import create_scrollable_tab
from dice_gui.accessibility import add_keyboard_shortcut_to_label, set_tab_order

from dice_gui.proximity_widget import ProximityHistogramWidget

if TYPE_CHECKING:
    from dice_gui.dice_gui import DiceGUI


def create_tab_analysis_settings(main_window: "DiceGUI") -> QWidget:
    """Create Tab 4: Analysis Settings."""
    tab, layout = create_scrollable_tab()

    # Fit method group
    fit_method_group = QGroupBox("Fit Method")
    fit_method_layout = QVBoxLayout(fit_method_group)

    main_window.plot_method_wls_radio = QRadioButton("&Weighted Least Squares (WLS)")
    main_window.plot_method_ols_radio = QRadioButton("&Ordinary Least Squares (OLS)")
    main_window.plot_method_wls_radio.setChecked(True)
    main_window.plot_method_wls_radio.setToolTip(
        "Weighted Least Squares regression.\n\n"
        "Weights each data point by the inverse of its squared relative error.\n"
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

    # The summary and threshold fraction describe the selected method, so
    # recompute them when it changes (no-op until results exist).
    main_window.plot_method_wls_radio.toggled.connect(
        lambda checked: main_window._populate_precision_from_results()
    )

    # Precision summary group (read-only, populated after simulation)
    precision_group = QGroupBox("Precision Summary")
    precision_layout = QFormLayout(precision_group)

    main_window.precision_std_label = QLabel("—")
    main_window.precision_rse_label = QLabel("—")
    main_window.precision_bias_label = QLabel("—")
    main_window.precision_mean_ratio_label = QLabel("—")

    precision_layout.addRow("Std Dev:", main_window.precision_std_label)
    precision_layout.addRow("Relative Std Error:", main_window.precision_rse_label)
    precision_layout.addRow("Bias (mean − nominal) ± standard error:", main_window.precision_bias_label)
    precision_layout.addRow("Mean D_est / D_nom:", main_window.precision_mean_ratio_label)

    main_window.precision_placeholder = QLabel("Run a simulation to see results.")
    main_window.precision_placeholder.setProperty("class", "info-text")
    precision_layout.addRow(main_window.precision_placeholder)

    # Top row: fit method + precision summary
    top_row = QHBoxLayout()
    top_row.addWidget(fit_method_group)
    top_row.addWidget(precision_group)
    layout.addLayout(top_row)

    # Explore accuracy threshold group (interactive, enabled after simulation)
    explore_group = QGroupBox("Explore Accuracy Threshold")
    explore_layout = QVBoxLayout(explore_group)
    main_window.explore_proximity_group = explore_group

    explore_description = QLabel(
        "Adjust the threshold to see what fraction of estimates\n"
        "fall within a given percentage of the nominal value.\n"
        "The histogram highlights that part of the estimates."
    )
    explore_description.setWordWrap(True)
    explore_description.setProperty("class", "info-text")
    explore_layout.addWidget(explore_description)

    # Spin box for proximity level
    proximity_label = QLabel("&Threshold:")
    main_window.proximity_spin = QDoubleSpinBox()
    main_window.proximity_spin.setMinimum(0.01)
    main_window.proximity_spin.setMaximum(0.50)
    main_window.proximity_spin.setValue(0.10)
    main_window.proximity_spin.setDecimals(2)
    main_window.proximity_spin.setSingleStep(0.01)
    main_window.proximity_spin.setToolTip(
        "Fractional tolerance for accuracy assessment.\n"
        "0.10 = ±10%, 0.20 = ±20%, 0.05 = ±5%"
    )

    main_window.proximity_pct_label = QLabel("(±10%)")
    main_window.proximity_spin.valueChanged.connect(
        lambda val: _on_proximity_changed(main_window, val)
    )

    add_keyboard_shortcut_to_label(proximity_label, main_window.proximity_spin)

    spin_row = QHBoxLayout()
    spin_row.addWidget(proximity_label)
    spin_row.addWidget(main_window.proximity_spin)
    spin_row.addWidget(main_window.proximity_pct_label)
    spin_row.addStretch()
    explore_layout.addLayout(spin_row)

    # Fraction result label
    main_window.proximity_fraction_label = QLabel("")
    main_window.proximity_fraction_label.setStyleSheet("font-weight: bold;")
    main_window.proximity_fraction_label.setWordWrap(True)
    explore_layout.addWidget(main_window.proximity_fraction_label)

    # Histogram of the estimates, with the part within the threshold highlighted
    main_window.proximity_histogram = ProximityHistogramWidget(proximity=0.10)
    explore_layout.addWidget(main_window.proximity_histogram)

    # Disabled until results exist
    explore_group.setEnabled(False)

    layout.addWidget(explore_group)
    layout.addStretch()

    set_tab_order([
        main_window.plot_method_wls_radio,
        main_window.plot_method_ols_radio,
        main_window.proximity_spin,
    ])

    return tab


def selected_estimator_key(main_window: "DiceGUI") -> str:
    """Key of the estimator the Fit Method radio buttons select (dice.models.estimators)."""
    return 'wls' if main_window.plot_method_wls_radio.isChecked() else 'ols'


def _on_proximity_changed(main_window: "DiceGUI", value: float) -> None:
    """Handle proximity spin box value change: update the label, the histogram, and the fraction."""
    pct = int(round(value * 100))
    main_window.proximity_pct_label.setText(f"(±{pct}%)")
    main_window.proximity_histogram.set_proximity(value)

    # Recalculate fraction if results are available
    if main_window.interface.last_result is not None:
        estimator = selected_estimator_key(main_window)
        main_window.proximity_histogram.set_ratios(main_window.interface.estimate_ratios(estimator))
        detail = main_window.interface.proximity_detail(value, estimator=estimator)
        if detail is not None:
            # The fraction comes from a finite number of runs, so it is shown
            # with its count and its 95% interval.
            main_window.proximity_fraction_label.setText(
                f"{detail['fraction']:.1%} of estimates fall within ±{pct}% of nominal "
                f"({detail['count']} of {detail['total']}; "
                f"95% interval {detail['lower']:.1%} to {detail['upper']:.1%})"
            )


def populate_precision_summary(main_window: "DiceGUI", metrics: dict, d_unit: Optional[str] = None) -> None:
    """Populate the read-only precision summary labels after simulation.

    The standard deviation and bias are diffusion coefficients in the unit
    d_unit (no unit is shown when it is None); the two ratios are unitless.
    The bias is shown with its standard error ('bias_standard_error'). Three
    significant digits: the statistics of a simulation are not known more
    closely.
    """
    unit_suffix = f" {d_unit}" if d_unit else ""
    main_window.precision_std_label.setText(f"{metrics['std']:.3g}{unit_suffix}")
    main_window.precision_rse_label.setText(f"{metrics['relative_std_error']:.3f}")
    main_window.precision_bias_label.setText(
        f"{metrics['bias']:.3g} ± {metrics['bias_standard_error']:.2g}{unit_suffix}")
    main_window.precision_mean_ratio_label.setText(f"{metrics['mean_relative_error']:.3f}")
    main_window.precision_placeholder.setVisible(False)

    # Enable the explore group
    main_window.explore_proximity_group.setEnabled(True)

    # Trigger initial proximity calculation
    _on_proximity_changed(main_window, main_window.proximity_spin.value())


def reset_precision_summary(main_window: "DiceGUI") -> None:
    """Reset precision summary to placeholder state."""
    main_window.precision_std_label.setText("—")
    main_window.precision_rse_label.setText("—")
    main_window.precision_bias_label.setText("—")
    main_window.precision_mean_ratio_label.setText("—")
    main_window.precision_placeholder.setVisible(True)
    main_window.proximity_fraction_label.setText("")
    main_window.proximity_histogram.set_ratios(None)
    main_window.explore_proximity_group.setEnabled(False)
