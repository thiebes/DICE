"""
Tab 4: Analysis

Fit method selection and post-hoc precision analysis.
"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING, Optional

from PyQt6.QtCore import Qt
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
    """Create Tab 4: Analysis."""
    tab, layout = create_scrollable_tab()

    # Fit method group
    fit_method_group = QGroupBox("Fit Method")
    fit_method_layout = QVBoxLayout(fit_method_group)

    main_window.plot_method_wls_radio = QRadioButton("&Weighted Least Squares (WLS)")
    main_window.plot_method_ols_radio = QRadioButton("&Ordinary Least Squares (OLS), for comparison")
    main_window.plot_method_wls_radio.setChecked(True)
    main_window.plot_method_wls_radio.setToolTip(
        "Weighted Least Squares regression.\n\n"
        "Weights each data point by the inverse of its squared relative error.\n"
        "Recommended for most applications as it accounts for\n"
        "heteroscedasticity in MSD measurements."
    )
    main_window.plot_method_ols_radio.setToolTip(
        "Ordinary Least Squares regression, shown for comparison.\n\n"
        "Treats all time points equally, although the later profiles are\n"
        "weaker and their fitted widths less certain. The weighted fit is\n"
        "the default."
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
    main_window.precision_rsd_label = QLabel("—")
    main_window.precision_bias_label = QLabel("—")
    main_window.precision_mean_ratio_label = QLabel("—")
    # The unit beside these values can come from a results file: no markup.
    for value_label in (main_window.precision_std_label, main_window.precision_rsd_label,
                        main_window.precision_bias_label, main_window.precision_mean_ratio_label):
        value_label.setTextFormat(Qt.TextFormat.PlainText)

    precision_layout.addRow("Std Dev:", main_window.precision_std_label)
    precision_layout.addRow("Relative Std Dev:", main_window.precision_rsd_label)
    precision_layout.addRow("Bias (mean − nominal) ± standard error:", main_window.precision_bias_label)
    precision_layout.addRow("Mean D_est / D_nom:", main_window.precision_mean_ratio_label)

    main_window.precision_placeholder = QLabel(NO_RESULTS_TEXT)
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
        "In the histogram, the blue part of each bar, from the bottom, is the\n"
        "share of its estimates within the threshold; dashed lines mark the limits."
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
    # Hidden while it has no text (set_fraction_text).
    main_window.proximity_fraction_label.setVisible(False)
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


def set_fraction_text(main_window: "DiceGUI", text: str) -> None:
    """Show the sentence about the estimates within the threshold, and hide its label when there is none.

    An empty label keeps a line of the tab. Without results the Precision
    Summary shows its placeholder row instead, and with both lines the tab is
    taller than its view in a window of the default size.
    """
    main_window.proximity_fraction_label.setText(text)
    main_window.proximity_fraction_label.setVisible(bool(text))


def _on_proximity_changed(main_window: "DiceGUI", value: float) -> None:
    """Handle proximity spin box value change: update the label, the histogram, and the fraction."""
    pct = int(round(value * 100))
    main_window.proximity_pct_label.setText(f"(±{pct}%)")
    main_window.proximity_histogram.set_proximity(value)

    # Recalculate the fraction if results are available. A fit without
    # estimates clears what an earlier fit or simulation left on the tab.
    if main_window.interface.last_result is not None:
        estimator = selected_estimator_key(main_window)
        main_window.proximity_histogram.set_ratios(main_window.interface.estimate_ratios(estimator))
        detail = main_window.interface.proximity_detail(value, estimator=estimator)
        if detail is None:
            set_fraction_text(main_window, "")
        else:
            # The fraction comes from a finite number of runs, so it is shown
            # with its count and its 95% interval, as in the summary file.
            from dice.io.results import fraction_text
            set_fraction_text(
                main_window,
                f"Within ±{pct}% of nominal: "
                f"{fraction_text(detail['fraction'], detail, interval_label='95% interval ')}"
            )


def value_with_uncertainty(value: float, uncertainty: float) -> str:
    """
    A value and its uncertainty in one notation, to the same decimal place.

    The uncertainty is given with two significant digits and the value is
    rounded to its last digit, for example '-0.00042 ± 0.00087'. Numbers
    that need more than eight decimals share one power of ten, that of the
    larger number, for example '(3.1 ± 8.7)e-10' or '(5.000 ± 0.046)e-10'.
    A value without a finite uncertainty (the standard
    error of one estimate) is given alone, with three significant digits.
    """
    if not math.isfinite(uncertainty) or uncertainty <= 0 or not math.isfinite(value):
        return f"{value:.3g}"

    def shown(number: float, decimals: int) -> str:
        rounded = round(number, decimals)
        return f"{rounded + 0.0:.{decimals}f}"  # + 0.0 turns -0.0 into 0.0

    decimals = 1 - math.floor(math.log10(uncertainty))
    if decimals <= 8:
        decimals = max(decimals, 0)
        return f"{shown(value, decimals)} ± {shown(uncertainty, decimals)}"
    # The power of ten of the larger number, as a number alone would be
    # written (5.0e-10), and decimals for two digits of the uncertainty.
    exponent = math.floor(math.log10(max(abs(value), uncertainty)))
    scale = 10.0 ** exponent
    decimals = max(1 - math.floor(math.log10(uncertainty / scale)), 1)
    return f"({shown(value / scale, decimals)} ± {shown(uncertainty / scale, decimals)})e{exponent}"


def populate_precision_summary(main_window: "DiceGUI", metrics: dict, d_unit: Optional[str] = None) -> None:
    """Populate the read-only precision summary labels after simulation.

    The standard deviation and bias are diffusion coefficients in the unit
    d_unit (no unit is shown when it is None); the two ratios are unitless.
    The bias is shown with its standard error ('bias_standard_error'), both
    to the same decimal place (value_with_uncertainty). The other values
    have three significant digits: the statistics of a simulation are not
    known more closely.
    """
    unit_suffix = f" {d_unit}" if d_unit else ""
    main_window.precision_std_label.setText(f"{metrics['std']:.3g}{unit_suffix}")
    main_window.precision_rsd_label.setText(f"{metrics['relative_std_dev']:.3f}")
    main_window.precision_bias_label.setText(
        f"{value_with_uncertainty(metrics['bias'], metrics['bias_standard_error'])}{unit_suffix}")
    main_window.precision_mean_ratio_label.setText(f"{metrics['mean_relative_error']:.3f}")
    main_window.precision_placeholder.setVisible(False)

    # Enable the explore group
    main_window.explore_proximity_group.setEnabled(True)

    # Trigger initial proximity calculation
    _on_proximity_changed(main_window, main_window.proximity_spin.value())


NO_RESULTS_TEXT = "Run a simulation to see results."


def reset_precision_summary(main_window: "DiceGUI", placeholder: str = NO_RESULTS_TEXT) -> None:
    """Reset precision summary to placeholder state, with `placeholder` as the text shown in its place."""
    main_window.precision_placeholder.setText(placeholder)
    main_window.precision_std_label.setText("—")
    main_window.precision_rsd_label.setText("—")
    main_window.precision_bias_label.setText("—")
    main_window.precision_mean_ratio_label.setText("—")
    main_window.precision_placeholder.setVisible(True)
    set_fraction_text(main_window, "")
    main_window.proximity_histogram.set_ratios(None)
    main_window.explore_proximity_group.setEnabled(False)
