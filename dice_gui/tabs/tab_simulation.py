"""
Tab 1: Simulation Setup

Configuration for simulation runs, output naming, and performance settings.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QSpinBox, QLineEdit, QLabel, QCheckBox
)
from PyQt6.QtCore import Qt

from dice_gui.tabs.base import create_error_label

if TYPE_CHECKING:
    from dice_gui.dice_gui import DiceGUI


def create_tab_simulation_setup(main_window: "DiceGUI") -> QWidget:
    """Create Tab 1: Simulation Setup."""
    tab = QWidget()
    layout = QVBoxLayout(tab)

    # Basic settings group
    basic_group = QGroupBox("Basic Settings")
    basic_layout = QFormLayout(basic_group)

    # Number of runs
    main_window.num_runs_spin = QSpinBox()
    main_window.num_runs_spin.setMinimum(1)
    main_window.num_runs_spin.setMaximum(1000000)
    main_window.num_runs_spin.setValue(1000)
    main_window.num_runs_spin.setToolTip(
        "Number of Monte Carlo simulation iterations to run.\n\n"
        "Higher values provide better statistical precision but take longer to compute.\n"
        "Typical values: 100-1000 for testing, 1000-10000 for publication-quality results.\n\n"
        "Each run generates a noisy profile, fits it, and estimates the diffusion coefficient."
    )
    basic_layout.addRow("Number of Runs:", main_window.num_runs_spin)

    # Filename slug
    main_window.filename_slug_input = QLineEdit()
    main_window.filename_slug_input.setText("dice_simulation")
    main_window.filename_slug_input.setToolTip(
        "Prefix for all output filenames.\n\n"
        "Output files will be saved as: output/<slug>/<slug>_results.csv, <slug>_accuracy_histogram.png, etc.\n\n"
        "Use descriptive names to organize multiple simulations (e.g., 'high_SNR_test' or 'sample_A_analysis')."
    )
    basic_layout.addRow("Filename Slug:", main_window.filename_slug_input)

    # Filename slug validation error label
    main_window._error_labels["filename_slug"] = create_error_label()
    basic_layout.addRow("", main_window._error_labels["filename_slug"])

    # Output path preview
    main_window.output_path_preview = QLabel()
    main_window.output_path_preview.setProperty("class", "output-path-preview")
    main_window.output_path_preview.setWordWrap(True)
    main_window.output_path_preview.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    basic_layout.addRow("Output Location:", main_window.output_path_preview)
    main_window.filename_slug_input.textChanged.connect(main_window._update_output_path_preview)

    # Performance settings group
    performance_group = QGroupBox("Performance Settings")
    performance_layout = QFormLayout(performance_group)

    # Multiprocessing checkbox
    main_window.multiprocessing_check = QCheckBox("Enable parallel processing")
    main_window.multiprocessing_check.setChecked(True)
    main_window.multiprocessing_check.setToolTip("Use multiple CPU cores to speed up simulation")
    performance_layout.addRow("Multiprocessing:", main_window.multiprocessing_check)

    # Retain profile data checkbox
    main_window.retain_profile_check = QCheckBox("Retain profile data")
    main_window.retain_profile_check.setChecked(False)
    main_window.retain_profile_check.setToolTip("Keep raw profile data (memory intensive)")
    performance_layout.addRow("Data Retention:", main_window.retain_profile_check)

    # Two-column layout for groups
    columns = QHBoxLayout()
    columns.addWidget(basic_group)
    columns.addWidget(performance_group)
    layout.addLayout(columns)

    layout.addStretch()

    return tab
