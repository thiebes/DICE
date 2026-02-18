"""
Tab 5: Output Settings

Configuration for plot output format, resolution, and typography.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QPushButton
)

from dice_gui.accessibility import add_keyboard_shortcut_to_label, set_tab_order

if TYPE_CHECKING:
    from dice_gui.dice_gui import DiceGUI


def create_tab_output_settings(main_window: "DiceGUI") -> QWidget:
    """Create Tab 5: Output Settings."""
    from dice_gui.presets import PRESETS, OUTPUT_DEFAULTS, get_preset

    tab = QWidget()
    layout = QVBoxLayout(tab)

    # === Preset Section ===
    preset_layout = QHBoxLayout()
    preset_label = QLabel("&Quick Setup:")
    main_window.preset_combo = QComboBox()
    main_window.preset_combo.setProperty("class", "preset-selector")
    main_window.preset_combo.addItem("Custom", None)
    main_window.preset_combo.addItem("Publication (journal-ready)", "publication")
    main_window.preset_combo.addItem("Presentation (large fonts)", "presentation")
    main_window.preset_combo.addItem("Draft (quick preview)", "draft")
    main_window.preset_combo.currentIndexChanged.connect(lambda idx: _apply_output_preset(main_window, idx))

    add_keyboard_shortcut_to_label(preset_label, main_window.preset_combo)
    preset_layout.addWidget(preset_label)
    preset_layout.addWidget(main_window.preset_combo)
    preset_layout.addStretch()
    layout.addLayout(preset_layout)

    # === Column 1: Image Format ===
    format_group = QGroupBox("Image Format")
    format_layout = QFormLayout(format_group)

    # Image type
    main_window.image_type_combo = QComboBox()
    main_window.image_type_combo.addItems(["png", "jpg", "svg", "tif"])
    main_window.image_type_combo.setCurrentText("png")
    main_window.image_type_combo.setToolTip(
        "File format for saved plots.\n\n"
        "- PNG: Best for general use, lossless compression (recommended)\n"
        "- JPG: Smaller files but lossy compression\n"
        "- SVG: Vector format, scalable, ideal for publications\n"
        "- TIF: Uncompressed, maximum quality"
    )
    file_type_label = QLabel("File &Type:")
    add_keyboard_shortcut_to_label(file_type_label, main_window.image_type_combo)
    format_layout.addRow(file_type_label, main_window.image_type_combo)
    main_window.image_type_combo.currentTextChanged.connect(main_window._update_output_path_preview)

    # Image width
    width_widget = QWidget()
    width_layout = QHBoxLayout(width_widget)
    width_layout.setContentsMargins(0, 0, 0, 0)
    main_window.image_width_spin = QDoubleSpinBox()
    main_window.image_width_spin.setMinimum(0.1)
    main_window.image_width_spin.setMaximum(100.0)
    main_window.image_width_spin.setValue(16.0)
    main_window.image_width_spin.setDecimals(2)
    main_window.image_width_unit_combo = QComboBox()
    main_window.image_width_unit_combo.addItems(["cm", "in", "mm"])
    width_layout.addWidget(main_window.image_width_spin)
    width_layout.addWidget(main_window.image_width_unit_combo)
    width_label = QLabel("&Width:")
    add_keyboard_shortcut_to_label(width_label, main_window.image_width_spin)
    format_layout.addRow(width_label, width_widget)

    # Image height
    height_widget = QWidget()
    height_layout = QHBoxLayout(height_widget)
    height_layout.setContentsMargins(0, 0, 0, 0)
    main_window.image_height_spin = QDoubleSpinBox()
    main_window.image_height_spin.setMinimum(0.1)
    main_window.image_height_spin.setMaximum(100.0)
    main_window.image_height_spin.setValue(10.0)
    main_window.image_height_spin.setDecimals(2)
    main_window.image_height_unit_combo = QComboBox()
    main_window.image_height_unit_combo.addItems(["cm", "in", "mm"])
    height_layout.addWidget(main_window.image_height_spin)
    height_layout.addWidget(main_window.image_height_unit_combo)
    height_label = QLabel("H&eight:")
    add_keyboard_shortcut_to_label(height_label, main_window.image_height_spin)
    format_layout.addRow(height_label, height_widget)

    # === Column 2: Resolution & Histogram ===
    resolution_group = QGroupBox("Resolution & Histogram")
    resolution_layout = QFormLayout(resolution_group)

    # DPI
    main_window.image_dpi_spin = QSpinBox()
    main_window.image_dpi_spin.setMinimum(50)
    main_window.image_dpi_spin.setMaximum(1200)
    main_window.image_dpi_spin.setValue(300)
    main_window.image_dpi_spin.setToolTip("Resolution: 300 DPI for publications, 96 for screen")
    dpi_label = QLabel("&DPI:")
    add_keyboard_shortcut_to_label(dpi_label, main_window.image_dpi_spin)
    resolution_layout.addRow(dpi_label, main_window.image_dpi_spin)

    # Number of bins
    main_window.image_numbins_spin = QSpinBox()
    main_window.image_numbins_spin.setMinimum(5)
    main_window.image_numbins_spin.setMaximum(200)
    main_window.image_numbins_spin.setValue(35)
    main_window.image_numbins_spin.setToolTip("Number of bins for accuracy histogram")
    bins_label = QLabel("Histogram &Bins:")
    add_keyboard_shortcut_to_label(bins_label, main_window.image_numbins_spin)
    resolution_layout.addRow(bins_label, main_window.image_numbins_spin)

    # === Column 3: Typography ===
    typography_group = QGroupBox("Typography")
    typography_layout = QFormLayout(typography_group)

    # Font size
    font_widget = QWidget()
    font_layout = QHBoxLayout(font_widget)
    font_layout.setContentsMargins(0, 0, 0, 0)
    main_window.image_font_size_spin = QSpinBox()
    main_window.image_font_size_spin.setMinimum(4)
    main_window.image_font_size_spin.setMaximum(72)
    main_window.image_font_size_spin.setValue(6)
    main_window.image_font_unit_combo = QComboBox()
    main_window.image_font_unit_combo.addItems(["pt", "px"])
    font_layout.addWidget(main_window.image_font_size_spin)
    font_layout.addWidget(main_window.image_font_unit_combo)
    font_size_label = QLabel("Fo&nt Size:")
    add_keyboard_shortcut_to_label(font_size_label, main_window.image_font_size_spin)
    typography_layout.addRow(font_size_label, font_widget)

    # Tick length
    tick_length_widget = QWidget()
    tick_length_layout = QHBoxLayout(tick_length_widget)
    tick_length_layout.setContentsMargins(0, 0, 0, 0)
    main_window.image_tick_length_spin = QSpinBox()
    main_window.image_tick_length_spin.setMinimum(1)
    main_window.image_tick_length_spin.setMaximum(50)
    main_window.image_tick_length_spin.setValue(6)
    main_window.image_tick_length_unit_combo = QComboBox()
    main_window.image_tick_length_unit_combo.addItems(["pt", "px"])
    tick_length_layout.addWidget(main_window.image_tick_length_spin)
    tick_length_layout.addWidget(main_window.image_tick_length_unit_combo)
    tick_length_label = QLabel("Tic&k Length:")
    add_keyboard_shortcut_to_label(tick_length_label, main_window.image_tick_length_spin)
    typography_layout.addRow(tick_length_label, tick_length_widget)

    # Tick width
    tick_width_widget = QWidget()
    tick_width_layout = QHBoxLayout(tick_width_widget)
    tick_width_layout.setContentsMargins(0, 0, 0, 0)
    main_window.image_tick_width_spin = QSpinBox()
    main_window.image_tick_width_spin.setMinimum(1)
    main_window.image_tick_width_spin.setMaximum(20)
    main_window.image_tick_width_spin.setValue(2)
    main_window.image_tick_width_unit_combo = QComboBox()
    main_window.image_tick_width_unit_combo.addItems(["pt", "px"])
    tick_width_layout.addWidget(main_window.image_tick_width_spin)
    tick_width_layout.addWidget(main_window.image_tick_width_unit_combo)
    tick_width_label = QLabel("Tick W&idth:")
    add_keyboard_shortcut_to_label(tick_width_label, main_window.image_tick_width_spin)
    typography_layout.addRow(tick_width_label, tick_width_widget)

    # Three-column layout
    columns = QHBoxLayout()
    columns.addWidget(format_group)
    columns.addWidget(resolution_group)
    columns.addWidget(typography_group)
    layout.addLayout(columns)

    # === Plot Actions ===
    actions_layout = QHBoxLayout()

    main_window.regenerate_plot_button = QPushButton("Regenerate Plot")
    main_window.regenerate_plot_button.setToolTip("Regenerate plot with current settings")
    main_window.regenerate_plot_button.clicked.connect(main_window.regenerate_plot)

    main_window.load_results_button = QPushButton("Load Results")
    main_window.load_results_button.setToolTip("Load results from CSV file")
    main_window.load_results_button.clicked.connect(main_window.load_and_plot_results)

    reset_btn = QPushButton("Reset to Defaults")
    reset_btn.setProperty("class", "reset-button")
    reset_btn.clicked.connect(lambda: _reset_all_plot_settings(main_window))

    actions_layout.addWidget(main_window.regenerate_plot_button)
    actions_layout.addWidget(main_window.load_results_button)
    actions_layout.addStretch()
    actions_layout.addWidget(reset_btn)

    layout.addLayout(actions_layout)
    layout.addStretch()

    set_tab_order([
        main_window.preset_combo,
        main_window.image_type_combo,
        main_window.image_width_spin,
        main_window.image_width_unit_combo,
        main_window.image_height_spin,
        main_window.image_height_unit_combo,
        main_window.image_dpi_spin,
        main_window.image_numbins_spin,
        main_window.image_font_size_spin,
        main_window.image_font_unit_combo,
        main_window.image_tick_length_spin,
        main_window.image_tick_length_unit_combo,
        main_window.image_tick_width_spin,
        main_window.image_tick_width_unit_combo,
        main_window.regenerate_plot_button,
        main_window.load_results_button,
    ])

    return tab


def _apply_output_preset(main_window: "DiceGUI", index: int) -> None:
    """Apply selected output preset."""
    from dice_gui.presets import get_preset

    preset_key = main_window.preset_combo.currentData()
    if preset_key is None:
        return

    preset = get_preset(preset_key)
    if preset is None:
        return

    settings = preset.settings

    main_window.image_type_combo.setCurrentText(settings.get("image_type", "png"))
    main_window.image_width_spin.setValue(settings.get("image_width", 16.0))
    main_window.image_width_unit_combo.setCurrentText(settings.get("image_width_unit", "cm"))
    main_window.image_height_spin.setValue(settings.get("image_height", 10.0))
    main_window.image_height_unit_combo.setCurrentText(settings.get("image_height_unit", "cm"))
    main_window.image_dpi_spin.setValue(settings.get("image_dpi", 300))
    main_window.image_numbins_spin.setValue(settings.get("image_numbins", 35))
    main_window.image_font_size_spin.setValue(settings.get("image_font_size", 6))
    main_window.image_font_unit_combo.setCurrentText(settings.get("image_font_unit", "pt"))
    main_window.image_tick_length_spin.setValue(settings.get("image_tick_length", 6))
    main_window.image_tick_width_spin.setValue(settings.get("image_tick_width", 2))


def _reset_all_plot_settings(main_window: "DiceGUI") -> None:
    """Reset all plot settings to defaults."""
    from dice_gui.presets import OUTPUT_DEFAULTS

    # Image format
    main_window.image_type_combo.setCurrentText(OUTPUT_DEFAULTS["image_type"])
    main_window.image_width_spin.setValue(OUTPUT_DEFAULTS["image_width"])
    main_window.image_width_unit_combo.setCurrentText(OUTPUT_DEFAULTS["image_width_unit"])
    main_window.image_height_spin.setValue(OUTPUT_DEFAULTS["image_height"])
    main_window.image_height_unit_combo.setCurrentText(OUTPUT_DEFAULTS["image_height_unit"])

    # Resolution & histogram
    main_window.image_dpi_spin.setValue(OUTPUT_DEFAULTS["image_dpi"])
    main_window.image_numbins_spin.setValue(OUTPUT_DEFAULTS["image_numbins"])

    # Typography
    main_window.image_font_size_spin.setValue(OUTPUT_DEFAULTS["image_font_size"])
    main_window.image_font_unit_combo.setCurrentText(OUTPUT_DEFAULTS["image_font_unit"])
    main_window.image_tick_length_spin.setValue(OUTPUT_DEFAULTS["image_tick_length"])
    main_window.image_tick_length_unit_combo.setCurrentText(OUTPUT_DEFAULTS["image_tick_length_unit"])
    main_window.image_tick_width_spin.setValue(OUTPUT_DEFAULTS["image_tick_width"])
    main_window.image_tick_width_unit_combo.setCurrentText(OUTPUT_DEFAULTS["image_tick_width_unit"])

    main_window.preset_combo.setCurrentIndex(0)
