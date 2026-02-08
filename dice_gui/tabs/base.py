"""
Base Tab Utilities

Shared utility functions for building tab UI components.
"""

from PyQt6.QtWidgets import (
    QFrame, QFormLayout, QVBoxLayout, QHBoxLayout,
    QWidget, QLineEdit, QLabel, QComboBox
)

# Unit lists ordered from smallest to largest for combo box display.
LENGTH_UNITS_DISPLAY = [
    "angstrom", "picometer", "nanometer", "micrometer",
    "millimeter", "centimeter", "meter",
]

TIME_UNITS_DISPLAY = [
    "attosecond", "femtosecond", "picosecond", "nanosecond",
    "microsecond", "millisecond", "second",
]


def create_option_card(layout_type: str = "form") -> tuple[QFrame, QFormLayout | QVBoxLayout]:
    """Create a styled option card with specified layout type.

    Args:
        layout_type: "form" for QFormLayout (default), "vbox" for QVBoxLayout

    Returns:
        Tuple of (frame, layout) for adding content.
    """
    frame = QFrame()
    frame.setProperty("class", "option-card")
    if layout_type == "vbox":
        layout = QVBoxLayout(frame)
    else:
        layout = QFormLayout(frame)
    layout.setContentsMargins(12, 8, 12, 8)
    return frame, layout


def create_field_with_unit(unit_label: str) -> tuple[QWidget, QLineEdit, QLabel]:
    """Create an input field with unit label suffix.

    Returns:
        Tuple of (container_widget, line_edit, unit_label).
    """
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    line_edit = QLineEdit()
    label = QLabel(unit_label)
    layout.addWidget(line_edit)
    layout.addWidget(label)
    return widget, line_edit, label


def create_unit_combo(unit_type: str, default_unit: str = None) -> QComboBox:
    """Create a combo box populated with valid units for per-parameter selection.

    The combo box has a '_user_modified' dynamic property that starts as False.
    When the user explicitly changes the selection (as opposed to programmatic
    updates from global unit changes), set this property to True so that global
    unit changes skip combos the user has customized.

    Args:
        unit_type: 'length' or 'time'.
        default_unit: Unit to select initially. Defaults to 'micrometer' for
            length or 'nanosecond' for time.

    Returns:
        Configured QComboBox.
    """
    combo = QComboBox()
    combo.setProperty("_user_modified", False)
    combo.setMaximumWidth(110)

    if unit_type == 'length':
        combo.addItems(LENGTH_UNITS_DISPLAY)
        combo.setCurrentText(default_unit or 'micrometer')
    elif unit_type == 'time':
        combo.addItems(TIME_UNITS_DISPLAY)
        combo.setCurrentText(default_unit or 'nanosecond')

    return combo


def create_error_label() -> QLabel:
    """Create an inline error label for a validated field."""
    error_label = QLabel("")
    error_label.setProperty("class", "validation-error")
    error_label.setWordWrap(True)
    error_label.setMinimumHeight(20)
    error_label.setVisible(False)
    return error_label
