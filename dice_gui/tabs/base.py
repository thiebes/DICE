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


def create_unit_combo(unit_type: str, default_unit: str = None,
                      squared: bool = False) -> QComboBox:
    """Create a combo box populated with valid units for per-parameter selection.

    The combo box has a '_user_modified' dynamic property that starts as False.
    When the user explicitly changes the selection (as opposed to programmatic
    updates from global unit changes), set this property to True so that global
    unit changes skip combos the user has customized.

    Args:
        unit_type: 'length' or 'time'.
        default_unit: Unit to select initially. Defaults to 'micrometer' for
            length or 'nanosecond' for time.
        squared: If True, display items with a superscript 2 suffix (e.g.,
            'micrometer\u00b2'). The underlying data stores the base unit name
            so that ``combo_value(combo)`` returns the unsuffixed name.

    Returns:
        Configured QComboBox.
    """
    combo = QComboBox()
    combo.setProperty("_user_modified", False)
    combo.setMaximumWidth(120 if squared else 110)

    if unit_type == 'length':
        items = LENGTH_UNITS_DISPLAY
        default = default_unit or 'micrometer'
    elif unit_type == 'time':
        items = TIME_UNITS_DISPLAY
        default = default_unit or 'nanosecond'
    else:
        items, default = [], default_unit or ''

    if squared:
        for item in items:
            combo.addItem(item + "\u00b2", item)
        idx = combo.findData(default)
        if idx >= 0:
            combo.setCurrentIndex(idx)
    else:
        combo.addItems(items)
        combo.setCurrentText(default)

    return combo


def combo_value(combo: QComboBox) -> str:
    """Get the unit name from a combo, using item data when available.

    Combos created with ``squared=True`` store the base unit name as item
    data so that callers always receive unsuffixed names like 'micrometer'
    regardless of what the combo displays.
    """
    data = combo.currentData()
    return data if data is not None else combo.currentText()


def set_combo_value(combo: QComboBox, unit_name: str) -> None:
    """Set a combo's selection by unit name.

    Uses data-based lookup first (for squared combos), falling back to
    text-based matching for regular combos.
    """
    idx = combo.findData(unit_name)
    if idx >= 0:
        combo.setCurrentIndex(idx)
    else:
        combo.setCurrentText(unit_name)


def create_error_label() -> QLabel:
    """Create an inline error label for a validated field."""
    error_label = QLabel("")
    error_label.setProperty("class", "validation-error")
    error_label.setWordWrap(True)
    error_label.setMinimumHeight(20)
    error_label.setVisible(False)
    return error_label
