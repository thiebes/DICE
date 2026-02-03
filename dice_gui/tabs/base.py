"""
Base Tab Utilities

Shared utility functions for building tab UI components.
"""

from PyQt6.QtWidgets import (
    QFrame, QFormLayout, QVBoxLayout, QHBoxLayout,
    QWidget, QLineEdit, QLabel
)


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


def create_error_label() -> QLabel:
    """Create an inline error label for a validated field."""
    error_label = QLabel("")
    error_label.setProperty("class", "validation-error")
    error_label.setWordWrap(True)
    error_label.setMinimumHeight(20)
    error_label.setVisible(False)
    return error_label
