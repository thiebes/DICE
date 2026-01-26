"""
Accessibility utilities for DICE GUI.

Provides helper functions for improving accessibility compliance with
WCAG 2.1 Level AA and AAA standards.
"""

from PyQt6.QtWidgets import (
    QWidget, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QRadioButton, QSlider, QPushButton, QGroupBox
)
from typing import Optional, List, Tuple


def add_accessible_name_and_description(
    widget: QWidget,
    name: str,
    description: Optional[str] = None
) -> None:
    """
    Add accessible name and description to a widget.

    Args:
        widget: Widget to add accessibility info to
        name: Accessible name (brief label)
        description: Accessible description (detailed explanation)
    """
    widget.setAccessibleName(name)
    if description:
        widget.setAccessibleDescription(description)


def add_tooltip_with_accessible_description(
    widget: QWidget,
    tooltip: str,
    accessible_desc: Optional[str] = None
) -> None:
    """
    Add tooltip and accessible description to a widget.

    Args:
        widget: Widget to add tooltip to
        tooltip: Tooltip text (visible on hover)
        accessible_desc: Accessible description for screen readers
                        (defaults to tooltip if not provided)
    """
    widget.setToolTip(tooltip)
    widget.setAccessibleDescription(accessible_desc or tooltip)


def set_tab_order(widgets: List[QWidget]) -> None:
    """
    Set explicit tab order for a list of widgets.

    Args:
        widgets: List of widgets in desired tab order
    """
    for i in range(len(widgets) - 1):
        QWidget.setTabOrder(widgets[i], widgets[i + 1])


def add_keyboard_shortcut_to_label(label: QLabel, buddy: QWidget) -> None:
    """
    Add keyboard shortcut to a label and associate with a widget.

    The label text should contain '&' before the accelerator key.
    Example: "&Number of Runs" creates Alt+N shortcut

    Args:
        label: QLabel with '&' prefix for accelerator
        buddy: Widget to focus when shortcut is activated
    """
    label.setBuddy(buddy)


def create_accessible_slider(
    minimum: int,
    maximum: int,
    default_value: int,
    accessible_name: str,
    value_suffix: str = ""
) -> Tuple[QSlider, QLabel]:
    """
    Create an accessible slider with value display.

    Args:
        minimum: Minimum slider value
        maximum: Maximum slider value
        default_value: Initial value
        accessible_name: Accessible name for the slider
        value_suffix: Suffix for value display (e.g., "%")

    Returns:
        Tuple of (slider widget, value display label)
    """
    slider = QSlider()
    slider.setMinimum(minimum)
    slider.setMaximum(maximum)
    slider.setValue(default_value)
    slider.setAccessibleName(accessible_name)
    slider.setAccessibleDescription(
        f"Slider control for {accessible_name}. "
        f"Range: {minimum} to {maximum}{value_suffix}. "
        f"Use arrow keys to adjust value."
    )

    # Value display for visual users
    value_label = QLabel()
    value_label.setObjectName("proximity-display")
    value_label.setAccessibleName(f"{accessible_name} current value")

    def update_value_display(value: int):
        value_label.setText(f"{value}{value_suffix}")
        # Update accessible description with current value
        slider.setAccessibleDescription(
            f"{accessible_name}: {value}{value_suffix}. "
            f"Range: {minimum} to {maximum}{value_suffix}. "
            f"Use arrow keys to adjust value."
        )

    slider.valueChanged.connect(update_value_display)
    update_value_display(default_value)

    return slider, value_label


def create_accessible_input(
    widget_type: type,
    label_text: str,
    tooltip: str,
    placeholder: str = "",
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
) -> Tuple[QLabel, QWidget]:
    """
    Create an accessible input field with label.

    Args:
        widget_type: Type of input widget (QLineEdit, QSpinBox, etc.)
        label_text: Label text (use & for keyboard shortcut)
        tooltip: Tooltip and accessible description
        placeholder: Placeholder text
        min_value: Minimum value (for numeric inputs)
        max_value: Maximum value (for numeric inputs)

    Returns:
        Tuple of (label, input widget)
    """
    label = QLabel(label_text)
    widget = widget_type()

    # Set placeholder if supported
    if hasattr(widget, 'setPlaceholderText') and placeholder:
        widget.setPlaceholderText(placeholder)

    # Set range if supported
    if hasattr(widget, 'setMinimum') and min_value is not None:
        widget.setMinimum(min_value)
    if hasattr(widget, 'setMaximum') and max_value is not None:
        widget.setMaximum(max_value)

    # Add accessibility attributes
    accessible_name = label_text.replace('&', '')
    widget.setAccessibleName(accessible_name)
    add_tooltip_with_accessible_description(widget, tooltip)

    # Link label to widget for keyboard navigation
    label.setBuddy(widget)

    return label, widget


def create_radio_group_with_accessibility(
    group_label: str,
    options: List[Tuple[str, str]],
    default_index: int = 0
) -> Tuple[QGroupBox, List[QRadioButton]]:
    """
    Create an accessible radio button group.

    Args:
        group_label: Label for the group box
        options: List of (label, description) tuples for each radio button
        default_index: Index of default selected option

    Returns:
        Tuple of (group box, list of radio buttons)
    """
    group = QGroupBox(group_label)
    group.setAccessibleName(group_label)

    radio_buttons = []
    for i, (label, description) in enumerate(options):
        radio = QRadioButton(label)
        radio.setAccessibleName(label)
        radio.setAccessibleDescription(description)
        radio.setChecked(i == default_index)
        radio_buttons.append(radio)

    return group, radio_buttons


def ensure_minimum_size(widget: QWidget, min_width: int = 44, min_height: int = 44) -> None:
    """
    Ensure widget meets minimum size requirements for touch targets.

    WCAG 2.5.5 (Level AAA) recommends 44x44px minimum for interactive elements.

    Args:
        widget: Widget to set minimum size for
        min_width: Minimum width in pixels (default: 44)
        min_height: Minimum height in pixels (default: 44)
    """
    widget.setMinimumSize(min_width, min_height)


def add_form_field_with_accessibility(
    label_text: str,
    widget: QWidget,
    tooltip: str,
    required: bool = True
) -> QLabel:
    """
    Prepare a form field with full accessibility attributes.

    Args:
        label_text: Label text (use & for keyboard shortcut)
        widget: Input widget
        tooltip: Tooltip and accessible description
        required: Whether field is required

    Returns:
        QLabel for the field
    """
    # Add required indicator to label if needed
    display_text = label_text
    if required and not display_text.endswith('*'):
        display_text = f"{display_text}:"
    else:
        display_text = f"{display_text}:"

    label = QLabel(display_text)

    # Extract clean name for accessibility
    accessible_name = label_text.replace('&', '').strip()
    if required:
        accessible_name = f"{accessible_name} (required)"

    # Set accessibility attributes
    widget.setAccessibleName(accessible_name)
    add_tooltip_with_accessible_description(widget, tooltip)

    # Link label to widget
    label.setBuddy(widget)

    return label
