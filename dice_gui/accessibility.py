"""
Accessibility utilities for DICE GUI.

Provides helper functions for improving accessibility compliance with
WCAG 2.1 Level AA and AAA standards.
"""

from PyQt6.QtWidgets import (
    QWidget, QLabel
)
from typing import Optional, List


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


# Each unit selector, by its attribute of the main window, and the quantity
# whose unit it sets. A unit selector shares its row with the field of that
# quantity, so no label names it.
UNIT_SELECTOR_QUANTITIES = {
    'length_unit_combo': 'Global length',
    'time_unit_combo': 'Global time',
    'diffusion_length_unit_combo': 'Diffusion length',
    'diffusion_coeff_length_unit_combo': 'Diffusion coefficient length',
    'diffusion_coeff_time_unit_combo': 'Diffusion coefficient time',
    'lifetime_unit_combo': 'Lifetime',
    'mean_unit_combo': 'Mean position',
    'width_unit_combo': 'Profile width',
    'spatial_width_unit_combo': 'Spatial width',
    'time_start_unit_combo': 'Start time',
    'time_stop_unit_combo': 'Stop time',
    'image_width_unit_combo': 'Image width',
    'image_height_unit_combo': 'Image height',
    'image_font_unit_combo': 'Font size',
    'image_tick_length_unit_combo': 'Tick length',
    'image_tick_width_unit_combo': 'Tick width',
}

# Fields whose label is too short to stand alone ("Stop", "File", "Width"),
# holds a symbol, or does not exist. A screen reader says these names, and
# the message of a blocked run starts with them.
FIELD_NAMES = {
    'diffusion_length_input': 'Diffusion length',
    'diffusion_coeff_input': 'Diffusion coefficient',
    'lifetime_input': 'Lifetime',
    'amplitude_input': 'Initial amplitude',
    'mean_input': 'Initial mean position',
    'width_input': 'Initial profile width',
    'noise_value_input': 'Noise standard deviation',
    'noise_cnr_input': 'Contrast-to-noise ratio',
    'noise_file_input': 'Measured profile file',
    'time_start_input': 'Start time',
    'time_stop_input': 'Stop time',
    'time_steps_input': 'Number of time steps',
    'time_series_input': 'Time series',
    'image_width_spin': 'Image width',
    'image_height_spin': 'Image height',
    'proximity_spin': 'Accuracy threshold',
}


def label_text_as_name(text: str) -> str:
    """The text of a field label as an accessible name: no shortcut marker and no final colon."""
    return text.replace('&', '').strip().rstrip(':').strip()


def name_inputs_for_screen_readers(window: QWidget) -> None:
    """
    Give every input of the window an accessible name.

    Qt takes the name of a field from its label only when the two are
    siblings in one layout. Most fields here share a row with a unit
    selector, so a screen reader would announce them as "edit" or "combo
    box" without a name. A field takes the text of the label it is the
    buddy of; a unit selector is named after its quantity. The fields of
    FIELD_NAMES get their own names first, and a name that is already set
    is kept.
    """
    for attribute, name in FIELD_NAMES.items():
        widget = getattr(window, attribute, None)
        if widget is not None and not widget.accessibleName():
            widget.setAccessibleName(name)

    for label in window.findChildren(QLabel):
        buddy = label.buddy()
        if buddy is not None and not buddy.accessibleName():
            buddy.setAccessibleName(label_text_as_name(label.text()))

    for attribute, quantity in UNIT_SELECTOR_QUANTITIES.items():
        selector = getattr(window, attribute, None)
        if selector is not None and not selector.accessibleName():
            selector.setAccessibleName(f"{quantity} unit")
