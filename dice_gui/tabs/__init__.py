"""
Tab Modules for DICE GUI

Each tab is created by a separate module for maintainability.
"""

from dice_gui.tabs.base import (
    create_option_card,
    create_field_with_unit,
    create_error_label,
)
from dice_gui.tabs.tab_simulation import create_tab_simulation_setup
from dice_gui.tabs.tab_physical import (
    create_tab_physical_parameters,
    toggle_diffusion_inputs,
    update_calculated_length,
    update_width_conversion,
)
from dice_gui.tabs.tab_experimental import (
    create_tab_experimental_conditions,
    toggle_noise_inputs,
    toggle_time_inputs,
    update_pixel_size,
    browse_noise_file,
)
from dice_gui.tabs.tab_analysis import (
    create_tab_analysis_settings,
    update_proximity_target,
)
from dice_gui.tabs.tab_output import create_tab_output_settings

__all__ = [
    # Base utilities
    "create_option_card",
    "create_field_with_unit",
    "create_error_label",
    # Tab creation functions
    "create_tab_simulation_setup",
    "create_tab_physical_parameters",
    "create_tab_experimental_conditions",
    "create_tab_analysis_settings",
    "create_tab_output_settings",
    # Tab helper functions
    "toggle_diffusion_inputs",
    "update_calculated_length",
    "update_width_conversion",
    "toggle_noise_inputs",
    "toggle_time_inputs",
    "update_pixel_size",
    "browse_noise_file",
    "update_proximity_target",
]
