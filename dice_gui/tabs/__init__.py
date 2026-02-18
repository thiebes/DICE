"""
Tab Modules for DICE GUI

Each tab is created by a separate module for maintainability.
"""

from dice_gui.tabs.base import (
    create_option_card,
    create_field_with_unit,
    create_error_label,
    combo_value,
    set_combo_value,
)
from dice_gui.tabs.tab_simulation import create_tab_simulation_setup
from dice_gui.tabs.tab_physical import (
    create_tab_physical_parameters,
    update_diffusion_fields,
    update_width_conversion,
)
from dice_gui.tabs.tab_experimental import (
    create_tab_experimental_conditions,
    toggle_noise_inputs,
    toggle_time_inputs,
    update_noise_cnr_display,
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
    "combo_value",
    "set_combo_value",
    # Tab creation functions
    "create_tab_simulation_setup",
    "create_tab_physical_parameters",
    "create_tab_experimental_conditions",
    "create_tab_analysis_settings",
    "create_tab_output_settings",
    # Tab helper functions
    "update_diffusion_fields",
    "update_width_conversion",
    "toggle_noise_inputs",
    "toggle_time_inputs",
    "update_noise_cnr_display",
    "update_pixel_size",
    "browse_noise_file",
    "update_proximity_target",
]
