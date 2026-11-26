"""
DICE GUI Package

Graphical user interface for the Diffusion Insight Computation Engine.
"""

# Note: __version__ is not imported here to keep package init lightweight
# Import it directly from dice where needed: from dice import __version__

from dice_gui.validation_manager import ValidationManager, apply_validation_style, clear_validation_style
from dice_gui.collapsible_group import CollapsibleGroupBox
from dice_gui.presets import PRESETS, OUTPUT_DEFAULTS, get_preset, list_presets

__all__ = [
    'ValidationManager',
    'apply_validation_style',
    'clear_validation_style',
    'CollapsibleGroupBox',
    'PRESETS',
    'OUTPUT_DEFAULTS',
    'get_preset',
    'list_presets',
]
