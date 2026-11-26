"""Output settings presets for common use cases."""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class OutputPreset:
    """Definition of an output settings preset."""
    name: str
    description: str
    settings: Dict[str, Any]


PRESETS = {
    "publication": OutputPreset(
        name="Publication",
        description="Journal-ready figures (300 DPI, compact size)",
        settings={
            "image_type": "png",
            "image_width": 8.5,
            "image_width_unit": "cm",
            "image_height": 6.0,
            "image_height_unit": "cm",
            "image_dpi": 300,
            "image_dpi_unit": "dpi",
            "image_font_size": 8,
            "image_font_unit": "pt",
            "image_tick_length": 5,
            "image_tick_length_unit": "pt",
            "image_tick_width": 1,
            "image_tick_width_unit": "pt",
            "image_numbins": 35,
        }
    ),
    "presentation": OutputPreset(
        name="Presentation",
        description="Large slides (150 DPI, big fonts)",
        settings={
            "image_type": "png",
            "image_width": 25.0,
            "image_width_unit": "cm",
            "image_height": 15.0,
            "image_height_unit": "cm",
            "image_dpi": 150,
            "image_dpi_unit": "dpi",
            "image_font_size": 14,
            "image_font_unit": "pt",
            "image_tick_length": 8,
            "image_tick_length_unit": "pt",
            "image_tick_width": 2,
            "image_tick_width_unit": "pt",
            "image_numbins": 30,
        }
    ),
    "draft": OutputPreset(
        name="Draft",
        description="Quick preview (96 DPI, screen resolution)",
        settings={
            "image_type": "png",
            "image_width": 16.0,
            "image_width_unit": "cm",
            "image_height": 10.0,
            "image_height_unit": "cm",
            "image_dpi": 96,
            "image_dpi_unit": "dpi",
            "image_font_size": 10,
            "image_font_unit": "pt",
            "image_tick_length": 6,
            "image_tick_length_unit": "pt",
            "image_tick_width": 2,
            "image_tick_width_unit": "pt",
            "image_numbins": 25,
        }
    ),
}


# Default values for reset functionality
OUTPUT_DEFAULTS = {
    "image_type": "png",
    "image_width": 16.0,
    "image_width_unit": "cm",
    "image_height": 10.0,
    "image_height_unit": "cm",
    "image_dpi": 300,
    "image_dpi_unit": "dpi",
    "image_font_size": 6,
    "image_font_unit": "pt",
    "image_tick_length": 6,
    "image_tick_length_unit": "pt",
    "image_tick_width": 2,
    "image_tick_width_unit": "pt",
    "image_numbins": 35,
}


def get_preset(name: str) -> OutputPreset:
    """Get a preset by name."""
    return PRESETS.get(name.lower())


def list_presets() -> list:
    """List all available preset names."""
    return list(PRESETS.keys())


def get_preset_choices() -> list:
    """Get list of (name, display_name, description) tuples for UI."""
    return [
        (key, preset.name, preset.description)
        for key, preset in PRESETS.items()
    ]
