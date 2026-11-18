"""
Output Settings tab: presets, reset, unit conversion of image settings, the
output path preview, and the settings reaching the saved histogram.
"""

from __future__ import annotations

import matplotlib.image
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QPushButton

from dice_gui.presets import OUTPUT_DEFAULTS, PRESETS
from gui_support import enter_text

pytestmark = pytest.mark.gui

WIDGET_FOR_SETTING = {
    "image_type": "image_type_combo",
    "image_width": "image_width_spin",
    "image_width_unit": "image_width_unit_combo",
    "image_height": "image_height_spin",
    "image_height_unit": "image_height_unit_combo",
    "image_dpi": "image_dpi_spin",
    "image_numbins": "image_numbins_spin",
    "image_font_size": "image_font_size_spin",
    "image_font_unit": "image_font_unit_combo",
    "image_tick_length": "image_tick_length_spin",
    "image_tick_width": "image_tick_width_spin",
}


def _widget_values(window, settings: dict) -> dict:
    widgets = {key: getattr(window, WIDGET_FOR_SETTING[key]) for key in settings}
    return {
        key: widget.currentText() if isinstance(widget, QComboBox) else widget.value()
        for key, widget in widgets.items()
    }


def _choose_preset(window, key: str | None) -> None:
    window.preset_combo.setCurrentIndex(window.preset_combo.findData(key))


class TestPresets:

    @pytest.mark.parametrize("preset_key", sorted(PRESETS))
    def test_preset_sets_the_image_widgets(self, window, preset_key):
        settings = {key: value for key, value in PRESETS[preset_key].settings.items() if key in WIDGET_FOR_SETTING}
        _choose_preset(window, preset_key)
        assert _widget_values(window, settings) == settings

    def test_reset_restores_defaults_and_custom_preset(self, qtbot, window):
        _choose_preset(window, "presentation")
        window.image_width_unit_combo.setCurrentText("in")
        window.image_font_unit_combo.setCurrentText("px")

        reset_button = next(b for b in window.tab5.findChildren(QPushButton) if b.text() == "Reset to Defaults")
        qtbot.mouseClick(reset_button, Qt.MouseButton.LeftButton)

        defaults = {key: value for key, value in OUTPUT_DEFAULTS.items() if key in WIDGET_FOR_SETTING}
        assert _widget_values(window, defaults) == defaults
        assert window.preset_combo.currentText() == "Custom"


class TestImageSettingConversion:

    @pytest.fixture
    def mixed_unit_window(self, window):
        window.image_width_spin.setValue(2.0)
        window.image_width_unit_combo.setCurrentText("in")
        window.image_height_spin.setValue(50.0)
        window.image_height_unit_combo.setCurrentText("mm")
        window.image_font_size_spin.setValue(12)
        window.image_font_unit_combo.setCurrentText("px")
        window.image_tick_length_spin.setValue(8)
        window.image_tick_length_unit_combo.setCurrentText("px")
        window.image_tick_width_spin.setValue(4)
        window.image_tick_width_unit_combo.setCurrentText("px")
        return window

    def test_sizes_are_converted_to_centimeters_and_points(self, mixed_unit_window):
        settings = mixed_unit_window.collect_image_settings()
        assert settings["image_width"] == pytest.approx(5.08)
        assert settings["image_height"] == pytest.approx(5.0)
        assert settings["image_font_size"] == pytest.approx(9.0)
        assert settings["image_tick_length"] == pytest.approx(6.0)
        assert settings["image_tick_width"] == pytest.approx(3.0)

    def test_run_parameters_carry_the_same_converted_values(self, mixed_unit_window):
        settings = mixed_unit_window.collect_image_settings()
        params = mixed_unit_window.collect_parameters()
        assert {key: params[key] for key in settings} == settings


class TestOutputPathPreview:

    def test_preview_follows_slug_and_image_type(self, qtbot, window, work_dir):
        enter_text(qtbot, window.filename_slug_input, "my_run")
        window.image_type_combo.setCurrentText("tif")

        preview = window.output_path_preview.text()
        assert f"<b>Directory:</b> {work_dir / 'output' / 'my_run'}<br>" in preview
        assert preview.endswith("my_run.csv, my_run_summary.txt, my_run_accuracy_histogram.tif")

    def test_preview_asks_for_a_slug_when_empty(self, qtbot, window):
        enter_text(qtbot, window.filename_slug_input, "")
        assert window.output_path_preview.text() == "Enter a filename slug to see output paths"


class TestSettingsReachThePlot:

    def test_regenerated_png_has_the_requested_pixel_size(self, window, completed_run, work_dir):
        window.simulation_finished(completed_run.as_thread_result())
        window.image_width_spin.setValue(4.0)
        window.image_width_unit_combo.setCurrentText("in")
        window.image_height_spin.setValue(2.0)
        window.image_height_unit_combo.setCurrentText("in")
        window.image_dpi_spin.setValue(100)

        window.regenerate_plot_button.click()

        pixels = matplotlib.image.imread(work_dir / "output" / "quick_test" / "quick_test_accuracy_histogram.png")
        height, width = pixels.shape[:2]
        assert (width, height) == (pytest.approx(400, abs=1), pytest.approx(200, abs=1))
