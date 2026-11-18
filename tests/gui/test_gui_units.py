"""
Global and per-parameter units: which selectors follow a global change, and
which `_unit` override keys reach the parameter dictionary as a result.
"""

from __future__ import annotations

import pytest

from dice_gui.global_units_dialog import GlobalUnitsDialog
from dice_gui.tabs.base import combo_value, set_combo_value
from gui_support import choose_global_units, enter_text, trigger_menu_action

pytestmark = pytest.mark.gui

LENGTH_SELECTORS = [
    "diffusion_length_unit_combo", "diffusion_coeff_length_unit_combo",
    "mean_unit_combo", "width_unit_combo", "spatial_width_unit_combo",
]
TIME_SELECTORS = [
    "diffusion_coeff_time_unit_combo", "lifetime_unit_combo",
    "time_start_unit_combo", "time_stop_unit_combo",
]


def _selected_units(window, selector_names: list[str]) -> dict[str, str]:
    return {name: combo_value(getattr(window, name)) for name in selector_names}


def _override_keys(window) -> dict[str, str]:
    return {key: value for key, value in window.collect_parameters().items() if key.endswith("_unit")
            and key not in ("length_unit", "time_unit")}


class TestGlobalUnits:

    def test_global_change_reaches_every_unmodified_selector(self, window, monkeypatch):
        choose_global_units(window, monkeypatch, "nanometer", "picosecond")

        assert set(_selected_units(window, LENGTH_SELECTORS).values()) == {"nanometer"}
        assert set(_selected_units(window, TIME_SELECTORS).values()) == {"picosecond"}
        params = window.collect_parameters()
        assert (params["length_unit"], params["time_unit"]) == ("nanometer", "picosecond")
        assert _override_keys(window) == {}

    def test_cancelled_dialog_changes_nothing(self, window, monkeypatch):
        def cancel(dialog: GlobalUnitsDialog) -> int:
            dialog.length_combo.setCurrentText("meter")
            return int(GlobalUnitsDialog.DialogCode.Rejected)

        monkeypatch.setattr(GlobalUnitsDialog, "exec", cancel)
        trigger_menu_action(window, "&Settings", "&Global Units...")

        assert window.length_unit_combo.currentText() == "micrometer"
        assert set(_selected_units(window, LENGTH_SELECTORS).values()) == {"micrometer"}

    def test_squared_selector_shows_squared_units_and_reports_the_base_unit(self, window, monkeypatch):
        choose_global_units(window, monkeypatch, "nanometer", "nanosecond")
        assert window.diffusion_coeff_length_unit_combo.currentText() == "nanometer²"
        assert combo_value(window.diffusion_coeff_length_unit_combo) == "nanometer"


class TestPerParameterUnits:

    def test_user_selection_survives_a_global_change(self, window, monkeypatch):
        set_combo_value(window.width_unit_combo, "angstrom")
        choose_global_units(window, monkeypatch, "nanometer", "nanosecond")

        assert combo_value(window.width_unit_combo) == "angstrom"
        others = set(LENGTH_SELECTORS) - {"width_unit_combo"}
        assert set(_selected_units(window, sorted(others)).values()) == {"nanometer"}
        assert _override_keys(window) == {"fwhm_0_unit": "angstrom"}

    def test_override_key_appears_only_while_unit_differs_from_global(self, window, monkeypatch):
        set_combo_value(window.lifetime_unit_combo, "picosecond")
        assert _override_keys(window) == {"lifetime_unit": "picosecond"}

        choose_global_units(window, monkeypatch, "micrometer", "picosecond")
        assert _override_keys(window) == {}

    def test_each_selector_maps_to_its_engine_override_key(self, window):
        selector_to_key = {
            "mean_unit_combo": "mu_0_unit",
            "spatial_width_unit_combo": "spatial_width_unit",
            "diffusion_coeff_length_unit_combo": "diffusion_coefficient_length_unit",
            "diffusion_coeff_time_unit_combo": "diffusion_coefficient_time_unit",
            "lifetime_unit_combo": "lifetime_unit",
            "time_start_unit_combo": "time_start_unit",
            "time_stop_unit_combo": "time_stop_unit",
        }
        for selector, key in selector_to_key.items():
            unit = "nanometer" if selector in LENGTH_SELECTORS else "picosecond"
            set_combo_value(getattr(window, selector), unit)
            assert _override_keys(window)[key] == unit, selector

    def test_width_override_key_follows_the_width_type(self, window):
        set_combo_value(window.width_unit_combo, "nanometer")
        assert _override_keys(window) == {"fwhm_0_unit": "nanometer"}
        window.sigma_radio.setChecked(True)
        assert _override_keys(window) == {"sigma_0_unit": "nanometer"}

    def test_reset_makes_every_selector_follow_global_again(self, window):
        set_combo_value(window.width_unit_combo, "angstrom")
        set_combo_value(window.lifetime_unit_combo, "femtosecond")
        window.reset_per_param_units()

        assert _override_keys(window) == {}
        assert set(_selected_units(window, LENGTH_SELECTORS).values()) == {"micrometer"}
        assert set(_selected_units(window, TIME_SELECTORS).values()) == {"nanosecond"}


class TestUnitAwareDisplays:

    def test_linked_diffusion_fields_convert_between_units(self, qtbot, window):
        set_combo_value(window.diffusion_length_unit_combo, "nanometer")
        enter_text(qtbot, window.diffusion_length_input, "100")
        enter_text(qtbot, window.diffusion_coeff_input, "0.01")

        # L = 100 nm = 0.1 micrometer, D = 0.01 micrometer^2/ns, so tau = L^2 / D = 1 ns.
        assert float(window.lifetime_input.text()) == pytest.approx(1.0)

    def test_conversion_labels_use_the_selected_unit(self, window):
        set_combo_value(window.width_unit_combo, "nanometer")
        set_combo_value(window.spatial_width_unit_combo, "nanometer")
        assert window.width_conversion_label.text().endswith(" nm")
        assert window.pixel_size_label.text().endswith(" nm/pixel")
