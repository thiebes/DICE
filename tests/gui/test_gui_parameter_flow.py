"""
Data flow from widgets to the engine.

Each boundary is checked against the engine's schema: the GUI dictionary that
collect_parameters() produces, the engine dictionary that
DiceInterface.build_parameters_dict() produces (keys known to
dice.models.parameter_keys), the unit overrides that resolve_units() must
consume, and the values parameter_parser() derives from them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from fft_cnr import fft_cnr

from dice.io.parameters import parameter_parser
from dice.models.parameter_keys import ParameterKey, normalize_parameter_key
from dice.utils.units import resolve_units
from dice_gui.dice_interface import DiceInterface
from dice_gui.tabs.base import set_combo_value
from gui_support import EXAMPLE_PROFILE_CSV, enter_text

pytestmark = pytest.mark.gui

ENGINE_CANONICAL_KEYS = {
    value for name, value in vars(ParameterKey).items() if not name.startswith("_")
}

GUI_KEYS_ALWAYS = {
    "number_of_runs", "filename_slug", "length_unit", "time_unit",
    "amplitude_0", "mean_0", "profile_width_type", "profile_width_value",
    "spatial_width", "pixel_width", "multiprocessing", "retain_profile_data",
    "image_type", "image_width", "image_height", "image_dpi", "image_font_size",
    "image_tick_length", "image_tick_width", "image_numbins",
    "diffusion_coefficient", "lifetime", "noise_type", "time_type",
}

ENGINE_KEYS_DEFAULT = {
    "number of runs", "filename slug", "length unit", "time unit",
    "amplitude_0", "mean_0", "FWHM_0",
    "nominal diffusion coefficient", "nominal lifetime (tau)",
    "noise value", "spatial width", "pixel width", "time range",
    "image type", "image width", "image height", "image dpi", "image font size",
    "image tick length", "image tick width", "image numbins",
    "retain profile data", "multiprocessing",
}

FWHM_TO_SIGMA = 1 / (2 * math.sqrt(2 * math.log(2)))


def _engine_parameters(window) -> dict:
    return DiceInterface().build_parameters_dict(window.collect_parameters())


def _choose_unit(combo, unit: str) -> None:
    """Select a unit in a per-parameter combo the way a user's selection arrives."""
    set_combo_value(combo, unit)


class TestGuiDictionarySchema:

    def test_default_widgets_produce_the_documented_gui_keys(self, window):
        gui_params = window.collect_parameters()
        assert set(gui_params) == GUI_KEYS_ALWAYS | {"noise_value", "time_start", "time_stop", "time_steps"}
        assert gui_params["profile_width_type"] == "fwhm"
        assert (gui_params["noise_type"], gui_params["time_type"]) == ("fixed", "range")

    def test_alternative_modes_swap_their_keys(self, qtbot, window):
        window.sigma_radio.setChecked(True)
        window.noise_estimate_radio.setChecked(True)
        window.noise_file_input.setText(str(EXAMPLE_PROFILE_CSV))
        window.time_series_radio.setChecked(True)
        window.time_series_input.setPlainText("0, 0.5, 1")

        gui_params = window.collect_parameters()

        assert set(gui_params) == GUI_KEYS_ALWAYS | {"noise_data_file", "time_series"}
        assert gui_params["profile_width_type"] == "sigma"
        assert (gui_params["noise_type"], gui_params["time_type"]) == ("estimate", "series")


class TestEngineDictionarySchema:

    def test_default_widgets_produce_engine_keys(self, window):
        assert set(_engine_parameters(window)) == ENGINE_KEYS_DEFAULT

    def test_every_engine_key_is_known_to_the_engine(self, qtbot, window):
        window.sigma_radio.setChecked(True)
        window.time_series_radio.setChecked(True)
        window.time_series_input.setPlainText("0, 1, 2")
        _choose_unit(window.width_unit_combo, "nanometer")

        unknown = {
            key for key in _engine_parameters(window)
            if normalize_parameter_key(key) not in ENGINE_CANONICAL_KEYS
        }
        assert unknown == set()

    def test_engine_values_match_the_widgets(self, window):
        params = _engine_parameters(window)
        assert params["number of runs"] == window.num_runs_spin.value()
        assert params["filename slug"] == window.filename_slug_input.text()
        assert params["FWHM_0"] == float(window.width_input.text())
        assert params["nominal diffusion coefficient"] == float(window.diffusion_coeff_input.text())
        assert params["nominal lifetime (tau)"] == float(window.lifetime_input.text())
        assert params["noise value"] == float(window.noise_value_input.text())
        assert params["spatial width"] == float(window.spatial_width_input.text())
        assert params["pixel width"] == window.pixel_width_input.value()
        assert params["time range"] == [
            float(window.time_start_input.text()),
            float(window.time_stop_input.text()),
            window.time_steps_input.value(),
        ]
        assert params["multiprocessing"] is window.multiprocessing_check.isChecked()

    def test_parser_derives_the_simulation_inputs(self, window):
        parsed = parameter_parser(resolve_units(_engine_parameters(window)))

        assert len(parsed["x array"]) == 101
        assert parsed["x array"].max() - parsed["x array"].min() == pytest.approx(10.0)
        assert list(parsed["time series"]) == pytest.approx(list(np.linspace(0, 2, 10)))
        assert parsed["noise series"] == [0.05]
        assert parsed["sigma^2_0"] == pytest.approx(FWHM_TO_SIGMA ** 2)
        assert parsed["nominal diffusion length"] == pytest.approx(math.sqrt(0.01 * 2.0))


class TestUnitOverridesReachTheEngine:

    def _enter_mixed_unit_inputs(self, qtbot, window) -> None:
        """Physics equal to FWHM 0.5, mean 0.1, width 5 (micrometer); D 0.1 per ns; tau 2, t 0..1 (ns)."""
        _choose_unit(window.width_unit_combo, "nanometer")
        _choose_unit(window.mean_unit_combo, "nanometer")
        _choose_unit(window.spatial_width_unit_combo, "nanometer")
        _choose_unit(window.diffusion_coeff_length_unit_combo, "centimeter")
        _choose_unit(window.diffusion_coeff_time_unit_combo, "second")
        _choose_unit(window.lifetime_unit_combo, "picosecond")
        _choose_unit(window.time_start_unit_combo, "picosecond")
        _choose_unit(window.time_stop_unit_combo, "picosecond")
        enter_text(qtbot, window.width_input, "500")
        enter_text(qtbot, window.mean_input, "100")
        enter_text(qtbot, window.spatial_width_input, "5000")
        enter_text(qtbot, window.diffusion_coeff_input, "1")
        enter_text(qtbot, window.lifetime_input, "2000")
        enter_text(qtbot, window.time_start_input, "0")
        enter_text(qtbot, window.time_stop_input, "1000")
        window.time_steps_input.setValue(5)

    def test_engine_dictionary_carries_one_known_override_key_per_changed_unit(self, qtbot, window):
        self._enter_mixed_unit_inputs(qtbot, window)
        params = _engine_parameters(window)

        unit_keys = {key for key in params if key.endswith("_unit")}
        assert unit_keys == {
            "fwhm_0_unit", "mu_0_unit", "spatial_width_unit",
            "diffusion_coefficient_length_unit", "diffusion_coefficient_time_unit",
            "lifetime_unit", "time_start_unit", "time_stop_unit",
        }
        assert {normalize_parameter_key(key) for key in unit_keys} <= ENGINE_CANONICAL_KEYS
        assert params["fwhm_0_unit"] == "nanometer"
        assert params["diffusion_coefficient_length_unit"] == "centimeter"
        assert params["lifetime_unit"] == "picosecond"

    def test_resolve_units_consumes_every_override(self, qtbot, window):
        self._enter_mixed_unit_inputs(qtbot, window)
        resolved = resolve_units(_engine_parameters(window))
        assert [key for key in resolved if key.endswith("_unit")] == []

    def test_parsed_values_are_converted_to_global_units(self, qtbot, window):
        self._enter_mixed_unit_inputs(qtbot, window)
        parsed = parameter_parser(resolve_units(_engine_parameters(window)))

        assert parsed["sigma^2_0"] == pytest.approx((0.5 * FWHM_TO_SIGMA) ** 2, rel=1e-12)
        assert parsed["mu_0"] == pytest.approx(0.1, rel=1e-12)
        assert parsed["spatial width"] == pytest.approx(5.0, rel=1e-12)
        assert parsed["nominal diffusion coefficient"] == pytest.approx(0.1, rel=1e-12)
        assert parsed["nominal lifetime (tau)"] == pytest.approx(2.0, rel=1e-12)
        assert list(parsed["time series"]) == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0], rel=1e-12)

    def test_sigma_width_override_uses_the_sigma_key(self, qtbot, window):
        window.sigma_radio.setChecked(True)
        _choose_unit(window.width_unit_combo, "nanometer")
        enter_text(qtbot, window.width_input, "400")

        params = _engine_parameters(window)

        assert (params["sigma_0"], params["sigma_0_unit"]) == (400.0, "nanometer")
        assert "FWHM_0" not in params and "fwhm_0_unit" not in params
        assert parameter_parser(resolve_units(params))["sigma^2_0"] == pytest.approx(0.4 ** 2)


class TestAlternativeInputModes:

    def test_time_series_is_sorted_and_parsed(self, window):
        window.time_series_radio.setChecked(True)
        window.time_series_input.setPlainText("2, 0.5, 1")

        params = _engine_parameters(window)

        assert params["time series"] == [0.5, 1.0, 2.0]
        assert "time range" not in params
        assert list(parameter_parser(resolve_units(params))["time series"]) == [0.5, 1.0, 2.0]

    def test_noise_estimated_from_a_profile_file(self, window):
        window.noise_estimate_radio.setChecked(True)
        window.noise_file_input.setText(str(EXAMPLE_PROFILE_CSV))

        params = _engine_parameters(window)

        assert params["estimate noise from data"] == str(EXAMPLE_PROFILE_CSV)
        assert "noise value" not in params
        profile = np.loadtxt(EXAMPLE_PROFILE_CSV, delimiter=",").tolist()
        with pytest.warns(UserWarning, match="4.6 pixels wide at half maximum"):
            parsed = parameter_parser(resolve_units(params))
        # The noise is set so that the simulated initial CNR equals the measured one.
        expected_noise = parsed["amplitude_0"] / fft_cnr(profile).cnr
        assert parsed["noise series"] == [pytest.approx(expected_noise)]


class TestMalformedInput:

    @pytest.mark.parametrize("missing_key", [
        "number_of_runs", "filename_slug", "amplitude_0", "mean_0", "profile_width_type",
        "profile_width_value", "diffusion_coefficient", "lifetime", "noise_type",
        "noise_value", "spatial_width", "pixel_width", "time_type", "time_start",
        "time_stop", "time_steps",
    ])
    def test_missing_gui_key_is_named_in_the_error(self, window, missing_key):
        gui_params = window.collect_parameters()
        del gui_params[missing_key]
        with pytest.raises(KeyError, match=missing_key):
            DiceInterface().build_parameters_dict(gui_params)

    @pytest.mark.parametrize(("series_text", "message"), [
        ("0, 1, 1", "duplicate"),
        ("0, one, 2", "comma-separated numbers"),
        (" , ", "at least two time points"),
        # The engine needs two time points; the form says so before the run starts.
        ("0.5", "at least two time points"),
    ])
    def test_malformed_time_series_is_rejected(self, window, series_text, message):
        window.time_series_radio.setChecked(True)
        gui_params = window.collect_parameters()
        gui_params["time_series"] = series_text
        with pytest.raises(ValueError, match=message):
            DiceInterface().build_parameters_dict(gui_params)

    @pytest.mark.parametrize(("change", "message"), [
        (lambda p: p.pop("pixel width"), "Missing required parameter: pixel width"),
        (lambda p: p.update({"sigma_0": 0.4}), "Cannot specify both FWHM_0 and sigma_0"),
        (lambda p: p.update({"nominal diffusion length": 0.1}), "Cannot specify both diffusion length"),
        (lambda p: p.update({"estimate noise from data": "x.csv"}), "Cannot specify both noise value"),
        (lambda p: p.update({"time series": [0, 1]}), "Cannot specify both time range and time series"),
        (lambda p: p.pop("FWHM_0"), "Must specify either FWHM_0 or sigma_0"),
        (lambda p: p.pop("nominal lifetime (tau)"), "Must specify either diffusion length"),
        (lambda p: p.pop("noise value"), "Must specify either noise value"),
        (lambda p: p.pop("time range"), "Must specify either time range or time series"),
    ])
    def test_interface_validation_names_the_problem(self, window, change, message):
        params = _engine_parameters(window)
        assert DiceInterface().validate_parameters(params) == (True, "")
        change(params)
        is_valid, error = DiceInterface().validate_parameters(params)
        assert not is_valid
        assert error.startswith(message)
