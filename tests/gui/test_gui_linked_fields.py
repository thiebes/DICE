"""
Linked and derived fields on the Physical Parameters and Experimental
Conditions tabs: L^2 = D * tau, CNR = amplitude / noise sigma, FWHM and
sigma equivalents, pixel size, and the mode radio buttons.
"""

from __future__ import annotations

import pytest

from dice.utils.units import length_abbreviation
from gui_support import EXAMPLE_PROFILE_CSV, enter_text, float_text

pytestmark = pytest.mark.gui

MICROMETER = length_abbreviation("micrometer")


class TestDiffusionTriangle:
    """Any two of L, D, and tau determine the third; the two most recently edited win."""

    def test_length_follows_coefficient_and_lifetime(self, qtbot, window):
        enter_text(qtbot, window.diffusion_coeff_input, "0.04")
        enter_text(qtbot, window.lifetime_input, "4")
        assert float_text(window.diffusion_length_input) == pytest.approx(0.4)

    def test_lifetime_follows_length_and_coefficient(self, qtbot, window):
        enter_text(qtbot, window.diffusion_length_input, "0.2")
        enter_text(qtbot, window.diffusion_coeff_input, "0.01")
        assert float_text(window.lifetime_input) == pytest.approx(4.0)

    def test_coefficient_follows_length_and_lifetime(self, qtbot, window):
        enter_text(qtbot, window.diffusion_length_input, "0.3")
        enter_text(qtbot, window.lifetime_input, "3")
        assert float_text(window.diffusion_coeff_input) == pytest.approx(0.03)

    def test_clearing_an_input_clears_the_derived_field(self, qtbot, window):
        window.lifetime_input.clear()
        assert window.diffusion_length_input.text() == ""

    def test_engine_receives_coefficient_and_lifetime_not_length(self, qtbot, window):
        enter_text(qtbot, window.diffusion_length_input, "0.3")
        enter_text(qtbot, window.lifetime_input, "3")
        params = window.collect_parameters()
        assert (params["diffusion_coefficient"], params["lifetime"]) == (pytest.approx(0.03), 3.0)
        assert "diffusion_length" not in params


class TestProfileWidth:

    def test_fwhm_shows_the_equivalent_sigma(self, qtbot, window):
        enter_text(qtbot, window.width_input, "2.3548")
        assert window.width_conversion_label.text() == f"Equivalent: σ = 1 {MICROMETER}"

    def test_sigma_shows_the_equivalent_fwhm(self, qtbot, window):
        window.sigma_radio.click()
        enter_text(qtbot, window.width_input, "1")
        assert window.width_conversion_label.text() == f"Equivalent: FWHM = 2.355 {MICROMETER}"

    def test_invalid_width_shows_no_equivalent(self, qtbot, window):
        enter_text(qtbot, window.width_input, "wide")
        assert window.width_conversion_label.text() == "Equivalent: ---"


class TestNoiseAndContrast:

    def test_noise_sigma_sets_cnr(self, qtbot, window):
        enter_text(qtbot, window.noise_value_input, "0.04")
        assert window.noise_cnr_input.text() == "25"

    def test_cnr_sets_noise_sigma(self, qtbot, window):
        enter_text(qtbot, window.noise_cnr_input, "50")
        assert window.noise_value_input.text() == "0.02"
        assert window.collect_parameters()["noise_value"] == pytest.approx(0.02)

    def test_amplitude_change_keeps_the_last_edited_quantity(self, qtbot, window):
        enter_text(qtbot, window.noise_cnr_input, "50")
        enter_text(qtbot, window.amplitude_input, "2")
        assert window.noise_cnr_input.text() == "50"
        assert window.noise_value_input.text() == "0.04"

        enter_text(qtbot, window.noise_value_input, "0.1")
        enter_text(qtbot, window.amplitude_input, "3")
        assert window.noise_value_input.text() == "0.1"
        assert window.noise_cnr_input.text() == "30"


class TestSpatialSampling:

    def test_pixel_size_is_the_spacing_of_the_simulation_axis(self, qtbot, window):
        enter_text(qtbot, window.spatial_width_input, "5")
        window.pixel_width_input.setValue(51)
        assert window.pixel_size_label.text() == f"Pixel Size: 0.1 {MICROMETER}/pixel"

    def test_single_pixel_shows_no_pixel_size(self, qtbot, window):
        enter_text(qtbot, window.spatial_width_input, "5")
        window.pixel_width_input.setValue(1)
        assert window.pixel_size_label.text() == "Pixel Size: ---"

    def test_invalid_width_shows_no_pixel_size(self, qtbot, window):
        enter_text(qtbot, window.spatial_width_input, "-1")
        assert window.pixel_size_label.text() == "Pixel Size: ---"


class TestModeSwitches:

    def test_noise_mode_switch_enables_one_input_set_and_clears_the_other(self, qtbot, window):
        window.noise_estimate_radio.click()
        assert window.noise_file_input.isEnabled() and window.noise_browse_button.isEnabled()
        assert not window.noise_value_input.isEnabled() and not window.noise_cnr_input.isEnabled()
        assert (window.noise_value_input.text(), window.noise_cnr_input.text()) == ("", "")

        enter_text(qtbot, window.noise_file_input, str(EXAMPLE_PROFILE_CSV))
        window.noise_fixed_radio.click()
        assert window.noise_value_input.isEnabled()
        assert not window.noise_file_input.isEnabled()
        assert window.noise_file_input.text() == ""

    def test_browse_fills_the_noise_file(self, qtbot, window, dialogs):
        window.noise_estimate_radio.click()
        dialogs.open_paths.append(str(EXAMPLE_PROFILE_CSV))

        window.noise_browse_button.click()

        assert dialogs.titles("open-file") == ["Select Noise Data File"]
        assert window.noise_file_input.text() == str(EXAMPLE_PROFILE_CSV)

    def test_time_mode_switch_enables_one_input_set_and_clears_the_other(self, qtbot, window):
        window.time_series_radio.click()
        assert window.time_series_input.isEnabled()
        assert not any(w.isEnabled() for w in (window.time_start_input, window.time_stop_input, window.time_steps_input))
        assert (window.time_start_input.text(), window.time_stop_input.text()) == ("", "")

        window.time_series_input.setPlainText("0, 1, 2")
        window.time_range_radio.click()
        assert window.time_start_input.isEnabled()
        assert window.time_series_input.toPlainText() == ""
