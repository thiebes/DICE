"""
Validation feedback: invalid input must be visible where the user looks
(inline label, field style, tab title, status bar, Run button) and must
block a run before any simulation thread starts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QFocusEvent
from PyQt6.QtWidgets import QApplication, QMessageBox

from gui_support import EXAMPLE_PROFILE_CSV, TAB_NAMES, enter_text, run_and_wait

pytestmark = pytest.mark.gui

WARNING_MARK = " ⚠"


def _visible_error(window, field_id: str) -> str | None:
    label = window._error_labels[field_id]
    return None if label.isHidden() else label.text()


def _leave_text_edit(widget) -> None:
    """QTextEdit validates on focus loss; deliver that event directly."""
    QApplication.sendEvent(widget, QFocusEvent(QEvent.Type.FocusOut))


def _click_run(qtbot, window) -> None:
    qtbot.mouseClick(window.run_button, Qt.MouseButton.LeftButton)


class TestInlineFeedback:

    def test_invalid_amplitude_is_flagged_everywhere_the_user_looks(self, qtbot, window):
        enter_text(qtbot, window.amplitude_input, "abc")

        assert _visible_error(window, "amplitude") == "Amplitude must be a valid number"
        assert window.amplitude_input.property("validation-state") == "invalid"
        assert window.amplitude_input.toolTip().startswith("⚠ Amplitude must be a valid number")
        assert window.tabs.tabText(1) == TAB_NAMES[1] + WARNING_MARK
        assert window.validation_status_label.text() == "1 validation error"
        assert not window.run_button.isEnabled()
        assert window.status_label.text() == "Please fix validation errors before running"

    def test_correcting_the_value_clears_feedback_while_typing(self, qtbot, window):
        original_tooltip = window.amplitude_input.toolTip()
        enter_text(qtbot, window.amplitude_input, "abc")

        window.amplitude_input.clear()
        qtbot.keyClicks(window.amplitude_input, "2")

        assert _visible_error(window, "amplitude") is None
        assert window.amplitude_input.property("validation-state") == "valid"
        assert window.amplitude_input.toolTip() == original_tooltip
        assert window.tabs.tabText(1) == TAB_NAMES[1]
        assert window.validation_status_label.text() == ""
        assert window.run_button.isEnabled()
        assert window.status_label.text() == "Ready to run"

    def test_errors_on_two_tabs_are_counted_and_both_tabs_flagged(self, qtbot, window):
        enter_text(qtbot, window.filename_slug_input, "a/b")
        enter_text(qtbot, window.spatial_width_input, "-5")

        assert _visible_error(window, "filename_slug").startswith("Filename slug cannot contain:")
        assert _visible_error(window, "spatial_width") == "Spatial width must be greater than 0"
        assert window.validation_status_label.text() == "2 validation errors"
        assert window.tabs.tabText(0) == TAB_NAMES[0] + WARNING_MARK
        assert window.tabs.tabText(2) == TAB_NAMES[2] + WARNING_MARK
        assert window.tabs.tabText(1) == TAB_NAMES[1]

    @pytest.mark.parametrize(("field_id", "widget_name", "text", "message"), [
        ("filename_slug", "filename_slug_input", "", "Filename slug is required"),
        ("mean", "mean_input", "left", "Mean position must be a valid number"),
        ("width", "width_input", "0", "Profile width must be greater than 0"),
        ("lifetime", "lifetime_input", "-1", "Lifetime"),
        ("noise_value", "noise_value_input", "-0.1", "Noise value"),
        ("time_start", "time_start_input", "soon", "Start time must be a valid number"),
    ])
    def test_field_messages(self, qtbot, window, field_id, widget_name, text, message):
        enter_text(qtbot, getattr(window, widget_name), text)
        assert _visible_error(window, field_id).startswith(message)

    def test_time_series_is_checked_when_focus_leaves(self, qtbot, window):
        window.time_series_radio.setChecked(True)
        window.time_series_input.setPlainText("0, 1, x")
        _leave_text_edit(window.time_series_input)

        assert _visible_error(window, "time_series") == "Time series must be comma-separated numbers"
        assert window.tabs.tabText(2) == TAB_NAMES[2] + WARNING_MARK

    def test_noise_file_must_exist_when_estimating_noise(self, qtbot, window, work_dir):
        window.noise_estimate_radio.setChecked(True)
        enter_text(qtbot, window.noise_file_input, str(work_dir / "absent.csv"))

        assert _visible_error(window, "noise_file").startswith("File")
        assert not window.run_button.isEnabled()

    def test_inactive_field_drops_its_error_everywhere_when_mode_changes(self, qtbot, window):
        enter_text(qtbot, window.noise_value_input, "-1")
        assert window.validation_status_label.text() == "1 validation error"

        window.noise_estimate_radio.setChecked(True)

        assert _visible_error(window, "noise_value") is None
        assert window.validation_status_label.text() == ""
        assert window.tabs.tabText(2) == TAB_NAMES[2]
        assert window.run_button.isEnabled()

    def test_kept_invalid_value_is_flagged_again_when_its_mode_returns(self, qtbot, window):
        enter_text(qtbot, window.noise_value_input, "-1")
        window.noise_estimate_radio.setChecked(True)

        window.noise_fixed_radio.setChecked(True)

        assert window.noise_value_input.text() == "-1"
        assert _visible_error(window, "noise_value").startswith("Noise value")
        assert window.validation_status_label.text() == "1 validation error"
        assert window.tabs.tabText(2) == TAB_NAMES[2] + WARNING_MARK
        assert not window.run_button.isEnabled()

    def test_kept_invalid_time_series_is_flagged_again_when_its_mode_returns(self, qtbot, window):
        window.time_series_radio.setChecked(True)
        window.time_series_input.setPlainText("0, 1, x")
        _leave_text_edit(window.time_series_input)
        window.time_range_radio.setChecked(True)
        assert window.run_button.isEnabled()

        window.time_series_radio.setChecked(True)

        assert _visible_error(window, "time_series") == "Time series must be comma-separated numbers"
        assert not window.run_button.isEnabled()

    def test_empty_inputs_of_a_returning_mode_are_not_flagged_before_the_user_visits_them(self, qtbot, window):
        window.noise_estimate_radio.setChecked(True)
        window.noise_fixed_radio.setChecked(True)
        window.noise_estimate_radio.setChecked(True)

        assert _visible_error(window, "noise_file") is None
        assert window.validation_status_label.text() == ""


class TestRunIsBlocked:

    def test_invalid_field_blocks_run_with_a_warning(self, qtbot, window, dialogs):
        window.amplitude_input.setText("abc")
        _click_run(qtbot, window)

        assert dialogs.titles("warning") == ["Missing or Invalid Parameters"]
        assert window.simulation_thread is None
        assert _visible_error(window, "amplitude") == "Amplitude must be a valid number"
        assert window.progress_bar.isHidden()

    def test_missing_noise_file_blocks_run(self, qtbot, window, dialogs):
        window.noise_estimate_radio.setChecked(True)
        _click_run(qtbot, window)

        assert dialogs.titles("warning") == ["Missing or Invalid Parameters"]
        assert _visible_error(window, "noise_file") == "File path is required"
        assert window.simulation_thread is None

    def test_reversed_time_range_blocks_run(self, qtbot, window, dialogs):
        enter_text(qtbot, window.time_start_input, "2")
        enter_text(qtbot, window.time_stop_input, "1")
        _click_run(qtbot, window)

        assert dialogs.titles("warning") == ["Validation Error"]
        assert dialogs.texts("warning")[0] == "Time range start must be less than stop, got 2.0 >= 1.0"
        assert window.simulation_thread is None


class TestZeroDiffusionInputs:
    """
    Lifetime 0 means no decay, which the engine supports; the derived diffusion
    length is then infinite, shows as "inf", and must not block the run. D = 0
    has no accuracy ratio (estimates are divided by D), so it must be rejected
    on the D field itself.
    """

    def test_zero_lifetime_runs(self, qtbot, window, dialogs, work_dir):
        window.num_runs_spin.setValue(3)
        window.multiprocessing_check.setChecked(False)
        enter_text(qtbot, window.filename_slug_input, "no_decay")
        enter_text(qtbot, window.lifetime_input, "0")
        assert window.diffusion_length_input.text() == "inf"

        run_and_wait(qtbot, window, dialogs)

        assert _visible_error(window, "diffusion_length") is None
        frame = pd.read_csv(work_dir / "output" / "no_decay" / "no_decay.csv")
        assert (frame["nominal lifetime [ns]"] == 0).all()
        assert (frame["nominal diffusion length [μm]"] == float("inf")).all()
        summary = (work_dir / "output" / "no_decay" / "no_decay_summary.txt").read_text(encoding="utf-8")
        assert "Nominal diffusion length:\n    inf micrometer\n" in summary

    def test_infinite_length_with_a_coefficient_gives_zero_lifetime(self, qtbot, window):
        enter_text(qtbot, window.diffusion_coeff_input, "0.01")
        enter_text(qtbot, window.diffusion_length_input, "inf")

        assert window.lifetime_input.text() == "0"
        assert _visible_error(window, "diffusion_length") is None
        assert _visible_error(window, "lifetime") is None

    def test_zero_diffusion_coefficient_is_rejected_on_its_own_field(self, qtbot, window, dialogs):
        enter_text(qtbot, window.diffusion_coeff_input, "0")
        _click_run(qtbot, window)

        assert not window.run_button.isEnabled()
        assert _visible_error(window, "diffusion_coeff") == "Diffusion coefficient must be greater than 0"
        assert _visible_error(window, "diffusion_length") is None
        assert window.simulation_thread is None


class TestConsistencyWarnings:
    """Valid but risky inputs ask before running; answering No starts nothing."""

    @pytest.mark.parametrize(("widget_name", "value", "expected"), [
        ("noise_value_input", "0.5", "⚠ Low final CNR: At t = 2 ns"),
        ("width_input", "0.1", "⚠ Undersampled profile: σ₀ = 0.04247 μm"),
        ("time_steps_input", 3, "ℹ Few time points: Only 3 time point(s)"),
    ])
    def test_warning_is_shown_and_no_aborts(self, qtbot, window, dialogs, widget_name, value, expected):
        widget = getattr(window, widget_name)
        if isinstance(value, int):
            widget.setValue(value)
        else:
            enter_text(qtbot, widget, value)
        dialogs.question_answers.append(QMessageBox.StandardButton.No)

        _click_run(qtbot, window)

        assert dialogs.titles("question") == ["Parameter Warnings"]
        assert expected in dialogs.texts("question")[0]
        assert dialogs.texts("question")[0].endswith("Proceed with simulation?")
        assert window.simulation_thread is None
        assert window.run_button.isEnabled()
        assert window.status_label.text() == "Ready to run"

    def test_final_cnr_counts_the_spreading_of_the_profile(self, qtbot, window, dialogs):
        """
        The peak falls by the decay and by sigma_0 / sigma_t. Here the decay
        alone leaves a CNR of 7.4, above the limit of 5; with the spreading
        the CNR is 4.1.
        """
        enter_text(qtbot, window.amplitude_input, "1")
        enter_text(qtbot, window.width_input, "0.5")
        enter_text(qtbot, window.diffusion_coeff_input, "0.05")
        enter_text(qtbot, window.lifetime_input, "1")
        enter_text(qtbot, window.time_stop_input, "1")
        enter_text(qtbot, window.noise_value_input, "0.05")
        dialogs.question_answers.append(QMessageBox.StandardButton.No)

        _click_run(qtbot, window)

        sigma_0 = 0.5 / (2 * np.sqrt(2 * np.log(2)))
        peak = np.exp(-1) * sigma_0 / np.sqrt(sigma_0 ** 2 + 2 * 0.05 * 1)
        assert np.exp(-1) / 0.05 > 5 > peak / 0.05
        assert dialogs.titles("question") == ["Parameter Warnings"]
        assert (f"Low final CNR: At t = 1 ns, decay and spreading lower the peak to {peak:.4g}, "
                f"giving CNR = {peak / 0.05:.2f}.") in dialogs.texts("question")[0]

    def test_narrow_measured_peak_warns(self, qtbot, window, dialogs):
        """The repository profile's peak is 4.6 pixels wide, too narrow for fft-cnr's CNR estimate."""
        window.noise_estimate_radio.setChecked(True)
        enter_text(qtbot, window.noise_file_input, str(EXAMPLE_PROFILE_CSV))
        dialogs.question_answers.append(QMessageBox.StandardButton.No)

        _click_run(qtbot, window)

        assert dialogs.titles("question") == ["Parameter Warnings"]
        assert (f"⚠ Narrow measured peak: The peak of the measured profile "
                f"'{EXAMPLE_PROFILE_CSV}' is 4.6 pixels wide at half maximum.") in dialogs.texts("question")[0]
        assert window.simulation_thread is None

    def test_wide_measured_peak_does_not_warn(self, qtbot, window, dialogs, work_dir):
        from dice.core.profiles import gaussian
        from dice.utils.converters import fwhm_to_sigma2
        x = np.arange(101) - 50
        rng = np.random.default_rng(5)
        wide = 1000 * gaussian(x, 0.3, fwhm_to_sigma2(15.0), 1.0) + rng.normal(0, 50, x.size)
        profile_file = work_dir / "wide_profile.csv"
        np.savetxt(profile_file, wide.reshape(1, -1), delimiter=",", fmt="%.17g")
        window.noise_estimate_radio.setChecked(True)
        enter_text(qtbot, window.noise_file_input, str(profile_file))

        assert not any("Narrow measured peak" in message
                       for message in window._check_parameter_consistency(window.collect_parameters()))

    def test_short_time_series_warns(self, qtbot, window, dialogs):
        window.time_series_radio.click()
        window.time_series_input.setPlainText("0, 2")
        dialogs.question_answers.append(QMessageBox.StandardButton.No)

        _click_run(qtbot, window)

        assert "Only 2 time point(s)" in dialogs.texts("question")[0]
        assert window.simulation_thread is None
