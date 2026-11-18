"""
Parameter files: loading the bundled examples, saving and reloading, the
unsaved-changes prompts, and how load errors are reported.
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtWidgets import QMessageBox

from dice.io.parameters import open_parameters, parameter_parser
from dice.utils.units import resolve_units
from dice_gui.dice_interface import DiceInterface
from dice_gui.tabs.base import set_combo_value
from gui_support import (
    EXAMPLE_PARAMETER_FILES, EXAMPLE_PROFILE_CSV, EXAMPLES_DIR,
    enter_text, read_example, trigger_menu_action,
)

pytestmark = pytest.mark.gui

DEFAULT_TITLE = "DICE - Diffusion Insight Computation Engine"


def _open_via_menu(window, dialogs, path) -> None:
    dialogs.open_paths.append(str(path))
    trigger_menu_action(window, "&File", "&Open Parameters...")


def _save_as_via_menu(window, dialogs, path) -> None:
    dialogs.save_paths.append(str(path))
    trigger_menu_action(window, "&File", "Save Parameters &As...")


def _engine_parameters(window) -> dict:
    return DiceInterface().build_parameters_dict(window.collect_parameters())


class TestLoadingExamples:

    @pytest.mark.parametrize("filename", EXAMPLE_PARAMETER_FILES)
    def test_example_populates_the_widgets(self, window, dialogs, filename):
        expected = read_example(filename)
        _open_via_menu(window, dialogs, EXAMPLES_DIR / filename)

        assert dialogs.titles("information") == ["Parameters Loaded"]
        assert window.windowTitle() == f"DICE - [{filename}]"
        assert window.parameters_modified is False
        assert window.filename_slug_input.text() == expected["filename slug"]
        assert window.num_runs_spin.value() == expected["number of runs"]
        assert window.fwhm_radio.isChecked()
        assert float(window.width_input.text()) == expected["FWHM_0"]
        assert float(window.amplitude_input.text()) == expected["amplitude_0"]
        assert float(window.mean_input.text()) == expected["mean_0"]
        assert float(window.noise_value_input.text()) == expected["noise value"]
        assert float(window.spatial_width_input.text()) == expected["spatial width"]
        assert window.pixel_width_input.value() == expected["pixel width"]
        start, stop, steps = expected["time range"]
        assert (float(window.time_start_input.text()), float(window.time_stop_input.text())) == (start, stop)
        assert window.time_steps_input.value() == steps
        assert window.multiprocessing_check.isChecked() is expected["multiprocessing"]
        assert (window.length_unit_combo.currentText(), window.time_unit_combo.currentText()) == (
            expected["length unit"], expected["time unit"],
        )

    @pytest.mark.parametrize("filename", EXAMPLE_PARAMETER_FILES)
    def test_diffusion_length_only_file_derives_lifetime_from_last_time(self, window, filename):
        """GUI convention: with only L given, tau is the final time and D = L^2 / tau."""
        expected = read_example(filename)
        window.load_parameters_from_file(str(EXAMPLES_DIR / filename))

        length = expected["nominal diffusion length"]
        lifetime = expected["time range"][1]
        assert float(window.diffusion_length_input.text()) == length
        assert float(window.lifetime_input.text()) == lifetime
        assert float(window.diffusion_coeff_input.text()) == pytest.approx(length ** 2 / lifetime, rel=1e-3)

    @pytest.mark.parametrize("filename", [
        pytest.param(name, marks=pytest.mark.xfail(
            strict=True,
            reason="For a diffusion-length-only file the GUI sets tau to the final time "
                   "(2 here) while the engine and CLI set tau = 1; the two front ends then "
                   "simulate different decay for the same file. Needs a decision on the convention.",
        )) if name == "high_precision.txt" else name
        for name in EXAMPLE_PARAMETER_FILES
    ])
    def test_gui_and_cli_simulate_the_same_physics_for_an_example(self, window, filename):
        path = EXAMPLES_DIR / filename
        window.load_parameters_from_file(str(path))
        from_gui = parameter_parser(resolve_units(_engine_parameters(window)))
        from_cli = open_parameters(path)

        for key in ("nominal diffusion coefficient", "nominal lifetime (tau)", "nominal diffusion length",
                    "sigma^2_0", "number of runs"):
            assert from_gui[key] == pytest.approx(from_cli[key], rel=1e-9), key
        for key in ("x array", "time series"):
            np.testing.assert_allclose(from_gui[key], from_cli[key], rtol=1e-12, err_msg=key)
        assert from_gui["noise series"] == from_cli["noise series"]


class TestSaveAndReload:

    def _configure_series_sigma_overrides(self, qtbot, window) -> None:
        enter_text(qtbot, window.filename_slug_input, "round_trip_a")
        window.num_runs_spin.setValue(250)
        window.sigma_radio.setChecked(True)
        enter_text(qtbot, window.width_input, "0.35")
        set_combo_value(window.mean_unit_combo, "nanometer")
        enter_text(qtbot, window.mean_input, "50")
        set_combo_value(window.lifetime_unit_combo, "picosecond")
        enter_text(qtbot, window.diffusion_coeff_input, "0.02")
        enter_text(qtbot, window.lifetime_input, "1500")
        enter_text(qtbot, window.noise_value_input, "0.03")
        window.time_series_radio.setChecked(True)
        window.time_series_input.setPlainText("0, 0.25, 0.5, 1, 1.5")
        window.preset_combo.setCurrentIndex(window.preset_combo.findData("publication"))
        window.multiprocessing_check.setChecked(False)
        window.retain_profile_check.setChecked(True)

    def _configure_range_estimate_overrides(self, qtbot, window) -> None:
        enter_text(qtbot, window.filename_slug_input, "round_trip_b")
        window.noise_estimate_radio.setChecked(True)
        enter_text(qtbot, window.noise_file_input, EXAMPLE_PROFILE_CSV.as_posix())
        set_combo_value(window.spatial_width_unit_combo, "nanometer")
        enter_text(qtbot, window.spatial_width_input, "8000")
        set_combo_value(window.time_start_unit_combo, "picosecond")
        set_combo_value(window.time_stop_unit_combo, "picosecond")
        enter_text(qtbot, window.time_start_input, "100")
        enter_text(qtbot, window.time_stop_input, "1900")
        window.time_steps_input.setValue(12)
        set_combo_value(window.diffusion_coeff_length_unit_combo, "centimeter")
        set_combo_value(window.diffusion_coeff_time_unit_combo, "second")
        enter_text(qtbot, window.diffusion_coeff_input, "0.5")
        enter_text(qtbot, window.lifetime_input, "3")

    def _round_trip(self, window, dialogs, path) -> tuple[dict, dict]:
        before = _engine_parameters(window)
        _save_as_via_menu(window, dialogs, path)
        trigger_menu_action(window, "&File", "&New Parameters")
        assert _engine_parameters(window) != before
        _open_via_menu(window, dialogs, path)
        return before, _engine_parameters(window)

    def test_series_sigma_and_overrides_survive(self, qtbot, window, dialogs, work_dir):
        self._configure_series_sigma_overrides(qtbot, window)
        before, after = self._round_trip(window, dialogs, work_dir / "a.txt")
        assert after == before
        assert dialogs.titles("critical") == []

    def test_range_estimate_and_overrides_survive(self, qtbot, window, dialogs, work_dir):
        self._configure_range_estimate_overrides(qtbot, window)
        before, after = self._round_trip(window, dialogs, work_dir / "b.txt")
        assert after == before
        assert dialogs.titles("critical") == []

    def test_saved_file_is_readable_by_the_cli(self, qtbot, window, dialogs, work_dir):
        self._configure_series_sigma_overrides(qtbot, window)
        saved = work_dir / "for_cli.txt"
        _save_as_via_menu(window, dialogs, saved)

        from_cli = open_parameters(saved)
        from_gui = parameter_parser(resolve_units(_engine_parameters(window)))

        for key in ("nominal diffusion coefficient", "nominal lifetime (tau)", "sigma^2_0", "mu_0"):
            assert from_cli[key] == pytest.approx(from_gui[key], rel=1e-12), key
        np.testing.assert_allclose(from_cli["time series"], from_gui["time series"])
        assert from_cli["nominal lifetime (tau)"] == pytest.approx(1.5)
        assert from_cli["mu_0"] == pytest.approx(0.05)

    def test_save_targets_the_opened_file_and_clears_the_modified_mark(self, qtbot, window, dialogs, work_dir):
        saved = work_dir / "current.txt"
        _save_as_via_menu(window, dialogs, saved)
        enter_text(qtbot, window.filename_slug_input, "edited")
        assert window.windowTitle() == "DICE - [current.txt]*"

        trigger_menu_action(window, "&File", "&Save Parameters")

        assert dialogs.titles("save-file") == ["Save Parameters As"]
        assert dialogs.titles("information")[-1] == "Parameters Saved"
        assert window.windowTitle() == "DICE - [current.txt]"
        assert "'filename slug': 'edited'" in saved.read_text()

    def test_cancelled_save_dialog_writes_nothing(self, qtbot, window, dialogs, work_dir):
        enter_text(qtbot, window.filename_slug_input, "edited")
        trigger_menu_action(window, "&File", "&Save Parameters")

        assert dialogs.titles("save-file") == ["Save Parameters As"]
        assert list(work_dir.iterdir()) == []
        assert window.windowTitle() == DEFAULT_TITLE + "*"


class TestRecentFiles:

    def _recent_names(self, window) -> list[str]:
        return [action.text() for action in window.recent_menu.actions()]

    def test_recent_menu_lists_newest_first_and_reopens(self, window, dialogs):
        assert self._recent_names(window) == ["No recent files"]
        window.load_parameters_from_file(str(EXAMPLES_DIR / "quick_test.txt"))
        window.load_parameters_from_file(str(EXAMPLES_DIR / "publication.txt"))
        assert self._recent_names(window) == ["publication.txt", "quick_test.txt"]

        window.recent_menu.actions()[1].trigger()

        assert window.filename_slug_input.text() == "quick_test"
        assert self._recent_names(window) == ["quick_test.txt", "publication.txt"]

    def test_single_recent_file_survives_a_restart(self, qtbot, window):
        """INI settings (the native format on Linux) store a one-item list as a plain string."""
        from dice_gui.dice_gui import DiceGUI

        window.load_parameters_from_file(str(EXAMPLES_DIR / "quick_test.txt"))
        window.settings.sync()

        restarted = DiceGUI()
        qtbot.addWidget(restarted)
        assert self._recent_names(restarted) == ["quick_test.txt"]


class TestUnsavedChanges:

    @pytest.fixture
    def modified_window(self, qtbot, window):
        enter_text(qtbot, window.filename_slug_input, "unsaved")
        assert window.windowTitle() == DEFAULT_TITLE + "*"
        return window

    def test_declining_to_discard_keeps_the_edits_on_open(self, modified_window, dialogs):
        dialogs.question_answers.append(QMessageBox.StandardButton.No)
        _open_via_menu(modified_window, dialogs, EXAMPLES_DIR / "quick_test.txt")

        assert dialogs.titles("question") == ["Unsaved Changes"]
        assert dialogs.titles("open-file") == []
        assert modified_window.filename_slug_input.text() == "unsaved"

    def test_declining_keeps_the_edits_on_new_and_on_example(self, modified_window, dialogs):
        dialogs.question_answers.extend([QMessageBox.StandardButton.No] * 2)
        trigger_menu_action(modified_window, "&File", "&New Parameters")
        modified_window.example_menu.actions()[0].trigger()

        assert dialogs.titles("question") == ["Unsaved Changes", "Unsaved Changes"]
        assert modified_window.filename_slug_input.text() == "unsaved"

    def test_accepting_new_restores_defaults(self, modified_window, dialogs):
        trigger_menu_action(modified_window, "&File", "&New Parameters")

        assert modified_window.filename_slug_input.text() == "DICE_results"
        assert modified_window.windowTitle() == DEFAULT_TITLE

    @pytest.mark.parametrize(("answer", "closes"), [
        (QMessageBox.StandardButton.Cancel, False),
        (QMessageBox.StandardButton.Discard, True),
        (QMessageBox.StandardButton.Save, False),  # the Save As dialog is then cancelled
    ])
    def test_close_asks_first(self, modified_window, dialogs, answer, closes):
        dialogs.question_answers.append(answer)
        assert modified_window.close() is closes
        assert dialogs.titles("question") == ["Unsaved Changes"]


class TestLoadErrors:

    def test_truncated_file_is_reported_and_leaves_the_form_unchanged(self, window, dialogs, work_dir):
        lines = (EXAMPLES_DIR / "quick_test.txt").read_text().splitlines()
        truncated = work_dir / "truncated.txt"
        truncated.write_text("\n".join(lines[: len(lines) // 2]))
        before = window.collect_parameters()

        _open_via_menu(window, dialogs, truncated)

        assert dialogs.titles("critical") == ["Error Loading Parameters"]
        assert window.collect_parameters() == before
        assert window.windowTitle() == DEFAULT_TITLE

    def test_missing_file_is_reported(self, window, dialogs, work_dir):
        _open_via_menu(window, dialogs, work_dir / "absent.txt")
        assert dialogs.titles("critical") == ["Error Loading Parameters"]
        assert "absent.txt" in dialogs.texts("critical")[0]
