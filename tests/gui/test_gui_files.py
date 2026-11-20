"""
Parameter files: loading the bundled examples, saving and reloading, the
unsaved-changes prompts, and how load errors are reported.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path, PureWindowsPath

import numpy as np
import pytest
from PyQt6.QtWidgets import QMessageBox

from dice.io.parameters import open_parameters, parameter_parser
from dice.utils.units import resolve_units
from dice_gui import dice_gui as dice_gui_window
from dice_gui.dice_interface import DiceInterface
from dice_gui.example_parameters import EXAMPLE_PARAMETERS
from dice_gui.tabs.base import set_combo_value
from gui_support import (
    EXAMPLE_PARAMETER_FILES, EXAMPLE_PROFILE_CSV, EXAMPLES_DIR,
    choose_global_units, enter_text, read_example, trigger_menu_action,
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
    def test_diffusion_length_only_file_uses_unit_lifetime(self, window, filename):
        """Engine convention: with only L given, tau is 1 and D = L^2."""
        expected = read_example(filename)
        window.load_parameters_from_file(str(EXAMPLES_DIR / filename))

        length = expected["nominal diffusion length"]
        assert float(window.diffusion_length_input.text()) == length
        assert float(window.lifetime_input.text()) == 1.0
        assert float(window.diffusion_coeff_input.text()) == pytest.approx(length ** 2, rel=1e-3)

    @pytest.mark.parametrize("filename", EXAMPLE_PARAMETER_FILES)
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

    @staticmethod
    def _existing_path_with_backslashes(work_dir: Path) -> str:
        """An existing absolute profile path whose text contains backslashes on this platform.

        On Windows that is the example profile's own path. Elsewhere a Windows-style
        string is a relative path, which a loaded file resolves against its folder,
        so the profile is copied to a file whose name contains backslashes instead.
        """
        if os.name == "nt":
            return str(PureWindowsPath(EXAMPLE_PROFILE_CSV))
        copy = work_dir / "measured\\profile.csv"
        shutil.copyfile(EXAMPLE_PROFILE_CSV, copy)
        return str(copy)

    @pytest.mark.parametrize(("widget_name", "text"), [
        ("filename_slug_input", "Joe's_run"),
        ("noise_file_input", None),
    ], ids=["apostrophe-in-slug", "path-with-backslashes"])
    def test_text_values_needing_escapes_survive(self, qtbot, window, dialogs, work_dir, widget_name, text):
        if text is None:
            text = self._existing_path_with_backslashes(work_dir)
            assert "\\" in text
        window.noise_estimate_radio.setChecked(True)
        enter_text(qtbot, window.noise_file_input, EXAMPLE_PROFILE_CSV.as_posix())
        enter_text(qtbot, getattr(window, widget_name), text)

        before, after = self._round_trip(window, dialogs, work_dir / "escapes.txt")

        assert dialogs.titles("critical") == []
        assert after == before
        assert getattr(window, widget_name).text() == text

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


class TestLoadExampleMenu:

    @pytest.mark.parametrize(("action_index", "filename"), [
        (0, "quick_test.txt"), (1, "high_precision.txt"), (2, "publication.txt"),
    ])
    def test_each_example_loads_its_bundled_values(self, window, dialogs, action_index, filename):
        expected = read_example(filename)
        window.example_menu.actions()[action_index].trigger()

        assert len(dialogs.titles("information")) == 1
        assert window.filename_slug_input.text() == expected["filename slug"]
        assert window.num_runs_spin.value() == expected["number of runs"]
        assert window.pixel_width_input.value() == expected["pixel width"]

    # A wheel install has no data/examples directory, so the Examples menu
    # loads the embedded copies.
    @pytest.mark.parametrize("example_name", ["quick_test", "high_precision", "publication"])
    def test_embedded_copy_matches_the_bundled_file(self, example_name):
        assert EXAMPLE_PARAMETERS[example_name] == read_example(f"{example_name}.txt")

    def test_standalone_build_loads_the_files_it_unpacked(self, window, dialogs, monkeypatch, tmp_path):
        import sys

        unpacked_examples = tmp_path / "data" / "examples"
        unpacked_examples.mkdir(parents=True)
        # One changed value shows which copy the menu read.
        (unpacked_examples / "quick_test.txt").write_text(
            (EXAMPLES_DIR / "quick_test.txt").read_text().replace("'number of runs': 100", "'number of runs': 123")
        )
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

        window.example_menu.actions()[0].trigger()

        assert window.num_runs_spin.value() == 123

    def test_example_without_the_bundled_files_sets_its_own_units(self, window, dialogs, monkeypatch, tmp_path):
        monkeypatch.setattr(dice_gui_window, "examples_directory", lambda: tmp_path / "data" / "examples")
        choose_global_units(window, monkeypatch, "nanometer", "picosecond")

        window.example_menu.actions()[0].trigger()

        expected = read_example("quick_test.txt")
        assert (window.length_unit_combo.currentText(), window.time_unit_combo.currentText()) == (
            expected["length unit"], expected["time unit"],
        )
        assert float(window.width_input.text()) == expected["FWHM_0"]

    def test_loaded_example_is_not_the_save_target(self, qtbot, window, dialogs):
        bundled = EXAMPLES_DIR / "quick_test.txt"
        original = bundled.read_bytes()
        window.example_menu.actions()[0].trigger()
        # Checked before saving, so a regression fails here instead of overwriting the example.
        assert window.windowTitle() == DEFAULT_TITLE
        assert [action.text() for action in window.recent_menu.actions()] == ["No recent files"]

        enter_text(qtbot, window.filename_slug_input, "my_quick_test")
        trigger_menu_action(window, "&File", "&Save Parameters")

        assert dialogs.titles("save-file") == ["Save Parameters As"]
        assert bundled.read_bytes() == original


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


class TestRelativeNoiseDataPath:
    """A relative profile path in an opened file is relative to that file's folder, as in the CLI."""

    def test_opened_file_resolves_against_its_folder(self, window, dialogs, work_dir):
        import shutil
        params_dir = work_dir / "project"
        params_dir.mkdir()
        shutil.copy(EXAMPLE_PROFILE_CSV, params_dir / "measured.csv")
        params = read_example("quick_test.txt")
        del params["noise value"]
        params["estimate noise from data"] = "measured.csv"
        params_file = params_dir / "relative.txt"
        params_file.write_text(str(params))

        _open_via_menu(window, dialogs, params_file)

        assert dialogs.titles("critical") == []
        assert window.noise_estimate_radio.isChecked()
        assert window.noise_file_input.text() == str(params_dir / "measured.csv")
        assert _engine_parameters(window)["estimate noise from data"] == str(params_dir / "measured.csv")
        assert window._error_labels["noise_file"].isHidden()


class TestLoadedFileEmptiesTheOtherMode:
    """A file selects one noise mode and one time mode; the inputs of the other mode keep nothing from before."""

    @staticmethod
    def _write(work_dir: Path, name: str, remove: str, add: dict) -> Path:
        params = read_example("quick_test.txt")
        del params[remove]
        params.update(add)
        params_file = work_dir / name
        params_file.write_text(str(params))
        return params_file

    def test_estimate_noise_file_empties_the_fixed_noise_inputs(self, window, dialogs, work_dir):
        params_file = self._write(work_dir, "estimate.txt", "noise value",
                                  {"estimate noise from data": EXAMPLE_PROFILE_CSV.as_posix()})
        assert window.noise_value_input.text() == "0.05"

        _open_via_menu(window, dialogs, params_file)

        assert dialogs.titles("critical") == []
        assert window.noise_estimate_radio.isChecked()
        assert (window.noise_value_input.text(), window.noise_cnr_input.text()) == ("", "")

    def test_fixed_noise_file_empties_the_noise_file_input(self, window, dialogs, work_dir):
        params_file = self._write(work_dir, "estimate.txt", "noise value",
                                  {"estimate noise from data": EXAMPLE_PROFILE_CSV.as_posix()})
        _open_via_menu(window, dialogs, params_file)
        assert window.noise_file_input.text() != ""

        _open_via_menu(window, dialogs, EXAMPLES_DIR / "quick_test.txt")

        assert dialogs.titles("critical") == []
        assert window.noise_fixed_radio.isChecked()
        assert window.noise_file_input.text() == ""
        assert window.noise_value_input.text() == str(read_example("quick_test.txt")["noise value"])

    def test_time_series_file_empties_the_time_range_inputs(self, window, dialogs, work_dir):
        params_file = self._write(work_dir, "series.txt", "time range", {"time series": [0, 0.5, 1, 2]})
        assert window.time_start_input.text() == "0"

        _open_via_menu(window, dialogs, params_file)

        assert dialogs.titles("critical") == []
        assert window.time_series_radio.isChecked()
        assert window.time_series_input.toPlainText() == "0, 0.5, 1, 2"
        assert (window.time_start_input.text(), window.time_stop_input.text()) == ("", "")

    def test_time_range_file_empties_the_time_series_input(self, window, dialogs, work_dir):
        params_file = self._write(work_dir, "series.txt", "time range", {"time series": [0, 0.5, 1, 2]})
        _open_via_menu(window, dialogs, params_file)

        _open_via_menu(window, dialogs, EXAMPLES_DIR / "quick_test.txt")

        assert dialogs.titles("critical") == []
        assert window.time_range_radio.isChecked()
        assert window.time_series_input.toPlainText() == ""

    def test_new_parameters_empties_what_the_user_kept_in_the_other_mode(self, qtbot, window, dialogs):
        window.noise_estimate_radio.setChecked(True)
        enter_text(qtbot, window.noise_file_input, EXAMPLE_PROFILE_CSV.as_posix())
        window.time_series_radio.setChecked(True)
        window.time_series_input.setPlainText("0, 1, 2")
        window.parameters_modified = False

        trigger_menu_action(window, "&File", "&New Parameters")

        assert window.noise_fixed_radio.isChecked() and window.time_range_radio.isChecked()
        assert window.noise_file_input.text() == ""
        assert window.time_series_input.toPlainText() == ""
        assert (window.noise_value_input.text(), window.time_start_input.text(),
                window.time_stop_input.text()) == ("0.05", "0", "2")


class TestNoiseRangeFile:
    """A file with a noise range opens with a warning that the GUI cannot edit or run the range."""

    @pytest.mark.parametrize("key", ["noise range, reciprocal", "noise_range_reciprocal_log"])
    def test_opening_warns_and_keeps_the_gui_noise(self, window, dialogs, work_dir, key):
        params = read_example("quick_test.txt")
        del params["noise value"]
        params[key] = [0.005, 0.1]
        params_file = work_dir / "range.txt"
        params_file.write_text(str(params))
        noise_before = (window.noise_fixed_radio.isChecked(), window.noise_value_input.text())

        _open_via_menu(window, dialogs, params_file)

        assert dialogs.titles("information") == []
        assert dialogs.titles("warning") == ["Noise Range Not Editable"]
        [text] = dialogs.texts("warning")
        assert f"Parameters loaded from:\n{params_file}" in text
        assert f"'{key}': [0.005, 0.1]" in text
        assert "The GUI cannot show, edit, or run a noise range." in text
        assert f'dice "{params_file}"' in text
        # The other values load; the noise fields keep the GUI's own setting.
        assert window.windowTitle() == "DICE - [range.txt]"
        assert window.num_runs_spin.value() == params["number of runs"]
        assert (window.noise_fixed_radio.isChecked(), window.noise_value_input.text()) == noise_before
        assert "noise range, reciprocal" not in _engine_parameters(window)

    def test_bundled_example_warns(self, window, dialogs):
        window.load_parameters_from_file(str(EXAMPLES_DIR / "noise_range.txt"))
        assert dialogs.titles() == ["Noise Range Not Editable"]
        assert "'noise range, reciprocal log': [0.005, 0.1]" in dialogs.texts("warning")[0]


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
