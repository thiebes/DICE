"""
Construction of the main window, its tabs, menus, and auxiliary dialogs.
"""

from __future__ import annotations

import re

import pytest
from PyQt6.QtCore import QLocale
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QAbstractButton, QDoubleSpinBox, QGroupBox, QLabel, QWidget

from dice import __version__
from dice.utils.units import length_abbreviation
from gui_support import TAB_NAMES, fit_method_radio

pytestmark = pytest.mark.gui


def _menu_action_texts(window, menu_title: str) -> list[str]:
    menu = next(a.menu() for a in window.menuBar().actions() if a.text() == menu_title)
    return [action.text() for action in menu.actions() if not action.isSeparator()]


def _mnemonics_on_tab(tab: QWidget) -> dict[str, str]:
    """Map Alt+letter to its widget text for every buddy label and button mnemonic on a tab."""
    labelled = [label for label in tab.findChildren(QLabel) if label.buddy() is not None]
    widgets = labelled + tab.findChildren(QAbstractButton)
    mnemonics = {}
    for widget in widgets:
        match = re.search(r"&(\w)", widget.text())
        if match:
            mnemonics[f"Alt+{match.group(1).upper()}"] = widget.text().replace("&", "").rstrip(":")
    return mnemonics


def _shortcuts_by_tab(dialog_html: str) -> list[dict[str, str]]:
    """Parse the shortcuts dialog into one {Alt+X: description} map per tab section."""
    sections = re.split(r"<h3>Tab \d+:[^<]*</h3>", dialog_html)[1:]
    return [
        dict(re.findall(r"<td><b>(Alt\+\w)</b></td><td>([^<]+)</td>", section))
        for section in sections
    ]


def _relative_luminance(color: QColor) -> float:
    """Relative luminance as the Web Content Accessibility Guidelines (WCAG) define it."""
    def linear(channel: int) -> float:
        value = channel / 255
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4
    return 0.2126 * linear(color.red()) + 0.7152 * linear(color.green()) + 0.0722 * linear(color.blue())


def _contrast_ratio(first: QColor, second: QColor) -> float:
    darker, lighter = sorted((_relative_luminance(first), _relative_luminance(second)))
    return (lighter + 0.05) / (darker + 0.05)


def _run_button_colors(stylesheet: str, state: str) -> list[QColor]:
    """The colors that the stylesheet rule for one Run button state sets, in the order written."""
    rule = re.search(r"QPushButton#run-button" + re.escape(state) + r"\s*\{([^}]*)\}", stylesheet).group(1)
    return [QColor(*map(int, channels)) for channels in re.findall(r"rgba\((\d+), (\d+), (\d+), \d+\)", rule)]


class TestMainWindow:

    def test_title_and_five_tabs_in_order(self, window):
        assert window.windowTitle() == "DICE - Diffusion Insight Computation Engine"
        assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == TAB_NAMES

    def test_status_bar_shows_version_and_no_loaded_data(self, window):
        permanent_texts = [label.text() for label in window.status_bar.findChildren(QLabel)]
        assert f"DICE v{__version__}" in permanent_texts
        assert window.status_bar.currentMessage() == "No data loaded"

    def test_menus_offer_the_file_and_plot_actions(self, window):
        assert _menu_action_texts(window, "&File") == [
            "&New Parameters", "&Open Parameters...", "&Save Parameters",
            "Save Parameters &As...", "Recent &Parameters", "Load &Results...", "E&xit",
        ]
        assert _menu_action_texts(window, "&Plots")[0] == "&Accuracy Histogram"
        assert _menu_action_texts(window, "&Settings") == ["&Global Units..."]
        assert _menu_action_texts(window, "&Help") == [
            "&Documentation", "&Keyboard Shortcuts", "&About DICE",
        ]

    def test_control_panel_starts_ready_to_run(self, window):
        assert window.run_button.isEnabled()
        assert not window.stop_button.isEnabled()
        assert window.progress_bar.isHidden()
        assert window.status_label.text() == "Ready to run"
        assert window.parameters_modified is False


class TestDefaultValues:
    """The defaults set_default_values() writes into each tab."""

    def test_simulation_setup_tab(self, window):
        assert window.num_runs_spin.value() == 1000
        assert window.filename_slug_input.text() == "DICE_results"
        assert window.multiprocessing_check.isChecked()
        assert not window.retain_profile_check.isChecked()

    def test_physical_parameters_tab_derives_diffusion_length(self, window):
        assert float(window.diffusion_coeff_input.text()) == 0.01
        assert float(window.lifetime_input.text()) == 2.0
        assert float(window.diffusion_length_input.text()) == pytest.approx((0.01 * 2.0) ** 0.5, rel=1e-3)
        assert window.fwhm_radio.isChecked()
        assert window.width_input.text() == "1.0"
        assert window.width_conversion_label.text() == f"Equivalent: σ = 0.4247 {length_abbreviation('micrometer')}"

    def test_experimental_conditions_tab(self, window):
        assert window.noise_fixed_radio.isChecked()
        assert window.noise_value_input.text() == "0.05"
        assert window.noise_cnr_input.text() == "20"
        assert window.spatial_width_input.text() == "10.0"
        assert window.pixel_width_input.value() == 101
        assert window.pixel_size_label.text() == f"Pixel Size: 0.1 {length_abbreviation('micrometer')}/pixel"
        assert window.time_range_radio.isChecked()
        assert (window.time_start_input.text(), window.time_stop_input.text()) == ("0", "2")
        assert window.time_steps_input.value() == 10
        assert not window.time_series_input.isEnabled()
        assert not window.noise_file_input.isEnabled()

    def test_analysis_tab_waits_for_results(self, window):
        assert fit_method_radio(window, "WLS").isChecked()
        assert not fit_method_radio(window, "OLS").isChecked()
        assert [window.precision_std_label.text(), window.precision_rse_label.text(),
                window.precision_bias_label.text(), window.precision_mean_ratio_label.text()] == ["—"] * 4
        assert not window.precision_placeholder.isHidden()
        assert window.precision_placeholder.text() == "Run a simulation to see results."
        assert not window.explore_proximity_group.isEnabled()
        assert window.proximity_spin.value() == pytest.approx(0.10)
        assert window.proximity_pct_label.text() == "(±10%)"

    def test_output_settings_tab(self, window):
        assert window.preset_combo.currentText() == "Custom"
        assert [window.image_type_combo.itemText(i) for i in range(window.image_type_combo.count())] == [
            "png", "jpg", "svg", "tif",
        ]
        assert window.image_dpi_spin.value() == 300
        assert window.image_numbins_spin.value() == 35


class TestStandaloneBuild:
    """A standalone (frozen) build offers parallel processing like a regular installation."""

    @pytest.fixture
    def frozen_window(self, qtbot, monkeypatch):
        import sys
        from dice_gui.dice_gui import DiceGUI

        monkeypatch.setattr(sys, "frozen", True, raising=False)
        gui = DiceGUI()
        qtbot.addWidget(gui)
        return gui

    def test_checkbox_is_on_and_enabled(self, frozen_window):
        checkbox = frozen_window.multiprocessing_check
        assert checkbox.isChecked()
        assert checkbox.isEnabled()
        assert checkbox.toolTip() == "Use multiple CPU cores to speed up simulation"

    def test_loaded_file_sets_parallel_processing(self, frozen_window, tmp_path):
        from pathlib import Path

        example = Path(__file__).parent.parent.parent / "data" / "examples" / "parameters.txt"
        serial_copy = tmp_path / "serial.txt"
        serial_copy.write_text(example.read_text().replace("'multiprocessing': True", "'multiprocessing': False"))

        frozen_window.load_parameters_from_file(str(serial_copy))
        assert not frozen_window.multiprocessing_check.isChecked()
        assert frozen_window.collect_parameters()["multiprocessing"] is False

        frozen_window.load_parameters_from_file(str(example))
        assert frozen_window.multiprocessing_check.isChecked()
        assert frozen_window.collect_parameters()["multiprocessing"] is True

    def test_regular_install_keeps_the_option(self, window):
        assert window.multiprocessing_check.isEnabled()
        assert window.multiprocessing_check.isChecked()


class TestAuxiliaryDialogs:

    def test_global_units_dialog_preselects_and_reports_units(self, qtbot):
        from dice_gui.global_units_dialog import GlobalUnitsDialog

        dialog = GlobalUnitsDialog(current_length="nanometer", current_time="picosecond")
        qtbot.addWidget(dialog)
        assert dialog.windowTitle() == "Global Units"
        assert (dialog.selected_length_unit(), dialog.selected_time_unit()) == ("nanometer", "picosecond")

        dialog.length_combo.setCurrentText("angstrom")
        dialog.time_combo.setCurrentText("femtosecond")
        assert (dialog.selected_length_unit(), dialog.selected_time_unit()) == ("angstrom", "femtosecond")

    def test_about_dialog_reports_version(self, window, dialogs):
        window.show_about_dialog()
        assert dialogs.titles("about") == ["About DICE"]
        assert f"<b>Version:</b> {__version__}" in dialogs.texts("about")[0]

    def test_documentation_opens_repository_readme(self, window, monkeypatch):
        opened = []
        monkeypatch.setattr("dice_gui.dice_gui.webbrowser.open", opened.append)
        window.open_documentation()
        assert opened == ["https://github.com/thiebes/DICE#readme"]

    def test_keyboard_shortcuts_dialog_lists_every_tab(self, window, dialogs):
        window.show_keyboard_shortcuts()
        assert dialogs.titles("information") == ["Keyboard Shortcuts"]
        assert len(_shortcuts_by_tab(dialogs.texts("information")[0])) == len(TAB_NAMES)

    @pytest.mark.parametrize("tab_index", range(len(TAB_NAMES)), ids=TAB_NAMES)
    def test_keyboard_shortcuts_dialog_matches_label_mnemonics(self, window, dialogs, tab_index):
        """Each Alt+letter the dialog lists must be the mnemonic of a label on that tab."""
        window.show_keyboard_shortcuts()
        listed = _shortcuts_by_tab(dialogs.texts("information")[0])[tab_index]
        actual = _mnemonics_on_tab(window.tabs.widget(tab_index))
        assert set(listed) <= set(actual), (
            f"Shortcuts listed but not wired on '{TAB_NAMES[tab_index]}': "
            f"{ {key: listed[key] for key in set(listed) - set(actual)} }; actual mnemonics: {actual}"
        )


class TestAppearance:

    def test_numbers_show_a_decimal_point_when_the_system_uses_a_decimal_comma(self, qtbot):
        """Text fields and parameter files use a decimal point, so the spin boxes do as well."""
        from dice_gui.dice_gui import DiceGUI

        system_locale = QLocale()
        QLocale.setDefault(QLocale(QLocale.Language.German, QLocale.Country.Germany))
        try:
            outside_the_window = QDoubleSpinBox()
            qtbot.addWidget(outside_the_window)
            outside_the_window.setValue(16.0)
            assert outside_the_window.cleanText() == "16,00"

            gui = DiceGUI()
            qtbot.addWidget(gui)
            assert gui.image_width_spin.cleanText() == "16.00"
            assert gui.image_height_spin.cleanText() == "10.00"
            assert gui.proximity_spin.cleanText() == "0.10"
        finally:
            QLocale.setDefault(system_locale)

    def test_group_titles_show_their_ampersands(self, window):
        """Qt reads a single ampersand as a shortcut marker and does not draw it."""
        titles = [group.title() for group in window.findChildren(QGroupBox)]
        assert "Resolution && Histogram" in titles
        assert [title for title in titles if "& " in title.replace("&&", "")] == []

    @pytest.mark.parametrize("dark", [False, True], ids=["light theme", "dark theme"])
    @pytest.mark.parametrize("state", [":enabled", ":hover:enabled", ":pressed:enabled"])
    def test_run_button_text_is_readable_in_each_theme(self, qapp, dark, state):
        """WCAG level AA asks for a contrast of at least 4.5:1 between text and its background."""
        from dice_gui.styles import DiceTheme

        theme = DiceTheme()
        theme.is_dark_mode = dark
        stylesheet = theme.get_stylesheet()
        text = _run_button_colors(stylesheet, ":enabled")[1]
        background = _run_button_colors(stylesheet, state)[0]

        assert _contrast_ratio(text, background) >= 4.5
