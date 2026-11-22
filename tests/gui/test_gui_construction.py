"""
Construction of the main window, its tabs, menus, and auxiliary dialogs.
"""

from __future__ import annotations

import re

import pytest
from PyQt6.QtCore import QLocale, Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QAbstractButton, QDoubleSpinBox, QGroupBox, QLabel, QScrollArea, QWidget

from dice import __version__
from dice.utils.units import length_abbreviation
from gui_support import TAB_NAMES, fit_method_radio, trigger_menu_action

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
        assert window.results_status_label.text() == "No data loaded"

    def test_results_text_returns_after_a_menu_status_tip(self, window):
        """A menu entry's status tip is a temporary message; the results text must outlive it."""
        window.show()
        window.status_bar.showMessage("Load parameters from file")
        assert window.results_status_label.isHidden()

        window.status_bar.clearMessage()

        assert not window.results_status_label.isHidden()
        assert window.results_status_label.text() == "No data loaded"
        assert window.status_bar.currentMessage() == ""

    def test_menus_offer_the_file_and_plot_actions(self, window):
        assert _menu_action_texts(window, "&File") == [
            "&New Parameters", "&Open Parameters...", "&Save Parameters",
            "Save Parameters &As...", "Recent &Parameters", "Load &Results...", "&Write Output Files", "E&xit",
        ]
        assert _menu_action_texts(window, "&Plots")[0] == "&Accuracy Histogram"
        assert _menu_action_texts(window, "&Settings") == ["&Global Units..."]
        assert _menu_action_texts(window, "&Help") == [
            "&Documentation", "&Keyboard Shortcuts", "&Licenses", "&About DICE",
        ]

    def test_licenses_of_an_installed_copy_name_the_gpl_of_pyqt(self, window, dialogs):
        trigger_menu_action(window, "&Help", "&Licenses")

        assert dialogs.titles("information") == ["Licenses"]
        text = dialogs.texts("information")[0]
        assert "MIT License" in text
        assert "PyQt6" in text and "GNU General Public License version 3" in text

    def test_licenses_of_a_standalone_application_show_its_notice(self, window, dialogs, tmp_path, monkeypatch):
        from PyQt6.QtWidgets import QDialog, QPlainTextEdit

        notice = tmp_path / "THIRD-PARTY-LICENSES.txt"
        notice.write_text("DICE: licenses of the standalone application\nGNU GENERAL PUBLIC LICENSE",
                          encoding="utf-8")
        monkeypatch.setattr("dice_gui.dice_gui.bundled_license_notice", lambda: notice)
        shown = []

        def record(dialog: QDialog) -> int:
            editor = dialog.findChild(QPlainTextEdit)
            shown.append((dialog.windowTitle(), editor.toPlainText(), editor.isReadOnly()))
            return 0

        monkeypatch.setattr(QDialog, "exec", record)

        trigger_menu_action(window, "&Help", "&Licenses")

        assert shown == [("Licenses", notice.read_text(encoding="utf-8"), True)]
        assert dialogs.titles("information") == []

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


def _scroll_area_ancestors(widget: QWidget) -> list[QScrollArea]:
    ancestors = []
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QScrollArea):
            ancestors.append(parent)
        parent = parent.parentWidget()
    return ancestors


def _tab_scroll_area(window, tab_index: int) -> QScrollArea:
    """The scroll area that holds all of a tab's content."""
    page = window.tabs.widget(tab_index)
    scroll_areas = [area for area in page.findChildren(QScrollArea) if area.parentWidget() is page]
    assert len(scroll_areas) == 1
    return scroll_areas[0]


class TestLayout:

    def test_tab_bar_and_control_panel_do_not_scroll_away(self, window):
        """Only the content of a tab scrolls; the tab bar and the Run button stay in view."""
        assert _scroll_area_ancestors(window.tabs) == []
        assert _scroll_area_ancestors(window.run_button) == []
        assert _scroll_area_ancestors(window.status_label) == []

    @pytest.mark.parametrize("tab_index", range(len(TAB_NAMES)), ids=TAB_NAMES)
    def test_each_tab_scrolls_its_own_content(self, window, tab_index):
        page = window.tabs.widget(tab_index)
        scroll_area = _tab_scroll_area(window, tab_index)

        assert scroll_area.widgetResizable()
        fields = [w for w in page.findChildren(QWidget)
                  if isinstance(w, (QAbstractButton, QDoubleSpinBox, QGroupBox))]
        assert fields, "the tab has no content to check"
        assert all(scroll_area in _scroll_area_ancestors(field) for field in fields)

    def test_long_output_path_does_not_widen_the_tab(self, window, monkeypatch, tmp_path):
        """A folder path has no place to wrap, so the preview must not make its width a minimum."""
        content = _tab_scroll_area(window, TAB_NAMES.index("Simulation Setup")).widget()
        window.filename_slug_input.setText("short")
        width_with_short_path = content.minimumSizeHint().width()

        long_root = tmp_path.joinpath(*["a_folder_with_a_long_name"] * 12, "output")
        monkeypatch.setattr("dice_gui.dice_gui.output_root", lambda: long_root)
        window._update_output_path_preview()

        assert len(str(long_root)) > 300
        assert f"<b>Directory:</b> {long_root / 'short'}<br>" in window.output_path_preview.text()
        assert window.output_path_preview.toolTip() == str(long_root / "short")
        assert content.minimumSizeHint().width() == width_with_short_path

    def test_output_preview_has_no_tool_tip_without_a_slug(self, window):
        window.filename_slug_input.setText("")
        assert window.output_path_preview.toolTip() == ""


class TestKeyboardAndScreenReader:
    """Every input has a name for a screen reader, and the keyboard reaches and leaves every control."""

    def test_every_input_has_an_accessible_name(self, window):
        from PyQt6.QtWidgets import QAbstractSpinBox, QComboBox, QLineEdit, QTextEdit

        unnamed = []
        for kind in (QLineEdit, QAbstractSpinBox, QComboBox, QTextEdit):
            for widget in window.findChildren(kind):
                # The line edit inside a spin box or an editable combo box takes the name of its owner.
                if isinstance(widget.parent(), (QAbstractSpinBox, QComboBox)):
                    continue
                if not widget.accessibleName():
                    unnamed.append(f"{type(widget).__name__} {widget.objectName()!r}")
        assert unnamed == []

    def test_names_come_from_the_labels_and_the_quantities(self, window):
        assert window.diffusion_length_input.accessibleName() == "Diffusion length"
        # A label that is one short word, or holds a symbol, is replaced by a name that stands alone.
        assert window.time_stop_input.accessibleName() == "Stop time"
        assert window.noise_file_input.accessibleName() == "Measured profile file"
        assert window.noise_value_input.accessibleName() == "Noise standard deviation"
        assert window.image_width_spin.accessibleName() == "Image width"
        assert window.proximity_spin.accessibleName() == "Accuracy threshold"
        assert window.diffusion_length_unit_combo.accessibleName() == "Diffusion length unit"
        assert window.diffusion_coeff_length_unit_combo.accessibleName() == "Diffusion coefficient length unit"
        assert window.time_stop_unit_combo.accessibleName() == "Stop time unit"
        assert window.pixel_width_input.accessibleName() == "Number of Pixels"
        assert window.time_series_input.accessibleName() == "Time series"
        assert window.length_unit_combo.accessibleName() == "Global length unit"

    @pytest.mark.parametrize("tab_index", range(5))
    def test_field_shortcuts_do_not_use_the_letters_of_the_menu_bar(self, window, tab_index):
        """Alt+F, P, S and H open the File, Plots, Settings and Help menus on every tab."""
        menu_letters = {"Alt+" + re.search(r"&(\w)", action.text()).group(1).upper()
                        for action in window.menuBar().actions()}
        assert menu_letters == {"Alt+F", "Alt+P", "Alt+S", "Alt+H"}
        assert set(_mnemonics_on_tab(window.tabs.widget(tab_index))) & menu_letters == set()

    def test_tab_key_leaves_the_time_series_box_and_skips_the_scroll_areas(self, window):
        from PyQt6.QtWidgets import QScrollArea

        assert window.time_series_input.tabChangesFocus()
        scroll_areas = [area for index in range(window.tabs.count())
                        for area in window.tabs.widget(index).findChildren(QScrollArea)]
        assert scroll_areas
        assert all(area.focusPolicy() == Qt.FocusPolicy.NoFocus for area in scroll_areas)

    def test_stop_has_a_shortcut_that_its_text_names(self, window):
        assert window.stop_button.shortcut().toString() == "Ctrl+."
        assert window.stop_button.text() == "Stop (Ctrl+.)"

    def test_blocked_run_names_the_field_and_takes_the_user_to_it(self, qtbot, window, dialogs):
        window.amplitude_input.setText("abc")
        window.tabs.setCurrentIndex(4)

        qtbot.mouseClick(window.run_button, Qt.MouseButton.LeftButton)

        assert dialogs.titles("warning") == ["Missing or Invalid Parameters"]
        assert dialogs.texts("warning")[0].startswith("Initial amplitude: ")
        assert "has the focus now" in dialogs.texts("warning")[0]
        assert window.tabs.currentIndex() == 1
        assert window.focusWidget() is window.amplitude_input
        assert window.simulation_thread is None


def test_field_that_blocks_a_run_is_scrolled_into_view(qtbot, window, dialogs):
    """A scroll area follows the Tab key, not a focus that the program sets."""
    window.show()
    window.resize(900, 320)
    window.tabs.setCurrentIndex(1)
    qtbot.wait(50)
    window.width_input.setText("abc")
    window.tabs.setCurrentIndex(0)

    qtbot.mouseClick(window.run_button, Qt.MouseButton.LeftButton)

    assert window.tabs.currentIndex() == 1
    assert not window.width_input.visibleRegion().isEmpty()
