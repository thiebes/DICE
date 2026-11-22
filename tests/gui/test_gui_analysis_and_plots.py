"""
The Analysis tab after a run, the post-hoc accuracy-threshold tool, and the
plot and results file operations that depend on having results.

One real serial run is shared per module (completed_run) and handed to the
window through simulation_finished(), the slot SimulationThread.finished
drives. Expected values are recomputed from the CSV that run exported.
"""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox

from gui_support import (
    EXAMPLE_PROFILE_CSV, NOMINAL_D_COLUMN, OLS_SLOPE_COLUMN, WLS_SLOPE_COLUMN,
    d_estimates_from_csv, fit_method_radio, fraction_sentence, fraction_within, summary_label_texts,
    trigger_menu_action,
)

pytestmark = pytest.mark.gui

FIT_METHOD_COLUMNS = {"wls": WLS_SLOPE_COLUMN, "ols": OLS_SLOPE_COLUMN}


def _summary_labels(window) -> list[str]:
    return [
        window.precision_std_label.text(),
        window.precision_rse_label.text(),
        window.precision_bias_label.text(),
        window.precision_mean_ratio_label.text(),
    ]


# The quick_test example runs in micrometers and nanoseconds.
RUN_D_UNIT = "μm²/ns"


def _expected_summary(
    estimates: np.ndarray, nominal: float, d_unit: str = RUN_D_UNIT, scale: float = 1.0,
) -> list[str]:
    """The four labels; the standard deviation and bias are diffusion coefficients in d_unit."""
    return summary_label_texts(estimates, nominal, d_unit, scale)


def _select_fit_method(window, method: str) -> None:
    fit_method_radio(window, method.upper()).click()


@pytest.fixture
def window_with_results(window, dialogs, completed_run):
    window.simulation_finished(completed_run.as_thread_result())
    assert dialogs.titles("information") == ["Simulation Complete"]
    return window


class TestPrecisionSummary:

    @pytest.mark.parametrize("method", ["wls", "ols"])
    def test_summary_describes_the_selected_fit_method(self, window, completed_run, method):
        _select_fit_method(window, method)
        window.simulation_finished(completed_run.as_thread_result())

        estimates, nominal = d_estimates_from_csv(completed_run.csv_path, FIT_METHOD_COLUMNS[method])
        assert _summary_labels(window) == _expected_summary(estimates, nominal)
        assert window.proximity_fraction_label.text() == fraction_sentence(estimates, nominal, 0.10)

    @pytest.mark.parametrize("method", ["ols", "wls"])
    def test_changing_fit_method_after_a_run_updates_the_summary(
        self, window_with_results, completed_run, method,
    ):
        _select_fit_method(window_with_results, "ols" if method == "wls" else "wls")
        window_with_results.proximity_spin.setValue(0.2)

        _select_fit_method(window_with_results, method)

        estimates, nominal = d_estimates_from_csv(completed_run.csv_path, FIT_METHOD_COLUMNS[method])
        assert _summary_labels(window_with_results) == _expected_summary(estimates, nominal)
        assert window_with_results.proximity_fraction_label.text() == fraction_sentence(estimates, nominal, 0.2)

    def test_results_enable_the_threshold_tool(self, window_with_results):
        assert window_with_results.precision_placeholder.isHidden()
        assert window_with_results.explore_proximity_group.isEnabled()

    def test_run_results_are_held_in_memory_not_as_a_loaded_file(self, window_with_results):
        assert window_with_results.results_status_label.text() == "Simulation results: quick_test (30 runs)"
        assert window_with_results.loaded_data_file is None
        assert window_with_results.interface.last_result.num_runs == 30

    def test_status_bar_names_a_loaded_file_then_the_next_run(self, window, dialogs, completed_run, work_dir):
        """The status bar names the results that the Analysis and Output tabs work on."""
        loaded_csv = work_dir / "earlier_run.csv"
        shutil.copy(completed_run.csv_path, loaded_csv)
        dialogs.open_paths.append(str(loaded_csv))
        window.load_results_button.click()
        assert window.results_status_label.text() == "Loaded: earlier_run.csv"

        window.simulation_finished(completed_run.as_thread_result())

        assert window.results_status_label.text() == "Simulation results: quick_test (30 runs)"


def _bar_pixel_counts(histogram) -> tuple[dict, dict]:
    """Count the pixels of the highlighted and of the other bar color in the painted histogram.

    Returns, for each of the two colors, the number of pixels and the set of
    pixel columns that hold the color. The bars are drawn without
    antialiasing, so their pixels have the exact colors.
    """
    image = histogram.grab().toImage()
    scale = image.devicePixelRatio()
    colors = [color.rgb() for color in histogram.bar_colors()]
    found = [{"pixels": 0, "columns": set()} for _ in colors]
    for y in range(image.height()):
        for x in range(image.width()):
            pixel = image.pixel(x, y)
            for color, entry in zip(colors, found):
                if pixel == color:
                    entry["pixels"] += 1
                    entry["columns"].add(int(x / scale))
    return found[0], found[1]


class TestAccuracyThresholdTool:

    @pytest.mark.parametrize("threshold", [0.01, 0.05, 0.2, 0.5])
    def test_fraction_follows_the_threshold(self, window_with_results, completed_run, threshold):
        window_with_results.proximity_spin.setValue(threshold)

        estimates, nominal = d_estimates_from_csv(completed_run.csv_path, WLS_SLOPE_COLUMN)
        assert window_with_results.proximity_pct_label.text() == f"(±{round(threshold * 100)}%)"
        assert window_with_results.proximity_fraction_label.text() == fraction_sentence(estimates, nominal, threshold)

    def test_fraction_never_decreases_as_the_threshold_widens(self, window_with_results):
        fractions = []
        for threshold in np.arange(0.01, 0.51, 0.07):
            window_with_results.proximity_spin.setValue(float(threshold))
            text = window_with_results.proximity_fraction_label.text()
            fractions.append(float(text.split("of nominal: ")[1].split("%")[0]))
        assert fractions == sorted(fractions)

    def test_histogram_shows_the_estimates_of_the_selected_fit(self, window_with_results, completed_run):
        histogram = window_with_results.proximity_histogram
        for abbreviation, column in (("WLS", WLS_SLOPE_COLUMN), ("OLS", OLS_SLOPE_COLUMN)):
            fit_method_radio(window_with_results, abbreviation).setChecked(True)
            estimates, nominal = d_estimates_from_csv(completed_run.csv_path, column)
            np.testing.assert_allclose(histogram.ratios(), estimates / nominal, rtol=1e-9)
            counts, edges = histogram.histogram()
            assert counts.sum() + histogram.count_outside_axis() == len(estimates)
            assert edges[0] < 1 < edges[-1]
            assert (edges[0] + edges[-1]) / 2 == pytest.approx(1.0)

    @pytest.mark.parametrize("threshold", [0.01, 0.05, 0.1, 0.2, 0.5])
    def test_highlighted_part_of_the_bars_is_the_fraction_within_the_threshold(
            self, window_with_results, completed_run, threshold):
        histogram = window_with_results.proximity_histogram
        histogram.resize(900, 300)
        window_with_results.proximity_spin.setValue(threshold)
        highlight, muted = _bar_pixel_counts(histogram)

        # Bar height is the count of a bin, and each bar is highlighted up to
        # the share of its estimates within the threshold, so the highlighted
        # share of the bar area is the fraction of the drawn estimates within it.
        counts, _ = histogram.histogram()
        within = histogram.counts_within_threshold()
        assert (within <= counts).all()
        expected = within.sum() / counts.sum()
        ratios = histogram.ratios()
        low, high = histogram.axis_range()
        drawn = ratios[(ratios >= low) & (ratios <= high)]
        assert expected == pytest.approx(np.mean((drawn >= 1 - threshold) & (drawn <= 1 + threshold)))
        assert highlight["pixels"] + muted["pixels"] > 0
        assert highlight["pixels"] / (highlight["pixels"] + muted["pixels"]) == pytest.approx(expected, abs=0.03)

        # A highlighted part is the foot of its bar: same columns, never taller.
        bars = {bar.left(): bar for bar in histogram.bar_rects()}
        for part in histogram.highlight_rects():
            bar = bars[part.left()]
            assert part.width() == bar.width() and part.height() <= bar.height()
            assert part.top() + part.height() == bar.top() + bar.height()

    def test_highlight_grows_with_the_threshold(self, window_with_results):
        histogram = window_with_results.proximity_histogram
        highlighted = []
        for threshold in (0.01, 0.05, 0.2, 0.5):
            window_with_results.proximity_spin.setValue(threshold)
            highlighted.append(_bar_pixel_counts(histogram)[0]["pixels"])
        assert highlighted == sorted(highlighted)
        assert highlighted[0] < highlighted[-1]

    def test_axis_does_not_move_when_the_threshold_changes(self, window_with_results):
        histogram = window_with_results.proximity_histogram
        window_with_results.proximity_spin.setValue(0.01)
        narrow = histogram.axis_range()
        window_with_results.proximity_spin.setValue(0.5)
        assert histogram.axis_range() == narrow

    def test_histogram_describes_itself_for_a_screen_reader(self, window_with_results, completed_run):
        window_with_results.proximity_spin.setValue(0.2)
        estimates, nominal = d_estimates_from_csv(completed_run.csv_path, WLS_SLOPE_COLUMN)
        within = int(round(fraction_within(estimates, nominal, 0.2) * len(estimates)))
        description = window_with_results.proximity_histogram.accessibleDescription()
        assert f"Histogram of {len(estimates)} estimates" in description
        assert f"{within} of them are within ±20%" in description

    def test_axis_is_centered_on_nominal_and_counts_the_estimates_beyond_it(
            self, window_with_results, completed_run):
        histogram = window_with_results.proximity_histogram
        estimates, nominal = d_estimates_from_csv(completed_run.csv_path, WLS_SLOPE_COLUMN)
        distance = np.abs(estimates / nominal - 1)

        low, high = histogram.axis_range()

        assert high - 1 == pytest.approx(1.08 * np.percentile(distance, 99.5))
        assert 1 - low == pytest.approx(high - 1)
        assert histogram.count_outside_axis() == np.count_nonzero(distance > high - 1)

    def test_values_that_are_not_numbers_are_left_out(self, window_with_results, completed_run):
        histogram = window_with_results.proximity_histogram
        estimates, nominal = d_estimates_from_csv(completed_run.csv_path, WLS_SLOPE_COLUMN)
        ratios = estimates / nominal
        axis_range = histogram.axis_range()

        histogram.set_ratios(np.concatenate([ratios, [np.nan, np.inf, -np.inf]]))

        np.testing.assert_allclose(histogram.ratios(), ratios, rtol=1e-9)
        assert histogram.axis_range() == pytest.approx(axis_range)
        assert not histogram.grab().isNull()

    def test_histogram_without_results_shows_no_bars(self, window):
        histogram = window.proximity_histogram
        assert histogram.ratios().size == 0
        assert histogram.accessibleDescription() == "No estimates to show."
        highlight, muted = _bar_pixel_counts(histogram)
        assert highlight["pixels"] == muted["pixels"] == 0


class TestRegeneratePlot:

    def test_without_results_it_offers_to_load_and_writes_nothing(self, window, dialogs, work_dir):
        dialogs.question_answers.append(QMessageBox.StandardButton.No)
        trigger_menu_action(window, "&Plots", "&Accuracy Histogram")

        assert dialogs.titles("question") == ["No Data in Memory"]
        assert dialogs.titles("open-file") == []
        assert list(work_dir.iterdir()) == []

    def test_without_results_accepting_opens_the_results_dialog(self, window, dialogs, work_dir):
        trigger_menu_action(window, "&Plots", "&Accuracy Histogram")

        assert dialogs.titles("open-file") == ["Select Results CSV File"]
        assert dialogs.titles("critical") == []
        assert list(work_dir.iterdir()) == []

    def test_with_results_writes_the_selected_image_type(self, window_with_results, dialogs, work_dir):
        window_with_results.image_type_combo.setCurrentText("svg")
        window_with_results.regenerate_plot_button.click()

        plot = work_dir / "output" / "quick_test" / "quick_test_accuracy_histogram.svg"
        assert plot.is_file() and plot.read_text().lstrip().startswith("<?xml")
        assert dialogs.titles("information")[-1] == "Plot Generated"

    def test_declining_overwrite_then_cancelling_writes_nothing(self, window_with_results, dialogs, work_dir):
        window_with_results.regenerate_plot_button.click()
        plot = work_dir / "output" / "quick_test" / "quick_test_accuracy_histogram.png"
        first_bytes = plot.read_bytes()
        dialogs.question_answers.append(QMessageBox.StandardButton.No)

        window_with_results.regenerate_plot_button.click()

        assert dialogs.titles("question") == ["File Exists"]
        assert dialogs.titles("save-file") == ["Save Plot As"]
        assert dialogs.titles("critical") == []
        assert plot.read_bytes() == first_bytes


class TestLoadResults:

    def test_exported_csv_reloads_into_the_analysis_tab(self, window, dialogs, completed_run, work_dir):
        loaded_csv = work_dir / "earlier_run.csv"
        shutil.copy(completed_run.csv_path, loaded_csv)
        dialogs.open_paths.append(str(loaded_csv))

        window.load_results_button.click()

        assert dialogs.titles("information") == ["Data Loaded and Plot Generated"]
        assert window.results_status_label.text() == "Loaded: earlier_run.csv"
        assert (work_dir / "earlier_run_accuracy_histogram.png").is_file()
        estimates, nominal = d_estimates_from_csv(loaded_csv, WLS_SLOPE_COLUMN)
        # The CSV does not name the simulation units, so the statistics take the
        # unit of its labeled nominal column: 1 micrometer^2/ns is 10 cm^2/s.
        frame = pd.read_csv(loaded_csv)
        scale = frame["nominal diffusion coeff [cm²/s]"].iloc[0] / frame[NOMINAL_D_COLUMN].iloc[0]
        assert scale == pytest.approx(10, rel=1e-12)
        assert _summary_labels(window) == _expected_summary(estimates, nominal, "cm²/s", scale)
        assert window.explore_proximity_group.isEnabled()

    def test_csv_without_results_columns_is_reported(self, window, dialogs, work_dir):
        profile_copy = work_dir / "profile.csv"
        shutil.copy(EXAMPLE_PROFILE_CSV, profile_copy)
        dialogs.open_paths.append(str(profile_copy))

        window.load_results_button.click()

        assert dialogs.titles("critical") == ["Error"]
        assert dialogs.texts("critical")[0].startswith(
            "The file has no column 'weighted fit diffusion slope' of the WLS fit.")
        assert window.results_status_label.text() == "No data loaded"
        assert window.interface.last_result is None
        assert sorted(p.name for p in work_dir.iterdir()) == ["profile.csv"]

    def test_cancelled_dialog_changes_nothing(self, window, dialogs):
        window.load_results_button.click()
        assert dialogs.titles() == ["Select Results CSV File"]
        assert window.interface.last_result is None


class TestFitWithoutEstimates:
    """Selecting a fit that has no estimates clears the numbers of the other fit."""

    def test_summary_and_histogram_are_cleared(self, window, dialogs, completed_run, work_dir):
        frame = pd.read_csv(completed_run.csv_path)
        weighted_only = work_dir / "weighted_only.csv"
        frame[[c for c in frame.columns if not c.startswith("unweighted fit")]].to_csv(weighted_only, index=False)
        dialogs.open_paths.append(str(weighted_only))
        window.load_results_button.click()
        assert "—" not in _summary_labels(window)
        assert window.proximity_histogram.ratios().size == len(frame)

        fit_method_radio(window, "OLS").setChecked(True)

        assert _summary_labels(window) == ["—"] * 4
        assert window.proximity_fraction_label.text() == ""
        assert window.proximity_histogram.ratios().size == 0
        assert not window.precision_placeholder.isHidden()
        assert window.precision_placeholder.text() == "The selected fit has no estimates in these results."
        assert not window.explore_proximity_group.isEnabled()

        fit_method_radio(window, "WLS").setChecked(True)

        assert "—" not in _summary_labels(window)
        assert window.proximity_histogram.ratios().size == len(frame)


class TestPlotThatCannotBeSaved:
    """A folder stands where the plot file would go: both plot paths of the bridge report it."""

    def test_plot_of_a_results_file_and_plot_from_memory(self, completed_run, tmp_path):
        from dice_gui.dice_interface import DiceInterface

        blocked = tmp_path / "plot.png"
        blocked.mkdir()
        interface = DiceInterface()

        with pytest.raises(RuntimeError, match="Failed to load and plot from CSV: .*plot.png"):
            interface.load_and_plot_from_csv(str(completed_run.csv_path), str(blocked), {"image_type": "png"})

        interface.load_and_plot_from_csv(
            str(completed_run.csv_path), str(tmp_path / "written.png"), {"image_type": "png"})
        assert (tmp_path / "written.png").is_file()
        with pytest.raises(RuntimeError, match="Failed to regenerate plot: .*plot.png"):
            interface.regenerate_plot_from_memory(str(blocked), {"image_type": "png"})


@pytest.mark.parametrize("value, uncertainty, text", [
    (-0.000415, 0.00087, "-0.00042 ± 0.00087"),
    (3.1e-05, 0.00087, "0.00003 ± 0.00087"),
    (0.0047, 0.47, "0.00 ± 0.47"),
    (12.345, 2.5, "12.3 ± 2.5"),
    (1234.6, 250.0, "1235 ± 250"),
    (3.1e-10, 8.7e-10, "(3.1 ± 8.7)e-10"),
    # A value much larger than its uncertainty keeps both digits of the uncertainty.
    # The power of ten is that of the larger number, as "5.0e-10" alone would be written.
    (5.0e-10, 4.6e-12, "(5.000 ± 0.046)e-10"),
    (3e-7, 2e-9, "(3.000 ± 0.020)e-7"),
    # A value that rounds to zero has no minus sign.
    (-0.00001, 0.47, "0.00 ± 0.47"),
    (0.5, float("nan"), "0.5"),
    (0.5, 0.0, "0.5"),
])
def test_value_and_uncertainty_share_one_notation_and_decimal_place(value, uncertainty, text):
    from dice_gui.tabs.tab_analysis import value_with_uncertainty

    assert value_with_uncertainty(value, uncertainty) == text


class TestTextFromAResultsFile:
    """The unit of a results file reaches the screen only if it is a unit DICE writes, and never as markup."""

    def test_results_file_of_dice_1_2_shows_its_unit(self, completed_run, tmp_path):
        """1.2.0 wrote the slopes without a unit and the diffusion coefficients in 'cm^2/s'."""
        from dice_gui.dice_interface import DiceInterface

        frame = pd.read_csv(completed_run.csv_path)
        frame = frame.rename(columns=lambda c: c.replace(" (2D)", "").replace("cm²/s", "cm^2/s")
                             if "cm²/s" in c else c.split(" [")[0].replace(" (2D)", ""))
        assert "nominal diffusion coeff" in frame.columns and "nominal diffusion coeff [cm^2/s]" in frame.columns
        old_file = tmp_path / "results_1_2.csv"
        frame.to_csv(old_file, index=False)

        interface = DiceInterface()
        interface.load_and_plot_from_csv(str(old_file), str(tmp_path / "plot.png"), {"image_type": "png"})

        scale, unit = interface.diffusion_coefficient_unit()
        assert unit == "cm^2/s" and scale == pytest.approx(10, rel=1e-12)

    def test_unit_that_dice_does_not_write_is_not_shown(self, window, dialogs, completed_run, work_dir):
        frame = pd.read_csv(completed_run.csv_path)
        markup = '<img src="//host/share/x.png">'
        crafted = work_dir / "crafted.csv"
        frame.rename(columns={"nominal diffusion coeff [cm²/s]": f"nominal diffusion coeff [{markup}]"}).to_csv(
            crafted, index=False)
        dialogs.open_paths.append(str(crafted))

        window.load_results_button.click()

        assert window.interface.diffusion_coefficient_unit() == (pytest.approx(10, rel=1e-12), None)
        assert all(markup not in text for text in _summary_labels(window))
        assert "<" not in window.precision_std_label.text()
        for label in (window.precision_std_label, window.precision_bias_label, window.results_status_label):
            assert label.textFormat() == Qt.TextFormat.PlainText

    @pytest.mark.parametrize("label, known", [
        ("cm²/s", True), ("μm²/ns", True), ("nm²/ps", True), ("cm^2/s", True),
        ("furlong²/s", False), ('<img src="x">', False), ("", False), (None, False),
    ])
    def test_only_labels_built_from_the_unit_tables_are_known(self, label, known):
        from dice.utils.units import known_diffusion_coefficient_label

        assert known_diffusion_coefficient_label(label) == (label if known else None)
