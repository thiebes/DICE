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
from PyQt6.QtWidgets import QMessageBox

from gui_support import (
    EXAMPLE_PROFILE_CSV, NOMINAL_D_COLUMN, OLS_SLOPE_COLUMN, WLS_SLOPE_COLUMN,
    d_estimates_from_csv, fit_method_radio, fraction_within, trigger_menu_action,
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
    return [
        f"{np.std(estimates) * scale:.5e} {d_unit}",
        f"{np.std(estimates) / nominal:.3f}",
        f"{(np.mean(estimates) - nominal) * scale:.5e} {d_unit}",
        f"{np.mean(estimates) / nominal:.3f}",
    ]


def _fraction_text(fraction: float, threshold: float) -> str:
    return f"{fraction:.1%} of estimates fall within ±{round(threshold * 100)}% of nominal"


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
        assert window.proximity_fraction_label.text() == _fraction_text(
            fraction_within(estimates, nominal, 0.10), 0.10
        )

    @pytest.mark.parametrize("method", ["ols", "wls"])
    def test_changing_fit_method_after_a_run_updates_the_summary(
        self, window_with_results, completed_run, method,
    ):
        _select_fit_method(window_with_results, "ols" if method == "wls" else "wls")
        window_with_results.proximity_spin.setValue(0.2)

        _select_fit_method(window_with_results, method)

        estimates, nominal = d_estimates_from_csv(completed_run.csv_path, FIT_METHOD_COLUMNS[method])
        assert _summary_labels(window_with_results) == _expected_summary(estimates, nominal)
        assert window_with_results.proximity_fraction_label.text() == _fraction_text(
            fraction_within(estimates, nominal, 0.2), 0.2
        )

    def test_results_enable_the_threshold_tool(self, window_with_results):
        assert window_with_results.precision_placeholder.isHidden()
        assert window_with_results.explore_proximity_group.isEnabled()

    def test_run_results_are_held_in_memory_not_as_a_loaded_file(self, window_with_results):
        assert window_with_results.status_bar.currentMessage() == "No data loaded"
        assert window_with_results.interface.last_result.num_runs == 30


class TestAccuracyThresholdTool:

    @pytest.mark.parametrize("threshold", [0.01, 0.05, 0.2, 0.5])
    def test_fraction_follows_the_threshold(self, window_with_results, completed_run, threshold):
        window_with_results.proximity_spin.setValue(threshold)

        estimates, nominal = d_estimates_from_csv(completed_run.csv_path, WLS_SLOPE_COLUMN)
        assert window_with_results.proximity_pct_label.text() == f"(±{round(threshold * 100)}%)"
        assert window_with_results.proximity_fraction_label.text() == _fraction_text(
            fraction_within(estimates, nominal, threshold), threshold
        )

    def test_fraction_never_decreases_as_the_threshold_widens(self, window_with_results):
        fractions = []
        for threshold in np.arange(0.01, 0.51, 0.07):
            window_with_results.proximity_spin.setValue(float(threshold))
            fractions.append(float(window_with_results.proximity_fraction_label.text().split("%")[0]))
        assert fractions == sorted(fractions)

    def test_target_graphic_paints_across_the_threshold_range(self, window_with_results):
        target = window_with_results.proximity_target
        for threshold in (window_with_results.proximity_spin.minimum(), 0.1, window_with_results.proximity_spin.maximum()):
            window_with_results.proximity_spin.setValue(threshold)
            assert not target.grab().isNull()


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
        assert window.status_bar.currentMessage() == "Loaded: earlier_run.csv"
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
        assert dialogs.texts("critical")[0].startswith("CSV missing WLS slope column.")
        assert window.status_bar.currentMessage() == "No data loaded"
        assert window.interface.last_result is None
        assert sorted(p.name for p in work_dir.iterdir()) == ["profile.csv"]

    def test_cancelled_dialog_changes_nothing(self, window, dialogs):
        window.load_results_button.click()
        assert dialogs.titles() == ["Select Results CSV File"]
        assert window.interface.last_result is None
