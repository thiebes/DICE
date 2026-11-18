"""
End-to-end runs started from the GUI: what it exports and what it reports.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gui_support import (
    EXAMPLES_DIR, NOMINAL_D_COLUMN, OLS_SLOPE_COLUMN, WLS_SLOPE_COLUMN,
    choose_global_units, d_estimates_from_csv, enter_text, fraction_within, run_and_wait,
)

pytestmark = pytest.mark.gui


class TestRunFromTheWindow:
    """Load the quick_test example, run it from the Run button, and follow the results."""

    RUNS = 10

    @pytest.fixture
    def finished_window(self, qtbot, window, dialogs):
        window.load_parameters_from_file(str(EXAMPLES_DIR / "quick_test.txt"))
        window.num_runs_spin.setValue(self.RUNS)
        window.multiprocessing_check.setChecked(False)
        run_and_wait(qtbot, window, dialogs)
        return window

    def test_controls_return_to_ready(self, finished_window):
        assert finished_window.status_label.text() == "Simulation completed successfully!"
        assert finished_window.run_button.isEnabled()
        assert not finished_window.stop_button.isEnabled()
        assert finished_window.progress_bar.isHidden()
        assert finished_window.progress_bar.maximum() == self.RUNS
        assert finished_window.simulation_thread is None
        assert finished_window.status_bar.currentMessage() == "No data loaded"

    def test_files_named_in_the_output_preview_exist(self, finished_window):
        preview = finished_window.output_path_preview.text()
        directory = Path(re.search(r"<b>Directory:</b> (.*?)<br>", preview).group(1))
        names = re.search(r"<b>Files:</b> (.*)$", preview).group(1).split(", ")
        assert names == ["quick_test.csv", "quick_test_summary.txt", "quick_test_accuracy_histogram.png"]
        assert all((directory / name).is_file() for name in names)

    def test_exported_csv_has_one_row_per_run_and_the_columns_the_gui_reads(self, finished_window, work_dir):
        frame = pd.read_csv(work_dir / "output" / "quick_test" / "quick_test.csv")
        assert list(frame["run number"]) == list(range(self.RUNS))
        assert {WLS_SLOPE_COLUMN, OLS_SLOPE_COLUMN, NOMINAL_D_COLUMN} <= set(frame.columns)
        assert frame[NOMINAL_D_COLUMN].to_numpy() == pytest.approx(np.full(self.RUNS, 0.01))

    def test_analysis_tab_reports_statistics_of_the_exported_estimates(self, finished_window, work_dir):
        estimates, nominal = d_estimates_from_csv(
            work_dir / "output" / "quick_test" / "quick_test.csv", WLS_SLOPE_COLUMN
        )
        window = finished_window
        assert window.precision_std_label.text() == f"{np.std(estimates):.5e}"
        assert window.precision_rse_label.text() == f"{np.std(estimates) / nominal:.3f}"
        assert window.precision_bias_label.text() == f"{np.mean(estimates) - nominal:.5e}"
        assert window.precision_mean_ratio_label.text() == f"{np.mean(estimates) / nominal:.3f}"
        assert window.precision_placeholder.isHidden()
        assert window.explore_proximity_group.isEnabled()
        assert window.proximity_fraction_label.text() == (
            f"{fraction_within(estimates, nominal, 0.10):.1%} of estimates fall within ±10% of nominal"
        )


def _enter_quick_test_physics(qtbot, window, fwhm: float, spatial_width: float, d: float) -> None:
    """The quick_test example's physics, entered in whatever global units are active."""
    window.num_runs_spin.setValue(3)
    window.multiprocessing_check.setChecked(False)
    enter_text(qtbot, window.width_input, str(fwhm))
    enter_text(qtbot, window.spatial_width_input, str(spatial_width))
    window.pixel_width_input.setValue(50)
    enter_text(qtbot, window.diffusion_coeff_input, str(d))
    enter_text(qtbot, window.lifetime_input, "1")
    enter_text(qtbot, window.noise_value_input, "0.02")
    enter_text(qtbot, window.time_start_input, "0")
    enter_text(qtbot, window.time_stop_input, "1")
    window.time_steps_input.setValue(5)


class TestExportUnits:
    """The exported CSV and summary must follow the global units chosen in the GUI."""

    @pytest.mark.parametrize(
        ("length_unit", "time_unit", "fwhm", "spatial_width", "d", "d_cm2_per_s"),
        [
            pytest.param("micrometer", "nanosecond", 1.0, 5.0, 0.01, 0.1, id="micrometer-nanosecond"),
            pytest.param("nanometer", "picosecond", 1000.0, 5000.0, 10000.0, 100.0, id="nanometer-picosecond"),
        ],
    )
    def test_csv_and_summary_use_the_global_units(
        self, qtbot, window, dialogs, monkeypatch, work_dir,
        length_unit, time_unit, fwhm, spatial_width, d, d_cm2_per_s,
    ):
        choose_global_units(window, monkeypatch, length_unit, time_unit)
        enter_text(qtbot, window.filename_slug_input, "units_check")
        _enter_quick_test_physics(qtbot, window, fwhm, spatial_width, d)

        run_and_wait(qtbot, window, dialogs)

        output_dir = work_dir / "output" / "units_check"
        frame = pd.read_csv(output_dir / "units_check.csv")
        nominal_columns = [c for c in frame.columns if c.startswith("nominal diffusion coeff [")]
        assert len(nominal_columns) == 1
        assert frame[nominal_columns[0]].iloc[0] == pytest.approx(d_cm2_per_s, rel=1e-9)

        wls_columns = [c for c in frame.columns if c.startswith("weighted fit diffusion coeff [")]
        assert frame[wls_columns[0]].to_numpy() == pytest.approx(
            (frame[WLS_SLOPE_COLUMN] / 2 / d * d_cm2_per_s).to_numpy(), rel=1e-9
        )

        summary = (output_dir / "units_check_summary.txt").read_text()
        assert f"Nominal diffusion coefficient:\n    {d:.5f} {length_unit}^2/{time_unit}\n" in summary
        assert f"Spatial width:\n    {spatial_width} {length_unit}\n" in summary
        assert f"Time range:\n    0.000 to 1.000 {time_unit}\n" in summary
