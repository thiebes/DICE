"""
End-to-end runs started from the GUI: what it exports and what it reports.
"""

from __future__ import annotations

import pandas as pd
import pytest

from gui_support import WLS_SLOPE_COLUMN, choose_global_units, enter_text, run_and_wait

pytestmark = pytest.mark.gui


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
