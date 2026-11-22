"""
Helpers shared by the GUI tests: real example inputs, user-style input, and
reference statistics computed independently from DICE's exported CSV.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLineEdit, QRadioButton

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "data" / "examples"
EXAMPLE_PARAMETER_FILES = ("parameters.txt", "quick_test.txt", "high_precision.txt", "publication.txt")
EXAMPLE_PROFILE_CSV = EXAMPLES_DIR / "example_profile.csv"

TAB_NAMES = [
    "Simulation Setup",
    "Physical Parameters",
    "Experimental Conditions",
    "Analysis Settings",
    "Output Settings",
]

# Column names the GUI reads back from the CSV that DICE exports.
WLS_SLOPE_COLUMN = "weighted fit diffusion slope"
OLS_SLOPE_COLUMN = "unweighted fit diffusion slope"
NOMINAL_D_COLUMN = "nominal diffusion coeff"


@dataclass
class CompletedRun:
    result: Any
    processed_parameters: dict[str, Any]
    output_dir: Path
    slug: str

    @property
    def csv_path(self) -> Path:
        return self.output_dir / f"{self.slug}.csv"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / f"{self.slug}_summary.txt"

    def as_thread_result(self) -> tuple[Any, dict[str, Any]]:
        """The (result, processed parameters) pair SimulationThread.finished carries."""
        return self.result, self.processed_parameters


def read_example(filename: str) -> dict[str, Any]:
    return ast.literal_eval((EXAMPLES_DIR / filename).read_text())


def quick_test_engine_parameters(number_of_runs: int) -> dict[str, Any]:
    """The bundled quick_test example, run serially with fewer runs to keep tests fast."""
    parameters = read_example("quick_test.txt")
    parameters["number of runs"] = number_of_runs
    parameters["multiprocessing"] = False
    return parameters


def enter_text(qtbot, line_edit: QLineEdit, text: str) -> None:
    """Replace a line edit's text by typing, then press Return to commit it like a user."""
    line_edit.clear()
    qtbot.keyClicks(line_edit, text)
    qtbot.keyClick(line_edit, Qt.Key.Key_Return)


def fit_method_radio(window, abbreviation: str) -> QRadioButton:
    """The Analysis tab's fit-method radio whose label holds '(WLS)' or '(OLS)'."""
    analysis_tab = window.tabs.widget(TAB_NAMES.index("Analysis Settings"))
    return next(
        radio for radio in analysis_tab.findChildren(QRadioButton)
        if f"({abbreviation})" in radio.text()
    )


def trigger_menu_action(window, menu_title: str, action_text: str) -> None:
    menu = next(a.menu() for a in window.menuBar().actions() if a.text() == menu_title)
    next(a for a in menu.actions() if a.text() == action_text).trigger()


def choose_global_units(window, monkeypatch, length_unit: str, time_unit: str) -> None:
    """Pick global units through Settings > Global Units, accepting the dialog as a user would."""
    from dice_gui.global_units_dialog import GlobalUnitsDialog

    def accept_choice(dialog: GlobalUnitsDialog) -> int:
        dialog.length_combo.setCurrentText(length_unit)
        dialog.time_combo.setCurrentText(time_unit)
        return int(GlobalUnitsDialog.DialogCode.Accepted)

    monkeypatch.setattr(GlobalUnitsDialog, "exec", accept_choice)
    trigger_menu_action(window, "&Settings", "&Global Units...")


def run_and_wait(qtbot, window, dialogs, timeout_ms: int = 120_000) -> None:
    """Click Run and wait until the GUI reports completion or failure."""
    outcome_titles = {"Simulation Complete", "Simulation Error"}
    qtbot.mouseClick(window.run_button, Qt.MouseButton.LeftButton)
    assert window.simulation_thread is not None, f"Run did not start: {dialogs.shown}"
    qtbot.waitUntil(lambda: bool(outcome_titles & set(dialogs.titles())), timeout=timeout_ms)
    assert "Simulation Complete" in dialogs.titles(), dialogs.shown


def d_estimates_from_csv(csv_path: Path, slope_column: str) -> tuple[np.ndarray, float]:
    """Diffusion coefficient estimates (slope / 2) and the nominal value, read from an exported CSV."""
    frame = pd.read_csv(csv_path)
    return frame[slope_column].dropna().to_numpy() / 2, float(frame[NOMINAL_D_COLUMN].iloc[0])


def fraction_within(estimates: np.ndarray, nominal: float, threshold: float) -> float:
    ratios = estimates / nominal
    return float(np.mean((ratios >= 1 - threshold) & (ratios <= 1 + threshold)))


def float_text(line_edit: QLineEdit) -> float:
    return float(line_edit.text())


def fraction_sentence(estimates: np.ndarray, nominal: float, threshold: float) -> str:
    """The Analysis tab's sentence about the estimates within a threshold.

    The count and the 95% interval are recomputed here with statsmodels,
    independently of DICE.
    """
    from statsmodels.stats.proportion import proportion_confint

    ratios = estimates / nominal
    count = int(np.sum((ratios >= 1 - threshold) & (ratios <= 1 + threshold)))
    lower, upper = proportion_confint(count, len(estimates), alpha=0.05, method="wilson")
    return (f"{count / len(estimates):.1%} of estimates fall within ±{round(threshold * 100)}% of nominal "
            f"({count} of {len(estimates)}; 95% interval {lower:.1%} to {upper:.1%})")


def summary_label_texts(estimates: np.ndarray, nominal: float, d_unit: str, scale: float = 1.0) -> list[str]:
    """The four labels of the Precision Summary, recomputed from the estimates with NumPy."""
    standard_error = np.std(estimates, ddof=1) / np.sqrt(len(estimates))
    return [
        f"{np.std(estimates, ddof=1) * scale:.3g} {d_unit}",
        f"{np.std(estimates, ddof=1) / nominal:.3f}",
        f"{(np.mean(estimates) - nominal) * scale:.3g} ± {standard_error * scale:.2g} {d_unit}",
        f"{np.mean(estimates) / nominal:.3f}",
    ]
