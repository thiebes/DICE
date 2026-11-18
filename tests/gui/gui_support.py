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
from PyQt6.QtWidgets import QLineEdit

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


def d_estimates_from_csv(csv_path: Path, slope_column: str) -> tuple[np.ndarray, float]:
    """Diffusion coefficient estimates (slope / 2) and the nominal value, read from an exported CSV."""
    frame = pd.read_csv(csv_path)
    return frame[slope_column].dropna().to_numpy() / 2, float(frame[NOMINAL_D_COLUMN].iloc[0])


def fraction_within(estimates: np.ndarray, nominal: float, threshold: float) -> float:
    ratios = estimates / nominal
    return float(np.mean((ratios >= 1 - threshold) & (ratios <= 1 + threshold)))


def float_text(line_edit: QLineEdit) -> float:
    return float(line_edit.text())
