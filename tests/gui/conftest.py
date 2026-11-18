"""
Fixtures for the PyQt6 GUI tests.

Every test here runs headless, writes outputs and Qt settings only under
pytest temporary directories, and never blocks on a modal dialog: message
boxes and file dialogs are replaced by a DialogScript that records what the
GUI showed and returns scripted answers.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from gui_support import CompletedRun, quick_test_engine_parameters


class ShownDialog(NamedTuple):
    kind: str
    title: str
    text: str


@dataclass
class DialogScript:
    """Stand-in for every modal dialog: records each one and answers from scripted queues."""

    shown: list[ShownDialog] = field(default_factory=list)
    question_answers: deque[QMessageBox.StandardButton] = field(default_factory=deque)
    open_paths: deque[str] = field(default_factory=deque)
    save_paths: deque[str] = field(default_factory=deque)

    def titles(self, kind: str | None = None) -> list[str]:
        return [dialog.title for dialog in self.shown if kind is None or dialog.kind == kind]

    def texts(self, kind: str | None = None) -> list[str]:
        return [dialog.text for dialog in self.shown if kind is None or dialog.kind == kind]

    def message(self, kind: str):
        def show(parent: Any, title: str, text: str, *args: Any, **kwargs: Any):
            self.shown.append(ShownDialog(kind, title, text))
            return QMessageBox.StandardButton.Ok
        return show

    def question(self, parent: Any, title: str, text: str, *args: Any, **kwargs: Any):
        self.shown.append(ShownDialog("question", title, text))
        if self.question_answers:
            return self.question_answers.popleft()
        return QMessageBox.StandardButton.Yes

    def open_file(self, parent: Any, caption: str = "", directory: str = "", *args: Any, **kwargs: Any):
        self.shown.append(ShownDialog("open-file", caption, directory))
        return (self.open_paths.popleft() if self.open_paths else "", "")

    def save_file(self, parent: Any, caption: str = "", directory: str = "", *args: Any, **kwargs: Any):
        self.shown.append(ShownDialog("save-file", caption, directory))
        return (self.save_paths.popleft() if self.save_paths else "", "")


@pytest.fixture(scope="session")
def qapp_args() -> list[str]:
    """Run Qt offscreen unless QT_QPA_PLATFORM already names a platform (pytest-qt qapp_args hook)."""
    args = ["dice-gui-tests"]
    if "QT_QPA_PLATFORM" not in os.environ:
        args += ["-platform", "offscreen"]
    return args


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Keep the recent-files list and other QSettings out of the developer's registry."""
    settings_dir = tmp_path_factory.mktemp("qsettings")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(settings_dir))


@pytest.fixture(autouse=True)
def work_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run each test in a temporary directory, since the GUI writes outputs under ./output."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def dialogs(monkeypatch: pytest.MonkeyPatch) -> DialogScript:
    """
    Replace modal dialogs for the whole test.

    Autouse makes this fixture set up before qtbot, so it is torn down after
    qtbot closes the test's widgets and a close-time prompt cannot block.
    """
    script = DialogScript()
    for kind in ("information", "warning", "critical", "about"):
        monkeypatch.setattr(QMessageBox, kind, staticmethod(script.message(kind)))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(script.question))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(script.open_file))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(script.save_file))
    return script


@pytest.fixture
def window(qtbot):
    from dice_gui.dice_gui import DiceGUI

    gui = DiceGUI()
    qtbot.addWidget(gui)
    return gui


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory: pytest.TempPathFactory) -> CompletedRun:
    """One real serial simulation of the quick_test example, shared by a module's tests."""
    from dice_gui.dice_interface import DiceInterface

    run_root = tmp_path_factory.mktemp("completed_run")
    parameters = quick_test_engine_parameters(number_of_runs=30)
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(run_root)
        result, processed_parameters = DiceInterface().run_simulation(parameters)
    slug = parameters["filename slug"]
    return CompletedRun(result, processed_parameters, run_root / "output" / slug, slug)
