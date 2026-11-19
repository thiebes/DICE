"""
Where the GUI saves results: output/ in the working directory when the GUI
can write there, otherwise DICE/output/ in the user's Documents folder. Every
test runs in a temporary working directory (work_dir) and, where it matters,
a temporary home directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dice_gui import output_location
from dice_gui.dice_interface import DiceInterface
from dice_gui.output_location import is_usable_folder, output_root
from gui_support import quick_test_engine_parameters, trigger_menu_action

pytestmark = pytest.mark.gui


@pytest.fixture
def home(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temporary home directory: Path.home() reads USERPROFILE on Windows and HOME elsewhere."""
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))
    assert Path.home() == home
    return home


@pytest.fixture
def documents(home: Path) -> Path:
    documents = home / "Documents"
    documents.mkdir()
    return documents


@pytest.fixture
def started_from_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Work in the filesystem root, as an application that the macOS Finder starts does."""
    monkeypatch.chdir(Path.cwd().anchor)


def no_file_can_be_created(*args, **kwargs):
    raise PermissionError("Access is denied")


class TestOutputRoot:

    def test_writable_working_directory_holds_the_output(self, work_dir, home):
        assert output_root() == work_dir / "output"
        assert list(work_dir.iterdir()) == []
        assert list(home.iterdir()) == []

    def test_usable_folder_check_leaves_no_file(self, work_dir):
        assert is_usable_folder(work_dir) is True
        assert list(work_dir.iterdir()) == []

    def test_folder_without_write_access_is_not_usable(self, work_dir, monkeypatch):
        monkeypatch.setattr(output_location.tempfile, "TemporaryFile", no_file_can_be_created)
        assert is_usable_folder(work_dir) is False

    def test_filesystem_root_is_not_usable(self):
        assert is_usable_folder(Path(Path.cwd().anchor)) is False

    def test_started_from_the_root_saves_in_documents(self, documents, started_from_root):
        root = output_root()
        assert root == documents / "DICE" / "output"
        assert root.is_dir()

    def test_without_a_documents_folder_saves_in_the_home_folder(self, home, started_from_root):
        root = output_root()
        assert root == home / "DICE" / "output"
        assert root.is_dir()

    def test_unwritable_working_directory_saves_in_documents(self, work_dir, documents, monkeypatch):
        monkeypatch.setattr(output_location.tempfile, "TemporaryFile", no_file_can_be_created)
        assert output_root() == documents / "DICE" / "output"
        assert list(work_dir.iterdir()) == []


class TestGuiStartedFromTheRoot:
    """The run, the output preview, and the file dialogs all use the Documents fallback."""

    def test_run_writes_its_files_to_the_fallback_folder(self, documents, started_from_root):
        assert DiceInterface().run_simulation(quick_test_engine_parameters(number_of_runs=3)) is not None

        output_dir = documents / "DICE" / "output" / "quick_test"
        assert sorted(path.name for path in output_dir.iterdir()) == [
            "quick_test.csv", "quick_test_accuracy_histogram.png", "quick_test_summary.txt",
        ]

    def test_preview_shows_the_fallback_folder(self, documents, started_from_root, window):
        window.filename_slug_input.setText("fallback_check")
        assert str(documents / "DICE" / "output" / "fallback_check") in window.output_path_preview.text()

    def test_file_dialogs_open_in_the_fallback_folder(self, documents, started_from_root, window, dialogs):
        dice_folder = documents / "DICE"

        trigger_menu_action(window, "&File", "&Open Parameters...")
        trigger_menu_action(window, "&File", "Save Parameters &As...")
        trigger_menu_action(window, "&File", "Load &Results...")
        window.noise_estimate_radio.setChecked(True)
        window.noise_browse_button.click()

        assert dialogs.texts("open-file") == [str(dice_folder), str(dice_folder / "output"), str(dice_folder)]
        assert dialogs.texts("save-file") == [str(dice_folder / "parameters.txt")]
