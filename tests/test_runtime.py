"""
Tests for dice.utils.runtime: what a standalone (frozen) build changes.
"""

import sys
from pathlib import Path

from dice.utils.runtime import examples_directory, is_standalone_build, joblib_workers_supported

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestStandaloneBuild:
    """PyInstaller marks a standalone build by setting sys.frozen."""

    def test_regular_install(self, monkeypatch):
        monkeypatch.delattr(sys, 'frozen', raising=False)
        assert is_standalone_build() is False
        assert joblib_workers_supported() is True

    def test_standalone_build(self, monkeypatch):
        monkeypatch.setattr(sys, 'frozen', True, raising=False)
        assert is_standalone_build() is True
        assert joblib_workers_supported() is False


class TestExamplesDirectory:

    def test_source_checkout_uses_the_repository_folder(self, monkeypatch):
        monkeypatch.delattr(sys, 'frozen', raising=False)
        assert examples_directory() == REPO_ROOT / 'data' / 'examples'
        assert (examples_directory() / 'quick_test.txt').is_file()

    def test_standalone_build_uses_the_unpacked_bundle(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, 'frozen', True, raising=False)
        monkeypatch.setattr(sys, '_MEIPASS', str(tmp_path), raising=False)
        assert examples_directory() == tmp_path / 'data' / 'examples'
