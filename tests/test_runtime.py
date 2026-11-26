"""
Tests for dice.utils.runtime: what a standalone (frozen) build changes.
"""

import sys
from pathlib import Path

from dice.utils.runtime import (
    LICENSE_NOTICE_FILE_NAME, bundled_license_notice, examples_directory, is_standalone_build,
    joblib_workers_supported,
)

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


class TestBundledLicenseNotice:
    """Only a standalone build carries the license notice of its bundled packages."""

    def test_installed_copy_has_no_notice(self, monkeypatch, tmp_path):
        monkeypatch.delattr(sys, 'frozen', raising=False)
        monkeypatch.setattr(sys, '_MEIPASS', str(tmp_path), raising=False)
        (tmp_path / LICENSE_NOTICE_FILE_NAME).write_text('notice', encoding='utf-8')
        assert bundled_license_notice() is None

    def test_standalone_build_finds_the_notice_in_its_bundle(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, 'frozen', True, raising=False)
        monkeypatch.setattr(sys, '_MEIPASS', str(tmp_path), raising=False)
        (tmp_path / LICENSE_NOTICE_FILE_NAME).write_text('notice', encoding='utf-8')
        assert bundled_license_notice() == tmp_path / LICENSE_NOTICE_FILE_NAME

    def test_standalone_build_without_the_file_reports_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, 'frozen', True, raising=False)
        monkeypatch.setattr(sys, '_MEIPASS', str(tmp_path), raising=False)
        assert bundled_license_notice() is None

    def test_file_name_matches_the_one_the_build_writes(self):
        sys.path.insert(0, str(REPO_ROOT / 'scripts'))
        try:
            import collect_licenses
        finally:
            sys.path.pop(0)
        assert collect_licenses.NOTICE_FILE_NAME == LICENSE_NOTICE_FILE_NAME
