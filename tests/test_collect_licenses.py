"""
Tests for scripts/collect_licenses.py, run on the packages installed in the
test environment, which has the dependencies a standalone application bundles.
"""

import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "collect_licenses.py"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
try:
    import collect_licenses
finally:
    sys.path.pop(0)


@pytest.fixture(scope="module")
def notice() -> str:
    return collect_licenses.collect_license_notice("9.8.7")


def entry(notice: str, package: str) -> str:
    """The entry of one package: from its heading to the next package heading."""
    version = metadata.version(package)
    heading = f"{collect_licenses.THIN_RULE}\n{package} {version}\nDeclared license: "
    assert notice.count(heading) == 1, package
    body = notice.split(heading)[1]
    return body.split(f"\n{collect_licenses.THIN_RULE}\n", 2)[0] + body.split(f"\n{collect_licenses.THIN_RULE}\n", 2)[1]


class TestNoticeContent:

    def test_header_states_the_gpl_and_the_source(self, notice):
        header = notice.split("1. DICE")[0]
        assert header.startswith("DICE 9.8.7: licenses of the standalone application")
        assert "distributed under the\nterms of the GPL version 3" in header
        assert "https://github.com/thiebes/DICE\nat the tag v9.8.7" in header
        assert "uv.lock" in header

    def test_header_of_a_development_build_names_no_tag(self):
        header = collect_licenses.notice_header("9.8.7.dev0", packaged_with_pyinstaller=True, qt_version="6.10.0")

        assert header.startswith("DICE 9.8.7.dev0: licenses of the standalone application")
        assert ("https://github.com/thiebes/DICE\n"
                "at the commit that this development build was made from, which has no tag.\n") in header
        assert "at the tag" not in header

    def test_header_names_both_files_that_pin_the_packages(self, notice):
        """Every installed package is pinned in one of the two files the header names."""
        import re
        header = notice.split("1. DICE")[0]
        root = Path(collect_licenses.REPOSITORY_ROOT)
        assert collect_licenses.BUILD_REQUIREMENTS_FILE in header
        assert (root / collect_licenses.BUILD_REQUIREMENTS_FILE).is_file()

        def normalized(name: str) -> str:
            return re.sub(r"[-_.]+", "-", name).lower()

        locked = set(re.findall(r'^name = "([^"]+)"', (root / "uv.lock").read_text(encoding="utf-8"), re.M))
        build = {normalized(name) for name in re.findall(
            r"^([A-Za-z0-9_.-]+)==", (root / collect_licenses.BUILD_REQUIREMENTS_FILE).read_text(encoding="utf-8"),
            re.M)}
        installed = {normalized(d.metadata["Name"]) for d in collect_licenses.installed_packages()}
        assert installed - locked - build == set()
        assert "pyinstaller" in build and "pyinstaller" not in locked

    def test_header_says_where_the_source_of_qt_is(self, notice):
        header = notice.split("1. DICE")[0]
        qt_version = metadata.version("PyQt6-Qt6")
        major_minor = ".".join(qt_version.split(".")[:2])
        assert f"the Qt libraries, version {qt_version}, in compiled" in header
        assert f"https://download.qt.io/archive/qt/{major_minor}/{qt_version}/single/" in header
        assert collect_licenses.qt_source_url("6.10.0") == "https://download.qt.io/archive/qt/6.10/6.10.0/single/"

    def test_dice_is_under_its_own_mit_license(self, notice):
        mit_text = (REPO_ROOT / "LICENSE.md").read_text(encoding="utf-8").strip()
        assert mit_text in notice
        assert f"{collect_licenses.THIN_RULE}\ndice " not in notice.lower()

    def test_pyqt6_entry_holds_the_text_of_the_gpl(self, notice):
        pyqt6 = entry(notice, "PyQt6")
        assert pyqt6.startswith("GPL-3.0-only")
        assert "GNU GENERAL PUBLIC LICENSE" in pyqt6
        assert "Version 3, 29 June 2007" in pyqt6
        assert "END OF TERMS AND CONDITIONS" in pyqt6

    def test_qt_entry_holds_the_text_of_the_lgpl(self, notice):
        qt = entry(notice, "PyQt6-Qt6")
        assert qt.startswith("LGPL v3")
        assert "GNU LESSER GENERAL PUBLIC LICENSE" in qt

    @pytest.mark.parametrize("package", ["numpy", "scipy", "pandas", "matplotlib", "seaborn", "statsmodels",
                                         "joblib", "fft-cnr"])
    def test_every_runtime_dependency_has_an_entry_with_a_license_text(self, notice, package):
        text = entry(notice, package)
        assert "The package installs no license file" not in text
        assert "[" in text and len(text) > 500

    def test_every_installed_package_is_listed_once(self, notice):
        names = {distribution.metadata["Name"] for distribution in metadata.distributions()}
        names -= {"dice", None}
        for name in names:
            assert notice.count(f"{collect_licenses.THIN_RULE}\n{name} ") == 1, name


class TestWriting:

    def test_script_writes_the_notice_as_utf8(self, tmp_path):
        output = tmp_path / "build" / collect_licenses.NOTICE_FILE_NAME

        completed = subprocess.run([sys.executable, str(SCRIPT), str(output)], capture_output=True, text=True)

        assert completed.returncode == 0, completed.stderr
        text = output.read_bytes().decode("utf-8")
        assert text.startswith(f"DICE {collect_licenses.project_version()}: licenses")
        assert b"\r\n" not in output.read_bytes()

    def test_environment_without_pyqt6_stops_the_build(self, tmp_path, monkeypatch):
        without_pyqt = [distribution for distribution in metadata.distributions()
                        if not distribution.metadata["Name"].lower().startswith("pyqt6")]
        monkeypatch.setattr(collect_licenses.metadata, "distributions", lambda: without_pyqt)
        output = tmp_path / collect_licenses.NOTICE_FILE_NAME

        with pytest.raises(SystemExit, match="PyQt6 is not installed; PyQt6-Qt6 is not installed"):
            collect_licenses.write_license_notice(output, "9.8.7")

        assert not output.exists()

    def test_pyqt6_without_its_license_file_stops_the_build(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_licenses, "license_files", lambda distribution: [("LICENSE", "MIT License")])

        with pytest.raises(SystemExit, match="PyQt6 has no license file with the text of the GNU GENERAL"):
            collect_licenses.write_license_notice(tmp_path / collect_licenses.NOTICE_FILE_NAME, "9.8.7")

    def test_script_without_an_output_file_prints_its_usage(self):
        completed = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
        assert completed.returncode == 2
        assert "collect_licenses.py OUTPUT_FILE" in completed.stderr
