"""Tests for scripts/bump_version.py, run against copies of the repository's own files."""

import importlib.util
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
VERSIONED_FILES = ("dice/__init__.py", "CITATION.cff", "pyproject.toml")


@pytest.fixture
def repo_copy(tmp_path: Path) -> Path:
    """Copy the script and the files it edits into an isolated directory tree."""
    for relative in ("scripts/bump_version.py", *VERSIONED_FILES):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / relative, destination)
    return tmp_path


def run_bump(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(repo / "scripts" / "bump_version.py"), *args, "--no-confirm"],
        capture_output=True,
        text=True,
    )


def current_version(repo: Path) -> str:
    content = (repo / "dice" / "__init__.py").read_text(encoding="utf-8")
    return re.search(r'__version__ = "([^"]+)"', content).group(1)


def test_same_version_with_date_succeeds_and_sets_date(repo_copy: Path) -> None:
    version = current_version(repo_copy)

    result = run_bump(repo_copy, version, "--date", "2031-02-03")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "ERROR" not in result.stdout
    citation = (repo_copy / "CITATION.cff").read_text(encoding="utf-8")
    assert re.findall(r'date-released: "([^"]+)"', citation) == ["2031-02-03"] * 2
    assert (repo_copy / "pyproject.toml").read_text(encoding="utf-8") == (
        REPO_ROOT / "pyproject.toml"
    ).read_text(encoding="utf-8")


def test_new_version_updates_every_file(repo_copy: Path) -> None:
    result = run_bump(repo_copy, "9.8.7", "--date", "2031-02-03")

    assert result.returncode == 0, result.stdout + result.stderr
    assert current_version(repo_copy) == "9.8.7"
    citation = (repo_copy / "CITATION.cff").read_text(encoding="utf-8")
    assert re.findall(r'^\s*version: "([^"]+)"', citation, flags=re.MULTILINE) == ["9.8.7"] * 2
    pyproject = (repo_copy / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^version = "9\.8\.7"$', pyproject, flags=re.MULTILINE)


def test_next_steps_show_the_version_and_the_lockfile_step(repo_copy: Path) -> None:
    result = run_bump(repo_copy, "9.8.7")

    assert "{args.version}" not in result.stdout
    assert "git tag -a v9.8.7" in result.stdout
    assert "uv lock" in result.stdout
    # The release workflow runs on a published release, not on a pushed tag.
    assert "Publish a GitHub release for v9.8.7" in result.stdout


def test_development_version_updates_every_file_and_asks_for_no_tag(repo_copy: Path) -> None:
    result = run_bump(repo_copy, "9.8.7.dev0", "--date", "2031-02-03")

    assert result.returncode == 0, result.stdout + result.stderr
    assert current_version(repo_copy) == "9.8.7.dev0"
    citation = (repo_copy / "CITATION.cff").read_text(encoding="utf-8")
    assert re.findall(r'^\s*version: "([^"]+)"', citation, flags=re.MULTILINE) == ["9.8.7.dev0"] * 2
    assert re.findall(r'date-released: "([^"]+)"', citation) == ["2031-02-03"] * 2
    pyproject = (repo_copy / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^version = "9\.8\.7\.dev0"$', pyproject, flags=re.MULTILINE)
    assert "uv lock" in result.stdout
    assert "git tag" not in result.stdout
    assert "Publish a GitHub release" not in result.stdout


def release_dates(repo: Path) -> list:
    return re.findall(r'date-released: "([^"]+)"', (repo / "CITATION.cff").read_text(encoding="utf-8"))


def test_release_without_a_date_gets_the_current_date(repo_copy: Path) -> None:
    # A date that is not the current one, so that the test fails if the script leaves it.
    assert run_bump(repo_copy, current_version(repo_copy), "--date", "2031-02-03").returncode == 0
    day_before_the_run = date.today()

    result = run_bump(repo_copy, "9.8.7")

    assert result.returncode == 0, result.stdout + result.stderr
    # Two dates are right if the run passes midnight.
    assert release_dates(repo_copy) in ([str(day_before_the_run)] * 2, [str(date.today())] * 2)


def test_development_version_without_a_date_keeps_the_release_date(repo_copy: Path) -> None:
    # A date that is not the current one, so that the test fails if the script writes the current date.
    assert run_bump(repo_copy, current_version(repo_copy), "--date", "2031-02-03").returncode == 0

    result = run_bump(repo_copy, "9.8.7.dev12")

    assert result.returncode == 0, result.stdout + result.stderr
    assert current_version(repo_copy) == "9.8.7.dev12"
    assert release_dates(repo_copy) == ["2031-02-03"] * 2


@pytest.mark.parametrize("version", [
    "9.8.7\n",            # a final newline, which $ would accept
    "9.8.1\u0667",        # ARABIC-INDIC DIGIT SEVEN, which \d would accept
    "9.8.7.dev1\u0661",   # ARABIC-INDIC DIGIT ONE
])
def test_version_check_refuses_a_final_newline_and_digits_of_other_scripts(version: str) -> None:
    """Checked in the process: such a version cannot be printed reliably by a child process on Windows."""
    spec = importlib.util.spec_from_file_location("bump_version", REPO_ROOT / "scripts" / "bump_version.py")
    bump_version = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bump_version)

    assert bump_version.validate_version(version) is False
    assert bump_version.validate_version(version.strip().encode("ascii", "ignore").decode()) is True


@pytest.mark.parametrize(
    "version", ["9.8", "v9.8.7", "9.8.7.dev", "9.8.7-dev0", "9.8.7.dev0.1", "9.8.7.dev01", "9.8.7rc1",
                "09.8.7", "9.08.7", "9.8.07"])
def test_version_of_another_form_is_refused_and_no_file_changes(repo_copy: Path, version: str) -> None:
    before = {relative: (repo_copy / relative).read_bytes() for relative in VERSIONED_FILES}

    result = run_bump(repo_copy, version)

    assert result.returncode == 1
    assert f"Invalid version format: {version}" in result.stdout
    assert {relative: (repo_copy / relative).read_bytes() for relative in VERSIONED_FILES} == before


def test_missing_version_field_fails(repo_copy: Path) -> None:
    pyproject = repo_copy / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")
    pyproject.write_text(
        re.sub(r'^version = "[^"]+"$\n', "", content, flags=re.MULTILINE), encoding="utf-8"
    )

    result = run_bump(repo_copy, "9.8.7")

    assert result.returncode == 1
    assert "Could not find version" in result.stdout


@pytest.mark.parametrize("line_ending", [b"\n", b"\r\n"])
def test_line_endings_are_kept(repo_copy: Path, line_ending: bytes) -> None:
    """A version bump changes the version lines only, on every platform."""
    for relative in VERSIONED_FILES:
        path = repo_copy / relative
        lines = path.read_bytes().replace(b"\r\n", b"\n").split(b"\n")
        path.write_bytes(line_ending.join(lines))
    before = {relative: (repo_copy / relative).read_bytes() for relative in VERSIONED_FILES}

    result = run_bump(repo_copy, "9.8.7", "--date", "2031-02-03")

    assert result.returncode == 0, result.stdout + result.stderr
    for relative in VERSIONED_FILES:
        after = (repo_copy / relative).read_bytes()
        assert after.count(line_ending) == before[relative].count(line_ending), relative
        if line_ending == b"\n":
            assert b"\r\n" not in after, relative
        changed = [
            (old, new)
            for old, new in zip(before[relative].split(line_ending), after.split(line_ending))
            if old != new
        ]
        assert changed, relative
        assert all(b"9.8.7" in new or b"2031-02-03" in new for _, new in changed), relative
