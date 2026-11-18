"""Tests for scripts/bump_version.py, run against copies of the repository's own files."""

import re
import shutil
import subprocess
import sys
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


def test_missing_version_field_fails(repo_copy: Path) -> None:
    pyproject = repo_copy / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")
    pyproject.write_text(
        re.sub(r'^version = "[^"]+"$\n', "", content, flags=re.MULTILINE), encoding="utf-8"
    )

    result = run_bump(repo_copy, "9.8.7")

    assert result.returncode == 1
    assert "Could not find version" in result.stdout
