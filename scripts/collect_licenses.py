#!/usr/bin/env python
"""
Write the license notice that a standalone DICE application carries.

A standalone application bundles DICE with Python packages under other
licenses. PyQt6 is under the GNU General Public License (GPL) version 3, so
the application is distributed under the GPL and must carry the license
texts. The notice states that, gives the MIT License of DICE, and gives the
declared license and the license files of every package installed in the
environment the application is built from.

dice-gui.spec calls write_license_notice() when PyInstaller builds the
application. To write the notice of the current environment by hand:

    python scripts/collect_licenses.py OUTPUT_FILE
"""

from __future__ import annotations

import re
import sys
from importlib import metadata
from pathlib import Path
from typing import List, Tuple

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
REPOSITORY_URL = "https://github.com/thiebes/DICE"
NOTICE_FILE_NAME = "THIRD-PARTY-LICENSES.txt"

# Names that license files have inside a package's .dist-info folder. Every
# file in a 'licenses' subfolder of it counts as well.
LICENSE_FILE_PREFIXES = ("license", "licence", "copying", "notice")

# The license texts the notice must hold, by the package that brings each:
# the GPL comes with PyQt6, the LGPL with the Qt libraries.
REQUIRED_TEXTS = {
    "PyQt6": "GNU GENERAL PUBLIC LICENSE",
    "PyQt6-Qt6": "GNU LESSER GENERAL PUBLIC LICENSE",
}

RULE = "=" * 78
THIN_RULE = "-" * 78


def project_version() -> str:
    """The version in dice/__init__.py, read as text so that DICE need not be importable."""
    source = (REPOSITORY_ROOT / "dice" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r"__version__\s*=\s*['\"](.+?)['\"]", source)
    if match is None:
        raise SystemExit("dice/__init__.py defines no __version__")
    return match.group(1)


def declared_license(distribution: metadata.Distribution) -> str:
    """The license a package declares in its metadata, in one line."""
    package = distribution.metadata
    expression = package.get("License-Expression")
    if expression:
        return expression
    # Some packages put the whole license text in the License field.
    text = (package.get("License") or "").strip()
    if text and "\n" not in text and len(text) <= 120:
        return text
    classifiers = [
        classifier.split("::")[-1].strip()
        for classifier in package.get_all("Classifier") or []
        if classifier.startswith("License ::")
    ]
    return "; ".join(classifiers) or "see the license text"


def license_files(distribution: metadata.Distribution) -> List[Tuple[str, str]]:
    """
    The license files a package installs in its .dist-info folder.

    Returns
    -------
    list of (str, str)
        File name and text of each license file, in name order.
    """
    found = []
    for file in distribution.files or []:
        folders, name = file.parts[:-1], file.parts[-1]
        if not any(folder.endswith(".dist-info") for folder in folders):
            continue
        if "licenses" not in folders and not name.lower().startswith(LICENSE_FILE_PREFIXES):
            continue
        path = Path(distribution.locate_file(file))
        if path.is_file():
            found.append((name, path.read_text(encoding="utf-8", errors="replace")))
    return sorted(found)


def installed_packages() -> List[metadata.Distribution]:
    """Every package installed in this environment except DICE itself, once each, in name order."""
    by_name = {}
    for distribution in metadata.distributions():
        name = distribution.metadata["Name"]
        if name and name.lower() != "dice":
            by_name.setdefault(name.lower(), distribution)
    return [by_name[name] for name in sorted(by_name)]


def missing_required_texts(packages: List[metadata.Distribution]) -> List[str]:
    """The required license texts that their packages do not bring, as messages."""
    by_name = {package.metadata["Name"].lower(): package for package in packages}
    missing = []
    for name, title in REQUIRED_TEXTS.items():
        package = by_name.get(name.lower())
        if package is None:
            missing.append(f"{name} is not installed")
        elif not any(title in text for _, text in license_files(package)):
            missing.append(f"{name} has no license file with the text of the {title}")
    return missing


# The file of the repository that pins the packages which build the
# application (PyInstaller and what it needs); uv.lock pins the others.
BUILD_REQUIREMENTS_FILE = ".github/pyinstaller-requirements.txt"


def qt_source_url(qt_version: str) -> str:
    """Where The Qt Company publishes the source code of this version of Qt, for example 6.10.0."""
    major_minor = ".".join(qt_version.split(".")[:2])
    return f"https://download.qt.io/archive/qt/{major_minor}/{qt_version}/single/"


def notice_header(version: str, packaged_with_pyinstaller: bool, qt_version: str) -> str:
    """The opening of the notice: what the application is licensed under and where its source is."""
    python_version = ".".join(str(part) for part in sys.version_info[:3])
    header = f"""DICE {version}: licenses of the standalone application
{RULE}

DICE (Diffusion Insight Computation Engine) is licensed under the MIT
License. Its text is in section 1.

This standalone application also contains the packages listed in section 2,
which names every package of the environment the application is built from.
One of them, PyQt6, is licensed under the GNU General Public License (GPL)
version 3. The application as a whole is therefore distributed under the
terms of the GPL version 3: you may copy, change, and redistribute it under
those terms, and it comes without warranty. The text of the GPL is in the
PyQt6 entry of section 2. The Qt libraries are used under the GNU Lesser
General Public License (LGPL) version 3, whose text is in the PyQt6-Qt6
entry.

Source code: the complete source of this application is the DICE repository,
{REPOSITORY_URL}
at the tag v{version}. In that repository, uv.lock gives the exact version
of every package of section 2 that the application runs with, and
{BUILD_REQUIREMENTS_FILE} gives the exact version of the
packages that build it (PyInstaller and the packages it needs). Each package
is available from the Python Package Index (https://pypi.org). The files
dice-gui.spec and .github/workflows/build-release.yml give the build
instructions.

The package PyQt6-Qt6 holds the Qt libraries, version {qt_version}, in compiled
form; the Python Package Index has no source archive of it. The Qt Company
publishes the source code of that version of Qt at
{qt_source_url(qt_version)}

The application also contains the Python interpreter, version {python_version},
which is licensed under the Python Software Foundation License Version 2
(https://docs.python.org/3/license.html).
"""
    if packaged_with_pyinstaller:
        header += """
The application is packaged with PyInstaller. The PyInstaller bootloader is
licensed under the GPL version 2 or later with an exception that permits its
use in applications under any license; the text is in the pyinstaller entry
of section 2.
"""
    return header


def collect_license_notice(version: str) -> str:
    """
    Build the license notice from the packages installed in this environment.

    Parameters
    ----------
    version : str
        Version of DICE that the application is built from.

    Returns
    -------
    str
        The notice: the header, the MIT License of DICE, and one entry for
        each installed package with its declared license and license files.
    """
    distributions = installed_packages()
    packaged_with_pyinstaller = any(
        distribution.metadata["Name"].lower() == "pyinstaller" for distribution in distributions
    )
    sections = [
        notice_header(version, packaged_with_pyinstaller, metadata.version("PyQt6-Qt6")),
        f"{RULE}\n1. DICE\n{RULE}\n",
        (REPOSITORY_ROOT / "LICENSE.md").read_text(encoding="utf-8").strip() + "\n",
        f"{RULE}\n2. Packages in the environment this application is built from\n{RULE}\n",
    ]

    for distribution in distributions:
        name = distribution.metadata["Name"]
        entry = [
            THIN_RULE,
            f"{name} {distribution.version}",
            f"Declared license: {declared_license(distribution)}",
            THIN_RULE,
            "",
        ]
        files = license_files(distribution)
        if not files:
            entry.append(f"The package installs no license file. See https://pypi.org/project/{name}/.\n")
        for file_name, text in files:
            entry.append(f"[{file_name}]\n\n{text.strip()}\n")
        sections.append("\n".join(entry))

    return "\n".join(sections)


def write_license_notice(output_file: Path, version: str) -> Path:
    """
    Write the license notice of this environment to a file.

    Parameters
    ----------
    output_file : Path
        File to write; its folder is created when missing.
    version : str
        Version of DICE that the application is built from.

    Returns
    -------
    Path
        The file written.

    Raises
    ------
    SystemExit
        If PyQt6 or the Qt package is missing or does not bring the text of
        its license (the GPL and the LGPL). A build stops here instead of
        producing an application without those texts.
    """
    missing = missing_required_texts(installed_packages())
    if missing:
        raise SystemExit("Cannot write the license notice: " + "; ".join(missing) + ".")
    notice = collect_license_notice(version)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(notice, encoding="utf-8", newline="\n")
    return output_file


def main(arguments: List[str]) -> int:
    if len(arguments) != 1:
        print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
        return 2
    written = write_license_notice(Path(arguments[0]), project_version())
    print(f"Wrote {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
