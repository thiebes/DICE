"""
Version bumping script for DICE.

This script updates the version number across all relevant files in the DICE project.
It ensures consistency by updating the single source of truth in dice/__init__.py and
all files that reference versions.

Usage:
    python scripts/bump_version.py <new_version> [--date YYYY-MM-DD]

Example:
    python scripts/bump_version.py 1.3.0 --date 2025-09-15

Between releases, the main branch carries a development version of the next
release, for example 1.3.0.dev0, so that its outputs do not name a release
that does not exist yet.
"""

import argparse
import re
import sys
from pathlib import Path
from datetime import datetime


def read_file(file_path: Path) -> str:
    """Read a file without translating its line endings."""
    with open(file_path, encoding='utf-8', newline='') as f:
        return f.read()


def write_file(file_path: Path, content: str) -> None:
    """Write a file without translating its line endings, so a one-line edit stays one line."""
    with open(file_path, 'w', encoding='utf-8', newline='') as f:
        f.write(content)


DEVELOPMENT_SUFFIX = r'\.dev\d+'


def validate_version(version: str) -> bool:
    """Validate that the version is X.Y.Z, or X.Y.Z.devN for a development version."""
    pattern = rf'^\d+\.\d+\.\d+({DEVELOPMENT_SUFFIX})?$'
    return bool(re.match(pattern, version))


def is_development_version(version: str) -> bool:
    """Whether the version is one between releases, such as 2.0.0.dev0."""
    return bool(re.search(rf'{DEVELOPMENT_SUFFIX}$', version))


def validate_date(date_str: str) -> bool:
    """Validate that date follows YYYY-MM-DD format."""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def update_dice_init(file_path: Path, new_version: str) -> bool:
    """Update version in dice/__init__.py."""
    content = read_file(file_path)
    pattern = r'__version__ = "[^"]+"'
    replacement = f'__version__ = "{new_version}"'

    if not re.search(pattern, content):
        print(f"ERROR: Could not find __version__ in {file_path}")
        return False

    new_content = re.sub(pattern, replacement, content)
    write_file(file_path, new_content)
    print(f"Updated {file_path}")
    return True


def update_citation_cff(file_path: Path, new_version: str, new_date: str) -> bool:
    """Update version and date in CITATION.cff."""
    content = read_file(file_path)

    # Update version (both occurrences)
    version_pattern = r'version: "[^"]+"'
    version_replacement = f'version: "{new_version}"'
    date_pattern = r'date-released: "\d{4}-\d{2}-\d{2}"'
    date_replacement = f'date-released: "{new_date}"'

    if not re.search(version_pattern, content):
        print(f"ERROR: Could not find version in {file_path}")
        return False

    if not re.search(date_pattern, content):
        print(f"ERROR: Could not find date-released in {file_path}")
        return False

    new_content = re.sub(version_pattern, version_replacement, content)
    new_content = re.sub(date_pattern, date_replacement, new_content)

    if content == new_content:
        print(f"No changes needed in {file_path}")
        return True

    write_file(file_path, new_content)
    print(f"Updated {file_path}")
    return True


def update_pyproject_toml(file_path: Path, new_version: str) -> bool:
    """Update version in pyproject.toml."""
    content = read_file(file_path)
    # (?=\r?$): a Windows (CRLF) checkout has a carriage return before the line end.
    pattern = r'(^version = ")[^"]+(")(?=\r?$)'
    replacement = rf'\g<1>{new_version}\g<2>'

    if not re.search(pattern, content, flags=re.MULTILINE):
        print(f"ERROR: Could not find version in {file_path}")
        return False

    new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    if content == new_content:
        print(f"No changes needed in {file_path}")
        return True

    write_file(file_path, new_content)
    print(f"Updated {file_path}")
    return True


def update_setup_py(file_path: Path, new_version: str) -> bool:
    """Update version in setup.py if it exists."""
    if not file_path.exists():
        print(f"Skipping {file_path} (file does not exist)")
        return True

    content = read_file(file_path)
    pattern = r'(version=")[^"]+("),'
    replacement = rf'\g<1>{new_version}\g<2>,'

    new_content = re.sub(pattern, replacement, content)

    if content == new_content:
        print(f"WARNING: No changes made to {file_path}")
        return True

    write_file(file_path, new_content)
    print(f"Updated {file_path}")
    return True


def get_current_version(repo_root: Path) -> str:
    """Get current version from dice/__init__.py."""
    init_file = repo_root / 'dice' / '__init__.py'
    content = read_file(init_file)
    match = re.search(r'__version__ = "([^"]+)"', content)
    if match:
        return match.group(1)
    return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description='Bump version number across DICE project files'
    )
    parser.add_argument(
        'version',
        help='New version number (e.g., 1.3.0)'
    )
    parser.add_argument(
        '--date',
        help='Release date in YYYY-MM-DD format (default: today)',
        default=datetime.now().strftime('%Y-%m-%d')
    )
    parser.add_argument(
        '--no-confirm',
        action='store_true',
        help='Skip confirmation prompt'
    )

    args = parser.parse_args()

    # Validate inputs
    if not validate_version(args.version):
        print(f"ERROR: Invalid version format: {args.version}")
        print("Version must be X.Y.Z (e.g., 1.2.0), or X.Y.Z.devN for a development version (e.g., 1.3.0.dev0)")
        sys.exit(1)

    if not validate_date(args.date):
        print(f"ERROR: Invalid date format: {args.date}")
        print("Date must be in YYYY-MM-DD format")
        sys.exit(1)

    # Find repository root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent

    # Get current version
    current_version = get_current_version(repo_root)

    # Confirm with user
    if not args.no_confirm:
        print(f"\nCurrent version: {current_version}")
        print(f"New version:     {args.version}")
        print(f"Release date:    {args.date}")
        print("\nFiles to be updated:")
        print("  - dice/__init__.py")
        print("  - CITATION.cff")
        print("  - pyproject.toml")
        print("  - setup.py (if exists)")

        response = input("\nProceed with version bump? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Aborted.")
            sys.exit(0)

    # Update files
    success = True

    print("\nUpdating files...")
    success &= update_dice_init(repo_root / 'dice' / '__init__.py', args.version)
    success &= update_citation_cff(repo_root / 'CITATION.cff', args.version, args.date)
    success &= update_pyproject_toml(repo_root / 'pyproject.toml', args.version)
    success &= update_setup_py(repo_root / 'setup.py', args.version)

    if success:
        print(f"\nVersion successfully bumped to {args.version}")
        print("\nNext steps:")
        print("  1. Refresh the lockfile: uv lock")
        print("  2. Review changes: git diff")
        print("  3. Run tests to ensure everything works")
        print(f"  4. Commit changes: git add -A && git commit -m \"Bump version to {args.version}\"")
        if is_development_version(args.version):
            print("  5. Push: git push. A development version gets no tag and no release.")
        else:
            print(f"  5. Tag release: git tag -a v{args.version} -m \"Release v{args.version}\"")
            print("  6. Push: git push && git push --tags")
            print(f"  7. Publish a GitHub release for v{args.version} to build and attach the executables")
    else:
        print("\nERROR: Some files could not be updated")
        sys.exit(1)


if __name__ == '__main__':
    main()
