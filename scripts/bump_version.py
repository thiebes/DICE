"""
Version bumping script for DICE.

This script updates the version number across all relevant files in the DICE project.
It ensures consistency by updating the single source of truth in dice/__init__.py and
all files that reference versions.

Usage:
    python scripts/bump_version.py <new_version> [--date YYYY-MM-DD]

Example:
    python scripts/bump_version.py 1.3.0 --date 2025-09-15
"""

import argparse
import re
import sys
from pathlib import Path
from datetime import datetime


def validate_version(version: str) -> bool:
    """Validate that version follows semantic versioning."""
    pattern = r'^\d+\.\d+\.\d+$'
    return bool(re.match(pattern, version))


def validate_date(date_str: str) -> bool:
    """Validate that date follows YYYY-MM-DD format."""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def update_dice_init(file_path: Path, new_version: str) -> bool:
    """Update version in dice/__init__.py."""
    content = file_path.read_text(encoding='utf-8')
    pattern = r'__version__ = "[^"]+"'
    replacement = f'__version__ = "{new_version}"'

    if not re.search(pattern, content):
        print(f"ERROR: Could not find __version__ in {file_path}")
        return False

    new_content = re.sub(pattern, replacement, content)
    file_path.write_text(new_content, encoding='utf-8')
    print(f"Updated {file_path}")
    return True


def update_citation_cff(file_path: Path, new_version: str, new_date: str) -> bool:
    """Update version and date in CITATION.cff."""
    content = file_path.read_text(encoding='utf-8')

    # Update version (both occurrences)
    version_pattern = r'version: "[^"]+"'
    version_replacement = f'version: "{new_version}"'
    new_content = re.sub(version_pattern, version_replacement, content)

    # Update date-released
    date_pattern = r'date-released: "\d{4}-\d{2}-\d{2}"'
    date_replacement = f'date-released: "{new_date}"'
    new_content = re.sub(date_pattern, date_replacement, new_content)

    if content == new_content:
        print(f"WARNING: No changes made to {file_path}")
        return False

    file_path.write_text(new_content, encoding='utf-8')
    print(f"Updated {file_path}")
    return True


def update_pyproject_toml(file_path: Path, new_version: str) -> bool:
    """Update version in pyproject.toml."""
    content = file_path.read_text(encoding='utf-8')
    pattern = r'(^version = ")[^"]+(")$'
    replacement = rf'\g<1>{new_version}\g<2>'

    new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    if content == new_content:
        print(f"ERROR: Could not find version in {file_path}")
        return False

    file_path.write_text(new_content, encoding='utf-8')
    print(f"Updated {file_path}")
    return True


def update_setup_py(file_path: Path, new_version: str) -> bool:
    """Update version in setup.py if it exists."""
    if not file_path.exists():
        print(f"Skipping {file_path} (file does not exist)")
        return True

    content = file_path.read_text(encoding='utf-8')
    pattern = r'(version=")[^"]+("),'
    replacement = rf'\g<1>{new_version}\g<2>,'

    new_content = re.sub(pattern, replacement, content)

    if content == new_content:
        print(f"WARNING: No changes made to {file_path}")
        return True

    file_path.write_text(new_content, encoding='utf-8')
    print(f"Updated {file_path}")
    return True


def get_current_version(repo_root: Path) -> str:
    """Get current version from dice/__init__.py."""
    init_file = repo_root / 'dice' / '__init__.py'
    content = init_file.read_text(encoding='utf-8')
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
        print("Version must follow semantic versioning (e.g., 1.2.0)")
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
        print("  1. Review changes: git diff")
        print("  2. Run tests to ensure everything works")
        print("  3. Commit changes: git add -A && git commit -m \"Bump version to {args.version}\"")
        print("  4. Tag release: git tag -a v{args.version} -m \"Release v{args.version}\"")
        print("  5. Push: git push && git push --tags")
    else:
        print("\nERROR: Some files could not be updated")
        sys.exit(1)


if __name__ == '__main__':
    main()
