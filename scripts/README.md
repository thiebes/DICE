# DICE Maintenance Scripts

The `scripts/` directory contains scripts for maintaining the DICE project.

## Version management

### bump_version.py

Sets a new version number in every file that stores it as text.

#### Where the version is defined

`dice/__init__.py` defines the version once, as `__version__`. Code that displays the version imports it from there:

- `dice/cli/arguments.py` imports `__version__` for `dice --version`
- `dice_gui/dice_gui.py` imports `__version__` and shows it in the GUI
- `dice-gui.spec` reads it from `dice/__init__.py` when PyInstaller builds the GUI

The script updates the files that cannot import the version:

- `dice/__init__.py`, the source of the version
- `pyproject.toml`, the package metadata
- `CITATION.cff`, the citation metadata (two version fields and two release dates)

#### Usage

```bash
python scripts/bump_version.py <new_version> [--date YYYY-MM-DD] [--no-confirm]
```

The release date defaults to the current date. Without `--no-confirm`, the script shows the planned change and asks for confirmation.

If a file already contains the requested version, the script reports that no change is needed in that file and continues. To set only the release date in `CITATION.cff`, run the script with the current version and `--date`. The script stops with an error when the version or the date has an invalid format, when a file has no version field, or when `CITATION.cff` has no `date-released` field. After an update, it prints the next steps of the release workflow.

#### Examples

```bash
# Set version 1.3.0 with the current date as the release date
python scripts/bump_version.py 1.3.0

# Set version 1.3.0 with a given release date
python scripts/bump_version.py 1.3.0 --date 2025-09-15

# Skip the confirmation prompt
python scripts/bump_version.py 1.3.0 --no-confirm
```

#### Release workflow

1. Run the script with the new version.
2. Run `uv lock`, so that `uv.lock` records the new version of the `dice` package.
3. Regenerate `.github/build-constraints.txt` and `.github/pyinstaller-requirements.txt` with the commands in their headers.
4. In `CHANGELOG.md`, set the release date of the new version.
5. Review the changes with `git diff`.
6. Run the tests: `uv run pytest`.
7. Commit the changes: `git add -A && git commit -m "Bump version to X.Y.Z"`.
8. Tag the release: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`.
9. Push the commit and the tag: `git push && git push --tags`.
10. On GitHub, publish a release for the tag. Publishing it, not pushing the tag, starts `.github/workflows/build-release.yml`, which builds the standalone executables and attaches them and `SHA256SUMS.txt` to the release.

#### Version format

Versions follow semantic versioning, `MAJOR.MINOR.PATCH` (for example, 1.2.0):

- MAJOR: changes that are not backward compatible
- MINOR: added features that are backward compatible
- PATCH: backward-compatible bug fixes
