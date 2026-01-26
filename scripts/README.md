# DICE Maintenance Scripts

This directory contains scripts for maintaining and managing the DICE project.

## Version Management

### bump_version.py

Updates the version number across all relevant files in the DICE project.

#### Single Source of Truth

The version is defined once in `dice/__init__.py` as `__version__`. All other files that display or reference the version import this value dynamically:

- `dice/cli/arguments.py` - imports `__version__`
- `dice_gui/__init__.py` - imports `__version__`
- `dice_gui/dice_gui.py` - imports and displays `__version__`

The bump script updates files that cannot dynamically import the version:

- `dice/__init__.py` - the source of truth
- `CITATION.cff` - citation metadata (2 occurrences + release date)
- `pyproject.toml` - build configuration
- `setup.py` - legacy build configuration (if present)

#### Usage

```bash
python scripts/bump_version.py <new_version> [--date YYYY-MM-DD]
```

#### Examples

```bash
# Bump to version 1.3.0 with today's date
python scripts/bump_version.py 1.3.0

# Bump to version 1.3.0 with specific release date
python scripts/bump_version.py 1.3.0 --date 2025-09-15

# Skip confirmation prompt
python scripts/bump_version.py 1.3.0 --no-confirm
```

#### Workflow

1. Run the bump script with the desired version
2. Review changes with `git diff`
3. Run tests to ensure everything works
4. Commit changes: `git add -A && git commit -m "Bump version to X.Y.Z"`
5. Tag the release: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
6. Push: `git push && git push --tags`

#### Version Format

Versions must follow semantic versioning: `MAJOR.MINOR.PATCH` (e.g., 1.2.0)

- MAJOR: Breaking changes
- MINOR: Feature additions (backward compatible)
- PATCH: Bug fixes (backward compatible)
