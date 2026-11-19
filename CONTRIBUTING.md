# Contributing to DICE

Thank you for your interest in contributing to DICE (Diffusion Insight Computation Engine). DICE helps researchers quantify the effect of noise on optical measurements of excited state transport.

## Ways to contribute

### Report bugs

- Use the [GitHub issue tracker](https://github.com/thiebes/DICE/issues).
- Include the parameters file (or the GUI settings), the full error message, and the expected and actual behavior.
- Provide a minimal example that reproduces the problem, when possible.

### Suggest enhancements

- Fitting models or diffusion models
- Noise models or experimental conditions
- Performance improvements
- Documentation improvements

### Submit examples

- Experimental use cases
- Validation studies that compare DICE predictions with experimental results
- Tutorials for specific material systems

### Improve documentation

- Usage examples
- Explanations of the scientific background
- Corrections where the documentation and the code disagree

## Code contributions

### Set up a development environment

DICE uses [uv](https://docs.astral.sh/uv/) to manage its development environment. Install uv as described in the [Installation Guide](docs/installation.md), then run from the repository root:

```bash
uv sync --extra dev
uv run pytest
```

`uv sync --extra dev` installs DICE in editable mode with the test tools and Flask. `uv run pytest` runs the test suite. With pip, use `pip install -e ".[dev]"` and `pytest` in an active virtual environment.

When you change dependencies in `pyproject.toml`, run `uv lock` and commit the updated `uv.lock`.

### Workflow

1. Fork the repository.
2. Create a branch for your change (`git checkout -b feature/short-description`).
3. Make your changes.
4. Add tests for new functionality.
5. Update the documentation, and describe user-visible changes in `CHANGELOG.md`.
6. Open a pull request.

### Code standards

- Follow the PEP 8 style guidelines. The repository includes a flake8 configuration (`.flake8`).
- Include docstrings for new functions and classes.
- Add type hints where appropriate.
- Keep compatibility with the dependency ranges declared in `pyproject.toml`.

### Testing

- Add tests for new features.
- Check that the existing tests still pass.
- Where possible, validate against known analytical solutions.

### Scientific rigor

- Provide references for new models or algorithms.
- Include parameter validation and error handling.
- Document assumptions and limitations.
- Consider edge cases and boundary conditions.

## Questions

Open an issue for discussion, or [contact the maintainer](mailto:joseph@thiebes.org).
