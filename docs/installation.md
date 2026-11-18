# DICE Installation Guide

DICE runs on Windows, macOS, and Linux and requires Python 3.11 or later. The recommended installation uses uv, which installs the dependency versions recorded in `uv.lock`. Installation with pip is also supported.

## Python

DICE requires Python 3.11 or later. The test suite runs on Python 3.11 and 3.12, on Windows and Linux.

uv can download a suitable Python version for you. If you install with pip, install Python first, for example from [python.org](https://www.python.org/downloads/) or with conda.

## Install with uv (recommended)

[uv](https://docs.astral.sh/uv/) is a Python package and project manager. For DICE, it creates a virtual environment and installs the exact dependency versions recorded in `uv.lock`, so every installation uses the versions that the tests ran against.

### Install uv

Install uv with a package manager:

```bash
# Windows
winget install --id=astral-sh.uv -e

# macOS (Homebrew)
brew install uv

# Any platform that already has Python and pipx
pipx install uv
```

The [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) lists the other methods.

### Install DICE

```bash
git clone https://github.com/thiebes/DICE.git
cd DICE
uv sync
```

`uv sync` creates a virtual environment in `.venv/` and installs DICE with its runtime dependencies, including the graphical user interface (GUI).

### Run DICE

From the `DICE` directory:

```bash
# Graphical user interface
uv run dice-gui

# Command-line interface with an example parameters file
uv run dice data/examples/parameters.txt
```

`uv run` runs the command inside the project environment. You do not need to activate the environment first.

## Install with pip

Create and activate a virtual environment, then install DICE from the repository directory:

```bash
git clone https://github.com/thiebes/DICE.git
cd DICE
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install .
```

`pip install .` installs DICE and the dependencies declared in `pyproject.toml`. pip selects the newest versions that satisfy the declared ranges, so they can differ from the versions in `uv.lock`.

For development, install in editable mode with the `dev` extra. Changes to the source code then take effect without a reinstall:

```bash
pip install -e ".[dev]"
```

While the environment is active, the `dice` and `dice-gui` commands are available from any directory:

```bash
dice-gui
dice data/examples/parameters.txt
```

### Install with conda

conda can provide Python; pip then installs DICE into the conda environment:

```bash
conda create -n dice python=3.12
conda activate dice
pip install .
```

Run the last command from the `DICE` directory.

## Optional extras

`pyproject.toml` defines three optional groups of dependencies (extras):

| Extra | Adds | Needed for |
| ------- | ------------- | ------------- |
| `webapp` | Flask | Running the CNR estimator web app in `webapp/` locally |
| `test` | pytest, pytest-cov, pytest-qt | Running the test suite |
| `dev` | Everything in `test`, plus Flask | Development |

With uv:

```bash
uv sync --extra webapp
uv sync --extra dev
uv sync --all-extras
```

`uv sync` installs only the extras named in the command and removes packages that belong to other extras. To keep all of them, use `uv sync --all-extras`.

With pip:

```bash
pip install ".[webapp]"
pip install -e ".[dev]"
```

## Dependencies

DICE depends on numpy, scipy, pandas, statsmodels, matplotlib, seaborn, joblib, PyQt6, and fft-cnr. Both installation methods install them automatically. The version ranges are declared in `pyproject.toml`, and the tested versions are recorded in `uv.lock`.

DICE estimates the contrast-to-noise ratio (CNR) of a profile with the [fft-cnr](https://pypi.org/project/fft-cnr/) package. Its documentation is at [fft-cnr.readthedocs.io](https://fft-cnr.readthedocs.io).

## Verify the installation

Print the installed version:

```bash
uv run dice --version       # with uv
dice --version              # with pip, environment active
```

Both commands print the installed version, for example `dice 1.3.0`.

Check that the GUI toolkit can be imported:

```bash
uv run python -c "from PyQt6.QtWidgets import QApplication; print('PyQt6 available')"
```

Run a short simulation (100 runs) from the `DICE` directory:

```bash
uv run dice data/examples/quick_test.txt
```

The results are written to `output/quick_test/`.

## Troubleshooting

### The GUI does not start on Linux

The PyQt6 packages include Qt, but Qt uses some system libraries. If the error message says that Qt could not load the platform plugin "xcb", install the XCB cursor library:

```bash
# Debian, Ubuntu
sudo apt-get install libxcb-cursor0

# Fedora
sudo dnf install xcb-util-cursor
```

### The GUI does not start over SSH or in a virtual machine

A GUI needs a display. Over SSH, connect with X forwarding (`ssh -X` or `ssh -Y`). In a virtual machine, configure display forwarding. Alternatively, use the command-line interface, which needs no display.

### The GUI does not start on Windows

Install the current Microsoft Visual C++ Redistributable, which PyQt6 requires.

### `No module named 'dice'` or `No module named 'fft_cnr'`

Python is running outside the DICE environment. Use `uv run` from the `DICE` directory, or activate the virtual environment before you run `python`, `dice`, or `dice-gui`. If the error remains, install again with `uv sync` or `pip install .`.

### Permission errors

Install into a virtual environment, as described above. Do not use `sudo pip`.

### Dependency conflicts

Create a new virtual environment and install DICE into it. With uv, `uv sync` installs the locked versions and avoids conflicts with other projects.

## Update DICE

```bash
cd DICE
git pull
uv sync
```

With pip, run `pip install .` (or `pip install -e ".[dev]"`) again after `git pull`, with the environment active.

## Uninstall

- pip installation: `pip uninstall dice`
- uv or venv environment: delete the `.venv` directory inside `DICE`
- conda environment: `conda env remove -n dice`
- source code: delete the `DICE` directory

## Get help

1. Search the existing issues: [GitHub Issues](https://github.com/thiebes/DICE/issues)
2. Open a new issue that includes:
   - operating system and version
   - Python version (`python --version`)
   - the full error message
   - the installation method
3. Contact the author: [joseph@thiebes.org](mailto:joseph@thiebes.org)

## See also

- [Parameter Reference](parameters.md)
- [CLI Guide](cli-guide.md)
- [GUI Guide](../dice_gui/README.md)
- [Contributing](../CONTRIBUTING.md)
