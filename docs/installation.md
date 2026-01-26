# DICE Installation Guide

Complete installation instructions for all platforms and methods.

## Python Requirements

DICE requires Python 3.8 or later and has been tested with Python 3.11, 3.12, and 3.13.

### Installing Python

- **Official website:** [python.org/downloads](https://www.python.org/downloads/)
- **Package managers:** conda, homebrew (macOS), apt (Linux)
- **Online environments:** [Google Colab](https://colab.research.google.com/), [DataLore](https://datalore.jetbrains.com/)

## Installation Methods

### Method 1: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver that simplifies dependency management.

#### Install uv

**macOS and Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Install DICE

1. **Clone the repository:**

   ```bash
   git clone https://github.com/thiebes/DICE.git
   cd DICE
   ```

2. **Sync dependencies:**

   ```bash
   uv sync
   ```

   This automatically creates a virtual environment and installs all dependencies including the GUI.

3. **Run DICE:**

   ```bash
   # Launch GUI
   uv run dice-gui

   # Or run command-line version
   uv run dice parameters.txt
   ```

#### Benefits of uv

- Fast dependency resolution and installation
- Automatic virtual environment management
- Reproducible environments with lock files
- No need to manually activate/deactivate environments

### Method 2: Install Dependencies and Use Directly

This method installs dependencies into your current Python environment.

1. **Clone or download** this repository to your local machine

2. **Navigate** to the DICE directory:

   ```bash
   cd DICE
   ```

3. **Install dependencies** using pip:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run DICE:**

   ```bash
   # Launch GUI
   python -m dice_gui.dice_gui

   # Or run command-line version
   python run_dice.py parameters.txt
   ```

### Method 3: Install as Editable Package

This method installs DICE as a package, allowing you to use the `dice` and `dice-gui` commands from anywhere.

1. **Clone** this repository:

   ```bash
   git clone https://github.com/thiebes/DICE.git
   cd DICE
   ```

2. **Install** in development mode:

   ```bash
   pip install -e .
   ```

   This installs DICE and all dependencies, and creates command-line entry points.

3. **Run DICE:**

   ```bash
   # Launch GUI (from any directory)
   dice-gui

   # Or run command-line version
   dice parameters.txt
   ```

## Dependencies

### Required Packages

DICE depends on the following Python packages:

- **numpy>=1.20.0** - Fundamental package for scientific computing
- **scipy>=1.7.0** - Scientific and technical computing
- **pandas>=1.3.0** - High-performance data structures and analysis
- **statsmodels>=0.12.0** - Statistical modeling and econometrics
- **matplotlib>=3.3.0** - Plotting and visualization
- **seaborn>=0.11.0** - Statistical data visualization
- **joblib>=1.0.0** - Lightweight pipelining and parallel processing
- **PyQt6>=6.0.0** - GUI framework (required for graphical interface)

### Optional Packages

For web application (CNR estimator):

- **flask>=2.0.0** - Web framework

For testing:

- **pytest>=6.0.0** - Testing framework
- **pytest-cov>=2.12.0** - Coverage plugin

### Installing Optional Dependencies

With uv:

```bash
# Install web app dependencies
uv sync --extra webapp

# Install test dependencies
uv sync --extra test

# Install all optional dependencies
uv sync --all-extras
```

With pip:

```bash
# Install web app dependencies
pip install -e .[webapp]

# Install test dependencies
pip install -e .[test]

# Install all optional dependencies
pip install -e .[dev]
```

### Standard Library Modules

The following are part of the Python Standard Library and do not need separate installation:

- **ast** - For working with abstract syntax trees
- **os** - Operating system dependent functionality
- **re** - Regular expression matching operations
- **typing** - Type hint support

## Virtual Environments

### Using uv (automatic)

When you run `uv sync`, uv automatically creates and manages a virtual environment in `.venv/`.

### Using venv manually

If not using uv, it's recommended to use a virtual environment:

**Create virtual environment:**

```bash
python -m venv venv
```

**Activate virtual environment:**

```bash
# macOS/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Deactivate when done:**

```bash
deactivate
```

### Using conda

```bash
# Create conda environment
conda create -n dice python=3.11
conda activate dice

# Install dependencies
pip install -r requirements.txt
```

## Verifying Installation

### Check Python version

```bash
python --version
```

Should show Python 3.8 or later.

### Check installed packages

With uv:

```bash
uv pip list
```

With pip:

```bash
pip list
```

Should show all required packages installed.

### Test DICE import

```python
python -c "import dice; print('DICE imported successfully')"
```

### Test GUI availability

```python
python -c "from PyQt6.QtWidgets import QApplication; print('PyQt6 available')"
```

## Troubleshooting

### PyQt6 Installation Issues

**Linux:**

PyQt6 may require additional system dependencies:

```bash
# Ubuntu/Debian
sudo apt-get install python3-pyqt6

# Fedora
sudo dnf install python3-pyqt6
```

**macOS:**

If you encounter issues, try:

```bash
pip install --upgrade pip
pip install PyQt6 --no-cache-dir
```

**Windows:**

Ensure you have the latest Visual C++ Redistributable installed.

### Display Issues

If the GUI won't launch, check:

1. **Display environment is available:** GUI applications require a display
2. **SSH with X forwarding:** If connecting via SSH, use `ssh -X` or `ssh -Y`
3. **Virtual machines:** Ensure display forwarding is configured

### Permission Errors

If you get permission errors during installation:

- Use a virtual environment (recommended)
- Or install with `--user` flag: `pip install --user -r requirements.txt`
- Avoid using `sudo pip` (not recommended)

### Dependency Conflicts

If you encounter version conflicts:

1. **Create fresh virtual environment:**

   ```bash
   python -m venv fresh_venv
   source fresh_venv/bin/activate  # or fresh_venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Use uv for better dependency resolution:**

   ```bash
   uv sync
   ```

### Import Errors

If you get import errors when running DICE:

1. **Check you're in the DICE directory**
2. **Verify virtual environment is activated**
3. **Reinstall dependencies:**

   ```bash
   pip install --force-reinstall -r requirements.txt
   ```

## Updating DICE

### With git

```bash
cd DICE
git pull origin main
uv sync  # or pip install -r requirements.txt
```

### Updating dependencies

```bash
# With uv
uv sync --upgrade

# With pip
pip install --upgrade -r requirements.txt
```

## Uninstalling

### If installed with pip -e

```bash
pip uninstall dice
```

### Remove virtual environment

```bash
# uv
rm -rf .venv

# venv
rm -rf venv

# conda
conda env remove -n dice
```

### Remove repository

```bash
cd ..
rm -rf DICE
```

## Getting Help

If you encounter issues during installation:

1. **Check existing issues:** [GitHub Issues](https://github.com/thiebes/DICE/issues)
2. **Create new issue:** Include:
   - Operating system and version
   - Python version (`python --version`)
   - Error messages (full traceback)
   - Installation method used
3. **Contact author:** [joseph@thiebes.org](mailto:joseph@thiebes.org)

Remember to regularly update your packages to ensure smooth functioning of DICE.

## See Also

- [Parameter Reference](parameters.md)
- [CLI Guide](cli-guide.md)
- [GUI Guide](../dice_gui/README.md)
- [Contributing](../CONTRIBUTING.md)
