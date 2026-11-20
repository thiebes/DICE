# DICE Installation Guide

DICE runs on Windows, macOS, and Linux and requires Python 3.11 or later. The recommended installation uses uv, which installs the dependency versions recorded in `uv.lock`. Installation with pip is also supported. If you only need the graphical user interface (GUI), you can instead download the [standalone application](#standalone-application), which needs no Python installation.

## Python

DICE requires Python 3.11 or later. The test suite runs on Python 3.11 and 3.12, on Windows, macOS, and Linux.

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

`uv sync` creates a virtual environment in `.venv/` and installs DICE with its runtime dependencies, including the GUI.

### Run DICE

From the `DICE` directory:

```bash
# Graphical user interface
uv run dice-gui

# Command-line interface with an example parameters file
uv run dice data/examples/parameters.txt
```

`uv run` runs the command inside the project environment. You do not need to activate the environment first.

The first start after an installation with uv takes longer than later starts, because Python compiles the scientific libraries to bytecode when it first imports them. To do this work during the installation instead, run `uv sync --compile-bytecode`, or set the environment variable `UV_COMPILE_BYTECODE=1` before `uv sync`.

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

## Standalone application

Each DICE release on [GitHub](https://github.com/thiebes/DICE/releases) includes the GUI as a standalone application for Windows, macOS, and Linux. It contains its own copy of Python and of every dependency, so you download one file and start it. The files have no code signature, so Windows and macOS warn you before the first start.

### Download

From the Assets list of the release, download the file for your system and `SHA256SUMS.txt`. Each file name gives the version, the operating system, and the processor architecture:

| System | File |
| ------- | ------------- |
| Windows, 64-bit Intel or AMD processor | `DICE-<version>-Windows-x86_64.exe` |
| macOS, Apple silicon (M1 or later) | `DICE-<version>-macOS-arm64.zip` |
| macOS, Intel processor | `DICE-<version>-macOS-x86_64.zip` |
| Linux, 64-bit Intel or AMD processor | `DICE-<version>-Linux-x86_64.tar.gz` |

For release 2.0.0, for example, the Windows file is `DICE-2.0.0-Windows-x86_64.exe`. The commands below use 2.0.0; replace it with the version you downloaded.

The Linux application is built on Ubuntu 22.04. It should run on any distribution with the same or a newer GNU C Library (glibc 2.35 or later).

### Check the download

`SHA256SUMS.txt` lists the SHA-256 checksum of every file of the release. If the checksum of your file matches its line, your copy is identical to the file that the release build produced. Run the command for your system in the folder that contains both files.

Windows (PowerShell):

```powershell
$file = "DICE-2.0.0-Windows-x86_64.exe"
$line = Get-Content SHA256SUMS.txt | Where-Object { $_.EndsWith(" $file") }
$expected = ($line -split " ")[0]
$actual = (Get-FileHash $file -Algorithm SHA256).Hash
if ($actual -eq $expected) { "OK: $file" } else { "MISMATCH: do not run $file" }
```

macOS (Terminal):

```bash
grep DICE-2.0.0-macOS-arm64.zip SHA256SUMS.txt | shasum -a 256 -c
```

Linux:

```bash
sha256sum -c --ignore-missing SHA256SUMS.txt
```

On macOS and Linux, a match prints the file name followed by `OK`. Any other result, such as `FAILED` or `MISMATCH`, means that the file is damaged or is not the released file. Delete it and download it again.

### Start on Windows

Double-click the `.exe` file. Because the file has no code signature, Microsoft Defender SmartScreen can show the message "Windows protected your PC". Select **More info**, check that the app name is the file you verified, and select **Run anyway**. Your browser can also warn about the download; choose the option that keeps the file.

The application is a single file that unpacks itself into a temporary folder at every start, so the window takes several seconds to appear.

### Start on macOS

Double-click the zip file in Finder to unpack `DICE.app`. You can move the app to the Applications folder.

macOS blocks the first start of a downloaded app that Apple has not notarized, with a message that it cannot verify the app or its developer. To allow DICE:

- macOS 15 (Sequoia) and later: open DICE and close the message. Then open System Settings > Privacy & Security, scroll down to Security, select **Open Anyway** next to the message about DICE, and confirm with your password.
- macOS 14 (Sonoma) and earlier: Control-click `DICE.app` in Finder, choose **Open**, and select **Open** in the dialog.

You can also remove the quarantine attribute, which the browser adds to a downloaded file and which causes the check. Remove it from the zip file before you unpack it:

```bash
xattr -d com.apple.quarantine DICE-2.0.0-macOS-arm64.zip
```

If you have already unpacked the app, remove the attribute from the app instead:

```bash
xattr -dr com.apple.quarantine DICE.app
```

### Start on Linux

Unpack the archive and start the application from a terminal:

```bash
tar -xzf DICE-2.0.0-Linux-x86_64.tar.gz
./DICE
```

The archive contains one file, the executable `DICE`, and keeps its execute permission. If the application reports that Qt could not load the platform plugin "xcb", install the libraries listed under [The GUI does not start on Linux](#the-gui-does-not-start-on-linux).

### Where results are saved

The application saves results in an `output` folder inside the folder it was started from, if it can write there. Started from File Explorer on Windows, that is the folder that contains the `.exe` file. Started from a terminal, it is the current folder of the terminal.

If DICE cannot write to that folder, it saves results in `DICE/output` inside your Documents folder, or inside your home folder if you have no Documents folder. On macOS this is where results go when you start DICE from the Finder, because the Finder starts applications in the system root folder (`/`). The Output Location on the Simulation Setup tab shows the full path, and the dialogs that open and save parameter files start in the same place.

### Standalone application and Python installation compared

The standalone application contains:

- the GUI, with parallel processing on all processor cores (the Enable parallel processing option on the Simulation Setup tab)
- the three examples of the Load Example menu (Quick Test, High Precision, Publication Quality)

It does not contain:

- the command-line interface (`dice`)
- the Python package, so you cannot `import dice` in your own scripts
- the CNR estimator web app
- the parameter files and the example profile in `data/examples/` as files you can open

For any of these, install DICE with Python as described above.

A parallel simulation starts one worker process per processor core, and each worker is another copy of the application. Task Manager or Activity Monitor therefore shows several DICE processes during a simulation; they end when the simulation ends. Starting the workers takes a few seconds, so a short simulation can finish sooner with parallel processing turned off.

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

Both commands print the installed version, for example `dice 2.0.0`.

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

For the standalone application, download the file of the new release and delete the old one.

## Uninstall

- standalone application: delete the `.exe` file, `DICE.app`, or the `DICE` executable
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
