# DICE GUI

The graphical user interface (GUI) of the Diffusion Insight Computation Engine (DICE). The GUI sets up and runs the same Monte Carlo simulation as the command-line interface, writes the same output files, and adds an interactive analysis of the results.

## Start the GUI

Install DICE first (see the [Installation Guide](../docs/installation.md)). Then start the GUI with one of these commands:

```bash
uv run dice-gui        # with uv, from the DICE directory
dice-gui               # with pip, environment active
python -m dice_gui     # as a module, environment active
```

All three show a splash screen while the scientific libraries load, then open the main window.

To start the GUI from Python:

```python
from dice_gui.__main__ import main

main()
```

## Main window

The main window has five tabs, a control panel at the bottom, a status bar, and a menu bar.

### Simulation Setup

- **Number of Runs**: the number of Monte Carlo runs (simulated experiments).
- **Filename Slug**: the prefix of the output files. The tab shows the output location that results from it.
- **Multiprocessing**: run the simulations in parallel on all CPU cores.
- **Data Retention**: keep the simulated profiles of every run in memory. Retained profiles use much memory and are not written to the output files.

### Physical Parameters

- **Diffusion Parameters**: diffusion length L, diffusion coefficient D, and lifetime τ, related by L² = D × τ. Enter any two of the three; the GUI computes the third.
- **Initial Profile**: amplitude, mean position, and width of the initial Gaussian profile. Choose whether the width is the full width at half maximum (FWHM) or the standard deviation (σ). The GUI shows the equivalent value of the other quantity.

### Experimental Conditions

- **Noise Parameters**: choose one of two options.
  - **Fixed Noise Value**: enter the noise standard deviation σ or the contrast-to-noise ratio (CNR). The two fields are linked through the amplitude (CNR = amplitude / σ).
  - **Estimate from Data**: select a CSV file that contains a measured initial profile as a single line of comma-separated values. DICE estimates the CNR of that profile with the [fft-cnr](https://pypi.org/project/fft-cnr/) package and sets the simulated noise level so that the simulated initial CNR equals the estimate. The amplitude and units of the measured profile do not matter.
- **Spatial Domain**: the total width of the spatial axis and the number of pixels. The GUI shows the resulting pixel size.
- **Temporal Domain**: choose one of two options.
  - **Time Range**: start, stop, and number of evenly spaced time points.
  - **Time Series**: an explicit, comma-separated list of time points. The GUI sorts the list and rejects duplicate values.

### Analysis Settings

The analysis acts on the results of the last simulation. It does not change the simulation.

- **Fit Method**: show results for the weighted least squares (WLS) or the ordinary least squares (OLS) fit of the mean squared displacement (MSD) against time.
- **Precision Summary**: after a run, shows the standard deviation of the diffusion coefficient estimates, their relative standard error (standard deviation divided by the nominal value), the bias (mean estimate minus nominal value), and the mean ratio of estimate to nominal value. The standard deviation and the bias are in the global units of the run, for example μm²/ns. For results loaded from a CSV file, which does not record the global units, they are in the unit of the file's `nominal diffusion coeff [...]` column, cm²/s by default.
- **Explore Accuracy Threshold**: after a run, set **Threshold** to a value between 0.01 and 0.50 (1% to 50%). The GUI shows the fraction of estimates that fall within that fraction of the nominal diffusion coefficient, for example "within ±10%". A target diagram shows the threshold. You can change the threshold as often as you like without running the simulation again.

### Output Settings

- **Quick Setup**: presets for publication, presentation, and draft figures.
- **Image Format**, **Resolution & Histogram**, **Typography**: file type (PNG, JPG, SVG, TIF), size, resolution, number of histogram bins, font size, and tick size of the accuracy histogram.
- **Regenerate Plot**, **Load Results**, **Reset to Defaults**: redraw the histogram with the current settings, plot a results CSV file from an earlier run, or restore the default output settings.

### Control panel

- **Load Example...**: fill all tabs with example parameters.
- **Run Simulation (Ctrl+R)**: check the inputs and start the simulation. Before the run starts, the GUI warns about settings that make the fits unreliable: a low CNR at the last time point, an initial profile that is narrow compared with the pixel size, or very few time points. You can continue or go back.
- **Stop**: stop a running simulation. A serial simulation stops after the current run; a parallel simulation stops after the current batch of runs (about a tenth of a large simulation). The status line reads "Stopping..." until then.
- A progress bar with elapsed and estimated remaining time.

### Menus

- **File**: new, open, and save parameters files (the same text format that the command-line interface reads), recently used parameters files, and **Load Results...** (Ctrl+L) for a results CSV file.
- **Plots**: **Accuracy Histogram** (Ctrl+H) redraws the histogram from the loaded results. The other entries in the menu are disabled.
- **Settings**: **Global Units...** sets the default length unit and time unit.
- **Help**: a link to the documentation, the list of keyboard shortcuts, and version information.

## Units

The Global Units dialog (**Settings > Global Units...**) sets the default length unit and time unit. Each dimensioned input field also has its own unit selector, so you can enter, for example, the FWHM in nanometers and the spatial width in micrometers. A unit selector follows the global unit until you change it. When you save a parameters file, the per-parameter units are saved as `_unit` keys (see the [Parameter Reference](../docs/parameters.md#per-parameter-units)).

Available units:

- Length: meter, centimeter, millimeter, micrometer, nanometer, angstrom, picometer
- Time: second, millisecond, microsecond, nanosecond, picosecond, femtosecond, attosecond

## Input validation

The GUI checks each field as you type and shows an error message below a field with an invalid value. A tab that contains an invalid field is marked. The rules are:

- Number of runs and number of pixels: positive integers
- Diffusion coefficient, profile width, and spatial width: positive numbers
- Diffusion length, lifetime, amplitude, and noise σ: zero or positive numbers. A lifetime of zero means no decay, and the diffusion length is then infinite and shows as `inf`.
- Mean position: any number
- Time range: numeric start and stop, with start less than stop
- Time series: comma-separated finite numbers, without duplicates
- Filename slug: not empty, with no characters that are invalid in file names
- Noise data file: the file must exist

## Output

The GUI writes the same files as the command-line interface, in `output/<slug>/` under the directory from which you started the GUI:

- `<slug>.csv`: results of every run
- `<slug>_summary.txt`: parameters and summary statistics
- `<slug>_accuracy_histogram.<type>`: histogram of the ratio of estimated to nominal diffusion coefficient (WLS fit)

If the GUI cannot write to that directory, it writes to `DICE/output/<slug>/` in your Documents folder instead, or in your home folder if there is no Documents folder. That happens, for example, when the macOS Finder starts the standalone application in the read-only root folder (`/`). The Output Location on the Simulation Setup tab shows the folder in use.

The [CLI Guide](../docs/cli-guide.md#output-files) describes the contents of these files.

## Code structure

- `__main__.py`: entry point with the splash screen
- `standalone.py`: entry point of the standalone (PyInstaller) application, which runs worker processes and `--self-test` before it imports Qt
- `dice_gui.py`: main window (`DiceGUI`), menus, and control panel
- `tabs/`: one module for each tab
- `dice_interface.py`: `DiceInterface`, which converts the GUI inputs to parameters and runs the simulation
- `simulation_thread.py`: `SimulationThread`, which runs the simulation in the background so that the window stays responsive
- `output_location.py`: `output_root()`, the folder where the GUI saves results and where its file dialogs open
- `validators.py`, `validation_manager.py`: input validation (`ValidationResult`, `ValidationManager`)
- `global_units_dialog.py`, `presets.py`, `proximity_widget.py`, `styles.py`, `accessibility.py`: units dialog, output presets, target diagram, theme, and keyboard access

## Troubleshooting

### The GUI does not start

1. Check that PyQt6 can be imported: `uv run python -c "import PyQt6"`.
2. Install the dependencies again: `uv sync`.
3. Check that a display is available. See the [Installation Guide](../docs/installation.md#troubleshooting) for Linux, SSH, and Windows.

### Validation errors

1. Read the error message below the marked field.
2. Check that each required field has a value and that values are positive where required.
3. For "Estimate from Data", check that the file exists.

### The simulation fails

1. Read the error message for the cause.
2. Check that the parameters are physically reasonable, for example that the spatial width is several times larger than the final profile width.
3. Reduce the number of runs, or turn off data retention, if memory is low.

## License

DICE, including the GUI, is licensed under the MIT License. See [LICENSE.md](../LICENSE.md).
