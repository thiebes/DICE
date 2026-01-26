# DICE GUI

Graphical User Interface for the Diffusion Insight Computation Engine.

## Features

The DICE GUI provides an intuitive interface for configuring and running DICE simulations:

- **Unit Selection**: Choose length and time units, with automatic label updates throughout the interface
- **Tabbed Organization**: Parameters organized into logical sections:
  - Simulation Setup: Number of runs, output filenames
  - Physical Parameters: Diffusion parameters and initial profile configuration
  - Experimental Conditions: Noise, spatial domain, and temporal domain settings
  - Analysis Settings: Proximity level for accuracy assessment

- **Smart Input Validation**: Real-time validation with clear error messages
- **Radio Button Groups**: Mutually exclusive parameters with automatic enable/disable logic:
  - Diffusion: Length vs (Coefficient + Lifetime)
  - Profile Width: FWHM vs Sigma
  - Noise: Fixed value vs Estimate from data
  - Time: Range vs Series

- **Live Calculations**: Automatic calculation and display of:
  - Diffusion length from D and τ
  - FWHM ↔ Sigma conversion
  - Pixel size from spatial parameters
  - Proximity level percentage

- **Progress Tracking**: Visual feedback during simulation execution
- **Error Handling**: Clear error messages for invalid inputs or simulation failures

## Installation

The GUI is included with DICE when you install using uv:

```bash
uv sync
```

This will install all dependencies including PyQt6.

## Running the GUI

### From Command Line

```bash
uv run dice-gui
```

Or:

```bash
uv run python -m dice_gui.dice_gui
```

### From Python

```python
from dice_gui.dice_gui import main
main()
```

## Usage

### Basic Workflow

1. **Select Units**: Choose length and time units from the dropdowns at the top
2. **Configure Simulation**:
   - Set number of Monte Carlo runs
   - Specify output filename slug
3. **Set Physical Parameters**:
   - Choose diffusion specification method (length OR coefficient+lifetime)
   - Set initial profile parameters (amplitude, mean, width)
   - Select width type (FWHM or Sigma)
4. **Configure Experimental Conditions**:
   - Choose noise specification (fixed value OR estimate from data)
   - Set spatial domain (width and number of pixels)
   - Set temporal domain (range OR explicit series)
5. **Analysis Settings**: Adjust proximity level for accuracy threshold
6. **Run Simulation**: Click "Run Simulation" button
7. **View Results**: Check output files in the working directory

### Parameter Dependencies

The GUI enforces parameter dependencies automatically:

- **Diffusion Parameters**:
  - Select "Diffusion Length" OR "Coefficient + Lifetime"
  - If Coefficient + Lifetime selected, both D and τ are required
  - Live calculation shows diffusion length when D and τ are entered

- **Profile Width**:
  - Select either FWHM or Sigma
  - Live conversion shows equivalent value in the other unit

- **Noise Specification**:
  - Fixed value: Enter noise standard deviation directly
  - Estimate from data: Browse to CSV file with experimental profile

- **Temporal Axis**:
  - Time Range: Evenly spaced points (start, stop, steps)
  - Time Series: Explicit comma-separated time values

### Unit System

The GUI automatically updates all unit labels when you change the length or time units:

- Length units: meter, centimeter, millimeter, micrometer, nanometer, angstrom, picometer
- Time units: second, millisecond, microsecond, nanosecond, picosecond, femtosecond, attosecond
- Derived units update automatically (e.g., μm²/ns for diffusion coefficient)

## Architecture

### Modules

- **dice_gui.py**: Main GUI application with PyQt6 interface
- **validators.py**: Input validation functions for all parameter types
- **dice_interface.py**: Bridge between GUI and DICE simulation engine

### Key Classes

- **DiceGUI**: Main window with tab interface and parameter controls
- **SimulationThread**: Background thread for running simulations without blocking UI
- **DiceInterface**: Handles parameter conversion and simulation execution
- **ValidationResult**: Encapsulates validation results with error messages

## Validation

All inputs are validated before running simulations:

- Positive integers: Number of runs, pixel width, time steps
- Positive floats: Diffusion parameters, profile width, spatial width, noise value
- Proximity level: Must be between 0 and 1
- Time range: Start must be less than stop
- Time series: Must be comma-separated finite values, no duplicates
- File paths: Must exist and be valid files

Invalid inputs show clear error messages indicating what needs to be corrected.

## Output

The GUI generates the same outputs as the command-line DICE tool:

- CSV files with simulation results
- Histogram plots of diffusion coefficient estimates
- Summary statistics (precision, accuracy metrics)
- Text summaries of simulation parameters and results

Output files are saved in the current working directory with the specified filename slug.

## Troubleshooting

### GUI Won't Start

If the GUI fails to start, check:

1. PyQt6 is installed: `uv run python -c "import PyQt6"`
2. All dependencies are installed: `uv sync`
3. Display environment is available (required for GUI applications)

### Validation Errors

If you receive validation errors:

1. Check that all required fields are filled
2. Verify numeric values are positive where required
3. Ensure mutually exclusive groups have exactly one option selected
4. Check that file paths exist for "estimate from data" option

### Simulation Errors

If simulation fails:

1. Verify dice.py is importable from the working directory
2. Check that all parameters are physically reasonable
3. Ensure sufficient memory for large simulations
4. Review error message for specific failure cause

## Future Enhancements

Planned features for future releases:

- Advanced settings dialog for visualization and performance parameters
- Results viewer window with embedded plots
- Parameter save/load functionality (JSON format)
- Parameter templates and presets
- Batch processing capabilities
- Enhanced CNR estimation with FFT visualization
- Help system with parameter descriptions and examples

## Requirements

- Python >= 3.9
- PyQt6 >= 6.0.0
- numpy >= 1.20.0
- scipy >= 1.7.0
- pandas >= 1.3.0
- matplotlib >= 3.3.0
- All other DICE dependencies

## License

Same license as DICE (see main repository LICENSE file).
