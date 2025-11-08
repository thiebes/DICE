# DICE Command-Line Interface Guide

Complete guide to using DICE from the command line.

## Quick Reference

```bash
# Run with parameters file
dice parameters.txt

# Using uv
uv run dice parameters.txt

# Using Python module
python run_dice.py parameters.txt
```

## Basic Usage

### Running Simulations

DICE reads parameters from a text file and runs Monte Carlo simulations to assess diffusion coefficient estimation accuracy.

**Command:**

```bash
dice parameters.txt
```

**Output:** Results are saved to `output/<filename_slug>/` directory.

### Parameters File

Create a `parameters.txt` file with simulation parameters as a Python dictionary:

```python
{
    'filename slug': 'my_simulation',
    'number of runs': 1000,
    'length unit': 'micrometer',
    'time unit': 'nanosecond',
    'nominal diffusion coefficient': 0.5,
    'nominal lifetime (tau)': 2.0,
    'amplitude_0': 1.0,
    'mean_0': 0.0,
    'FWHM_0': 1.0,
    'noise value': 0.01,
    'spatial width': 10.0,
    'pixel width': 100,
    'time range': [0.0, 10.0, 11],
    'proximity level': 0.1,
}
```

See [Parameter Reference](parameters.md) for complete documentation.

## Running Methods

### Method 1: Installed Package

If you installed DICE with `pip install -e .`:

```bash
dice parameters.txt
```

The `dice` command is available from any directory.

### Method 2: Using uv

With uv, you don't need to install DICE separately:

```bash
cd DICE
uv run dice parameters.txt
```

### Method 3: Direct Script Execution

From the DICE directory:

```bash
python run_dice.py parameters.txt
```

### Method 4: Python Module

```bash
python -m dice.cli.main parameters.txt
```

## Programmatic Usage

### From Python Scripts

```python
import dice

# Run simulation with parameters file
result = dice.dice_runner("parameters.txt")
```

### Using Parameters Dictionary

```python
from dice.analysis.simulation import run_monte_carlo_simulation
from dice.io.parameters import load_parameters

# Load parameters from file
parameters = load_parameters("parameters.txt")

# Run simulation
result = run_monte_carlo_simulation(parameters)
```

### Building Parameters in Code

```python
parameters = {
    'filename slug': 'code_simulation',
    'number of runs': 1000,
    'length unit': 'micrometer',
    'time unit': 'nanosecond',
    'nominal diffusion coefficient': 0.5,
    'nominal lifetime (tau)': 2.0,
    'amplitude_0': 1.0,
    'mean_0': 0.0,
    'FWHM_0': 1.0,
    'noise value': 0.01,
    'spatial width': 10.0,
    'pixel width': 100,
    'time range': [0.0, 10.0, 11],
    'proximity level': 0.1,
}

import dice
result = dice.dice_runner(parameters)
```

## Output Files

DICE generates three main output files in `output/<filename_slug>/`:

### 1. Results CSV

**File:** `<slug>.csv`

Contains raw simulation data:
- Each row is one Monte Carlo iteration
- Columns include estimated diffusion coefficient, fit parameters, and statistics

**Usage:**

```python
import pandas as pd

# Load results
df = pd.read_csv('output/my_simulation/my_simulation.csv')

# Analyze
print(df['D_est'].describe())
print(df['D_est/D_nom'].hist())
```

### 2. Summary Text

**File:** `<slug>_summary.txt`

Human-readable summary including:
- All simulation parameters
- Statistical analysis (mean, median, std dev)
- Proximity analysis results
- Timestamp and run information

### 3. Histogram Plot

**File:** `<slug>_accuracy_histogram.png`

Visualization of D_est/D_nom distribution showing:
- Histogram of estimate ratios
- Proximity level boundaries
- Statistical markers

## Advanced Usage

### Batch Processing

Process multiple parameter files:

```bash
for params in params_*.txt; do
    dice "$params"
done
```

Or with Python:

```python
import glob
import dice

for params_file in glob.glob("params_*.txt"):
    print(f"Processing {params_file}")
    dice.dice_runner(params_file)
```

### Parameter Scanning

Systematically vary parameters:

```python
import dice
import numpy as np

base_params = {
    'filename slug': 'scan',
    'number of runs': 500,
    'length unit': 'micrometer',
    'time unit': 'nanosecond',
    'amplitude_0': 1.0,
    'mean_0': 0.0,
    'FWHM_0': 1.0,
    'spatial width': 10.0,
    'pixel width': 100,
    'time range': [0.0, 10.0, 11],
    'proximity level': 0.1,
}

# Scan over noise values
noise_values = [0.005, 0.01, 0.02, 0.05, 0.1]

for noise in noise_values:
    params = base_params.copy()
    params['filename slug'] = f'scan_noise_{noise}'
    params['noise value'] = noise
    params['nominal diffusion coefficient'] = 0.5
    params['nominal lifetime (tau)'] = 2.0

    print(f"Running with noise = {noise}")
    dice.dice_runner(params)
```

### Custom Analysis

Access simulation results programmatically:

```python
from dice.analysis.simulation import run_monte_carlo_simulation
from dice.io.parameters import load_parameters
import pandas as pd

# Load and run
parameters = load_parameters("parameters.txt")
result = run_monte_carlo_simulation(parameters)

# result is a DataFrame with all simulation data
print(f"Mean D_est: {result['D_est'].mean()}")
print(f"Std Dev: {result['D_est'].std()}")

# Custom proximity analysis
nominal_D = parameters.get('nominal diffusion coefficient',
                           parameters.get('nominal diffusion length')**2)
ratios = result['D_est'] / nominal_D
within_5pct = ((ratios > 0.95) & (ratios < 1.05)).sum() / len(ratios)
print(f"Within ±5%: {within_5pct:.1%}")
```

## Performance Optimization

### Parallel Processing

Enable multiprocessing in parameters file:

```python
{
    # ... other parameters ...
    'multiprocessing': True,  # Use all cores
    # or
    'multiprocessing': 4,     # Use 4 cores
}
```

### Memory Management

For large simulations, disable profile data retention:

```python
{
    # ... other parameters ...
    'retain profile data': False,  # Save memory
}
```

### Batch Size

For very long parameter scans, process in batches:

```python
import dice

def run_batch(param_files, batch_size=10):
    for i in range(0, len(param_files), batch_size):
        batch = param_files[i:i+batch_size]
        for params_file in batch:
            dice.dice_runner(params_file)
        print(f"Completed batch {i//batch_size + 1}")
```

## Troubleshooting

### Common Errors

**"File not found"**
- Check that parameters file path is correct
- Use absolute paths if running from different directory

**"Invalid parameter"**
- Verify parameter file syntax (must be valid Python dict)
- Check for required parameters (see [Parameter Reference](parameters.md))

**"Simulation failed"**
- Check parameter values are physically reasonable
- Ensure sufficient memory for large simulations
- Verify all units are consistent

### Debugging

Add print statements to see progress:

```python
import dice

# DICE prints progress information to stdout
result = dice.dice_runner("parameters.txt")
```

### Checking Parameters

Validate parameters before running:

```python
from dice.io.parameters import load_parameters, validate_parameters

params = load_parameters("parameters.txt")
is_valid, error_msg = validate_parameters(params)

if not is_valid:
    print(f"Invalid parameters: {error_msg}")
else:
    print("Parameters are valid")
```

## Integration with Workflows

### Jupyter Notebooks

```python
import dice
import pandas as pd
import matplotlib.pyplot as plt

# Run simulation
result = dice.dice_runner("parameters.txt")

# Load and plot results
df = pd.read_csv('output/my_simulation/my_simulation.csv')
df['D_est/D_nom'].hist(bins=50)
plt.xlabel('D_est / D_nom')
plt.ylabel('Frequency')
plt.show()
```

### Shell Scripts

```bash
#!/bin/bash

# Run multiple simulations
for noise in 0.01 0.02 0.05 0.1; do
    # Create parameters file
    cat > params_${noise}.txt <<EOF
{
    'filename slug': 'noise_${noise}',
    'number of runs': 1000,
    'noise value': ${noise},
    # ... other parameters ...
}
EOF

    # Run simulation
    dice params_${noise}.txt
done

# Combine results
python analyze_results.py
```

### Make files

```makefile
PARAMS = $(wildcard params_*.txt)
RESULTS = $(PARAMS:params_%.txt=output/%/%.csv)

all: $(RESULTS)

output/%/%.csv: params_%.txt
	dice $<

clean:
	rm -rf output/

.PHONY: all clean
```

## See Also

- [Parameter Reference](parameters.md)
- [Installation Guide](installation.md)
- [GUI Guide](../dice_gui/README.md)
- [Glossary](glossary.md)
