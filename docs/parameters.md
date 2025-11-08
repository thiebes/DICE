# DICE Parameter Reference

Complete reference for all parameters used in DICE simulations. Parameters are specified in the `parameters.txt` file or through the GUI.

## Overview

Each simulation models one experimental diffusion measurement, comprising a series of time-evolved Gaussian profiles with noise. Each simulation uses one set of parameters.

## Simulation Control

### Filename slug

A prefix for your output files to help identify them later. The output file names will also automatically include the parameters below, so your slug could be a textual identifier connected to a set of experiments. Avoid special characters in the slug.

**Example:**

If the Filename slug is set to `'example'`, simulation results are saved in the `output/example/` directory:

- `output/example/example.csv` - Simulation results data
- `output/example/example_summary.txt` - Human-readable summary
- `output/example/example_accuracy_histogram.png` - Visualization plot

All simulation parameters and settings are included in the summary text file for reference.

### Number of simulation iterations

Specify the number of Monte Carlo simulations to run. More iterations provide better statistical power but take longer to complete.

**Typical values:** 1000-10000

## Units

### Length unit

Specify the singular unit for length, like `'micrometer'` or `'nanometer'`. This unit applies to all parameters representing physical lengths.

**Available units:** meter, centimeter, millimeter, micrometer, nanometer, angstrom, picometer

### Time unit

Specify the singular unit for time, like `'nanosecond'` or `'picosecond'`. This unit applies to all parameters representing time.

**Available units:** second, millisecond, microsecond, nanosecond, picosecond, femtosecond, attosecond

**Important:** Ensure that all parameter magnitudes are appropriately scaled to these units.

## Diffusion and Lifetime Parameters

Provide **either** diffusion length **or** (diffusion coefficient and lifetime). Do not provide both.

### Option 1: Diffusion Length

**nominal diffusion length**: The characteristic transport distance, $L_D = \sqrt{D\tau}$

**Units:** Specified by length unit

### Option 2: Diffusion Coefficient and Lifetime

**nominal diffusion coefficient (D)**: The diffusion coefficient

**Units:** (length unit)² / (time unit)

**nominal lifetime (tau)**: The excited state lifetime

**Units:** Specified by time unit

### Scaling Examples

**Example 1:** For a diffusion length of 100 nm, but the length unit is `'micrometer'`, use:

```python
'nominal diffusion length': 0.1,
```

**Example 2:** For a nominal diffusion coefficient of 1.0 cm²/s, but your length unit parameter is `'micrometer'` and your time unit is `'nanosecond'`, scale to μm²/ns as follows:

```python
'nominal diffusion coefficient': 0.1,
'nominal lifetime (tau)': 1,
```

## Initial Profile Parameters

Configure the amplitude, mean, and width of the initial Gaussian profile.

### Amplitude

**amplitude_0**: Initial amplitude of the Gaussian profile

**Typical value:** 1.0 (unity)

**Note:** Amplitudes other than unity have not been tested.

### Mean Position

**mean_0**: Mean position of the initial Gaussian

**Typical value:** 0.0 (centered at origin)

**Units:** Specified by length unit

### Profile Width

Provide **either** FWHM **or** sigma. Do not provide both.

**FWHM_0**: Full Width at Half Maximum of the initial Gaussian

**sigma_0**: Standard deviation of the initial Gaussian

**Relationship:** FWHM = σ × 2 × √(2 ln 2) ≈ σ × 2.355

**Units:** Specified by length unit

## Noise Parameters

Specify how noise is added to the simulated profiles. Provide **one** of the following options:

### Option 1: Fixed Noise Value

**noise value**: Standard deviation of white noise added to each profile pixel

**Units:** Same as amplitude (typically unitless if amplitude is 1.0)

### Option 2: Estimate from Data

**estimate noise from data**: Path to a CSV file containing an initial profile

The script will use FFT-based analysis to estimate the CNR (Contrast-to-Noise Ratio) from the experimental data.

**Format:** Single line of comma-separated values representing the profile

### Advanced Options (Not commonly used)

**noise range, reciprocal**: Range for noise with uniform distribution in reciprocal (CNR) space

**Format:** [min, max]

**noise range, reciprocal log**: Range for noise with logarithmic distribution in reciprocal space

**Format:** [min, max]

## Spatial Axis Parameters

### Spatial width

**spatial width**: Total width of the spatial axis

**Units:** Specified by length unit

### Pixel width

**pixel width**: Number of pixels (points) across the spatial axis

**Derived value:** pixel size = spatial width / pixel width

## Temporal Axis Parameters

Provide **either** time range **or** time series. Do not provide both.

### Option 1: Time Range

**time range**: Evenly spaced time points

**Format:** [start, stop, steps]

**Example:**

```python
'time range': [0.0, 10.0, 11]  # 11 points from 0 to 10
```

**Units:** start and stop in time unit specified

### Option 2: Time Series

**time series**: Explicit list of time points

**Format:** [t₁, t₂, t₃, ...]

**Example:**

```python
'time series': [0.1, 0.3, 0.5, 0.7, 0.9]
```

**Units:** Specified by time unit

**Note:** Values will be automatically sorted and must be unique.

## Analysis Parameters

### Proximity level

This parameter determines the accuracy threshold for analysis.

**proximity level**: Fraction defining the accuracy window

**Range:** 0 < value < 1

**Example:** 0.1 means estimates within ±10% of the nominal value are considered accurate

### How it works

- The simulations produce diffusion coefficient estimates, $D_{est}$
- The accuracy of each estimate is quantified by the relative proximity to the nominal value, $D_{nom}$
- The relative proximity is $D_{est} / D_{nom}$
- If the estimate is perfectly accurate, the relative proximity will be 1.0
- The fraction of estimates within the proximity level is reported

**For example,** if you enter 0.1 for your proximity level, the program will report what fraction of all estimates were within ±10% of the nominal value.

**Note:** This metric provides a single evaluation of the combined impact of precision and accuracy on the results, which may be more intuitive and practical than separate statistical measures.

## Plotting Parameters

Optional parameters for customizing output plots.

### Image type

**image type**: File format for plots

**Options:** 'jpg', 'png', 'svg', 'tif', 'pdf'

**Default:** 'png'

### Image dimensions

**image width**: Width of output images in centimeters

**Default:** 10 cm

**image height**: Height of output images in centimeters

**Default:** 6 cm

### Image resolution

**image dpi**: Resolution in dots per inch

**Default:** 100

### Typography

**image font size**: Font size for plot text in points

**Default:** 12

**image tick length**: Axis tick length in points

**image tick width**: Axis tick width in points

### Histogram settings

**image numbins**: Number of bins for histograms

**Default:** 50

**image x_lim**: X-axis limits for plots

**Format:** [min, max] or None

**Default:** None (auto-calculated as ±3 standard deviations)

## Performance Parameters

### Data retention

**retain profile data**: Whether to retain all profile data

**Options:**
- `True` or `1`: Keep all data (necessary for plotting individual profiles)
- `False` or `0`: Discard raw data (retain only fitting results)

**Default:** False

**Warning:** Keeping profile data may significantly increase memory usage, depending on the number of profiles generated.

### Parallel processing

**multiprocessing**: Enable parallel processing

**Options:**
- `False` or `0`: Disable parallel processing
- `True` or `1`: Enable with all available cores
- `-1`: Use all available cores
- `n > 0`: Use n specific cores

**Default:** True

**Note:** Parallel processing can significantly speed up simulations with many iterations.

## Parameter File Format

Parameters are specified as a Python dictionary literal in `parameters.txt`:

```python
{
    'filename slug': 'example',
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

## See Also

- [Installation Guide](installation.md)
- [CLI Guide](cli-guide.md)
- [GUI Guide](../dice_gui/README.md)
- [Glossary](glossary.md)
