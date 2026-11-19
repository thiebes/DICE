# DICE Parameter Reference

Every parameter of a DICE simulation is described below. You set the parameters in a parameters file for the command-line interface (CLI) or in the tabs of the graphical user interface (GUI). The GUI can save and open the same parameters files.

## Overview

A simulation consists of many runs. Each run simulates one diffusion measurement: a series of Gaussian profiles that spread by diffusion and decay over time, with random noise added. DICE fits each noisy profile, estimates the diffusion coefficient from the fits, and compares the estimates of all runs with the nominal diffusion coefficient. All runs use the same parameters; only the random noise differs.

The table lists the parameters by group. "One of" means that you give exactly one option of the group.

| Group | Keys | Required |
| ------- | ------------- | ------------- |
| Simulation control | `filename slug`, `number of runs` | `number of runs` |
| Units | `length unit`, `time unit`, keys that end in `_unit` | No |
| Diffusion | `nominal diffusion length`, or `nominal diffusion coefficient` with `nominal lifetime (tau)` | One of |
| Initial profile | `amplitude_0`, `mean_0`, and `FWHM_0` or `sigma_0` | Yes |
| Noise | `noise value`, `estimate noise from data`, `noise range, reciprocal`, `noise range, reciprocal log` | One of |
| Spatial axis | `spatial width`, `pixel width` | Yes |
| Temporal axis | `time range` or `time series` | One of |
| Plot | keys that start with `image` | No |
| Performance | `retain profile data`, `multiprocessing` | No |

## Simulation control

### `filename slug`

A prefix for the output files. DICE writes the output to the directory `output/<slug>/` under the current working directory. For the slug `'example'`, the files are:

- `output/example/example.csv`: results of every run
- `output/example/example_summary.txt`: parameters and summary statistics
- `output/example/example_accuracy_histogram.png`: histogram of the estimates

Use only characters that are valid in file names. If the key is missing, the slug is `dice_simulation`.

### `number of runs`

The number of runs (Monte Carlo iterations). More runs give more precise statistics and take longer.

**Typical values:** 100 for a quick check; 1000 to 10000 for final results.

## Units

### `length unit` and `time unit`

The global units of the simulation. All values with a length or time dimension are in these units, unless a per-parameter unit key (see below) gives another unit. The summary file and the results file report values in these units, except for the results-file columns marked `[cm²/s]`, which DICE converts to cm²/s.

- **Length units:** meter, centimeter, millimeter, micrometer, nanometer, angstrom, picometer
- **Time units:** second, millisecond, microsecond, nanosecond, picosecond, femtosecond, attosecond

Write the unit name in the singular, for example `'micrometer'`. If a key is missing, DICE uses micrometer and nanosecond.

### Per-parameter units

A unit key gives the unit of one value. DICE converts that value to the global unit before the simulation. Without a unit key, a value is in the global unit.

| Unit key | Applies to | Dimension |
| ------- | ------------- | ------------- |
| `FWHM_0_unit` | `FWHM_0` | length |
| `sigma_0_unit` | `sigma_0` | length |
| `mu_0_unit` | `mean_0` | length |
| `spatial_width_unit` | `spatial width` | length |
| `diffusion_length_unit` | `nominal diffusion length` | length |
| `diffusion_coefficient_length_unit` | `nominal diffusion coefficient` (length part) | length |
| `diffusion_coefficient_time_unit` | `nominal diffusion coefficient` (time part) | time |
| `lifetime_unit` | `nominal lifetime (tau)` | time |
| `time_range_unit` | start and stop of `time range` | time |
| `time_start_unit`, `time_stop_unit` | start or stop of `time range` | time |
| `time_series_unit` | `time series` | time |

The unit key for `mean_0` is `mu_0_unit`.

**Example:** the global units are micrometer and nanosecond, the initial FWHM is 800 nm, the diffusion coefficient is 0.01 cm²/s, and the lifetime is 500 ps:

```python
'length unit': 'micrometer',
'time unit': 'nanosecond',
'FWHM_0': 800,
'FWHM_0_unit': 'nanometer',
'nominal diffusion coefficient': 0.01,
'diffusion_coefficient_length_unit': 'centimeter',
'diffusion_coefficient_time_unit': 'second',
'nominal lifetime (tau)': 500,
'lifetime_unit': 'picosecond',
```

DICE converts these values to 0.8 μm, 0.001 μm²/ns, and 0.5 ns.

## Diffusion and lifetime

Give **either** the diffusion length **or** both the diffusion coefficient and the lifetime. Do not give all three.

### `nominal diffusion length`

The diffusion length, L_D = √(Dτ).

**Units:** length unit

If you give only the diffusion length, DICE sets the lifetime to 1 time unit and the diffusion coefficient to L_D² per time unit. The time points of the temporal axis are then in units of the lifetime.

### `nominal diffusion coefficient` and `nominal lifetime (tau)`

The diffusion coefficient D and the excited state lifetime τ. DICE computes the diffusion length from them.

**Units:** D in (length unit)² / (time unit); τ in time unit

### Scaling examples

**Example 1:** for a diffusion length of 100 nm with the length unit `'micrometer'`:

```python
'nominal diffusion length': 0.1,
```

**Example 2:** for a diffusion coefficient of 1.0 cm²/s and a lifetime of 1 ns, with the length unit `'micrometer'` and the time unit `'nanosecond'`. Because 1 cm²/s equals 0.1 μm²/ns:

```python
'nominal diffusion coefficient': 0.1,
'nominal lifetime (tau)': 1,
```

Alternatively, give the value in cm²/s with the unit keys `diffusion_coefficient_length_unit` and `diffusion_coefficient_time_unit` (see [Per-parameter units](#per-parameter-units)).

## Initial profile

The initial profile is a Gaussian function with an amplitude, a mean position, and a width.

### `amplitude_0`

The amplitude of the initial Gaussian profile.

**Typical value:** 1.0

DICE reports the nominal contrast-to-noise ratio (CNR) as 1 / `noise value`, which equals the CNR only for an amplitude of 1. Amplitudes other than 1 have not been tested.

### `mean_0`

The mean position (center) of the initial Gaussian profile.

**Typical value:** 0.0 (the center of the spatial axis)

**Units:** length unit

### `FWHM_0` or `sigma_0`

The width of the initial profile, as the full width at half maximum (FWHM) or as the standard deviation σ. Give exactly one of the two.

**Relationship:** FWHM = 2√(2 ln 2) σ ≈ 2.355 σ

**Units:** length unit

## Noise

DICE adds white Gaussian noise of standard deviation σ to every pixel of every profile, with the same σ at all time points. Give exactly one of the following options.

### `noise value`

The standard deviation σ of the noise.

**Units:** the units of the amplitude. For an amplitude of 1, the initial CNR is 1 / σ.

### `estimate noise from data`

The path to a CSV file that contains a measured initial profile, as a single line of comma-separated values. A relative path is relative to the current working directory.

DICE estimates the CNR of the measured profile with the [fft-cnr](https://pypi.org/project/fft-cnr/) package. It then sets the noise level so that the initial CNR of the simulated profiles equals that estimate. The amplitude and units of the measured profile do not matter. The [fft-cnr documentation](https://fft-cnr.readthedocs.io) describes the method and its limits.

Use a profile whose peak is at least about 9 pixels wide at half maximum, with flat background on both sides. For narrower peaks fft-cnr overestimates the noise and the CNR comes out too low. In simulated Gaussian profiles with white noise, the estimate was within 2.4% of the true CNR for peaks 9 to 14 pixels wide, and 14% to 27% too low for a peak 4.6 pixels wide. Check the summary file's initial CNR against your own estimate for the measured data.

### `noise range, reciprocal` and `noise range, reciprocal log`

**Format:** `[low, high]`, two noise standard deviations with low < high

DICE draws one noise value at random from this range. With `noise range, reciprocal`, the reciprocal of the noise value (the CNR for an amplitude of 1) is uniformly distributed between 1/high and 1/low. With `noise range, reciprocal log`, its base-10 logarithm is uniformly distributed. All runs of the simulation use the one drawn value, which the summary file reports.

## Spatial axis

### `spatial width`

The total width of the spatial axis. The axis is centered on zero and runs from −width/2 to +width/2.

**Units:** length unit

Choose a width several times larger than the width of the profile at the last time point, so that the profile is not cut off at the edges.

### `pixel width`

The number of points (pixels) on the spatial axis. The axis includes both end points, so the spacing between points is `spatial width` / (`pixel width` − 1).

## Temporal axis

Give **either** a time range **or** a time series. The linear fit needs at least two time points.

### `time range`

Evenly spaced time points.

**Format:** `[start, stop, steps]`, where `steps` is the number of time points. Both `start` and `stop` are included, and `start` must be less than `stop`.

**Example:**

```python
'time range': [0.0, 10.0, 11]  # 11 points: 0, 1, 2, ..., 10
```

**Units:** time unit

### `time series`

An explicit list of time points.

**Format:** `[t1, t2, t3, ...]`

**Example:**

```python
'time series': [0.1, 0.3, 0.5, 0.7, 0.9]
```

**Units:** time unit

List the time points in increasing order, without duplicates. The results file reports the fitted variance and the CNR estimate of the profile at the first time point as `estimated sigma^2_0` and `estimated CNR`. The GUI sorts the list and rejects duplicate values.

## Analysis output

For each run, DICE fits a Gaussian function to each noisy profile. It then fits the variance of these Gaussians against time with two linear fits: weighted least squares (WLS) and ordinary least squares (OLS). In one dimension the slope equals 2D, so each linear fit gives one estimate of the diffusion coefficient. For all runs together, DICE reports:

- the standard deviation of the estimates
- the relative standard error (standard deviation divided by the nominal diffusion coefficient)
- the mean estimate, the bias (mean estimate minus nominal value), and the root-mean-square error (RMSE)
- the fraction of estimates within ±5%, ±10%, ±20%, and ±50% of the nominal value

The proximity level (the fraction of the nominal value within which an estimate counts as close) is not a simulation parameter. The summary file reports the four standard levels listed above. In the GUI, choose any level from 1% to 50% in the Analysis Settings tab after a run. From Python or a results file, you can compute the fraction for any level (see the [CLI Guide](cli-guide.md#analyze-the-results-in-python)). DICE ignores a `proximity level` key in a parameters file and warns about it.

## Plot parameters

Optional parameters for the accuracy histogram.

### `image type`

The file format of the plot.

**Options:** `'png'`, `'jpg'`, `'svg'`, `'tif'`, `'pdf'`

**Default:** `'png'`

### `image width` and `image height`

The width and height of the plot in centimeters.

**Defaults:** 10 cm and 6 cm

### `image dpi`

The resolution in dots per inch.

**Default:** 100

### `image font size`, `image tick length`, `image tick width`, `image numbins`

The font size of the plot text (points), the length and width of the axis ticks (points), and the number of histogram bins.

The command-line interface and the GUI both apply these values. The GUI takes them from its Output Settings tab.

**Defaults:** a font size of 12 points, ticks 4 points long and 1 point wide, and 50 bins

### `image x_lim`

The limits of the x-axis of the histogram.

**Format:** `[low, high]`, two numbers with low < high

**Default:** the mean ± 3 standard deviations of the plotted ratios

## Performance parameters

### `retain profile data`

Whether to keep the simulated profiles, the noisy profiles, and the fitted widths of every run in memory.

**Options:** `True` or `1` to keep them; `False` or `0` to discard them

**Default:** `False`

The output files do not include the retained data. The data are available from Python, in the result of `dice.dice_runner()`. On the command line, the `--retain-profiles` option turns retention on for a file that does not set this key.

**Warning:** retained profile data can use a large amount of memory, in proportion to the number of runs.

### `multiprocessing`

Whether to run the simulations in parallel.

**Options:**

- `False` or `0`: run the simulations one after another
- `True` or any other number: run the simulations in parallel on all available CPU cores

**Default:** `True`

To set the number of worker processes, use the command-line option `--multiprocessing N`, which overrides this key (see the [CLI Guide](cli-guide.md#command-line-options)).

## Parameters file format

A parameters file contains one Python dictionary literal. Each entry has the form `'key': value,`. Text after `#` on a line is a comment.

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
}
```

The directory `data/examples/` contains example files:

- `parameters.txt`: all parameters, with comments
- `quick_test.txt`: 100 runs, for a quick check
- `publication.txt`: 1000 runs
- `high_precision.txt`: 10000 runs on a finer grid

## See also

- [Installation Guide](installation.md)
- [CLI Guide](cli-guide.md)
- [GUI Guide](../dice_gui/README.md)
- [Glossary](glossary.md)
