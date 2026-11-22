# DICE Command-Line Interface Guide

The command-line interface (CLI) runs a DICE simulation from a parameters file and writes the results to files. It suits batch processing, scripts, and computers without a display.

## Quick reference

```bash
# Installed with pip, environment active
dice data/examples/parameters.txt

# With uv, from the DICE directory
uv run dice data/examples/parameters.txt

# Short test run (100 runs)
uv run dice data/examples/quick_test.txt

# Accuracy over a range of initial CNR (100 runs, a noise level drawn for each)
uv run dice data/examples/noise_range.txt
```

The examples on this page use the files in `data/examples/` and assume that you run them from the `DICE` directory. With pip, leave out `uv run`.

## Run a simulation

```bash
dice parameters.txt
```

DICE reads the parameters file, runs the simulation, and writes three files to `output/<slug>/` under the current working directory, where `<slug>` is the `filename slug` parameter.

### Parameters file

A parameters file contains one Python dictionary. For example:

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
}
```

The [Parameter Reference](parameters.md) describes every parameter. `data/examples/parameters.txt` lists all parameters with comments.

## Command-line options

```text
dice [options] parameters_file
```

| Option | Effect |
| ------- | ------------- |
| `--verbose`, `-v` | Print the settings, the size of the simulation, and progress messages. |
| `--quiet`, `-q` | Print warnings and errors only. |
| `--output-dir DIR`, `-o DIR` | Write the output files to `DIR` instead of `output/<slug>/`. |
| `--no-plots` | Do not create the plot: the histogram, or the scatter plot of a noise range. |
| `--multiprocessing N` | `0` runs the simulations one after another. A positive `N` runs them in parallel in `N` worker processes, and `-1` uses all CPU cores. Overrides the `multiprocessing` parameter. |
| `--seed N` | Use the random seed `N` (a non-negative integer) for the noise. See [Reproducible results](#reproducible-results). |
| `--export-format {csv,json,both}` | `csv` (default) writes the results file `<slug>.csv`. `json` writes `<slug>.json` with the parameters and the summary statistics, but no per-run results. Its values are in the units of the simulation, which `summary.units` names. With a noise range, it also has the noise standard deviation of each run and the statistics of each CNR bin. `both` writes both files. |
| `--progress-interval N` | With `--verbose`, the number of runs between progress messages (default 100). |
| `--retain-profiles` | Keep the profiles of every run in memory. The output files do not include them. |
| `--dry-run`, `--check` | Load and validate the parameters file as a real run would, then stop without a simulation and without writing files. |
| `--version` | Print the version and stop. |
| `--help`, `-h` | Print the list of options and stop. |

### Reproducible results

With `--seed N`, repeated runs with the same parameters and the same seed give the same results. The serial and the parallel calculation also give the same results as each other. With a noise range, the seed also reproduces the noise level that each run draws. Without `--seed`, each run draws new random numbers, so the results differ slightly from one run to the next.

```bash
dice data/examples/quick_test.txt --seed 12345
```

Results are reproducible for the same versions of DICE and its dependencies on the same platform. On Linux, Windows, and Macs with Apple silicon they repeat bit for bit. On Macs with Intel processors, the linear algebra library that NumPy and SciPy use there can change the last digit of a few fitted values from one run to the next; the relative differences are about 1e-15, far below any statistical effect. Identical numbers across platforms or across dependency versions are not guaranteed. The GUI has no seed option.

## Output files

### Results file: `<slug>.csv`

One row for each run. The main columns are:

| Columns | Content |
| ------- | ------------- |
| `run number` | Index of the run |
| `nominal diffusion coeff [μm²/ns]`, `nominal lifetime [ns]`, `nominal diffusion length [μm]` | Nominal values, in the units of the simulation |
| `noise standard deviation` | Noise standard deviation of the run, in the units of the signal. Only with a noise range, where each run draws its own. |
| `nominal CNR` | `amplitude_0` / noise standard deviation |
| `estimated CNR` | CNR of the noisy profile at the first time point, estimated with the [fft-cnr](https://pypi.org/project/fft-cnr/) package |
| `nominal sigma^2_0 [μm²]`, `estimated sigma^2_0 [μm²]` | Variance of the initial profile: nominal value, and fitted value at the first time point |
| `weighted fit diffusion slope (2D) [μm²/ns]`, `... slope stderr [μm²/ns]`, `... intercept [μm²]`, `... intercept stderr [μm²]` | WLS fit of variance against time, in the units of the simulation. The slope is twice the diffusion coefficient. |
| `unweighted fit ...` | The same quantities for the OLS fit |
| `weighted fit diffusion coeff [cm²/s]`, `weighted fit diffusion stderr [cm²/s]` | WLS estimate of the diffusion coefficient and its standard error, in cm²/s |
| `unweighted fit diffusion coeff [cm²/s]`, `unweighted fit diffusion stderr [cm²/s]` | The same for the OLS fit |
| `nominal diffusion coeff [cm²/s]` | Nominal diffusion coefficient in cm²/s |

WLS stands for weighted least squares, OLS for ordinary least squares. In one dimension the slope of the variance against time equals 2D. If a fit fails in a run, the columns of that fit are empty in that row. The file is encoded in UTF-8.

Each column whose value has a unit names it in brackets. The table shows the names for a simulation in micrometers and nanoseconds; a simulation in nanometers and picoseconds writes `nominal lifetime [ps]`, `weighted fit diffusion slope (2D) [nm²/ps]`, and so on. The function `dice.io.results.results_column` finds a column from its name without the unit, so that a script does not depend on the units (see [Analyze the results in Python](#analyze-the-results-in-python)).

A diffusion coefficient estimate can be negative. At a low CNR the fitted widths scatter, and the line through them can slope downward. DICE keeps these estimates in the results file and in every statistic: an experiment at that CNR would obtain them too, and leaving them out would bias the mean upward.

The parameters `output_diffusion_length_unit` and `output_diffusion_time_unit` replace cm²/s in the columns marked `[cm²/s]` with another unit, for example `[nm²/ps]` (see [Output units](parameters.md#output_diffusion_length_unit-and-output_diffusion_time_unit)). When that unit is the unit of the simulation, the file has one `nominal diffusion coeff` column.

### Summary file: `<slug>_summary.txt`

A text report that contains:

- the DICE version and the seed, which together with the parameters file reproduce the simulation (see [Reproducible results](#reproducible-results)). The seed is the number given to `--seed`. A simulation started from Python with a NumPy `SeedSequence` that one integer does not describe records its entropy and its spawn key
- the main simulation parameters: number of runs, spatial axis, time points, noise level, initial CNR, initial variance, and the nominal diffusion length, diffusion coefficient, and lifetime
- for the WLS fit and, for comparison, the OLS fit: the number of runs with an estimate, and the standard deviation, relative standard error, mean, standard error of the mean, bias, and root-mean-square error (RMSE) of the diffusion coefficient estimates
- the fraction of estimates within ±5%, ±10%, ±20%, and ±50% of the nominal value, each with its count and its 95% interval

The statistics come from a finite number of runs, so each has an uncertainty of its own. DICE reports it for the mean, the bias and the fractions. The standard deviation, the relative standard error and the root mean square error come without one; treat their last digit with care when the number of runs is small. The standard error of the mean is also the standard error of the bias. Compare a bias with it before you read the bias as real: a bias smaller than about twice its standard error may be chance. A fraction is written as `19.0% (19 of 100; 12.5% to 27.8%)`: 19 of the 100 estimates lie within the level, and the fraction that a much longer simulation would give lies between 12.5% and 27.8% with 95% confidence. More runs narrow the interval. The interval is the score interval for a binomial proportion (the Wilson interval).

A run has no estimate when its linear fit fails. If no run has an estimate, the summary ends with a line that says so.

Each value with a unit is followed by that unit. The standard deviation, mean, bias, and RMSE are in the unit of the nominal diffusion coefficient, for example `micrometer^2/nanosecond`: the global units, or the output units that `output_diffusion_length_unit` and `output_diffusion_time_unit` set. The relative standard error and the fractions have no unit.

With a [noise range](parameters.md#noise-range-reciprocal-and-noise-range-reciprocal-log), the runs differ in initial CNR, and statistics of all runs together would mix low-CNR and high-CNR runs. In place of the noise level and the initial CNR, the summary reports:

- the number of noise standard deviations drawn, their lowest and highest values, and the range they were drawn from
- the lowest and highest initial CNR of the runs, the CNR range of the parameters, and whether the CNR or its logarithm is uniformly distributed

In place of the statistics of all runs, it gives the same statistics for each bin of initial CNR. DICE sorts the runs by initial CNR and splits them into consecutive bins with equal numbers of runs; the sizes differ by one at most, and the bins with lower CNR take the extra runs. The number of bins is 5, or fewer so that each bin has at least 20 runs: the number of runs divided by 20 and rounded down, but at least 1 and at most 5. For example, 100 or more runs give 5 bins, 50 runs give 2 bins, and fewer than 40 runs give 1 bin. Each bin is headed by its lowest and highest initial CNR and its number of runs, for example `Initial CNR 10.259 to 19.884 (20 runs):`.

### Histogram: `<slug>_accuracy_histogram.png`

A histogram of the ratio D_est/D_nom of the WLS estimates to the nominal diffusion coefficient. The y-axis shows the probability of each bin. A normal distribution with the same mean and standard deviation is drawn over the histogram, a dashed line marks the nominal value at 1, and a legend reports the mean, median, and standard deviation of the ratio. The x-axis spans the mean ± 3 standard deviations, or the limits that `image x_lim` sets. Estimates beyond the axis are not drawn; a note above the plot gives their number on each side. The file type follows the `image type` parameter.

### Scatter plot: `<slug>_accuracy_vs_cnr.png`

A simulation with a noise range writes this plot in place of the histogram. Each point is one run: the ratio D_est/D_nom of its WLS estimate on the y-axis, and its initial CNR on the x-axis. A dashed line marks D_est/D_nom = 1. The CNR axis is logarithmic for `noise range, reciprocal log`. The y-axis spans the mean ± 3 standard deviations of the ratios, or the limits that `image x_lim` sets. The other plot parameters apply as to the histogram, except `image numbins`.

## Use DICE from Python

### Run a simulation and analyze it in memory

`dice.dice_runner()` runs the simulation of a parameters file and returns the results. It does not write output files.

```python
import dice

result = dice.dice_runner("data/examples/quick_test.txt")
analysis = dice.analyze_simulation_results(result, proximity_levels=[0.05, 0.15])

wls = analysis["accuracy"]["wls"]
print(f"Mean estimate: {wls['mean']:.5f}")
print(f"Standard deviation: {wls['std']:.5f}")
print(f"Within ±15%: {analysis['precision']['15%']['wls']:.1%}")
```

`analysis["accuracy"]` contains the mean, median, standard deviation, bias, relative bias, RMSE, and relative RMSE of the estimates for each fit (`"wls"` and `"ols"`). `analysis["precision"]` contains the fraction of estimates within each proximity level. The estimates are in the simulation units.

For a simulation with a noise range, `analysis["cnr_bins"]` lists the CNR bins of the summary file, from the lowest CNR to the highest. Each bin has its lowest and highest initial CNR (`"cnr_min"`, `"cnr_max"`), its number of runs (`"num_runs"`), and `"precision"`, `"accuracy"`, and `"statistics"` entries for its runs, in the same form as those of the whole simulation. `result.noise_values` holds the noise standard deviation of each run, in run order.

### Run the CLI from Python

`dice.cli.main.main()` takes the same arguments as the `dice` command, writes the same files, and returns the exit code (0 for success):

```python
from dice.cli.main import main

exit_code = main(["data/examples/quick_test.txt", "--quiet"])
```

### Analyze the results in Python

The results file contains everything needed to compute the fraction of estimates within any proximity level. The slope and the first nominal diffusion coefficient column are both in the units of the simulation, so their ratio does not depend on the units. `results_column` gives the full name of a column, with its unit, from the name without it:

```python
import pandas as pd

from dice.io.results import results_column

df = pd.read_csv("output/quick_test/quick_test.csv")
slope = df[results_column(df.columns, "weighted fit diffusion slope")]  # "... (2D) [μm²/ns]"
nominal = df[results_column(df.columns, "nominal diffusion coeff")]     # "... [μm²/ns]"
ratio = (slope / 2 / nominal).dropna()  # one dimension: MSD = 2Dt

print(ratio.describe())
print(f"Within ±15%: {ratio.between(0.85, 1.15).mean():.1%}")
```

`dropna()` removes the runs in which the fit failed. In a Jupyter notebook, `ratio.plot.hist(bins=35)` draws a histogram of the ratio.

### Check a parameters file

`open_parameters()` reads and parses a parameters file and raises `ValueError` with a description if the file is invalid. `validate_parameters()` checks that the number of runs, the spatial width, and the number of pixels are positive, that the global length and time units are known, and that `image x_lim`, if given, is two increasing numbers:

```python
from dice.io.parameters import open_parameters, validate_parameters

params = open_parameters("data/examples/parameters.txt")
validate_parameters(params)
print("Parameters are valid")
```

## Batch processing

### Several parameters files

```bash
for params in params_*.txt; do
    dice "$params" --quiet
done
```

### Parameter scan

To vary one parameter, write one parameters file for each value and run each file. The following example varies the noise level of `quick_test.txt`:

```python
import ast
from pathlib import Path

from dice.cli.main import main

base = ast.literal_eval(Path("data/examples/quick_test.txt").read_text())

for noise in [0.01, 0.02, 0.05]:
    params = dict(base)
    params["noise value"] = noise
    params["filename slug"] = f"scan_noise_{noise}"
    path = Path(f"params_noise_{noise}.txt")
    path.write_text(repr(params))
    main([str(path), "--quiet"])
```

Each value writes its results to `output/scan_noise_<value>/`.

To study the effect of the noise level, a noise range does the same in one simulation: each run draws its own noise level, and the summary reports the statistics for each bin of initial CNR (see [`noise range, reciprocal` and `noise range, reciprocal log`](parameters.md#noise-range-reciprocal-and-noise-range-reciprocal-log)).

## Performance

- Parallel processing is on by default and uses all CPU cores. Turn it off with `'multiprocessing': False` in the parameters file or with `--multiprocessing 0`. To leave some cores free for other work, set the number of worker processes, for example `--multiprocessing 4`.
- The run time grows in proportion to the number of runs, and with the number of pixels and time points. Use `data/examples/quick_test.txt` (100 runs) to check a setup before a long simulation.
- Keep `--retain-profiles` off for large simulations, because retained profiles use much memory.

## Troubleshooting

When a simulation cannot start or fails, DICE prints an error message and exits with a non-zero code.

| Message | Cause and solution |
| ------- | ------------- |
| `Parameters file not found: <file>` | The path is wrong. Check the path, or give an absolute path. |
| `Error evaluating the file's contents as a dictionary` | The file is not a valid Python dictionary. Check for missing commas, quotes, or braces. |
| `The required parameter '<name>' ... is missing` | Add the parameter (see the [Parameter Reference](parameters.md)). |
| `More than one parameter provided for ...` or `No parameter provided for ...` | Give exactly one option from the group, for example `FWHM_0` or `sigma_0`. |
| `Provide either 'nominal diffusion length' or both ...` | Give either the diffusion length or both the diffusion coefficient and the lifetime. |

Add `--verbose` to print the full traceback of an error.

## See also

- [Parameter Reference](parameters.md)
- [Installation Guide](installation.md)
- [GUI Guide](../dice_gui/README.md)
- [Glossary](glossary.md)
