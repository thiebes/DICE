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
| `--quiet`, `-q` | Print errors only. |
| `--output-dir DIR`, `-o DIR` | Write the output files to `DIR` instead of `output/<slug>/`. |
| `--no-plots` | Do not create the histogram. |
| `--multiprocessing N` | `0` runs the simulations one after another. A positive `N` runs them in parallel in `N` worker processes, and `-1` uses all CPU cores. Overrides the `multiprocessing` parameter. |
| `--seed N` | Use the random seed `N` (a non-negative integer) for the noise. See [Reproducible results](#reproducible-results). |
| `--export-format {csv,json,both}` | `csv` (default) writes the results file `<slug>.csv`. `json` writes `<slug>.json` with the parameters and the summary statistics, but no per-run results. `both` writes both files. |
| `--progress-interval N` | With `--verbose`, the number of runs between progress messages (default 100). |
| `--retain-profiles` | Keep the profiles of every run in memory. The output files do not include them. |
| `--dry-run`, `--check` | Load and validate the parameters file as a real run would, then stop without a simulation and without writing files. |
| `--version` | Print the version and stop. |
| `--help`, `-h` | Print the list of options and stop. |

### Reproducible results

With `--seed N`, repeated runs with the same parameters and the same seed give identical results. The serial and the parallel calculation also give the same results as each other. Without `--seed`, each run draws new random numbers, so the results differ slightly from one run to the next.

```bash
dice data/examples/quick_test.txt --seed 12345
```

Results are reproducible for the same versions of DICE and its dependencies on the same platform. Identical numbers across platforms or across dependency versions are not guaranteed. The GUI has no seed option.

## Output files

### Results file: `<slug>.csv`

One row for each run. The main columns are:

| Columns | Content |
| ------- | ------------- |
| `run number` | Index of the run |
| `nominal diffusion coeff`, `nominal lifetime`, `nominal diffusion length` | Nominal values, in the simulation units |
| `nominal CNR` | `amplitude_0` / noise standard deviation |
| `estimated CNR` | CNR of the noisy profile at the first time point, estimated with the [fft-cnr](https://pypi.org/project/fft-cnr/) package |
| `nominal sigma^2_0`, `estimated sigma^2_0` | Variance of the initial profile: nominal value, and fitted value at the first time point |
| `weighted fit diffusion slope`, `... slope stderr`, `... intercept`, `... intercept stderr` | WLS fit of variance against time, in the simulation units |
| `unweighted fit ...` | The same quantities for the OLS fit |
| `weighted fit diffusion coeff [cm²/s]`, `weighted fit diffusion stderr [cm²/s]` | WLS estimate of the diffusion coefficient and its standard error, in cm²/s |
| `unweighted fit diffusion coeff [cm²/s]`, `unweighted fit diffusion stderr [cm²/s]` | The same for the OLS fit |
| `nominal diffusion coeff [cm²/s]` | Nominal diffusion coefficient in cm²/s |

WLS stands for weighted least squares, OLS for ordinary least squares. In one dimension the slope of the variance against time equals 2D. If a fit fails in a run, the columns of that fit are empty in that row. The file is encoded in UTF-8.

### Summary file: `<slug>_summary.txt`

A text report that contains:

- the main simulation parameters: number of runs, spatial axis, time points, noise level, initial CNR, initial variance, and the nominal diffusion length, diffusion coefficient, and lifetime
- for the WLS and the OLS fits: the standard deviation, relative standard error, mean, bias, and root-mean-square error (RMSE) of the diffusion coefficient estimates
- the fraction of estimates within ±5%, ±10%, ±20%, and ±50% of the nominal value

### Histogram: `<slug>_accuracy_histogram.png`

A histogram of the ratio D_est/D_nom of the WLS estimates to the nominal diffusion coefficient. The y-axis shows the probability of each bin. A normal distribution with the same mean and standard deviation is drawn over the histogram, and a legend reports the mean, median, and standard deviation of the ratio. The file type follows the `image type` parameter.

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

### Run the CLI from Python

`dice.cli.main.main()` takes the same arguments as the `dice` command, writes the same files, and returns the exit code (0 for success):

```python
from dice.cli.main import main

exit_code = main(["data/examples/quick_test.txt", "--quiet"])
```

### Analyze the results in Python

The results file contains everything needed to compute the fraction of estimates within any proximity level. The slope and the nominal diffusion coefficient are both in the simulation units, so their ratio does not depend on the units:

```python
import pandas as pd

df = pd.read_csv("output/quick_test/quick_test.csv")
d_est = df["weighted fit diffusion slope"] / 2  # one dimension: MSD = 2Dt
ratio = (d_est / df["nominal diffusion coeff"]).dropna()

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
