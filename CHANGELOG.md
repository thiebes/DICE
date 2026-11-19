# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - Unreleased

### Added

- Graphical user interface (GUI), started with `dice-gui` or
  `python -m dice_gui`. It has five tabs (Simulation Setup, Physical
  Parameters, Experimental Conditions, Analysis Settings, Output Settings) and
  runs the simulation in the background with a progress bar and an estimate
  of the remaining time. Inputs are checked as you type. Linked fields compute
  the diffusion length, diffusion coefficient, or lifetime from the other two,
  convert between the full width at half maximum (FWHM) and the standard
  deviation σ, and convert between the noise σ and the contrast-to-noise
  ratio (CNR). The GUI opens and saves parameters files in the format that
  the command-line interface reads, plots results files from earlier runs,
  and offers output presets for publication, presentation, and draft figures.
  It has light and dark themes, Alt-key shortcuts for the input fields, and a
  list of the shortcuts under Help.
- GUI warnings before a run for settings that make the fits unreliable: a low
  CNR at the last time point, an initial profile that is narrow compared with
  the pixel size, or very few time points.
- Post-hoc accuracy analysis in the GUI Analysis Settings tab. After a run,
  the tab shows the standard deviation, relative standard error, and bias of
  the estimates, and the fraction of estimates within any proximity level
  from 1% to 50% of the nominal value, for the weighted least squares (WLS)
  or the ordinary least squares (OLS) fit.
- Per-parameter units. In a parameters file, a key that ends in `_unit` (for
  example `FWHM_0_unit` or `lifetime_unit`) gives the unit of one value, and
  DICE converts it to the global `length unit` and `time unit`. In the GUI,
  every input with a length or time dimension has its own unit selector, and
  Settings > Global Units sets the defaults.
- Time range of the simulation in the summary file.
- Example files in `data/examples/`: `quick_test.txt` (100 runs),
  `publication.txt` (1000 runs), and `high_precision.txt` (10000 runs).
- Standalone GUI executables for Windows, macOS, and Linux, built with
  PyInstaller when a GitHub release is published. The standalone executables
  run simulations on one CPU core; parallel processing requires an
  installation with Python.
- `CHANGELOG.md`, contributor guidelines, a code of conduct, issue templates,
  and user guides in `docs/` (installation, parameters, command-line
  interface, glossary).

### Changed

- Behavior change: `proximity level` is no longer a simulation parameter. A
  `proximity level` key in a parameters file has no effect. DICE reports the
  standard deviation, relative standard error, mean, bias, and root-mean-square
  error (RMSE) of the diffusion coefficient estimates, and the fraction of
  estimates within ±5%, ±10%, ±20%, and ±50% of the nominal value. Choose any
  other proximity level after the simulation, in the GUI or from the results
  file.
- Behavior change: CNR estimation uses the
  [fft-cnr](https://pypi.org/project/fft-cnr/) package, version 0.4.1 or
  later, in place of the built-in implementation. The change applies to the
  `estimated CNR` column of the results file, to `estimate noise from data`,
  and to the web app. The reported CNR values shift slightly.
- The summary file lists the dispersion statistics of the estimates first,
  followed by the fractions within the standard proximity levels.
- The accuracy histogram shows the probability of each bin on the y-axis,
  with the normal curve scaled to match, and draws no proximity limits.
- The diffusion coefficient columns of the results file are labeled
  `[cm²/s]` instead of `[cm^2/s]`. Scripts that select these columns by name
  need the new label.
- DICE requires Python 3.11 or later.
- Packaging uses `pyproject.toml`, and `uv.lock` records the tested
  dependency versions. Install with `uv sync`, or with `pip install .` from
  the repository directory.
- `parameters.txt` and `example_profile.csv` moved from the repository root
  to `data/examples/`.
- `import dice` no longer loads SciPy and Matplotlib. They load when a
  function that needs them is first used.
- Library functions report problems through Python warnings and logging
  instead of printing to the console.
- Parallel simulations with few runs start faster, because each batch holds
  at least one task per worker process.
- Python API: `SimulationResults` is renamed `ProcessedSimulationResult`, and
  `dice.dice_runner()` returns a `MonteCarloOutput`. `RunResult` holds the
  fit results as flat fields, and its `cnr_estimate` field is renamed
  `cnr_0_estimate`. The `GaussianParameters` fields `amplitude`, `sigma2`, and
  `mu` are renamed `amplitude_0`, `sigma2_0`, and `mu_0`. The ratio key of
  `plot_accuracy_histogram()` is renamed from `d_wls_over_d_nom` to
  `d_est_over_d_nom`.

### Fixed

- `--seed N` makes runs reproducible: repeated runs with the same parameters
  and seed give identical results, in serial and in parallel. The flag had no
  effect on the noise draws.
- `--multiprocessing N` sets the number of worker processes. Any value other
  than 0 used all CPU cores.
- Gaussian fits no longer fail because the starting point lies outside the
  parameter bounds. The initial center and amplitude come from the largest
  absolute value within the allowed range of the center, and the initial
  amplitude is positive.
- `ProcessedSimulationResult.filter_by_cnr()` recomputes the statistics for
  the selected runs. It returned the statistics of all runs.
- The command-line interface applies the `image font size`,
  `image tick length`, `image tick width`, `image numbins`, and `image x_lim`
  keys of the parameter file to the histogram, and it honors
  `retain profile data`. It ignored these keys. A malformed `image x_lim` is
  rejected before the simulation starts.
- `CITATION.cff` declares the MIT License, in agreement with `LICENSE.md`.

### Removed

- `setup.py`, `requirements.txt`, `requirements-dev.txt`, and
  `webapp/requirements.txt`. The dependencies are declared in
  `pyproject.toml`; the `webapp`, `test`, and `dev` extras add the optional
  ones.
- The built-in CNR implementation based on the fast Fourier transform (FFT),
  `dice.core.noise.fft_cnr`. Import `fft_cnr` from the `fft_cnr` package. It
  returns a result object; the CNR estimate is its `cnr` attribute.

Changes before 1.3.0 are recorded in the Git history.

[1.3.0]: https://github.com/thiebes/DICE/compare/v1.2.0...v1.3.0
