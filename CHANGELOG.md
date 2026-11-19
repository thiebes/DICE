# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - Unreleased

2.0.0 is a major release because the Python interface changes in ways that
break existing code: classes and fields are renamed, and
`dice.core.noise.fft_cnr` is removed (see Changed and Removed). Several
reported numbers also change; read Changed and Fixed before comparing
results with 1.2.0.

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
- Python interface: an estimator registry (`dice.models.estimators`) with the
  ordinary and weighted least-squares fits as its entries.
  `RunResult.slope_fit()`, `RunResult.accuracy()`, and
  `MonteCarloOutput.d_estimates()` address an estimator by its key (`'ols'`
  or `'wls'`).
- `CHANGELOG.md`, contributor guidelines, a code of conduct, issue templates,
  and user guides in `docs/` (installation, parameters, command-line
  interface, glossary).

### Changed

- Behavior change: `proximity level` is no longer a simulation parameter. A
  `proximity level` key in a parameters file triggers a warning and is
  otherwise ignored. DICE reports the standard deviation, relative standard
  error, mean, bias, and root-mean-square error (RMSE) of the diffusion
  coefficient estimates, and the fraction of
  estimates within ±5%, ±10%, ±20%, and ±50% of the nominal value. Choose any
  other proximity level after the simulation, in the GUI or from the results
  file.
- Behavior change: CNR estimation uses the
  [fft-cnr](https://pypi.org/project/fft-cnr/) package, version 0.4.1 or
  later, in place of the built-in implementation. The change applies to the
  `estimated CNR` column of the results file, to `estimate noise from data`,
  and to the web app. On simulated initial profiles from the four example
  files (CNR 50, 4000 profiles each), the estimates are 0.5% to 2.3% lower
  than in 1.2.0 and closer to the nominal CNR: 0.3% to 1.0% high, against
  1.1% to 2.7% high in 1.2.0. At CNR 1 to 3 they are 17% to 39% lower: 10%
  to 55% high, against 34% to 160% high in 1.2.0. For a peak only a few
  pixels wide, fft-cnr overestimates the noise, so the CNR comes out too low.
  The measured profile `data/examples/example_profile.csv`, whose peak is
  4.6 pixels wide at half maximum, gives a CNR of 8.9, where 1.2.0 gave 176.
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

- Results in length or time units other than micrometer and nanosecond are
  converted and labeled correctly. In 1.2.0 the command-line interface wrote
  the results file and the summary as if the units were micrometer and
  nanosecond: for lengths in nanometers, the diffusion coefficients in cm²/s
  were 10^6 times too large, and the summary labeled lengths in micrometers.
  Results from 1.2.0 in micrometers and nanoseconds are not affected, and
  releases before 1.2.0 converted correctly.
- The nominal CNR in the results file and the initial CNR in the summary are
  `amplitude_0` divided by the noise standard deviation. They were the
  reciprocal of the noise standard deviation, which is correct only for
  `amplitude_0` = 1.
- `estimate noise from data` sets the noise so that the initial CNR of the
  simulated profiles equals the CNR measured from the profile, for any
  `amplitude_0`. The noise was the reciprocal of the measured CNR, which is
  correct only for `amplitude_0` = 1.
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
- `--dry-run` loads and validates the parameters file and reports the first
  problem it finds, without running the simulation. It reported success
  without reading the file.
- Summary and other text output files are written as UTF-8 on every
  platform. On Windows they used the system code page, so the ± sign was not
  valid UTF-8.
- Proximity levels are labeled by their exact percentage. They were cut down
  to whole percentages, so 0.29 was labeled 28%, and two levels such as 0.565
  and 0.57 shared one label, so one result replaced the other.
- `CITATION.cff` declares the MIT License, in agreement with `LICENSE.md`.

### Removed

- `setup.py`, `requirements.txt`, `requirements-dev.txt`, and
  `webapp/requirements.txt`. The dependencies are declared in
  `pyproject.toml`; the `webapp`, `test`, and `dev` extras add the optional
  ones.
- The built-in CNR implementation based on the fast Fourier transform (FFT),
  `dice.core.noise.fft_cnr`. Import `fft_cnr` from the `fft_cnr` package. It
  returns a result object; the CNR estimate is its `cnr` attribute.

Changes before 2.0.0 are recorded in the Git history.

[2.0.0]: https://github.com/thiebes/DICE/compare/v1.2.0...v2.0.0
