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
  list of the shortcuts under Help. Stop ends a serial run after the current
  run and a parallel run after the current batch of runs. Closing the window
  during a run asks first, then waits in the same way. If the output files
  cannot be written, for example because a results file is open in another
  program, the GUI keeps the results and offers to write the files again. The
  GUI saves
  results in `output/<slug>/` in the folder it was started from, or in
  `DICE/output/<slug>/` in the user's Documents folder when it cannot write
  there, as when the macOS Finder starts the standalone application.
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
  Settings > Global Units sets the defaults. A key that ends in `_unit` but
  belongs to no parameter, such as a misspelled one, is rejected with the
  closest key named.
- Time range of the simulation in the summary file.
- Example files in `data/examples/`: `quick_test.txt` (100 runs),
  `publication.txt` (1000 runs), `high_precision.txt` (10000 runs), and
  `noise_range.txt` (100 runs over a range of CNR).
- For a noise-range simulation, a scatter plot of the estimated over the
  nominal diffusion coefficient against the initial CNR, in place of the
  accuracy histogram, and summary statistics in bins of initial CNR with
  equal numbers of runs: one bin per 20 runs, at most five.
- License notice of the standalone applications. They contain PyQt6, which
  is under the GNU General Public License (GPL) version 3, so they are
  distributed under the GPL version 3; the source code of DICE stays under
  the MIT License. Each release has a `-licenses.txt` file for each
  application, with the license text of every package it contains, and the
  application shows the same notice under Help > Licenses.
- The GUI warns when it opens a parameters file with a noise range, which the
  GUI cannot edit, instead of silently keeping its own noise setting.
- Python interface: `run_monte_carlo_simulation()` takes `noise_range`, a
  `NoiseRange` (`dice.models.parameters`), and then gives each run its own
  noise level; `MonteCarloOutput.noise_range` records it.
- Standalone GUI applications for Windows (x86_64), macOS (Apple silicon
  and Intel), and Linux (x86_64), built with PyInstaller when a GitHub
  release is published, with SHA-256 checksums in `SHA256SUMS.txt`. They need
  no Python installation and run simulations in parallel on all CPU cores.
  `docs/installation.md` describes how to download, check, and start them.
  The release workflow tests each packaged application with `--self-test`,
  which runs a seeded simulation serially and in parallel and requires the
  results to agree to within floating-point rounding.
- Python interface: `run_monte_carlo_simulation()` takes `parallel_backend`,
  `'joblib'` or `'process_pool'` (the standard library's
  `ProcessPoolExecutor`). Both give the serial results for the same seed. The
  default is joblib, or the process pool in a standalone application.
- A warning when the measured profile for `estimate noise from data` has a
  peak narrower than about 9 pixels at half maximum, where fft-cnr reads the
  CNR too low (14% to 27% low for a peak 4.6 pixels wide in simulated
  profiles). The command-line interface prints it, and the GUI shows it with
  the other warnings before a run. The example profile
  `data/examples/example_profile.csv` gives this warning.
- The parameter reference explains how to match a detector: for N pixels at
  pitch p, set `pixel width` to N and `spatial width` to p × (N − 1). A
  spatial width of N × p makes the pixel spacing N/(N − 1) times the pitch.
- Python interface: an estimator registry (`dice.models.estimators`) with the
  ordinary and weighted least-squares fits as its entries.
  `RunResult.slope_fit()`, `RunResult.accuracy()`, and
  `MonteCarloOutput.d_estimates()` address an estimator by its key (`'ols'`
  or `'wls'`).
- `CHANGELOG.md`, contributor guidelines, a code of conduct, issue templates,
  and user guides in `docs/` (installation, parameters, command-line
  interface, glossary).

### Changed

- Behavior change: `noise range, reciprocal` and `noise range, reciprocal log`
  draw a new noise standard deviation for each run, so one simulation covers
  a range of CNR (#20). They drew one value for the whole simulation. The
  results file of such a simulation records each run's noise standard
  deviation, and the summary reports the drawn noise and CNR ranges instead
  of a single value.
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
- Behavior change: `filename slug` must be a plain file name. A slug with a
  path separator, the slugs `.` and `..`, and a slug that ends with a dot or
  a space are rejected before the simulation runs. Such a slug used to write
  the output outside `output/`, or fail after the simulation had run.
- Parameters files are read as UTF-8, with or without a byte-order mark. A
  file that is not valid UTF-8 is read in the encoding of the locale, as
  before.
- Python interface: `export_collated_results()` requires `length_unit` and
  `time_unit`, the units the results are in. `write_summary_file()` takes
  them from its `parameters` argument when they are not given. Both used to
  assume micrometers and nanoseconds.
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
- Python API: `dice.dice_runner()` returns a `MonteCarloOutput`. `RunResult` holds the
  fit results as flat fields, and its `cnr_estimate` field is renamed
  `cnr_0_estimate`. The `GaussianParameters` fields `amplitude`, `sigma2`, and
  `mu` are renamed `amplitude_0`, `sigma2_0`, and `mu_0`. The ratio key of
  `plot_accuracy_histogram()` is renamed from `d_wls_over_d_nom` to
  `d_est_over_d_nom`.
- Python API: `load_time_series_profiles()` takes the first column as the
  time axis only when called with `time_column=True`. It guessed, and took any
  first column whose values increased as time, including the first pixel of a
  profile.
- Behavior change: a relative `estimate noise from data` path is relative to
  the folder of the parameters file, in the command-line interface and when
  the GUI opens the file. It was relative to the current working directory,
  so the same file failed or read a different profile when run from another
  directory.
- The `output_length_unit` and `output_time_unit` keys are rejected before
  the simulation. No output used them; lengths and times are reported in the
  global `length unit` and `time unit`.
- A negative diffusion coefficient, lifetime, or diffusion length is rejected
  when the parameters are loaded.

### Fixed

- The Gaussian fit of each frame gives the same estimates in any length
  unit and for any `amplitude_0`. The fit stopped on tolerances relative to
  its parameter values, so it ended early when the numbers were small: in
  meters the spread of the diffusion coefficient estimates was 28 times too
  large, and with `amplitude_0` = 1e-5 every estimate was 0. The error
  exceeds 1e-4 of the estimate when the spatial width is below about 0.05 in
  the chosen unit or `amplitude_0` is below about 0.01. Every release since
  1.0.0 fits the same way, so results from earlier releases with lengths in
  centimeters or meters, or with a small amplitude, are affected. Results in
  micrometers or nanometers with an amplitude near 1 change by less than
  1e-5 of the estimate.
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
  and seed give the same results, in serial and in parallel. The flag had no
  effect on the noise draws. The results repeat bit for bit on Linux, Windows,
  and Macs with Apple silicon. On Macs with Intel processors, the linear
  algebra library of NumPy and SciPy can change the last digit of a few fitted
  values from one run to the next (relative differences up to about 1e-15).
- `--multiprocessing N` sets the number of worker processes. Any value other
  than 0 used all CPU cores.
- Gaussian fits no longer fail because the starting point lies outside the
  parameter bounds. The initial center and amplitude come from the largest
  absolute value within the allowed range of the center, and the initial
  amplitude is positive.
- A lifetime of 0, which means no decay, gives an infinite diffusion length
  in the summary, the results file, the JSON file, and the GUI. The diffusion
  length was reported as 0.
- The summary file prints the unit after the standard deviation, mean, bias,
  and root-mean-square error of the estimates, and the GUI Analysis Settings
  tab prints it after the standard deviation and bias. These values had no
  unit, although they are in the units of the diffusion coefficient.
- The summary file shows values too small for its fixed decimals in
  scientific notation. In meters and seconds, for example, the diffusion
  coefficient printed as 0.00000.
- An `image type` that matplotlib cannot write, such as 'bmp', is rejected
  before the simulation with a list of the valid types. The simulation ran to
  the end, reported the histogram as saved, wrote no histogram, and exited
  without an error.
- The `output_diffusion_length_unit` and `output_diffusion_time_unit` keys
  set the unit of the diffusion coefficients in the results file and the
  summary file in the command-line interface. They were read and ignored. The
  GUI does not set them. Give both or neither; an unknown unit name is rejected
  before the simulation.
- `load_experimental_data()` reads a CSV whose first line is numbers as data.
  A single-line profile, the documented format, came back empty because its
  only line was taken as the header.
- `load_time_series_profiles()` reads a file with one row or one column. It
  failed with an index error.
- The command-line interface applies the `image font size`,
  `image tick length`, `image tick width`, `image numbins`, and `image x_lim`
  keys of the parameter file to the histogram, and it honors
  `retain profile data`. It ignored these keys. A malformed `image x_lim` is
  rejected before the simulation starts.
- `--dry-run` loads and validates the parameters file and reports the first
  problem it finds, without running the simulation. It reported success
  without reading the file.
- The command-line interface reports a plot as saved only when its file was
  written, and warns otherwise.
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
- Python API functions and classes that nothing in DICE used:
  `SimulationResults` and `StatisticalAnalysis` (use `MonteCarloOutput` and
  `analyze_simulation_results()`); `export_legacy_format()` and
  `export_to_json()` (use `export_collated_results()` or
  `dice --export-format json`); `estimates_precision()`, `precision_counts()`,
  and `analyze_cnr_dependence()` (use `calculate_precision()`);
  `scan_runner_compatibility()` and `convert_legacy_result_to_dict()`; and
  `plot_diffusion_msd()`.
- The built-in CNR implementation based on the fast Fourier transform (FFT),
  `dice.core.noise.fft_cnr`. Import `fft_cnr` from the `fft_cnr` package. It
  returns a result object; the CNR estimate is its `cnr` attribute.

Changes before 2.0.0 are recorded in the Git history.

[2.0.0]: https://github.com/thiebes/DICE/compare/v1.2.0...v2.0.0
