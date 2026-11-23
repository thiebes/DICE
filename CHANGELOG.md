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

### Results from 1.2.0: what to rerun

Find the row that describes your 1.2.0 simulation. The entries under Changed
and Fixed give the details.

| Your 1.2.0 simulation | What was wrong | What to do |
| ------- | ------------- | ------------- |
| Micrometers and nanoseconds, `amplitude_0` = 1, a fixed `noise value`, `mean_0` within 0.2 × `spatial width` of 0, and a CNR above about 4 in every frame | Nothing that changes the diffusion coefficient statistics: the mean and the standard deviation of the estimates agree to four digits. Single estimates change a little, because the fit runs in scaled coordinates and starts its search for the center at the center of the window. With a diffusion length of 0.1 × `FWHM_0`, a window 5 FWHM wide, 100 pixels and 10 frames, they change by less than 1e-4 of their value at an initial CNR of 50; at an initial CNR of 20, one estimate in ten changes by more than 1e-5 and none by more than 1%. The `estimated CNR` column changes with the new CNR estimator. | No rerun needed. |
| A frame with a CNR below about 4. The CNR falls from frame to frame with the decay and the spreading of the profile. | The Gaussian fit of a noisy frame could start outside its bounds and fail. 1.2.0 printed a warning and used (`spatial width` / 4)² as the variance of that frame, which put an outlier in the estimates of that run. In a window 5 FWHM wide with 100 pixels, 0.4% of the frames failed at a CNR of 3 and 6% at a CNR of 2. | Rerun. |
| `mean_0` farther than 0.2 × `spatial width` from 0, the center of the window | The Gaussian fit looks for the center of a profile only in the central 40% of the window. It held the center at that limit and reported a wrong width, so the diffusion coefficient estimates were wrong (10% low for `mean_0` = 1.3 in a window 5 wide, at low noise). | Rerun with `mean_0` inside the central 40% of the window. 2.0.0 rejects the old value. |
| Another length or time unit | The diffusion coefficients in cm²/s were too large by the factor (1 micrometer / length unit)² × (time unit / 1 nanosecond): 10^6 for nanometers and nanoseconds, 10^-3 for micrometers and picoseconds. The summary labeled lengths in micrometers. Ratios of estimate to nominal value were correct. | Divide the cm²/s columns by the factor, or rerun. |
| Lengths in centimeters or meters, a spatial width below about 0.05 in its unit, or `amplitude_0` below about 0.01 | The Gaussian fit stopped early, so the estimates themselves were wrong. | Rerun. |
| `estimate noise from data` | The noise level came from another CNR estimator, and it was wrong for `amplitude_0` other than 1. For the example profile, whose peak is 4.6 pixels wide, 1.2.0 read a CNR of 176, where the residuals of a Gaussian fit give 59. It simulated too little noise and reported a precision that was too good. | Rerun with a profile whose peak is at least 9 pixels wide, or set `noise value` from your own CNR estimate. For a narrower peak, 2.0.0 reads too low (8.9 for the example profile) and warns. |
| `amplitude_0` other than 1, with a fixed `noise value` | The nominal CNR in the results file and the summary was 1 / noise, not `amplitude_0` / noise. | Recompute the CNR; rerun only if you chose the noise from that CNR. |
| A noise-range key | One noise level was drawn for the whole simulation. | The old result is a valid simulation at that one level. Rerun to cover the range. |

### Added

- Graphical user interface (GUI), started with `dice-gui` or
  `python -m dice_gui`. It has five tabs (Simulation Setup, Physical
  Parameters, Experimental Conditions, Analysis, Output) and
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
  run and a parallel run within a few seconds. Closing the window
  during a run asks first; the window then stays, with a message, until the
  run has stopped in the same way, and closes. If the output files cannot be
  written, for example because a results file is open in another program,
  the GUI keeps the results and offers to write the files again, at once or
  later with File > Write Output Files, and asks before a new simulation,
  a loaded results file, or the exit discards them. A value of a parameters file that a field of the form cannot hold is
  reported; the form is not changed. The
  GUI saves
  results in `output/<slug>/` in the folder it was started from, or in
  `DICE/output/<slug>/` in the user's Documents folder when it cannot write
  there, as when the macOS Finder starts the standalone application.
- GUI use with the keyboard and a screen reader (#25). Every input and every
  unit selector has an accessible name. The Alt-key shortcuts of the fields
  do not use the letters of the menus. The Tab key leaves the time-series
  box and does not stop on the scroll areas. Ctrl+. stops a run; the focus
  moves to Stop when a run starts and back to Run when it ends. When a run
  is blocked by an invalid field, the message names the field, and the GUI
  opens its tab and gives it the focus.
- The GUI compares the start and the stop of a time range in one unit. A
  start of 500 ps with a stop of 1 ns was refused.
- GUI warnings before a run for settings that make the fits unreliable: a low
  CNR at the last time point (the peak falls by the decay and by the
  spreading of the profile), an initial profile that is narrow compared with
  the pixel size, or very few time points.
- Post-hoc accuracy analysis in the GUI Analysis tab. After a run,
  the tab shows the standard deviation, relative standard deviation, and bias of
  the estimates, and the fraction of estimates within any proximity level
  from 1% to 50% of the nominal value, for the weighted least squares (WLS)
  or the ordinary least squares (OLS) fit. A histogram of the estimates
  highlights the part within the chosen level: each bar from its foot up to
  the share of its estimates within the level, so the highlighted area is
  the fraction.
- Per-parameter units. In a parameters file, a key that ends in `_unit` (for
  example `FWHM_0_unit` or `lifetime_unit`) gives the unit of one value, and
  DICE converts it to the global `length unit` and `time unit`. In the GUI,
  every input with a length or time dimension has its own unit selector, and
  Settings > Global Units sets the defaults. A key that ends in `_unit` but
  belongs to no parameter, such as a misspelled one, is rejected with the
  closest key named.
- Time range of the simulation in the summary file.
- Uncertainty of the reported statistics. The summary file gives, for each
  fit, the number of runs with an estimate, the standard error of the mean
  and of the bias, and each fraction within a proximity level with its count
  and its 95% interval, for example `19.0% (19 of 100; 12.5% to 27.8%)`. The
  GUI Analysis tab shows the same for the bias and the fraction.
  The analysis dictionary and the JSON file carry them as `standard_error`
  and `precision_intervals`.
- The summary file gives the ratio of the mean estimate to the nominal value
  ("Mean estimate / nominal"), the form in which the 2024 paper reports the
  bias. "Bias" in DICE is the difference, mean minus nominal, with its
  standard error.
- The summary file and the JSON file record the DICE version and the seed of
  the simulation. A simulation in which no run gives an estimate says so in
  the summary, on the console and in the GUI, in place of "completed
  successfully".
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
- Behavior change: the standard deviation of the estimates is the sample
  standard deviation (divisor n − 1), in the summary, the JSON file, the
  plots and the GUI. It was the population form (divisor n). The reported
  standard deviation and relative standard deviation are larger by the factor
  √(n / (n − 1)): 0.5% for 100 runs, 0.05% for 1000. The standard error of
  the mean is this standard deviation divided by √n.
- Behavior change: `proximity level` is no longer a simulation parameter. A
  `proximity level` key in a parameters file triggers a warning and is
  otherwise ignored. DICE reports the standard deviation, relative standard
  deviation, mean, bias, and root-mean-square error (RMSE) of the diffusion
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
  The residuals of a Gaussian fit to that profile give a CNR of 59, so
  neither estimator is close for a peak that narrow: 1.2.0 read it three
  times too high, and 2.0.0 reads it more than six times too low and warns.
  The percentages are one-time measurements made for this release on
  simulated profiles; the scripts are not part of the repository. The test
  suite checks only that the summary reports the CNR that fft-cnr gives for
  the example profile.
- Behavior change: `mean_0` must lie in the central 40% of the spatial
  window, which is where the Gaussian fit looks for the center of a profile.
  A profile centered outside it was fitted with its center held at the
  limit: the fit reported a wrong width without failing, and the diffusion
  coefficient estimates were wrong (10% low for `mean_0` = 1.3 in a window 5
  wide, at low noise). Such a value is rejected before the simulation.
- Behavior change: `filename slug` must be a plain file name. A slug with a
  path separator, a tab or a line break, the slugs `.` and `..`, a slug that
  ends with a dot or a space, and a name that Windows reserves for a device
  (`NUL`, `CON`, `COM1`, ...) are rejected before the simulation runs, and so
  is a slug of more than 64 characters. Such a
  slug used to write the output outside `output/`, or fail after the
  simulation had run.
- Behavior change: a parameters file that cannot run is rejected when it is
  read, with a message that names its key, so `--dry-run` finds it. The
  checks: `number of runs` and `pixel width` are whole numbers (50.0 stands
  for 50); `pixel width` is 16 or more, the number of points the CNR
  estimate needs; `time range` has a whole number of steps, 2 or more, and
  a start below its stop; `time series` has two or more time points and
  repeats none; no time is below 0; `noise value` is 0 or greater; `FWHM_0` or `sigma_0` is
  greater than 0; the nominal diffusion coefficient is greater than 0;
  `image numbins` is a whole number, 1 or more. These files used to fail
  during the run or at the plot, or to run without an estimate. The command-line interface used to cut a `time
  range` of 5.5 steps to 5.
- Behavior change: a `time series` is sorted into ascending order, as the
  GUI sorts it. The command-line interface used to run the series in the
  order given.
- Behavior change: a unit key given in two spellings, for example
  `FWHM_0_unit` and `fwhm_0_unit`, is rejected. One of the two used to be
  ignored.
- `dice.dice_runner()` validates the parameters file as the command-line
  interface does.
- The measured profile of `estimate noise from data` is read as UTF-8, with
  or without a byte-order mark, and a file that cannot be read is reported
  with its path and the cause.
- Parameters files are read as UTF-8, with or without a byte-order mark. A
  file that is not valid UTF-8 is read in the encoding of the locale, as
  before.
- Behavior change: every column of the results file whose value has a unit
  names it in brackets, and the slope columns state that the slope is twice
  the diffusion coefficient: `nominal lifetime` is now `nominal lifetime
  [ns]`, and `weighted fit diffusion slope` is `weighted fit diffusion slope
  (2D) [μm²/ns]`, for a simulation in micrometers and nanoseconds. A script
  that reads columns by name must change. `dice.io.results.results_column`
  finds a column from its name without the unit, in a results file of this
  version and of 1.2.0. When the output unit of the diffusion coefficients
  is the unit of the simulation, the file has one `nominal diffusion coeff`
  column. The GUI still plots a results file of 1.2.0. The results file is
  written as UTF-8 with a byte-order mark, so that a spreadsheet program
  shows the units in the column names; with Python's `csv` module, open it
  with `encoding="utf-8-sig"`.
- The JSON file names the units of its values in `summary.units`.
- The summary of the command-line interface marks the lifetime as assumed when the parameters file gives
  only the diffusion length; it printed "1 nanosecond" as if it were given.
- Python interface: `export_collated_results()` requires `length_unit` and
  `time_unit`, the units the results are in. `write_summary_file()` takes
  them from its `parameters` argument when they are not given. Both used to
  assume micrometers and nanoseconds.
- The summary file lists the dispersion statistics of the estimates first,
  followed by the fractions within the standard proximity levels. It marks
  the ordinary least squares fit as given for comparison; the weighted fit
  is the default and the one the plots show.
- The accuracy histogram shows the probability of each bin on the y-axis,
  with the normal curve scaled to match, and draws no proximity limits. A
  dashed line marks the nominal value, the legend names the normal curve,
  and a note counts the estimates beyond the limits of the axis, which the
  plot does not show. The scatter plot of a noise-range simulation has the
  same note. The fractions in the summary file have one decimal.
- The diffusion coefficient columns of the results file are labeled
  `[cm²/s]` instead of `[cm^2/s]`. Scripts that select these columns by name
  need the new label.
- DICE requires Python 3.11 or later, and joblib 1.3 or later.
- Packaging uses `pyproject.toml`, and `uv.lock` records the tested
  dependency versions. Install with `uv sync`, or with `pip install .` from
  the repository directory.
- `parameters.txt` and `example_profile.csv` moved from the repository root
  to `data/examples/`.
- `import dice` no longer loads SciPy and Matplotlib. They load when a
  function that needs them is first used.
- Library functions report problems through Python warnings and logging
  instead of printing to the console.
- A parallel simulation reports its progress while it runs: to the progress
  callback of `run_monte_carlo_simulation()`, with `--verbose` on the command
  line, and in the GUI. It reported nothing until it had finished.
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

- The Gaussian fit of each frame gives the same estimates in the length
  units and at the amplitudes that the tests cover: nanometers, millimeters,
  centimeters, and meters against micrometers, and `amplitude_0` from 1e-5 to
  1e3, on the geometry of the `quick_test.txt` example, with agreement to a
  relative 1e-6. The fit stopped on tolerances relative to
  its parameter values, so it ended early when the numbers were small: in
  meters the spread of the diffusion coefficient estimates was 28 times too
  large, and with `amplitude_0` = 1e-5 every estimate was 0. The error
  exceeds 1e-4 of the estimate when the spatial width is below about 0.05 in
  the chosen unit or `amplitude_0` is below about 0.01. Every release since
  1.0.0 fits the same way, so results from earlier releases with lengths in
  centimeters or meters, or with a small amplitude, are affected. For results
  in micrometers or nanometers with an amplitude near 1 the error was small;
  the first row of the table above says how much their single estimates
  change.
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
  effect on the noise draws. The tests require agreement to within
  floating-point rounding (a relative difference of 1e-12), for the same
  versions of DICE and its dependencies on the same platform. On Linux,
  Windows, and Macs with Apple silicon, the results have repeated bit for bit.
  On Macs with Intel processors, the linear algebra library of NumPy and
  SciPy can change the last digit of a few fitted values from one run to the
  next (relative differences up to about 1e-15).
- `--multiprocessing N` sets the number of worker processes. Any value other
  than 0 used all CPU cores.
- Gaussian fits no longer fail because the starting point lies outside the
  parameter bounds. The fit starts its search for the center at the center of
  the window and the amplitude starts at the largest absolute value of the
  profile, as Table II of the 2024 paper gives them. In 1.2.0 the search for
  the center started at the position of the largest absolute value of
  the whole profile, and the amplitude started at the signed value there.
  When noise put that position outside the allowed range, or made that value
  negative, the fit failed and (`spatial width` / 4)² stood in for the
  variance of the frame.
- A lifetime of 0, which means no decay, gives an infinite diffusion length
  in the summary, the results file, the JSON file, and the GUI. The diffusion
  length was reported as 0.
- The summary file prints the unit after the standard deviation, mean, bias,
  and root-mean-square error of the estimates, and the GUI Analysis
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
- Importing `dice.visualization.histograms` no longer switches Matplotlib to
  the Agg backend, so a Python session that imports DICE keeps its own. The
  command-line interface and the GUI select Agg when they plot
  (`use_file_backend`).
- The command-line interface prints a warning as `Warning: <message>`. It
  printed Python's default form, with the path and the line of the source
  file. The help of `--quiet` says that warnings are still printed; a
  simulation in which no run gives an estimate is reported as a warning.
- A run removes the accuracy plots that an earlier run with the same slug
  left in the output folder, in every image type, also with `--no-plots`.
  An earlier plot could stay beside results it did not show: after a run
  with another image type, after a change between a noise value and a noise
  range, and after a run without plots.
- A plot that cannot be saved is an error. The plot functions raise, and the
  command-line interface ends with exit code 1 and a message that the
  results were saved and the plot was not. The plot functions used to log
  the failure and return, so a caller could report a file that was never
  written.
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
