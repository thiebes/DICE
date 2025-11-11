# Diffusion Insight Computation Engine (DICE)

![DICE logo](logo/dice_logo_640w.png)

[Download the latest release on Zenodo](https://doi.org/10.5281/zenodo.10258191) | [GitHub repository](https://github.com/thiebes/DICE)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10258191.svg)](https://doi.org/10.5281/zenodo.10258191)

DICE is an open-source tool for researchers in time-resolved microscopy and related fields. It evaluates the precision and accuracy of diffusion coefficient estimates derived from optical measures of excited state transport. DICE provides a robust method for assessing experimental accuracy and precision by simulating parameters that mirror your experimental setup.

DICE was developed alongside research that the tool was used to perform, published in the following paper:

Joseph J. Thiebes, Erik M. Grumstrup; Quantifying noise effects in optical measures of excited state transport. J. Chem. Phys. 28 March 2024; 160 (12): 124201. [https://doi.org/10.1063/5.0190347](https://doi.org/10.1063/5.0190347).

## What Does DICE Do?

DICE quantifies the precision and accuracy of diffusion estimates by reporting the fraction of estimates found within a user-specified proximity to the nominal diffusion parameter. For example: "50% of the diffusion coefficient estimates are within ±10% of the nominal value."

Thus, DICE measures the likelihood that your estimated diffusion coefficient will attain the accuracy and precision required for your investigations based on your experimental parameters, such as the initial spot width, nominal diffusion length, and initial contrast-to-noise ratio.

### How it works

1. **Generation of time-series profiles:**  DICE starts by simulating a population of excited states as a Gaussian distribution profile that undergoes decay and diffusion. The initial profile has an amplitude of unity, a width parameterized by either the full-width half-maximum or the Gaussian standard deviation, a mean position of zero, and a baseline background of zero.
2. **Noise addition:** DICE incorporates white noise into the generated profiles to better resemble real-world scenarios. The magnitude of the noise is the same for every time frame and is parameterized by its standard deviation.
3. **Gaussian fit:** The time-evolved noisy profiles are fitted with Gaussian functions, estimating the Mean Squared Displacement (MSD) at each time point.
4. **Linear fit and diffusion coefficient estimation:** The MSD values are fitted to a linear function using a weighted least squares method. The slope of the linear fit is proportional to the estimated diffusion coefficient.
5. **Analysis:** DICE compares the estimated diffusion coefficient to the nominal parameter used to generate the series of profiles. By conducting multiple simulations with the same parameters, DICE provides a statistical view of the precision and accuracy of diffusion estimates.
6. **Presentation:** Several customizable plotting functions are provided to present the results.

## Quick Start

### Option 1: GUI (Recommended for New Users)

The graphical interface provides the easiest way to get started with DICE.

1. **Install uv** (if not already installed):

   ```bash
   # macOS/Linux:
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows:
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Clone and run:**

   ```bash
   git clone https://github.com/thiebes/DICE.git
   cd DICE
   uv sync
   uv run dice-gui
   ```

The GUI provides an intuitive interface with tab-based parameter organization, real-time validation, unit-aware input fields, and live calculations.

See [GUI Guide](dice_gui/README.md) for detailed documentation.

### Option 2: Command Line

For batch processing and programmatic control, use the command-line interface.

1. **Install uv and clone** (same as above)

2. **Run with parameters file:**

   ```bash
   uv run dice parameters.txt
   ```

See [CLI Guide](docs/cli-guide.md) for detailed documentation and [Parameter Reference](docs/parameters.md) for all available parameters.

### Option 3: Online CNR Estimator

Use [this link](https://dice-thiebes.pythonanywhere.com) to access an online CNR estimator for your 1D noisy Gaussian profile. No Python installation necessary.

## Documentation

- **[Installation Guide](docs/installation.md)** - Detailed installation instructions for all methods
- **[Parameter Reference](docs/parameters.md)** - Complete parameter documentation
- **[CLI Guide](docs/cli-guide.md)** - Command-line usage and examples
- **[GUI Guide](dice_gui/README.md)** - Graphical interface documentation
- **[Glossary](docs/glossary.md)** - Technical terms and acronyms
- **[Contributing](CONTRIBUTING.md)** - How to contribute to DICE

## Installation

### Quick Install (Recommended)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
# OR
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Clone and install DICE
git clone https://github.com/thiebes/DICE.git
cd DICE
uv sync
```

### Other Installation Methods

- **Install with pip:** See [Installation Guide](docs/installation.md#method-2-install-dependencies-and-use-directly)
- **Editable install:** See [Installation Guide](docs/installation.md#method-3-install-as-editable-package)
- **Troubleshooting:** See [Installation Guide](docs/installation.md#troubleshooting)

## Outputs

DICE generates three main output files in `output/<filename_slug>/`:

- **Summary text file** (`<slug>_summary.txt`): Human-readable report with statistical analysis
- **Results CSV file** (`<slug>.csv`): Raw simulation data for further analysis
- **Histogram plot** (`<slug>_accuracy_histogram.png`): Visualization of diffusion coefficient estimates

All simulation parameters and settings are included in the summary text file for reference.

## Interpreting Results

The result of simulations based on a given set of parameters is a distribution of diffusion coefficient estimates. By analyzing this distribution, users can assess whether a given experimental measurement with the same parameters is representative of the range of possible measurements that might be taken if the experiment were performed repeatedly.

When the parameters are input, the user chooses a diffusion coefficient and lifetime or a diffusion length. The "true" diffusion coefficient is inexorably unknown, so the input parameter is a best guess or an initial measurement.

Importantly, the results of the simulation are not to be interpreted as a comment on whether the diffusion coefficient and other input parameters are correct. Rather, the results indicate a range of possible estimates that would arise in experiment if the input parameters are correct.

## Known Issues and Limitations

- The data format in an imported initial profile for estimation of CNR is limited to a single line of comma-separated values
- This program only evaluates Gaussian function fits for distribution profiles
- Profiles analyzed are 1-dimensional
- Non-Fickian diffusion (e.g., subdiffusion) is not considered
- Anisotropic diffusion is not considered
- Lifetime is modeled as single-exponential

## Acknowledgements

Thanks to Professor Erik M. Grumstrup, Skyler Hollinbeck, Sajia Afrin, and my spouse, Julia K. Thiebes, for their invaluable support and feedback.

This material is based upon work supported by the National Science Foundation under Grant No. 2154448. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.

## How to Cite

```text
Thiebes, J. J. (2023). Diffusion Insight Computation Engine (DICE) [Software]. Zenodo. https://doi.org/10.5281/zenodo.10258191
```

## License

Diffusion Insight Computation Engine (DICE) by [Joseph J. Thiebes](http://thiebes.org) is licensed under the [MIT License](https://opensource.org/licenses/MIT).

Copyright (c) 2023-2025 Joseph J. Thiebes

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
