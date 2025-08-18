# Diffusion Insight Computation Engine (DICE)
![DICE logo](logo/dice_logo_640w.png)

[Download the latest release on Zenodo](https://doi.org/10.5281/zenodo.10258191) | [GitHub repository](https://github.com/thiebes/DICE) 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10258191.svg)](https://doi.org/10.5281/zenodo.10258191)

DICE is an open-source tool for researchers in time-resolved microscopy and related fields. It evaluates the precision and accuracy of diffusion coefficient estimates derived from optical measures of excited state transport. DICE provides a robust method for assessing experimental accuracy and precision by simulating parameters that mirror your experimental setup.

DICE was developed alongside research that the tool was used to perform, published in the following paper:

Joseph J. Thiebes, Erik M. Grumstrup; Quantifying noise effects in optical measures of excited state transport. J. Chem. Phys. 28 March 2024; 160 (12): 124201. [https://doi.org/10.1063/5.0190347](https://doi.org/10.1063/5.0190347).

# Introduction

DICE's primary advantage lies in its ability to quantify the precision and accuracy of diffusion estimates. This is accomplished by reporting the fraction of the diffusion estimates found within a user-specified proximity to the nominal diffusion parameter. For example: "50% of the diffusion coefficient estimates are within ±10% of the nominal value."

Thus, DICE measures the likelihood that your estimated diffusion coefficient will attain the accuracy and precision required for your investigations based on your experimental parameters, such as the initial spot width, nominal diffusion length, and initial contrast-to-noise ratio.

DICE aims to support ongoing improvements in the reliability and reproducibility of diffusion coefficient estimates derived from time-resolved microscopy methods. If you enjoy this program and found it helpful, please share it.

## How it works

1. **Generation of time-series profiles:**  DICE starts by simulating a population of excited states as a Gaussian distribution profile that undergoes decay and diffusion. The *initial profile* has an amplitude of unity, a width parameterized by either the full-width half-maximum or the Gaussian standard deviation, a mean position of zero, and a baseline background of zero.
2. **Noise addition:** DICE incorporates white noise into the generated profiles to better resemble real-world scenarios. The magnitude of the noise is the same for every time frame and is parameterized by its standard deviation.
3. **Gaussian fit:** The time-evolved noisy profiles are fitted with Gaussian functions, estimating the Mean Squared Displacement (*MSD*) at each time point.
4. **Linear fit and diffusion coefficient estimation:** The *MSD* values are fitted to a linear function using a weighted least squares method. The slope of the linear fit is proportional to the estimated diffusion coefficient.
5. **Analysis:** DICE compares the estimated diffusion coefficient to the nominal parameter used to generate the series of profiles. By conducting multiple simulations with the same parameters, DICE provides a statistical view of the precision and accuracy of diffusion estimates.
6. **Presentation:** Several customizable plotting functions are provided to present the results. 

# Table of Contents

- [Quick start](#quick-start)
  - [Online CNR Estimator](#online-cnr-estimator)
  - [Full DICE Software](#full-dice-software)
- [Installation](#installation)
- [Parameter inputs](#parameter-inputs)
- [Functions](#functions)
- [Outputs](#outputs)
- [Interpreting Results](#interpreting-results)
- [Glossary](#glossary)
- [Known issues and limitations](#known-issues-and-limitations)
- [Contributions and feedback](#contributions-and-feedback)
- [How to cite](#how-to-cite)
- [License](license)

---

# Quick Start

## Online CNR Estimator

Use [this link](https://dice-thiebes.pythonanywhere.com) to access an online CNR estimator for your 1D noisy Gaussian profile. No Python installation necessary. 

## Full DICE software

Follow these steps to get started with DICE quickly:

0. **Install Python**:
   See [Installation] below.

1. **Download the Repository**: 
   Clone or download this repository to your local machine and navigate to the project directory in your Python environment.

2. **Edit Parameters**: 
   Modify the `parameters.txt` file to align with your experimental parameters. This file is the primary input for the simulation.

3. **Run the Simulation**: 
   DICE can be used in several ways depending on your installation method:
   
   - **Using the console command** (if installed with `pip install -e .`):
     ```bash
     dice parameters.txt
     ```

   - **From the command line using the script**:
     Navigate to the DICE directory and run:
     ```bash
     python run_dice.py parameters.txt
     ```

   - **From within a Python environment**:
     ```python
     from dice.cli.main import main
     import sys
     
     # Set the parameters file as command line argument
     sys.argv = ['dice', 'parameters.txt']
     main()
     ```
     
     Or use the analysis functions directly:
     ```python
     from dice.analysis.simulation import run_monte_carlo_simulation
     from dice.io.parameters import load_parameters
     
     # Load and run simulation
     parameters = load_parameters("parameters.txt")
     result = run_monte_carlo_simulation(parameters)
     ```

   *Note:* You can run the simulation immediately with the included `parameters.txt` file to see example results.

4. **View the Results**: 
   After the simulation completes, the results will be summarized and saved, including a histogram plot for visual analysis.

[Back to table of contents](table-of-contents)

# Installation

## Python Requirements
DICE requires Python 3.8 or later and has been tested with Python 3.11 and 3.12. You can install Python from the [official website](https://www.python.org/downloads/) or use package managers like conda. For online development, you can use services like [Google Colab](https://colab.research.google.com/) or [DataLore](https://datalore.jetbrains.com/).

## Installation Methods

### Method 1: Install Dependencies and Use Directly
1. **Clone or download** this repository to your local machine
2. **Navigate** to the DICE directory
3. **Install dependencies** using pip:
   ```bash
   pip install -r requirements.txt
   ```

### Method 2: Install as Editable Package
1. **Clone** this repository:
   ```bash
   git clone https://github.com/thiebes/DICE.git
   cd DICE
   ```
2. **Install** in development mode:
   ```bash
   pip install -e .
   ```
   This allows you to use the `dice` command from anywhere and imports the package.

## Dependencies
DICE depends on the following Python packages (minimum versions):
- **`numpy>=1.20.0`**: Fundamental package for scientific computing
- **`pandas>=1.3.0`**: High-performance data structures and analysis
- **`matplotlib>=3.3.0`**: Plotting and visualization
- **`seaborn>=0.11.0`**: Statistical data visualization
- **`scipy>=1.7.0`**: Scientific and technical computing
- **`statsmodels>=0.12.0`**: Statistical modeling and econometrics
- **`joblib>=1.0.0`**: Lightweight pipelining and parallel processing

**Standard Python Libraries:** The following are part of the Python Standard Library and do not need to be installed separately:
- **`ast`**: For working with abstract syntax trees.
- **`os`**: Provides a way of using operating system-dependent functionality.
- **`re`**: Provides regular expression matching operations.
- **`typing`**: Used to support type hints (available in Python 3.5 and later).

Remember to regularly update your packages to their latest versions using pip to ensure the smooth functioning of DICE. If you encounter issues during installation, please **[contact the author](http://thiebes.org/contact)** or raise an issue on GitHub.

[Back to table of contents](table-of-contents)

# Parameter inputs
Edit the `parameters.txt` file to set up your simulation. Each simulation models one experimental diffusion measurement, comprising a series of time-evolved Gaussian profiles with noise. Each simulation uses one set of parameters.

## Filename slug
A prefix for your output files to help identify them later. The output file names will also automatically include the parameters below, so your slug could be a textual identifier connected to a set of experiments. Avoid special characters in the slug. 

### Example:
If the Filename slug is set to `'example'`, simulation results are saved in the `output/example/` directory:
- `output/example/example.csv` - Simulation results data
- `output/example/example_summary.txt` - Human-readable summary
- `output/example/example_accuracy_histogram.png` - Visualization plot

All simulation parameters and settings are included in the summary text file for reference.

## Number of simulation iterations
Specify the number of simulations to run. 

## Units
Specify the singular units for length and time, like `'micrometer'` and `'nanosecond'`. These units apply to all parameters representing physical quantities of length and time. Ensure that all parameter magnitudes are appropriately scaled.

## Nominal diffusion and lifetime parameters
Provide either:
- Nominal diffusion coefficient $D$ and lifetime $\tau$, or
- Nominal diffusion length $\sqrt{D\tau}$

*Note*: Scale these parameters for the units parameterized above (or, in the case of the diffusion coefficient, the square of your length unit over your time unit)
*Examples*: 
- For a diffusion length of 100 nm, but the length unit is `'micrometer'`, use:
  - `'nominal diffusion length': 0.1,`
- For a nominal diffusion coefficient of 1.0 cm<sup>2</sup>/s, but your length unit parameter is `'micrometer'` and your time unit is `'nanosecond'`, scale to μm<sup>2</sup>/ns as follows:
  - `'nominal diffusion coefficient': 0.1,` and
  - `'nominal lifetime (tau)': 1,`
 
## Initial profile parameters
Configure the amplitude, mean, and width of the initial Gaussian profile. Note: Amplitudes other than unity have not been tested.
- The amplitude is usually unity, but it can be changed here if needed. Other amplitude values have not been tested, however.
- For profiles centered at the origin, the mean should be set to zero.
- The width may be given as sigma (the standard deviation) or FWHM (full-width, half-maximum).

## Noise value
Set the standard deviation of noise added to each profile pixel. You can provide:
- A single value for all runs, or
- An initial profile as a text file; the script will estimate the noise.

## Spatial axis parameters
Set the spatial width and pixel number for the scan. The spatial width should align with the length unit specified.

### Temporal axis parameters
Provide time-axis information using one of the following forms:
- `'time range'`: Format `[start, stop, steps]` for evenly timed frames.
- `'time series'`: Specific time values like `[0.1, 0.3, 0.5, 0.7, 0.9]`.

## Proximity level
This parameter determines the accuracy threshold to analyze. For instance, a 0.1 proximity level means the program will evaluate how many estimates are within $\pm 10$% of the nominal value.

In other words:
- The simulations will produce diffusion coefficient estimates, $D_{est}$. 
- The *accuracy* of each estimate is quantified by the relative proximity of the estimate to the nominal value, $D_{nom}$.
- The relative proximity is evaluated by taking the quotient of the estimate and the nominal value, *i.e.*, $D_{est} / D_{nom}$. 
- If the estimate is perfectly accurate, it will equal the nominal value, and the relative proximity will be 1. 
- The portion of estimates within the proximity level is reported.

*For example,* if you enter 0.1 for your precision level, the program will tell you what portion of all the estimates were within $\pm 10$% of the nominal value.

*Note:* Conventional statistical methods of characterizing precision and accuracy are to report the standard deviation and the mean, median, and mode, respectively. The portion of estimates that are within proximity to the nominal value, reported by this script, offers a single evaluation of the impact of both precision and accuracy on the results. In some ways, this figure may be more intuitive and practical in experimental settings.

## Plotting (image) parameters
- Choose the file type for the default plot image. Options include `'jpg'`, `'png'`, `'svg'`, `'tif'`.
- Choose the width and height of the overall plot image in cm.
- Specify resolution in DPI. 
- Indicate the font size, tick length, and tick width in points.
- Select the number of bins to use in your histogram.
- Set limits for the x-axis, e.g., [0,20]. If 'None' is chosen, the script will estimate 3 standard deviations of the data and use the negative and positive of that value to set the x-axis limits. 

## Data retention
Decide whether to retain all profile data. Note that keeping profile data may impact performance, depending on the number of profiles generated. 
- `True`: Keep all data (necessary for plotting individual profiles).
- `False`: Discard raw data (retain only the fitting results).

## Parallel processing
Enable parallel processing by setting this parameter to `True`.
 
[Back to table of contents](table-of-contents)

# Functions

The source code contains extensive documentation describing the functions and how they work. 

[Back to table of contents](table-of-contents)

# Outputs

DICE generates three main output files, organized in the `output/<filename_slug>/` directory:

- **Summary text file** (`<slug>_summary.txt`): Human-readable report with statistical analysis
- **Results CSV file** (`<slug>.csv`): Raw simulation data for further analysis  
- **Histogram plot** (`<slug>_accuracy_histogram.png`): Visualization of diffusion coefficient estimates

All outputs are saved in `output/<filename_slug>/` to keep your workspace organized.

[Back to table of contents](table-of-contents)

# Interpreting Results

The result of simulations based on a given set of parameters is a distribution of diffusion coefficient estimates. By analyzing this distribution, users can assess whether a given experimental measurement with the same parameters is representative of the range of possible measurements that might be taken if the experiment were performed repeatedly. 

When the parameters are input, the user chooses a diffusion coefficient and lifetime or a diffusion length. The "true" diffusion coefficient is inexorably unknown, so the input parameter is a best guess or an initial measurement. 

Importantly, the results of the simulation are *not* to be interpreted as a comment on whether the diffusion coefficient and other input parameters are correct. Rather, the results indicate a range of possible estimates that would arise in experiment *if* the input parameters are correct. 

[Back to table of contents](table-of-contents)

# Glossary
Jargon terms and acronyms used in this documentation include:
- CNR: contrast-to-noise ratio, *i.e.,* the ratio of the signal amplitude to the standard deviation of the noise
- OLS: ordinary least-squares fitting algorithm
- WLS: weighted least-squares fitting algorithm

[Back to table of contents](table-of-contents)

# Known issues and limitations
- The data format in an imported initial profile for estimation of $CNR_0$ is limited to a single line of comma-separated values. Data that do not strictly follow this format will fail.
- This program only evaluates Gaussian function fits for distribution profiles. Lorentzian, Voigt, Green's function, *etc.* are not considered. 
- Profiles analyzed are 1-dimensional. 
- Non-Fickian diffusion (*e.g.*, subdiffusion) is not considered.
- Anisotropic diffusion is not considered.
- Lifetime is modeled as single-exponential.

[Back to table of contents](table-of-contents)

# Contributions and feedback

We welcome contributions and feedback from the community to improve DICE. If you're interested in contributing, here's how you can help:

### Reporting Bugs or Issues
- **Bug Reports**: If you encounter a bug or an issue while using DICE, please open an issue on our [GitHub Issues page](https://github.com/thiebes/DICE/issues). Include detailed information about the problem, steps to reproduce it, and any relevant logs or screenshots.

### Feature Requests
- **Suggesting Enhancements**: We are always looking for ways to make DICE better. If you have ideas for new features or enhancements, please share them with us through a new issue labeled as a feature request on our GitHub repository, or email [joseph@thiebes.org](mailto:joseph@thiebes.org).

### Contributing Code
- **Pull Requests**: Contributions in the form of code are particularly appreciated. Before you start working on a contribution, please check the existing issues and pull requests to see if someone else is working on a similar contribution. 
  - Fork the repository.
  - Create a new branch for your feature or fix.
  - Commit your changes with clear and concise commit messages.
  - Push your branch and submit a pull request to the main DICE repository. Include a detailed description of your changes and the reason for them.

### Documentation Improvements
- Good documentation is crucial to any project. If you notice errors or see an opportunity for improvement, please don't hesitate to edit and submit a pull request or email [joseph@thiebes.org](mailto:joseph@thiebes.org).

### Questions and General Feedback
- If you have questions or general feedback about DICE, feel free to contact the project maintainer at [joseph@thiebes.org](mailto:joseph@thiebes.org) or open an issue for discussion.

Your contributions, feedback, and questions are invaluable to us. They help us make DICE a better tool for everyone in the research community. Thank you for your support and collaboration!

[Back to table of contents](table-of-contents)

# Acknowledgements

Thanks to Professor Erik M. Grumstrup, Skyler Hollinbeck, Sajia Afrin, and my spouse, Julia K. Thiebes, for their invaluable support and feedback.

This material is based upon work supported by the National Science Foundation under Grant No. 2154448. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.

[Back to table of contents](table-of-contents)

# How to cite
```
Joseph J. Thiebes. 2023. thiebes/DICE. Zenodo. https://doi.org/10.5281/zenodo.10258191 
```

[Back to table of contents](table-of-contents)

# License

Diffusion Insight Computation Engine (DICE) by [Joseph J. Thiebes](http://thiebes.org) is licensed under the [MIT License](https://opensource.org/licenses/MIT).

Copyright (c) 2023-2025 Joseph J. Thiebes

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

[Back to table of contents](table-of-contents)
