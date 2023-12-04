# Diffusion Insight Computation Engine (DICE)
![DICE logo](logo/dice_logo_640w.png)

## Introduction

DICE is an open-source tool for researchers in time-resolved microscopy and related fields. It evaluates the precision and accuracy of diffusion coefficient estimates derived from optical measures of excited state transport. DICE provides a robust method for assessing experimental accuracy and precision by simulating parameters that mirror your experimental setup.

DICE's primary advantage lies in its ability to quantify the precision and accuracy of diffusion estimates. This is accomplished by reporting the fraction of the diffusion estimates found within a user-specified proximity to the nominal diffusion parameter. For example: "50% of the diffusion coefficient estimates are within ±10% of the nominal value."

Thus, DICE measures the likelihood that your estimated diffusion coefficient will attain the accuracy and precision required for your investigations based on your experimental parameters, such as the initial spot width, nominal diffusion length, and initial contrast-to-noise ratio.

DICE aims to support ongoing improvements in the reliability and reproducibility of diffusion coefficient estimates derived from time-resolved microscopy methods. If you enjoy this program and found it helpful, please share it.

This software accompanies the paper titled 'Quantifying noise effects in optical measures of excited state transport' by Joseph J. Thiebes and Erik M. Grumstrup, currently under review. The full citation will be updated upon publication. 

### How it works

1. **Generation of time-series profiles:**  DICE starts by simulating a population of excited states as a Gaussian distribution profile that undergoes decay and diffusion. The *initial profile* has an amplitude of unity, a width parameterized by either the full-width half-maximum or the Gaussian standard deviation, a mean position of zero, and a baseline background of zero.
2. **Noise addition:** DICE incorporates white noise into the generated profiles to better resemble real-world scenarios. The magnitude of the noise is the same for every time frame and is parameterized by its standard deviation.
3. **Gaussian fit:** The time-evolved noisy profiles are fitted with Gaussian functions, estimating the Mean Squared Displacement (*MSD*) at each time point.
4. **Linear fit and diffusion coefficient estimation:** The *MSD* values are fitted to a linear function using a weighted least squares method. The slope of the linear fit is proportional to the estimated diffusion coefficient.
5. **Analysis:** DICE compares the estimated diffusion coefficient to the nominal parameter used to generate the series of profiles. By conducting multiple simulations with the same parameters, DICE provides a statistical view of the precision and accuracy of diffusion estimates.
6. **Presentation:** Several customizable plotting functions are provided to present the results. 

# Table of Contents
- [Installation](#installation)
- [How to use](#how-to-use)
  - [Quick start](#quick-start)
  - [Glossary](#glossary)
  - [Parameter inputs](#parameter-inputs)
  - [Functions](#functions)
  - [Outputs](#outputs)
  - [Known issues and limitations](#known-issues-and-limitations)
- [How to cite](#how-to-cite)
- [License](license)

# Installation
## Python
This program is written in Python. We recommend using the latest version if you don't have Python installed. You can install Python from the [official website](https://www.python.org/downloads/). Alternatively, you can use online Python notebook services (for example, [DataLore](https://datalore.jetbrains.com/)).

### Packages
DICE depends on several Python packages. Some of these might already be installed if you have Python installed. If not, you can install these packages using pip, Python's package installer. 

You can do this by typing the following command in your terminal:

```bash
pip install numpy pandas matplotlib scipy statsmodels joblib
```

Here's a brief overview of what each package is used for:
- **`numpy`**: Fundamental package for scientific computing with Python.
- **`pandas`**: Library providing high-performance, easy-to-use data structures.
- **`matplotlib`**: Plotting library for creating static, animated, and interactive visualizations.
- **`scipy`**: Used for scientific and technical computing.
- **`statsmodels`**: Provides classes and functions for estimating many different statistical models.
- **`joblib`**: Used for lightweight pipelining in Python.

**Standard Python Libraries:** The following are part of the Python Standard Library and do not need to be installed separately:
- **`ast`**: For working with abstract syntax trees.
- **`os`**: Provides a way of using operating system-dependent functionality.
- **`re`**: Provides regular expression matching operations.
- **`typing`**: Used to support type hints (available in Python 3.5 and later).

Remember to regularly update your packages to their latest versions using pip to ensure the smooth functioning of DICE. If you encounter issues during installation, please **[contact the author](http://thiebes.org/contact)** or raise an issue on GitHub.

[Back to table of contents](table-of-contents)

---

## Quick Start

Follow these steps to get started with DICE quickly:

1. **Download the Repository**: 
   Clone or download this repository to your local machine and navigate to the project directory in your Python environment.

2. **Edit Parameters**: 
   Modify the `parameters.txt` file to align with your experimental parameters. This file is the primary input for the simulation.

3. **Run the Simulation**: 
   The DICE module can be used in two different environments: the command line and a Jupyter-like environment.
   
   - **From the command line/console**:
     Utilize the `run_dice.py` script to run the DICE module. This script requires the parameters file as an argument.
       - Open your console or terminal.
       - Navigate to the directory containing `run_dice.py`.
       - Execute the script with your parameters file:
         ```bash
         python run_dice.py parameters.txt
         ```

   - **From within a Jupyter-like environment**:
     - Import the DICE module.
     - Invoke the `dice_runner` function, passing the filename of your parameters.
       ```python
       import dice

       # Replace "parameters.txt" with your parameters file
       result = dice.dice_runner("parameters.txt")
       ```
     This function processes the parameters from the specified file and returns the results in a dictionary named `result`.

   *Note:* The script can be executed immediately after downloading to experience the simulation with default parameters, which serve as an illustrative example.

4. **View the Results**: 
   After the simulation completes, the results will be summarized and saved, including a histogram plot for visual analysis.

[Back to table of contents](table-of-contents)

## Glossary
Jargon terms and acronyms used in this documentation include:
- CNR: contrast-to-noise ratio, *i.e.,* the ratio of the signal amplitude to the standard deviation of the noise
- OLS: ordinary least-squares fitting algorithm
- WLS: weighted least-squares fitting algorithm

[Back to table of contents](table-of-contents)

## Parameter inputs
Edit the `parameters.txt` file to set up your simulation. Each simulation models one experimental diffusion measurement, comprising a series of time-evolved Gaussian profiles with noise. Each simulation uses one set of parameters.

### Filename slug
A prefix for your output files to help identify them later. The output file names will also automatically include the parameters below, so your slug could be a textual identifier connected to a set of experiments. Avoid special characters in the slug. 

Example:
If the Filename slug is set to `'slug-example'`, and a simulation with specific parameters is run, the files generated might include:
- `slug-example_LD-0.1_CNR-20.0_px-100_tx-10_runs-1000_results.csv`
- `slug-example_LD-0.1_CNR-20.0_px-100_tx-10_runs-1000_summary.txt`
- `slug-example_LD-0.1_CNR-20.0_px-100_tx-10_runs-1000_histogram.svg`

The parameters in the above examples include:
- `LD-0.1' means that the nominal diffusion length was 0.1 relative to the initial FWHM.
- `CNR-20.0` indicates an initial contrast-to-noise ratio of 20.0.
- `px-100` means 100 pixels across the *x*-axis.
- `tx-10` means that there are 10 time frames in each simulated experiment.
- `runs-1000` means the file contains the results from 1000 simulation runs.

### Image type
Choose the file type for the default plot image. Options include `'jpg'`, `'png'`, `'svg'`, `'tif'`.

### Retain profile data?
Decide whether to retain all profile data. Note that keeping profile data may impact performance, depending on the number of profiles generated. 
- `True`: Keep all data (necessary for plotting individual profiles).
- `False`: Discard raw data (retain only the fitting results).

### Parallel processing
Enable parallel processing by setting this parameter to `True`.

### Units
Specify the singular units for length and time, like `'micrometer'` and `'nanosecond'`. These units apply to all parameters representing physical quantities of length and time. Ensure that all parameter magnitudes are appropriately scaled.

### Number of simulation iterations
Specify the number of simulations to run. 

### Spatial parameters
Set the spatial width and pixel number for the scan. The spatial width should align with the length unit specified.

### Temporal parameters
Provide time-axis information using one of the following forms:
- `'time range'`: Format `[start, stop, steps]` for evenly timed frames.
- `'time series'`: Specific time values like `[0.1, 0.3, 0.5, 0.7, 0.9]`.

### Initial profile parameters
Configure the amplitude, mean, and width of the initial Gaussian profile. Note: Amplitudes other than unity have not been tested.
- The amplitude is usually unity, but it can be changed here if needed. Other amplitude values have not been tested, however.
- For profiles centered at the origin, the mean should be set to zero.
- The width may be given as sigma (the standard deviation) or FWHM (full-width, half-maximum).

### Diffusion length parameters
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

### Noise standard deviation
Set the standard deviation of noise added to each profile pixel. You can provide:
- A single value for all runs, or
- An initial profile as a text file; the script will estimate the noise.

### Proximity level
This parameter determines the accuracy threshold. For instance, a 0.1 proximity level means the program will evaluate how many estimates are within $\pm 10$% of the nominal value.

In other words:
- The simulations will produce diffusion coefficient estimates, $D_{est}$. 
- The *accuracy* of each estimate is quantified by the relative proximity of the estimate to the nominal value, $D_{nom}$.
- The relative proximity is evaluated by taking the quotient of the estimate and the nominal value, *i.e.*, $D_{est} / D_{nom}$. 
- If the estimate is perfectly accurate, it will equal the nominal value, and the relative proximity will be 1. 
- The portion of estimates within the proximity level is reported.

*For example,* if you enter 0.1 for your precision level, the program will tell you what portion of all the estimates were within $\pm 10$% of the nominal value.

*Note:* Conventional statistical methods of characterizing precision and accuracy are to report the standard deviation and the mean, median, and mode, respectively. The portion of estimates that are within proximity to the nominal value, reported by this script, offers a single evaluation of the impact of both precision and accuracy on the results. In some ways, this figure may be more intuitive and practical in experimental settings. 

[Back to table of contents](table-of-contents)

# Functions

The source code contains extensive documentation describing the functions and how they work. 

[Back to table of contents](table-of-contents)

# Outputs
- Summary text file
- Results CSV file
- Plot

[Back to table of contents](table-of-contents)

# Known issues and limitations
- The data format in an imported initial profile for estimation of $CNR_0$ is limited to a single line of comma-separated values. Data that do not strictly follow this format will fail.
- This program only evaluates Gaussian function fits for distribution profiles. Lorentzian, Voigt, Green's function, *etc.* are not considered. 
- Profiles analyzed are 1-dimensional. 
- Non-Fickian diffusion (*e.g.*, subdiffusion) is not considered.
- Anisotropic diffusion is not considered.
- Lifetime is modeled as single-exponential.

[Back to table of contents](table-of-contents)

# Acknowledgements

Thanks to Professor Erik M. Grumstrup, Skyler Hollinbeck, Sajia Afrin, and my spouse, Julia K. Thiebes, for their invaluable support and feedback.

This material is based upon work supported by the National Science Foundation under Grant No. 2154448. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.

[Back to table of contents](table-of-contents)

# How to cite
Coming soon.

[Back to table of contents](table-of-contents)

# License
[![Creative Commons License](https://i.creativecommons.org/l/by/4.0/88x31.png)](http://creativecommons.org/licenses/by/4.0/)  
Diffusion Insight Computation Engine (DICE) by [Joseph J. Thiebes](http://thiebes.org) is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).  
Based on a work at [https://github.com/thiebes/DICE](https://github.com/thiebes/DICE).

[Back to table of contents](table-of-contents)
