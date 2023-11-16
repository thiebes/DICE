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

## Quick Start

Here are some steps to quickly get started with DICE:

1. **Download the Repository**: Clone or download this repository and navigate to the project directory in your Python environment.

2. **Edit Parameters**: Open the `parameters.txt` file and adjust the values to match your experimental parameters. This file serves as the primary means of input to the simulation.

3. **Run the Simulation**: Execute the `main.py` script in your Python environment to start the simulation. The script initializes all necessary functions and runs the simulation based on the parameters specified in `parameters.txt`.
   **From the console**:
    ```bash
    python main.py
    ```
   - `python main.py`: This command is used in the console (like Command Prompt, PowerShell, or a terminal in Linux/Mac). It tells Python to execute the script named `main.py`. This is a common way to run Python scripts.
   
   **From within a Jupyter-like environment**:
    ```
    import main
    main
    ```
   - `import main`: This line imports the `main` module into the Jupyter notebook or similar interactive environment.
   - `main`: After importing, this line calls the main function or the main executable part of your main.py script.

   You can run the script immediately after downloading if you want to get a feel of the simulation before editing the parameters. It is set to work with default parameters that serve as an example.

5. **View the Results**: Upon successful execution, the simulation results will be displayed and saved.

[Back to table of contents](table-of-contents)

## Glossary
Jargon terms and acronyms used in this documentation include:
- CNR: contrast-to-noise ratio, *i.e.,* the ratio of the signal amplitude to the standard deviation of the noise
- OLS: ordinary least-squares fitting algorithm
- WLS: weighted least-squares fitting algorithm

[Back to table of contents](table-of-contents)

## Parameter inputs
- Edit the parameters.txt file to set up your simulation.
- Each simulation is a model of one experimental diffusion measurement comprising a series of time-evolved Gaussian profiles with noise.
- Each simulation will be executed using one set of parameters.
- Some parameters can be given as a series of values, and the program will run with each value in the series in turn.

### Filename slug
A prefix for your output files to help you identify them later. The output files will already have several parameters indicated, so your slug could be a textual identifier to connect it to a set of experiments.

### Image type
The file type for the default plot image to be saved as. For example, `'jpg'`

### Retain profile data?
You may or may not care about keeping all the profile data. If you are generating a large number of profiles, keeping all that data may become a performance or memory issue. 
- To keep all profile data (needed for *e.g.*, plotting profiles), choose `1`
- To discard profile data and only retain the fitting results, choose `0`

### Parallel processing
It is possible to take advantage of parallel processing with this program. To do so, select `1` for this parameter. 

### Units
Provide the length and time units that apply to all length and time parameter values. For example, `'micrometers'` and `'nanoseconds'`. **Important!** The units you enter here for length and time will be applied to ***all parameters*** that represent physical quantities of length and/or time, respectively. Therefore, ensure that the magnitudes you enter for all parameters are appropriately scaled to match your specified units.

### Number of simulation iterations
Specify a number of simulations to run for each set of parameters. 

### Spatial parameters
Provide the spatial width and number of pixels for the simulated scan. Note that the value given for spatial width here will be assumed to be in the units of length specified above. 

### Temporal parameters
Provide time axis information to be used in each simulation. You can provide:
- **`'time range'`**: start, stop, and number of steps (inclusive) for an evenly-timed series of frames, *e.g.,* `[0, 1, 10]` or
- **`'time series'`**: an explicit series of time values, *e.g.,* `[0.1, 0.3, 0.5, 0.7, 0.9]`

### Initial profile parameters
Provide the amplitude, mean (*i.e.,* center), and width of the initial Gaussian signal profile.
- The amplitude is usually unity, but it can be changed here if needed. Note, however, that the noise standard deviation values are relative to a normalized initial amplitude.
- The mean is 0 for profiles centered at the origin.
- The width may be given as sigma (the standard deviation of the Gaussian) or as FWHM (full-width, half-maximum).

### Diffusion length
Provide *one* of the following:
- the diffusion coefficient and lifetime, or
- the diffusion length
If you enter diffusion length, the script will generate corresponding nominal values for the diffusion and lifetime (and *vice-versa*).

***Remember to use the units you specified in the unit parameters.*** For example, if your unit parameters are micrometers and nanoseconds, then the units for the diffusion coefficient will be assumed as \[$`\text{µm}^2`$ $`\text{ns}^{-1} `$\]. Thus, ensure that all the magnitudes you enter are appropriately scaled to correspond to the units you have specified.

### Noise standard deviation
Provide the standard deviation of the white noise to be added to each profile pixel in a simulation. You can provide:
- A single value, which will be repeatedly used for every run of the simulation
- An experimental profile at $t=0$, in the form of a comma-separated text file with the profile values in one row — the script will estimate the noise based on this initial profile.

### Proximity level
The proximity to the nominal value is used to analyze accuracy and precision. Proximity is expressed as the ratio of estimate to nominal diffusion coefficient values. The program will calculate what portion of the simulations are within the proximity level specified here. For example, if you want to know how many diffusion estimates are within 10% of the nominal value, you should enter 0.1 for the precision level. 

In other words:
- The simulations will produce diffusion coefficient estimates, $D_{est}$. 
- The *accuracy* of each estimate is quantified by the relative proximity of the estimate to the nominal value, $D_{nom}$.
- The relative proximity is evaluated by taking the quotient of the estimate and the nominal value, *i.e.*, $D_{est} / D_{nom}$. 
- If the estimate is perfectly accurate, it will equal the nominal value, and the relative proximity will be 1. 
- Proximity statistics are calculated from the number of estimates within the proximity level.
- For example, if you enter 0.1 for your precision level, the program will tell you what portion of all the estimates were within $\pm 10$% of the nominal value. 

[Back to table of contents](table-of-contents)

# Functions

The source code contains extensive documentation at almost every line, describing the functions and how they work. Therefore, the following will describe a few of the main functions that you may want to use from the command line:

- ```nds_runner(parameters_filename)```
  - this will run the simulations according to the parameters stored in a parameters text file
  - it outputs a dictionary with the results of each simulation, along with a table of results from all simulations
  - Example: 
  ```
  my_simulation_result = scan_iterator(my_parameters.txt)
  ```

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

[Back to table of contents](table-of-contents)

# How to cite
Coming soon.

[Back to table of contents](table-of-contents)

# License
[![Creative Commons License](https://i.creativecommons.org/l/by/4.0/88x31.png)](http://creativecommons.org/licenses/by/4.0/)  
Diffusion Insight Computation Engine (DICE) by [Joseph J. Thiebes](http://thiebes.org) is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).  
Based on a work at [https://github.com/thiebes/DICE](https://github.com/thiebes/DICE).

[Back to table of contents](table-of-contents)
