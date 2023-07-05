# Diffusion Investigation and Validation Engine (DIVE)

## Introduction

The Diffusion Investigation and Validation Engine (DIVE) is a powerful, open-source tool that facilitates precise analysis and interpretation of diffusion experiments. It empowers researchers to assess the precision of their composite fits of Gaussian distributions that undergo decay and diffusion over time.

With DIVE, you can input parameters that mirror your experimental setup, and the tool will conduct a series of simulations. Here's how it works:

1. **Generation of Time-Series Profiles:** DIVE creates decaying and diffusing Gaussian distribution profiles that evolve over time.
2. **Noise Addition:** Normal white noise is integrated into the generated profiles to simulate real-world experimental scenarios.
3. **Gaussian Fit:** The noisy profiles are then fitted with Gaussian functions, deriving the Mean Squared Displacement (MSD).
4. **Linear Fit and Diffusion Coefficient Estimation:** The MSD values of the Gaussian functions are fitted to a linear function using a weighted least squares method. The slope of this linear fit provides an estimate of the diffusion coefficient.
5. **Analysis:** DIVE then compares this diffusion coefficient estimate to the nominal diffusion parameter used to generate the profiles. Through numerous simulations under the same parameter set, DIVE constructs a statistical overview of the fit's precision.

The tool's primary value comes from determining the fraction of total fits that align within an acceptable proximity to the nominal diffusion value. In essence, DIVE provides the probability that your estimated diffusion coefficient achieves a degree of precision suitable for your research needs. This offers researchers a robust, quantifiable method of assessing experimental accuracy and precision, thereby contributing to more reliable and reproducible research.

If you enjoy this program and found it helpful, please share it.

# Table of Contents
- [Installation](#installation)
- [How to use](#how-to-use)
  - [Quick start](#quick-start)
  - [Acronyms](acronyms)
  - [Parameter inputs](#parameter-inputs)
  - [Functions](#functions)
  - [Outputs](#outputs)
  - [Known issues and limitations](#known-issues-and-limitations)
- [How to cite](#how-to-cite)
- [License](license)

# Installation
## Python
This program is written in Python. If you don't have Python installed on your system, we recommend using the latest version. You can download and install Python from the [official website](https://www.python.org/downloads/). Alternatively, you can use online Python notebook services like [DataLore](https://datalore.jetbrains.com/).

### Packages

DIVE depends on several Python packages. If you have Python already installed, some of these might be pre-installed. If not, you can install these packages using pip, Python's package installer. 

You can do this by typing the following command in your terminal:

```bash
pip install ast numpy pandas matplotlib scipy statsmodels pint
```

Here are the packages required:

- **[ast](https://docs.python.org/3/library/ast.html)**: A built-in Python library for Abstract Syntax Trees.
- **[NumPy](https://numpy.org/)**: Fundamental package for numerical computation in Python.
- **[Pandas](https://pandas.pydata.org/)**: Provides high-performance, easy-to-use data structures and data analysis tools.
- **[Matplotlib](https://matplotlib.org/)**: A comprehensive library for creating static, animated, and interactive visualizations in Python.
- **[SciPy](https://www.scipy.org/)**: A Python library used for scientific and technical computing.
- **[Statsmodels](https://www.statsmodels.org/stable/index.html)**: A Python module that provides classes and functions for the estimation of many different statistical models.
- **[Pint](https://pint.readthedocs.io/en/stable/)**: A Python package to define, operate and manipulate physical quantities.

Remember to regularly update your packages to their latest versions using pip to ensure the smooth functioning of DIVE. If you encounter issues during installation, feel free to contact us or raise an issue on GitHub.

## Quick Start

Here are some steps to quickly get started with DIVE:

1. **Download the Repository**: Clone or download this repository and navigate to the project directory in your Python environment.

2. **Edit Parameters**: Open the `parameters.txt` file and adjust the values to match your experimental parameters. This file serves as the primary means of input to the simulation.

3. **Run the Simulation**: Execute the `main.py` script in your Python environment to start the simulation. The script initializes all necessary functions and runs the simulation based on the parameters specified in `parameters.txt`.

    ```bash
    python main.py
    ```
   
    If you want to get a feel of the simulation before editing the parameters, you can run the script immediately after downloading. It is set to work with default parameters that serve as an example.

4. **View the Results**: Upon successful execution, the simulation results will be displayed.

5. **Modifying the Simulation**: If you wish to use the functions without automatically running a simulation, you can comment or remove the lines in `main.py` that execute the simulation. This can be useful if you want to load data from a previous simulation or if you wish to create a separate script for running the simulation.

[Back to table of contents](table-of-contents)

## Glossary
Jargon terms and acronyms used in this documentation include:
- CNR: contrast-to-noise ratio, *i.e.,* the ratio of the signal amplitude to the standard deviation of the noise
- OLS: ordinary least-squares fitting algorithm
- WLS: weighted least-squares fitting algorithm

[Back to table of contents](table-of-contents)

## Parameter inputs
- Edit the parameters.txt file to set up your simulation.
- Each simulation is a model of one experimental diffusion measurement, comprised of a series of time-evolved Gaussian profiles with noise.
- Each simulation will be executed using one set of parameters. 
- A series of many simulations may have a range of parameters, or copies of the same parameters. 

### Filename slug
The filename slug is a prefix for your output files to help you identify them later. The output files will already have several of your parameters indicated, so your slug could be a textual identifier of some other kind, for example to connect it to a set of experiments.

### Verbose or brief
You may or may not care about keeping all the profile data. If you are generating a large number of profiles, keeping all that data may become a performance or memory issue. 
- To keep all profile data (needed for e.g. plotting profiles), choose 'verbose'
- To discard profile data and only retain the fitting results, choose 'brief'

### Units
Provide the units that apply to all length and time parameter values.

### Number of simulation runs
Provide a number of simulations to run for each value of diffusion length.

### Spatial parameters
Provide the spatial width and number of pixels for the simulated scan. 

### Temporal parameters
Provide information about the time axis to be used for each simulation. You can provide:
- Start, stop, and number of steps (inclusive) for an evenly-timed series of frames
- An explicit series of time values

### Initial profile parameters
Provide the amplitude, mean (i.e. center), and width of the initial Gaussian signal profile.
- The amplitude is usually 1 arbitrary unit, but you can change it if needed. Note, however, that the noise standard deviation values are based on a normalized initial amplitude.
- The mean is usually 0, for profiles centered at the origin, but you can change it if needed.
- The width may be given as sigma (the standard deviation of the Gaussian) or as FWHM (full-width, half-maximum)

### Profile evolution parameters
Provide *one* of the following:
- the diffusion coefficient and lifetime, or
- the diffusion length
If you enter diffusion length, the script will generate corresponding nominal values for the diffusion and lifetime (and *vice-versa*).

Remember to use the units you specified in the unit parameters. 
For example, if your unit parameters are micrometers and nanoseconds, 
then the units for the diffusion coefficient will be assumed as
$\text{µm}^2$ $\text{ns}^{-1}$

### Noise parameters
  Provide the standard deviation of the additive normal white noise to be added to each pixel of each profile in a simulation. You can provide:
- A single value, which will be repeatedly used for every run of the simulation
- An experimental profile at $t=0$, in the form of a comma-separated text file with 
  the profile values in one row — the script will estimate the noise using Fourier transform

### Precision level
Briefly, if you want to know how many of the diffusion estimates are within 10% of the nominal value, then you should enter 0.1 for precision level. For a more detailed explanation, read on.

- The simulations will produce diffusion coefficient estimates, $D_{est}$. 
- The *accuracy* of each estimate is quantified by the relative proximity of the estimate to the nominal value, $D_{nom}$. The relative proximity is evaluated by taking the quotient of the estimate and the nominal value, *i.e.*, $D_{est} / D_{nom}$. 
- If the estimate is perfectly accurate, it will be equal to the nominal value, and the relative proximity will be 1. 
- Precision is evaluated based on how many of the estimates lie within your specified arbitrary precision level. For example, if you enter 0.1 for your precision level, the program will tell you what portion of all the estimates were within $\pm 10$% of the nominal value. 

[Back to table of contents](table-of-contents)

# Functions

The source code contains extensive documentation at almost every line, describing the functions and how they work. Therefore the following will just describe a few of the main functions that you may want to use from the command line:

- ```nds_runner(parameters_filename)```
  - this will run the simulations according to the parameters stored in a parameters text file
  - it outputs a dictionary with the results of each individual simulation, along with a table of results from all simulations
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
- Performing OLS fits is totally unnecessary but was done for comparison purposes. Probably adding a lot of computational time.
- This program only evaluates Gaussian function fits for distribution profiles. Could incorporate options of Lorentzian, Voigt, Green's functions, etc.
- Distributions could be non-Gaussian to begin with. Would require some thought about how to approach modeling the initial population.
- Diffusion could be anomalous and/or non-Fickian.
- Lifetime is currently only modeled as single-exponential. 

[Back to table of contents](table-of-contents)

# Acknowledgements

This software could not have been written without the valuable feedback provided by [Professor Erik M. Grumstrup in the Chemistry & Biochemistry Department at Montana State University](https://www.montana.edu/grumstruplab/). I also acknowledge the support of my colleagues Skyler Hollinbeck, Sajia Afrin, and Alex King, who each offered input on this project. This effort would also not have been possible without the kindness and patience of my spouse and our two children.

[Back to table of contents](table-of-contents)

# How to cite
Coming soon.

[Back to table of contents](table-of-contents)

# License

Noisy Diffusion Simulator simulates diffusion under experimental parameters and evaluates the precision of composite fitting methods of estimating the diffusion coefficient.
Copyright © 2022 Joseph J. Thiebes

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

[Back to table of contents](table-of-contents)
