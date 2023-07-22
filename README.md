# Diffusion Insight Computation Engine (DICE)
![DICE logo](dice_logo_620w.png)

## Introduction

DICE is an open-source tool that empowers researchers to evaluate the precision of their diffusion coefficient estimates derived from composite fits of time-resolved microscopy experiments. By simulating parameters that mirror your experimental setup, DICE provides a robust and quantifiable method for assessing experimental precision.

### How it works

1. **Generation of time-series profiles:**  DICE starts by generating Gaussian distribution profiles that undergo decay and diffusion as they evolve over time.
2. **Noise addition:** To better resemble real-world scenarios, DICE incorporates white noise into these generated profiles.
3. **Gaussian fit:** The noisy profiles are then fitted with Gaussian functions, resulting in the derivation of the estimated Mean Squared Displacement (MSD) at each time point.
4. **Linear fit and diffusion coefficient estimation:** The MSD values are fitted to a linear function using a weighted least squares method. The slope of this function is proportional to the estimated diffusion coefficient.
5. **Analysis:** DICE concludes by comparing the estimated diffusion coefficient to the nominal diffusion parameter used to generate the initial profiles. By conducting multiple simulations with the same parameters, DICE provides a statistical overview of the precision of the diffusion estimate.

DICE's primary advantage lies in its ability to quantify the fraction of total fits aligned within a user-specified proximity to the nominal diffusion parameter. This provides a measure of the likelihood that your estimated diffusion coefficient will attain the precision required for your investigations. DICE aims to support ongoing improvements in the reliability and reproducibility of diffusion coefficient estimates derived from time-resolved microscopy methods.

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

The source code contains extensive documentation at almost every line, describing the functions and how they work. Therefore the following will describe a few of the main functions that you may want to use from the command line:

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
- Performing OLS fits is unnecessary but was done for comparison purposes. It adds a lot of computational time. Need to make it optional or eliminate it.
- This program only evaluates Gaussian function fits for distribution profiles. Could incorporate options of Lorentzian, Voigt, Green's functions, etc.
- Non-Fickian diffusion is not analyzed by this program.
- Lifetime is currently only modeled as single-exponential. In an experimental result with two (or more) very different lifetimes, this could mean that the experimental probability of precision is lower than predicted by the model.
- The format of data in an imported profile for estimation of CNR is limited to single-line comma-separated values. Files that do not strictly follow this format will fail. This could be made more versatile.

[Back to table of contents](table-of-contents)

# Acknowledgements

I am deeply thankful to Professor Erik M. Grumstrup from the Chemistry & Biochemistry Department at Montana State University for his invaluable feedback and guidance throughout the development of this software. His expertise and commitment have been instrumental in refining this package. A link to his lab can be found [here](https://www.montana.edu/grumstruplab/).

My colleagues, Skyler Hollinbeck and Sajia Afrin, deserve special mention for their thoughtful suggestions and probing questions, which consistently challenged me to improve and fine-tune this software.

Lastly, I want to express my heartfelt gratitude to my spouse, Julia K. Thiebes. Beyond her unwavering support and patience, her insightful questions, informed by her past scientific education in a different field, have helped shape how I communicate about this project to a broader audience. Our children, Leila, Alexandria, and Zoe, have shown understanding and provided joy amidst the long hours dedicated to this work.

Through the collective efforts of these individuals, this software has been made possible; for that, I am profoundly grateful.

[Back to table of contents](table-of-contents)

# How to cite
Coming soon.

[Back to table of contents](table-of-contents)

# License

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">Diffusion Insight Computation Engine (DICE)</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="http://thiebes.org" property="cc:attributionName" rel="cc:attributionURL">Joseph J. Thiebes</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.<br />Based on a work at <a xmlns:dct="http://purl.org/dc/terms/" href="https://github.com/thiebes/DICE" rel="dct:source">https://github.com/thiebes/DICE</a>.

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 

[Back to table of contents](table-of-contents)
