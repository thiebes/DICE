# Diffusion Insight Computation Engine (DICE)
![DICE logo](logo/dice_logo_620w.png)

## Introduction

DICE is an open-source tool that empowers researchers to evaluate the precision and accuracy of their diffusion coefficient estimates derived from composite fits of time-resolved microscopy experiments. By simulating parameters that mirror your experimental setup, DICE provides a robust method for assessing the likelihood of experimental accuracy.

This software accompanies the paper titled 'Quantifying noise effects in optical measures of excited state transport' by Joseph J. Thiebes and Erik M. Grumstrup, currently under review. The full citation will be updated upon publication. 

Updates to this software can be found at [the DICE GitHub repository](https://github.com/thiebes/DICE).

### How it works

1. **Generation of time-series profiles:**  DICE starts by simulating a population of excited states in the form of a Gaussian distribution profile that undergoes decay and diffusion. The *initial profile* has an amplitude of unity, width parameterized by either the full-width half-maximum or the Gaussian standard deviation, mean position of zero, and baseline background of zero.
2. **Noise addition:** DICE incorporates white noise into the generated profiles to better resemble real-world scenarios. The magnitude of the noise is the same for every time frame and is parameterized by its standard deviation.
3. **Gaussian fit:** The time-evolved noisy profiles are fitted with Gaussian functions, deriving the estimated Mean Squared Displacement (*MSD*) at each time point.
4. **Linear fit and diffusion coefficient estimation:** The *MSD* values are fitted to a linear function using a weighted least squares method. The weights are assigned as the normalized inverse of the relative variance of the squared width parameter reported by the Gaussian fit. The slope of the linear fit is proportional to the estimated diffusion coefficient.
5. **Analysis:** DICE compares the estimated diffusion coefficient to the nominal parameter used to generate the series of profiles. By conducting multiple simulations with the same parameters, DICE provides a statistical overview of the precision and accuracy of the diffusion estimate.
6. **Presentation:** Several customizable plotting functions are provided to present the results. 

DICE's primary advantage lies in its ability to quantify the precision and accuracy of diffusion estimates. This is accomplished by reporting the fraction of the diffusion estimates within a user-specified proximity to the nominal diffusion parameter. 

For example: "50% of the diffusion coefficient estimates are within $\pm 10 \\%$ of the nominal value." 

Thus, DICE provides a measure of the likelihood that your estimated diffusion coefficient will attain the precision required for your investigations based on your experimental parameters, such as the initial spot width, decay lifetime, and contrast-to-noise ratio. Other traditional quantifications of precision and accuracy, such as mean and standard deviation of estimated diffusion coefficients, are also reported.

DICE aims to support ongoing improvements in the reliability and reproducibility of diffusion coefficient estimates derived from time-resolved microscopy methods.

If you enjoy this program and found it helpful, please share it.

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
This program is written in Python. We recommend using the latest version if you don't have Python installed on your system. You can install Python from the [official website](https://www.python.org/downloads/). Alternatively, you can use online Python notebook services (for example, [DataLore](https://datalore.jetbrains.com/)).

### Packages
DICE depends on several Python packages. Some of these might already be installed if you have Python installed. If not, you can install these packages using pip, Python's package installer. 

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
- **[Pint](https://pint.readthedocs.io/en/stable/)**: A Python package to define, operate and manipulate physical quantities (*i.e.*, units).

Remember to regularly update your packages to their latest versions using pip to ensure the smooth functioning of DICE. If you encounter issues during installation, feel free to contact us or raise an issue on GitHub.

## Quick Start

Here are some steps to quickly get started with DICE:

1. **Download the Repository**: Clone or download this repository and navigate to the project directory in your Python environment.

2. **Edit Parameters**: Open the `parameters.txt` file and adjust the values to match your experimental parameters. This file serves as the primary means of input to the simulation.

3. **Run the Simulation**: Execute the `main.py` script in your Python environment to start the simulation. The script initializes all necessary functions and runs the simulation based on the parameters specified in `parameters.txt`.

    ```bash
    python main.py
    ```
   
    You can run the script immediately after downloading if you want to get a feel of the simulation before editing the parameters. It is set to work with default parameters that serve as an example.

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
Provide a number of simulations to run for each set of parameters. 

### Spatial parameters
Provide the spatial width and number of pixels for the simulated scan. Note that the value given for spatial width here will be assumed to be in the units of length specified above. 

### Temporal parameters
Provide time axis information to be used in each simulation. You can provide:
- **`'time range'`**: start, stop, and number of steps (inclusive) for an evenly-timed series of frames, *e.g.,* `[0, 1, 10]` or
- **`'time series'`**: an explicit series of time values, *e.g.,* `[0.1, 0.3, 0.5, 0.7, 0.9]`

### Initial profile parameters
Provide the amplitude, mean (*i.e.,* center), and width of the initial Gaussian signal profile.
- The amplitude is usually 1 arbitrary unit, but it can be changed here if needed. Note, however, that the noise standard deviation values are relative to a normalized initial amplitude.
- The mean is 0 for profiles centered at the origin.
- The width may be given as sigma (the standard deviation of the Gaussian) or as FWHM (full-width, half-maximum).

### Diffusion length
Provide *one* of the following:
- the diffusion coefficient and lifetime, or
- the diffusion length
If you enter diffusion length, the script will generate corresponding nominal values for the diffusion and lifetime (and *vice-versa*).

***Remember to use the units you specified in the unit parameters.*** For example, if your unit parameters are micrometers and nanoseconds, then the units for the diffusion coefficient will be assumed as \[$`\text{µm}^2`$ $`\text{ns}^{-1} `$\]. Thus, ensure that all the magnitudes you enter are appropriately scaled to correspond to the units you have specified.

### Noise standard deviation
Provide the standard deviation of the white noise to be added to each pixel of each profile in a simulation. You can provide:
- A single value, which will be repeatedly used for every run of the simulation
- An experimental profile at $t=0$, in the form of a comma-separated text file with the profile values in one row — the script will estimate the noise based on this initial profile.

### Proximity level
The proximity to the nominal value is used to analyze accuracy and precision. Proximity is expressed as the ratio of estimate to nominal diffusion coefficient values. The program will calculate what portion of the simulations are within the proximity level specified here. For example, if you want to know how many diffusion estimates are within 10% of the nominal value, you should enter 0.1 for the precision level. 

In other words:
- The simulations will produce diffusion coefficient estimates, $D_{est}$. 
- The *accuracy* of each estimate is quantified by the relative proximity of the estimate to the nominal value, $D_{nom}$.
- The relative proximity is evaluated by taking the quotient of the estimate and the nominal value, *i.e.*, $D_{est} / D_{nom}$. 
- If the estimate is perfectly accurate, it will be equal to the nominal value, and the relative proximity will be 1. 
- Proximity statistics are calculated from the number of estimates within the proximity level.
- For example, if you enter 0.1 for your precision level, the program will tell you what portion of all the estimates were within $\pm 10$% of the nominal value. 

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
- The format of data in an imported initial profile for estimation of $CNR_0$ is limited to a single line of comma-separated values. Data that do not strictly follow this format will fail.
- This program only evaluates Gaussian function fits for distribution profiles. Lorentzian, Voigt, Green's function, *etc.* are not considered. 
- Profiles analyzed are 1-dimensional. 
- Non-Fickian diffusion (*e.g.*, subdiffusion) is not considered.
- Anisotropic diffusion is not considered.
- Lifetime is modeled as single-exponential.

[Back to table of contents](table-of-contents)

# Acknowledgements

I am deeply thankful to Professor Erik M. Grumstrup from the Chemistry & Biochemistry Department at Montana State University for his invaluable feedback and guidance throughout the development of this software. His expertise and commitment have been instrumental in refining this package. A link to his lab can be found [here](https://www.montana.edu/grumstruplab/).

My colleagues Skyler Hollinbeck and Sajia Afrin offered thoughtful suggestions and probing questions, which consistently challenged me to improve and fine-tune this software.

Lastly, I want to express my heartfelt gratitude to my spouse, Julia K. Thiebes, for her unwavering support and patience. Moreover, her insightful questions, informed by her past scientific education in a different field, have helped shape how I communicate about this project to a broader audience. Our children, Leila, Alexandria, and Minerva, have provided joy amidst the long hours dedicated to this work.

Through the collective efforts of these individuals, this software has been made possible; for that, I am profoundly grateful.

[Back to table of contents](table-of-contents)

# How to cite
Coming soon.

[Back to table of contents](table-of-contents)

# License
[![Creative Commons License](https://i.creativecommons.org/l/by/4.0/88x31.png)](http://creativecommons.org/licenses/by/4.0/)  
Diffusion Insight Computation Engine (DICE) by [Joseph J. Thiebes](http://thiebes.org) is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).  
Based on a work at [https://github.com/thiebes/DICE](https://github.com/thiebes/DICE).

[Back to table of contents](table-of-contents)
