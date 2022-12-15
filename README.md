# Noisy Diffusion Simulator

This tool is designed to help gain a quantitative sense of the precision of composite fits of time-evolved Gaussian distributions that undergo decay and diffusion. After entering parameters to match your experimental setup, the program will perform simulations by:
1. generating a temporal series of profiles as decaying and diffusing Gaussian distributions,
2. adding normal white noise to the generated profiles,
3. fitting the noisy profiles with Gaussian functions, and
4. fitting the variances of the fitted Gaussian functions to a linear function using weighted least squares.

The slope of the resulting linear fit is proportional to a diffusion coefficient estimate. This estimate is then compared to the nominal diffusion parameter used to generate the profiles to begin with.

By running many simulations for a given set of parameters, a statistical view of the precision of the fits is built up. The principal figure of merit that the tool will offer is the portion of the total number of fits that fall within an arbitrary proximity to the nominal diffusion value. e.g., what percentage of fitted diffusion coefficients are within ±10% (or your selected precision level) of the nominal diffusion coefficient. 

If you enjoy this program and found it helpful, please share it.

# Table of Contents
- [Installation](#installation)
- [How to use](#how-to-use)
  - [Quick start](#quick-start)
  - [Parameter inputs](#parameter-inputs)
  - [Functions](#functions)
  - [Outputs](#outputs)
- [License](license)

# Installation
## Python
This program is written in Python. If you don't have Python installed on your system, you may want to [download and install it](https://wiki.python.org/moin/BeginnersGuide/Download). Another option is to use an online service like [DataLore](https://datalore.jetbrains.com/).

## Packages
The Noisy Diffusion Simulator relies on the following packages to function. If you have Python installed, or if you use an online service, some of these may already be installed as well. Installation of packages is a relatively simple process that usually just requires typing a single command. Package installation is [described here](https://packaging.python.org/en/latest/tutorials/installing-packages/). 
* [NumPy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [MatPlotLib](https://matplotlib.org/)
* [SciPy](https://scipy.org/)
* [statsmodels](https://www.statsmodels.org/stable/index.html)
* [sigfig](https://pypi.org/project/sigfig/)
* [Pint](https://pint.readthedocs.io/en/stable/)

[Back to table of contents](table-of-contents)

# How to use

## Quick start
To get started right away, simply edit the parameters.txt file to match your experimental parameters and run the main.py script.

To see an example before you edit anything, you can just run the main.py script right off the bat, and it will run based on the default parameters.

The main.py script, when run, will define all the functions and then run the simulations according to the parameters.txt file. It is set up this way, in a single file, to make it easiest for researchers who may be new to using Python. 

You may wish to remove the final few lines of main.py that execute the simulation, if you wish to define all the functions without running a simulation. For example, you may want to use the functions with data loaded from a file of previous simulation results, or you might want to create a separate script that runs the simulation. 

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

# Outputs
- Summary text file
- Results CSV file
- Plot

# Known issues and limitations
- Performing OLS fits is totally unnecessary but was done for comparison purposes. Probably adding a lot of computational time.
- This program only evaluates Gaussian function fits for distribution profiles. Could incorporate options of Lorentzian, Voigt, Green's functions, etc.
- Distributions could be non-Gaussian to begin with. Would require some thought about how to approach modeling the initial population.
- Diffusion could be anomalous and/or non-Fickian.
- Lifetime is currently only modeled as single-exponential. 

# Acknowledgements

This software could not have been written without the valuable feedback provided by
[Professor Erik M. Grumstrup in the Chemistry & Biochemistry Department at 
Montana State University](https://www.montana.edu/grumstruplab/). I also 
acknowledge the support of my colleagues Skyler Hollinbeck, Sajia Afrin, and Alex 
King, who each offered input on this project. This effort would also not have 
been possible without the kindness and patience of my spouse and our two children.

# License

Noisy Diffusion Simulator simulates diffusion under experimental parameters and evaluates 
the precision of composite fitting methods of estimating the diffusion coefficient.
Copyright (C) 2022 Joseph J. Thiebes

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
