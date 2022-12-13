# Noisy Diffusion Estimate Simulator

This tool is designed to help gain a quantitative sense of the precision of composite fits of time-evolved Gaussian distributions that undergo decay and diffusion. After entering parameters to match your experimental setup, the program will perform simulations by:
- generating a temporal series of profiles as Gaussian distributions,
- adding normal white noise to the generated profiles,
- fitting the noisy profiles with Gaussian functions, and
- fitting the variances of the fitted Gaussian functions to a linear function.

The slope of the resulting linear fit is proportional to a diffusion coefficient estimate. This estimate is then compared to the nominal diffusion parameter used to generate the profiles to begin with.

By running many simulations for a given set of parameters, a statistical view of the precision of the fits is built up. The principal figure of merit that the tool will offer is the portion of the total number of fits that fall within an arbitrary proximity to the nominal diffusion value. e.g., what percentage of fitted diffusion coefficients are within ±10% of the nominal diffusion coefficient. 

# Parameter inputs
- Edit the parameters.txt file to set up your simulation.
- Each simulation is a model of one experimental diffusion measurement, 
  comprised of a series of time-evolved Gaussian profiles with noise.
- Each simulation will have one set of parameters. 
- A series of many simulation runs may have a range of parameters, or copies of the same parameters.

## General parameters

### Verbose or brief
You may or may not care about keeping all the profile data. If you are generating a 
large number of profiles, keeping all that data may become a performance or memory issue. 
- To keep all profile data (needed for e.g. plotting profiles), choose 'verbose'
- To discard profile data and only retain the fitting results, choose 'brief'

### Units
Provide the units that apply to all length and time parameter values.

### Number of simulation runs
Provide a number of simulations to run for each value of diffusion length.

## Simulation parameters

### Spatial parameters
Provide the spatial width and number of pixels for the simulated scan. 

### Temporal parameters
Provide information about the time axis to be used for each simulation. You can provide:
- Start, stop, and number of steps (inclusive) for an evenly-timed series of frames
- An explicit series of time values

### Initial profile parameters
Provide the amplitude, mean (i.e. center), and width of 
the initial Gaussian signal profile.
- The amplitude is usually 1 arbitrary unit, but you can change it if needed. 
  Note, however, that the noise standard deviation values are based on a normalized initial amplitude.
- The mean is usually 0, for profiles centered at the origin, but you can change it if needed.
- The width may be given as sigma (the standard deviation of the Gaussian)
  or as FWHM (full-width, half-maximum)

### Profile evolution parameters
Provide *one* of the following:
- the diffusion coefficient and lifetime, or
- the diffusion length
The script will generate diffusion length for you from the diffusion coefficient and lifetime, 
or you can enter the diffusion length yourself.

Remember to use your units specified in the above unit parameters. 
For example, if your units were entered as 
micrometers and nanoseconds, then your units for the diffusion coefficient will be 
µm$^2$ ns$^{-1}$.

### Noise parameters
Provide the standard deviation of the additive normal white noise
to be added to each pixel of each profile in a simulation. You can provide:
- High and low values for a distribution that is uniform in reciprocal log space
- High and low values for a distribution that is uniform in reciprocal space
- A single value, which will be repeatedly used for every run of the simulation

If a range is provided, the number of values generated in the range will equal 
the number of simulation iterations. Each value in the range will be used once
for each simulation.

Noisy Diffusion Estimate Simulator is © 2022, Joseph Thiebes

Noisy Diffusion Estimate Simulator is published and distributed under the Academic Software License v1.0 (ASL).

Noisy Diffusion Estimate Simulator is distributed in the hope that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the ASL for more details.

You should have received a copy of the ASL along with this program; if not, write to joseph@thiebes.org.  It is also published at [this link](https://github.com/thiebes/noisy_diffusion_estimate_simulator).

You may contact the original licensor at joseph@thiebes.org.
