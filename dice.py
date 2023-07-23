#########################################################################
# Noisy Diffusion Simulator models diffusion using experimental         #
# parameters and evaluates the precision of composite fitting           #
# methods of estimating the diffusion coefficient.                      #
# Copyright (C) 2022 Joseph J. Thiebes                                  #
#                                                                       #
# This program is free software: you can redistribute it and/or modify  #
# it under the terms of the GNU General Public License as published by  #
# the Free Software Foundation, either version 3 of the License, or     #
# (at your option) any later version.                                   #
#                                                                       #
# This program is distributed in the hope that it will be useful,       #
# but WITHOUT ANY WARRANTY; without even the implied warranty of        #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         #
# GNU General Public License for more details.                          #
#                                                                       #
# You should have received a copy of the GNU General Public License     #
# along with this program.  If not, see <http://www.gnu.org/licenses/>. #
#########################################################################
# See the README.md file for information about how to use this tool     #
# and to cite it in publications.                                       #
#########################################################################

import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.signal import find_peaks
# import matplotlib.pylab as pl
from numpy.random import default_rng
from scipy.optimize import curve_fit
import statsmodels.api as sm
from pint import UnitRegistry
from sigfig import round

#################################
# simple calculators and models #
#################################

# Make an x-axis array
def make_x_axis(scan_width, scan_width_pixels, psf_mu):
    x_start = psf_mu - scan_width/2
    x_end = psf_mu + scan_width/2
    x_values = np.linspace(x_start, x_end, scan_width_pixels)
    return x_values

# Calculate the time points with given start, end, and number of time points
def timeline(t_start, t_end, tixels):
    this_time = np.linspace(t_start, t_end, tixels)
    return this_time

# Convert sigma value to fwhm of a Gaussian
def sigma_to_fwhm(sigma):
    return sigma * 2 * np.sqrt(2 * np.log(2))

# Convert fwhm value to sigma of a Gaussian
def fwhm_to_sigma(fwhm):
    return fwhm / (2 * np.sqrt(2 * np.log(2)))

# Definition of a Gaussian
def gaussian(x, mu, sig, amp):
    return amp * np.exp(-1 * np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

# Calculate the integrated intensity of a pure Gaussian
def integrated_intensity(psf_sigma, psf_amp):
    return psf_amp * np.sqrt(2 * np.pi * np.power(psf_sigma,2.))

#################################################################################
# Make a distribution of noise sigmas that is uniform in reciprocal log space
def make_invexp_noise_dist(noise_low, noise_high, num):
    # get the -log10 of the reciprocal of the noise range
    noise_low_inv = np.log10(np.power(noise_low, -1.))
    noise_high_inv = np.log10(np.power(noise_high, -1.))

    # instantiate a PCG-64 pseudo-random number generator 
    # (new seed from chaotic environment)
    rng = default_rng()
    # create the uniform distribution
    noise_sigmas_inv = rng.uniform(low=noise_high_inv, high=noise_low_inv, size=num)

    # un-invert and exponentiate the values
    noise_sigmas = [np.power(10, -1 * pHsigma) for pHsigma in noise_sigmas_inv]

    return noise_sigmas

#################################################################################
# Make a distribution of noise sigmas that is uniform in reciprocal space
def make_inv_noise_dist(noise_low, noise_high, num):
    # get the -log10 of the reciprocal of the noise range
    noise_low_inv = np.power(noise_low, -1.)
    noise_high_inv = np.power(noise_high, -1.)

    # instantiate a PCG-64 pseudo-random number generator 
    # (new seed from chaotic environment)
    rng = default_rng()

    # create the uniform distribution
    noise_sigmas_inv = rng.uniform(low=noise_high_inv, high=noise_low_inv, size=num)

    # un-invert the values
    noise_sigmas = [np.power(sigma, -1) for sigma in noise_sigmas_inv]

    return noise_sigmas

#################################################################################
# Convert the slope in a MSD vs. time analysis
# to a diffusion constant with conventional units
def slope_to_diffusion_constant(slope, l_unit, t_unit, dims):
    ureg = UnitRegistry()
    speed = slope * ureg(l_unit) ** 2 / ureg(t_unit)
    dc = speed.to(ureg('centimeter squared') / ureg('second')) / (2. * dims)
    return dc

#########################################
# Data management and parsing functions #
#########################################

#################################################################################
# Get parameters from file
def open_parameters(filename):
    with open(filename) as f:
        parms_txt = f.read()
        parms_dict = ast.literal_eval(parms_txt)
        result = parameter_parser(parms_dict)
    return result

#################################################################################
# look at the parameters input by the user and set them up to pass to the iterator
def parameter_parser(parameters_dictionary):

    # most parameters are just copied in
    result_dictionary = parameters_dictionary

    num_runs = parameters_dictionary['number of runs']

    # set up the x-axis based on user selection
    wid = parameters_dictionary['spatial width']
    pix = parameters_dictionary['pixel width']
    mu = parameters_dictionary['t0 mean']
    result_dictionary['x array'] = make_x_axis(wid, pix, mu)

    # set up the time series based on user selection
    # first check to see if more than one parameter were provided, or if none were provided
    time_provided = [
        'time range' in parameters_dictionary.keys(),
        'time series' in parameters_dictionary.keys()
    ]
    if time_provided.count(True) > 1:
        print('More than one time parameter was provided. Please only provide one.')
        return
    elif time_provided.count(True) == 0:
        print('No time parameters provided.')
        return
    # if we are given a range, generate a series and get rid of the range
    elif 'time range' in parameters_dictionary.keys():
        tix_start, tix_end, tix_steps = \
            parameters_dictionary['time range'][0], \
            parameters_dictionary['time range'][1], \
            parameters_dictionary['time range'][2]
        result_dictionary['time series'] = timeline(tix_start, tix_end, tix_steps)
        del result_dictionary['time range']
    # else, we already have a series, so nothing more to do
    
    # generate noise sigma values as a series
    # first check to see if more than one were provided, or if none were provided
    noise_provided = [
        'noise range, reciprocal log' in parameters_dictionary.keys(),
        'noise range, reciprocal' in parameters_dictionary.keys(),
        'noise value' in parameters_dictionary.keys(),
        'estimate noise from data' in parameters_dictionary.keys()
    ]
    if noise_provided.count(True) > 1:
        print('More than one noise parameter was provided. Please only provide one.')
        return
    elif noise_provided.count(True) == 0:
        print('No noise parameters provided.')
        return
    elif 'noise value' in parameters_dictionary.keys():
        result_dictionary['noise series'] =  [parameters_dictionary['noise value']] * num_runs
    elif 'estimate noise from data' in parameters_dictionary.keys():
        # read the file
        t0_profile_strings = pd.read_csv(parameters_dictionary['estimate noise from data'])
        # convert to floats
        t0_profile_y = [float(y) for y in t0_profile_strings]
        # get the CNR estimate
        cnr = fft_snr_hsc([t0_profile_y])
        # calculate the noise sigma based on the CNR
        sigma_n = np.power(cnr['snrs_float'][0],-1)
        # store the result
        result_dictionary['noise series'] =  [sigma_n] * num_runs
    elif 'noise range, reciprocal log' in parameters_dictionary.keys():
        noise_low, noise_high, num = \
            parameters_dictionary['noise range, reciprocal log'][0], \
            parameters_dictionary['noise range, reciprocal log'][1], \
            num_runs
        result_dictionary['noise series'] = make_invexp_noise_dist(noise_low, noise_high, num)
    elif 'noise range, reciprocal' in parameters_dictionary.keys():
        noise_low, noise_high, num = \
            parameters_dictionary['noise range, reciprocal'][0], \
            parameters_dictionary['noise range, reciprocal'][1], \
            num_runs
        result_dictionary['noise series'] = make_inv_noise_dist(noise_low, noise_high, num)
    
    # profile width info
    # check to see if fwhm or sigma was given, or if both or neither were given
    t0width_provided = [
        't0 FWHM' in parameters_dictionary.keys(),
        't0 sigma' in parameters_dictionary.keys(),
    ]
    if t0width_provided.count(True) > 1:
        print('More than one initial profile width parameter provided. Please only provide one.')
        return
    elif t0width_provided.count(True) == 0:
        print('No profile width parameter provided.')
        return
    elif 't0 FWHM' in parameters_dictionary.keys():
        t0width = fwhm_to_sigma(parameters_dictionary['t0 FWHM'])
    else:
        t0width = parameters_dictionary['t0 sigma']

    t0amp, t0mean = parameters_dictionary['t0 amplitude'], parameters_dictionary['t0 mean']
    result_dictionary['t0 Gaussian sigma, amplitude, mean'] = [t0width, t0amp, t0mean]

    # diffusion length info
    # check to see if diff + tau were given, or if L_D was given, or if both or neither were given
    ld_provided = [
        'diffusion coefficient' in parameters_dictionary.keys(),
        'lifetime (tau)' in parameters_dictionary.keys(),
        'diffusion length' in parameters_dictionary.keys()
    ]
    if ((ld_provided[0] == True) and (ld_provided[1] == True)) ^ (ld_provided[2] == True):
        if (ld_provided[0] == True) and (ld_provided[1] == True):
            diff = parameters_dictionary['diffusion coefficient']
            tau =  parameters_dictionary['lifetime (tau)']
            result_dictionary['diffusion length'] = np.sqrt(diff * tau)
        else:
            ld = result_dictionary['diffusion length']
            parameters_dictionary['diffusion coefficient'] = np.power(ld,2)
            parameters_dictionary['lifetime (tau)'] = 1
    else:
        print('Provide nominal values for either a) the diffusion length, or b) the diffusion coefficient and lifetime.')

    return result_dictionary

#################################################################################
# print to console and append to summary file
def print_and_append(summary_filename, text):
    # print it
    print(text)
    # appending to file
    with open(summary_filename, 'a') as file1:
        file1.write(text + '\n')

##################################
# model run management functions #
##################################

#################################################################################
# Iterate through variable parameters to generate a large number of scans
def nds_runner(parameters_filename):

    # open and process the parameters
    parameters_dictionary = open_parameters(parameters_filename)

    # aliases of dictionary entries for brevity
    ##### Units #####
    l_unit = parameters_dictionary['length unit']
    t_unit= parameters_dictionary['time unit']

    ##### Number of simulation iterations #####
    numruns = parameters_dictionary['number of runs']

    ##### Spatial parameters #####
    wid = parameters_dictionary['spatial width']
    pix = parameters_dictionary['pixel width']
    x_axis = parameters_dictionary['x array']

    # time axis
    t_axis = parameters_dictionary['time series']
    tix = len(t_axis)

    # Gaussian parameters for t0
    sig, amp, mu = parameters_dictionary['t0 Gaussian sigma, amplitude, mean']
    ld = parameters_dictionary['diffusion length']
    diff = parameters_dictionary['diffusion coefficient']
    tau = parameters_dictionary['lifetime (tau)']

    # series of standard deviations of noise to be added
    noise_sigmas = parameters_dictionary['noise series']

    # do we keep all the profile data in memory or just the analysis results
    verbose_or_brief = parameters_dictionary['verbose or brief']

    # precision level
    precision_level = parameters_dictionary['precision level']

    # initialize the result dictionary
    result_dictionary = {
        'indices': {
            'time_axis': t_axis, 
            'x_axis': x_axis,
            'all_noise_sigmas': noise_sigmas,
            'total_runs': numruns
        },
        'parameters':{
            't0_sigma': sig,
            't0_amplitude': amp,
            't0_mu': mu,
            'scan_width': wid,
            'scan_pixels': pix,
            'diffusion_length': ld,
            'diffusion_coeff': diff,
            'lifetime': tau,
            'length_units': l_unit,
            'time_units': t_unit,
        },
        'run_results': {},
    }

    # aliases for subdictionaries
    indices = result_dictionary['indices']
    parameters = result_dictionary['parameters']

    # parameter text slugs for file naming
    file_prefix = parameters_dictionary['filename slug']
    ld_txt = str(round(parameters_dictionary['diffusion length'], decimals=3))
    cnr_txt = str(round(1/parameters_dictionary['noise series'][0], decimals=3))
    pix_txt =  str(parameters_dictionary['pixel width'])
    tix_txt = str(len(parameters_dictionary['time series']))
    runs_txt = str(parameters_dictionary['number of runs'])

    filename_slug = file_prefix + '_LD-' + ld_txt + '_CNR-' + cnr_txt + "_px-" + pix_txt.rjust(4, "0") + "_tx-" + tix_txt.rjust(3,"0") + '_runs-' + runs_txt
    result_dictionary['filename slug'] = filename_slug
    summary_filename = filename_slug + '_summary.txt'
    result_filename = filename_slug + '_results.csv'

    result_dictionary['image type'] = parameters_dictionary['image type']
    
    # make array of run numbers corresponding to noise sigmas for iterative simulations
    run_numbers = list(range(numruns))
    parameter_sets = zip(run_numbers, noise_sigmas)

    # create summary text file and record parameters
    print_and_append(summary_filename, 'Running ' + str(indices['total_runs']) + ' simulations with the following parameters (rounded):')
    print_and_append(summary_filename, '')
    print_and_append(summary_filename, 'Spatial width: ' + str(wid) + ' ' + l_unit)
    print_and_append(summary_filename, 'Pixel width: ' + str(pix) + ' pixels')
    print_and_append(summary_filename, 'Number of time frames: ' + str(tix) + ' frames')
    print_and_append(summary_filename, 'Initial distribution sigma: ' + str(round(sig, decimals=3)) + ' ' + l_unit)
    print_and_append(summary_filename, 'Nominal diffusion length: ' + str(round(ld, decimals=3)) + ' ' + l_unit)
    print_and_append(summary_filename, 'Nominal diffusion coefficient: ' + str(round(diff, decimals=3)) + ' ' + l_unit + '^2 per ' + t_unit )
    print_and_append(summary_filename, 'Nominal lifetime: ' + str(tau) + ' ' + t_unit)
    print_and_append(summary_filename, 'Noise standard deviation: ' + str(round(noise_sigmas[0], decimals=3)))
    print_and_append(summary_filename, 'Initial contrast-to-noise ratio (CNR): ' + str(round(1/noise_sigmas[0], decimals=3)))
    print_and_append(summary_filename, '')

    result = [scan_runner(indices, parameters, ld, diff, tau, this_noise_sigma, this_run, verbose_or_brief) for this_run, this_noise_sigma in parameter_sets]
                
    # update the result dictionary
    [result_dictionary['run_results'].update(this_result) for this_result in result]

    # Collate data from runs
    print_and_append(summary_filename, 'Model runs completed. Collating results.')
    print_and_append(summary_filename, '')
    collated_results = pd.DataFrame(
        np.asarray(
            [
            [result_dictionary['run_results'][run]['run'], 
             result_dictionary['run_results'][run]['run_parameters']['run_diff'],
             result_dictionary['run_results'][run]['run_parameters']['run_tau'],
             result_dictionary['run_results'][run]['run_parameters']['run_ld'],
             1/result_dictionary['run_results'][run]['run_parameters']['run_noise'], 
             result_dictionary['run_results'][run]['fft_snr_hsc']['snrs_float'][0],
             result_dictionary['run_results'][run]['nominal_profiles']['parameters_t']['sigmas'][0],
             result_dictionary['run_results'][run]['noisy_profile_fits']['parameter_estimates']['sigmas_t'][0],
             result_dictionary['run_results'][run]['diffusion']['ols'],
             result_dictionary['run_results'][run]['diffusion']['wls']] 
             for run in result_dictionary['run_results'].keys()
            ]),
            columns=['run number', 
                     'nominal diffusion', 'nominal lifetime', 'nominal diffusion length', 
                     'nominal CNR', 'estimated CNR', 
                     'nominal initial sigma', 'estimated initial sigma',
                     'OLS fit diffusion', 'WLS fit diffusion'])
    

    print_and_append(summary_filename, 'Converting diffusion constants to conventional units')
    print_and_append(summary_filename,'')
    # convert diffusion to constant in cm^2 s^-1
    ureg = UnitRegistry()

    true_slopes = [slope * ureg(l_unit) ** 2 / ureg(t_unit) for slope in collated_results['nominal diffusion']]
    true_dcs = [slope.to(ureg('centimeter squared') / ureg('second')) for slope in true_slopes]
    true_dcs_mag = [true_dc.magnitude for true_dc in true_dcs]

    collated_results['nominal diffusion in cm2/s'] = true_dcs_mag

    fit_wls_slopes = [slope * ureg(l_unit) ** 2 / ureg(t_unit) for slope in collated_results['WLS fit diffusion']]
    fit_wls_dcs = [slope.to(ureg('centimeter squared') / ureg('second')) / 2. for slope in fit_wls_slopes]
    fit_wls_dcs_mag = [fit_dc.magnitude for fit_dc in fit_wls_dcs]

    fit_ols_slopes = [slope * ureg(l_unit) ** 2 / ureg(t_unit) for slope in collated_results['OLS fit diffusion']]
    fit_ols_dcs = [slope.to(ureg('centimeter squared') / ureg('second')) / 2. for slope in fit_ols_slopes]
    fit_ols_dcs_mag = [fit_dc.magnitude for fit_dc in fit_ols_dcs]

    collated_results['WLS fit diffusion in cm2/s'] = fit_wls_dcs_mag
    collated_results['OLS fit diffusion in cm2/s'] = fit_ols_dcs_mag

    print_and_append(summary_filename, 'Analyzing precision')
    result_dictionary['analysis'] = estimates_precision(collated_results, precision_level)
    wls_proxpct = result_dictionary['analysis']['WLS % fits within proximity']
    ols_proxpct = result_dictionary['analysis']['OLS % fits within proximity']
    print_and_append(summary_filename,'Portion of fits where D_fit / D_nominal = 1 ± ' + str(precision_level) + ': ')
    print_and_append(summary_filename,'----- from WLS: ' + str(round(wls_proxpct, decimals=2)))
    print_and_append(summary_filename,'----- from OLS: ' + str(round(ols_proxpct, decimals=2)))
    print_and_append(summary_filename,'')

    #store collated results
    result_dictionary['collated results'] = collated_results

    # export file of collated results
    # filename includes several parameters for identification
    print_and_append(summary_filename, 'Exporting result data')
    print_and_append(summary_filename, 'Filename: ' + result_filename)
    collated_results.to_csv(result_filename, index = False)
    print_and_append(summary_filename, '')
    print_and_append(summary_filename, 'Done!')
    return result_dictionary

#################################################################################
# Create one scan, generating Gaussian PSF fits and linear diffusion fits
def scan_runner(indices, parameters, ld, this_diff, this_tau, this_noise, this_run, verbose_or_brief):
    # run a single scan and generate a comprehensive dictionary of results
    # result dictionary is produced for each scan and placed as a subdictionary for the run

    # alias for brevity
    x_axis = indices['x_axis']
    time_axis = indices['time_axis']

    # initialize the run subdictionary and store the parameters for this run
    result_dictionary = {'run_' + str(this_run): {
                            'run': this_run,
                            'run_parameters': {
                                'run_diff': this_diff,
                                'run_tau': this_tau,
                                'run_ld': ld, 
                                'run_noise': this_noise
                            }
                            }
                        }
    
    # make a dictrionary of parameters to pass for diffusion and decay generation
    diff_decay_parameters = {
        'x_axis': x_axis,
        'time_axis': time_axis,
        't0_sigma': parameters['t0_sigma'],
        't0_amplitude': parameters['t0_amplitude'],
        't0_mu': parameters['t0_mu'],
        'diff': this_diff,
        'tau': this_tau,
    }
    # generate pure Gaussian PSF profiles
    nominal_profiles = make_diffusion_decay(diff_decay_parameters)

    # add noise to the profiles
    noisy_profiles = add_noise(this_noise, nominal_profiles['y_values_tx'])

    # fit the noisy profiles
    noisy_profile_fits = gauss_fitting(x_axis, noisy_profiles['y_values_tx'])

    # get the fittted sigmas and stdevs of sigmas of the Gaussians
    gaussfit_sigmas = noisy_profile_fits['parameter_estimates']['sigmas_t']
    gaussfit_sigmas_stdevs = list(zip(
        noisy_profile_fits['parameter_estimates']['sigmas_t'], 
        noisy_profile_fits['parameter_stdevs']['sigma_stdevs']
    ))

    # calculate the weights for diffusion fitting
    # zero weight if sigma or stdev are zero
    # otherwise, weight is the reciprocal of the variance of the fitted variance, normalized
    weights = [np.power(2. * sigma * stdev, -2.) if (sigma !=0 and stdev != 0) else 0 for sigma,stdev in gaussfit_sigmas_stdevs]
    weights = [weight/np.max(weights) for weight in weights]
    
    fft_snr_hsc_dict = fft_snr_hsc(noisy_profiles['y_values_tx'])
    # snrs = fft_snr_hsc_dict['snrs_float']

    # get the OLS and WLS fits of the change in sigma^2
    diff_ols_fit = diffusion_ols_fit(time_axis, gaussfit_sigmas)
    diff_wls_fit = diffusion_wls_fit(time_axis, gaussfit_sigmas, weights)

    # delete the profile data if brief
    if verbose_or_brief == 'brief':
        del noisy_profiles['y_values_tx']
        del nominal_profiles['y_values_tx']

    result_dictionary['run_' + str(this_run)].update({'nominal_profiles': nominal_profiles, 
                              'noisy_profiles': noisy_profiles,
                              'noisy_profile_fits': noisy_profile_fits,
                              'fft_snr_hsc': fft_snr_hsc_dict,
                              'diffusion': {'ols': diff_ols_fit,
                                            'wls': diff_wls_fit,
                                            'weights': weights}
                              })

    return result_dictionary

############################################################
# functions that create Gaussians with decay and diffusion #
############################################################

# A "scan" refers to a single run of the model, which produces
# a set of Gaussian distributions with decay, diffusion, and noise.
#
# Within a given scan, the noise amount added is the same for all 
# Gaussians, but because the signal is decaying and diffusing,
# the CNR is also diminishing over the course of the scan.
#
# Any units can be used, but note that they should be the same units 
# used throughout. So for example if you use ns for the lifetime, then
# anywhere there is an input of arbitrary time units, you must use ns.
# At the outset, you will specify your units so that they will be 
# handled and converted appropriately throughout.

#################################################################################
# Make a temporal series of Gaussians with diffusion and decay
def make_diffusion_decay(parameters):

    # aliases of relevant parameters
    x_axis = parameters['x_axis']
    time_axis = parameters['time_axis']

    t0_sigma = parameters['t0_sigma']
    t0_amplitude = parameters['t0_amplitude']
    t0_mu = parameters['t0_mu']

    this_diff = parameters['diff']
    this_tau = parameters['tau']

    # initialize result dictionary for this scan
    result_dictionary = {
        'parameters_t': {
            'amplitudes': [], 'sigmas': [], 'fwhms': [], 'mus':[], 'integrated_intensities': []}
    }
    
    # calculate initial integrated intensity
    t0_ii = integrated_intensity(t0_sigma, t0_amplitude)
    
    # calculate integrated intensities with decay
    kdis = kinetic_decay_intensities(t0_ii, this_tau, time_axis)
    result_dictionary['parameters_t']['integrated_intensities'] = kdis

    # calculate sigmas with diffusion
    dsigs = diffusion_sigmas(this_diff, t0_sigma, time_axis)
    result_dictionary['parameters_t']['sigmas'] = dsigs

    # calcuilate and store fwhms with diffusion
    result_dictionary['parameters_t']['fwhms'] = [sigma_to_fwhm(this_sig) for this_sig in dsigs]

    # calculate amplitudes from intensities and sigmas
    intensity_sigs = list(zip(kdis,dsigs)) # array of decay intensities and diffusion sigmas for iterating
    amps = [intensity / np.sqrt(2. * np.pi * np.power(sigma,2)) for (intensity,sigma) in intensity_sigs]
    result_dictionary['parameters_t']['amplitudes'] = amps

    # calculate y-values of Gaussians for each time point
    times_amps_sigs = list(zip(time_axis, amps, dsigs))     # array of times, amplitudes, and sigmas for iterating
    y_values = []                                           # initialize y-values array
    for this_time, this_amp, this_sig in times_amps_sigs:               # iterate over each time point
        this_gaussian = gaussian_psf(x_axis, this_sig, t0_mu, this_amp) # make the gaussian for this time
        y_values.append(this_gaussian)                                  # store the gaussian profile

    result_dictionary.update({'y_values_tx': y_values})

    return result_dictionary

#################################################################################
# Create a Gaussian distribution:
def gaussian_psf(x_axis, sigma, mu, amp):
    y_values = gaussian(x_axis, mu, sigma, amp)
    return y_values

#################################################################################
# Calculate integrated intensities with decay
def kinetic_decay_intensities(initial_integrated_intensity, tau, t_values):
    if tau == 0:
        y_values = [initial_integrated_intensity + t * 0 for t in t_values]
    else:
        y_values = [initial_integrated_intensity * np.exp(-t / tau) for t in t_values]

    return y_values

#################################################################################
# Calculate Gaussian sigmas with diffusion
def diffusion_sigmas(diffusion_constant, psf_sigma, t_values):
    if diffusion_constant == 0: # diffusion of 0 means turn off diffusion
        y_values = [psf_sigma + t * 0. for t in t_values]
    else:
        y_values = [np.sqrt(np.power(psf_sigma,2.) + 2. * diffusion_constant * t) for t in t_values]
    return y_values
 
####################################################
# functions that add white noise to pure Gaussians #
####################################################

#################################################################################
# Add noise to pure Gaussian profiles
def add_noise(noise_sigma, nominal_profiles):
    # initialize array for resulting profiles
    noisy_profiles = []

    # generate noise and add it to the signal to make t-by-x noisy signal matrix
    for this_gaussian in nominal_profiles: # each row is a profile
        # get the width in pixels
        pixels = len(this_gaussian)

        # calculate noise and add it to the profile
        this_noise_profile = make_normal_noise(noise_sigma, pixels)            # get pseudorandom number for each pixel
        noise_gauss = list(zip(this_noise_profile, this_gaussian))             # array of noise and signal profiles
        this_noisy_profile = [noise + gauss for (noise, gauss) in noise_gauss] # add the noise to the signal
        noisy_profiles.append(this_noisy_profile)                              # append the noisy Gaussian to the result array

    noisy_signal = {'y_values_tx': noisy_profiles}
    return noisy_signal

#################################################################################
# Using a given noise standard deviation and number of pixels, make an array of 
# normally-distrributed values to be added to the pure Gaussian profiles.
def make_normal_noise(sigma, pix):
    # instantiate a PCG-64 pseudo-random number generator
    # (with new seed from chaotic environment)
    rng = default_rng()
    # get random numbers in normal distribution
    return rng.normal(loc=0, scale=sigma, size=pix)

##################################################
# functions that analyze scans with fitting etc. #
##################################################

#################################################################################
# Gaussian fitting of noisy PSFs
def gauss_fitting(x_axis, noisy_profiles):
    # this will fit each gaussian at every time point in a single run

    # initialize results dictionary
    fit_dictionary = {
        'parameter_estimates': {
            'sigmas_t': [], 
        },
        'parameter_stdevs': {
            'sigma_stdevs': [], 
        },
    } 
    
    # array of times and noisy gaussians
    profiles = list(noisy_profiles)

    for this_profile in profiles:
        # this fitting algorithm is ignorant of the input parameters
        
        ##############################################
        # Set guesses and bounds.                   
        # Handle errors so the script keeps running.
        
        # get x min, max and width
        xpix = len(x_axis)
        xmin = np.min(x_axis)
        xmax = np.max(x_axis)
        xwid = np.abs(xmax - xmin)

        # get index of max abs amp
        max_amp_idx = np.argmax(this_profile)

        # guesses
        mu0 = 0                                             # mu guess: 0
        sigma0 = xwid / 4                                   # sigma guess: 1/4 of full scan width
        a0 = this_profile[max_amp_idx]                      # amplitude guess: max value

        # set bounds of mu0 to the central fifth of the window
        fifthwidth = xwid / 5
        mu0_min = xmin + 2 * fifthwidth
        mu0_max = xmin + (3 * fifthwidth)
        if mu0_min > mu0_max:
            print('error: mu0 bounds are inverted')
            break

        # set bounds of sigma0
        sigma0_min = xwid / xpix                            # sigma minimum is 1 pixel
        sigma0_max = xwid                                   # sigma maximum is the entire window width
        if sigma0_min > sigma0_max:
            print('error: sigma0 bounds are inverted')
            break

        # set bounds of amp0
        a0_max = 2 * a0                                     # amplitude maximum is 2x the maximum y-value
        a0_min = 0                                          # amplitude minimum is 0
        if a0_min > a0_max:
            print('error: a0 bounds are inverted')
            break

        #############################################
        # Do the fit and store results.            
        parms, covars = curve_fit(
            gaussian, x_axis, this_profile, 
            p0 = [mu0, sigma0, a0],
            bounds=(
                (mu0_min, sigma0_min, a0_min),
                (mu0_max, sigma0_max, a0_max)
            ),
            maxfev=5000)

        ###################################################################
        # get the parameter estimates, covariances, variances, and stdevs
        fit_dictionary['parameter_estimates']['sigmas_t'].append(parms[1])

        coefficient_variance_table = np.diag(covars) # diagonalize the covariance table to get parameter variances
        coefficient_stdev_table = np.sqrt(coefficient_variance_table) # square root of variance is stdev
        fit_dictionary['parameter_stdevs']['sigma_stdevs'].append(coefficient_stdev_table[1])

    return fit_dictionary

#################################################################################
# Estimate the SNRs and heteroscedasticisy (hsc) for a scan
def fft_snr_hsc(input_noisy_profiles):
    # get SNR and heteroscedascticity of a run
    # input should be a matrix of time-evolved noisy profiles

    # fft_moduli = []      # moduli of fourier transforms
    # peak_indices = []    # indices of the peak maximum of the moduli
    # peak_bases = []      # indices of the first minimum of the moduli
    # signal_amps = []     # mean value of signal regime
    # noise_amps = []      # mean value of noise regime
    snrs_float = []      # SNR at each time point
    # snrs_rounded = []    # SNR rounded to nearest multiple of snr_round

    for this_profile in input_noisy_profiles:
        # fourier transform this profile
        transform = np.fft.rfft(this_profile)       # single-sided fft
        fft_modulus =  list(np.abs(transform))      # abs gives the modulus of complex values

        # prep the moduli for peak finding
        fft_modulus.insert(0,0)                           # insert a zero at the beginning so that the edge peak can be found
        neg_fft_modulus = [-1 * n for n in fft_modulus]   # negative modulus for finding minima

        peaks, _ = find_peaks(fft_modulus)          # get the peaks
        minima, _ = find_peaks(neg_fft_modulus)     # get the minima

        # corrections based on preparatory changes    
        peaks = [peak - 1 for peak in peaks]        # subtract 1 from peaks indices
        minima = [peak - 1 for peak in minima]      # subtract 1 from minima indices
        fft_modulus.pop(0)                          # remove leading 0 from modulus

        # get the index of the first peak
        first_peak_idx = peaks[0]

        # get the indices of all minima to the right of the first peak
        first_min_idx = np.min([b for b in minima if b - first_peak_idx >= 0])

        # get the peak mean from zero to the base, not including the first minimum
        signal_amp = sum(fft_modulus[0:first_min_idx]) / len(fft_modulus[0:first_min_idx])

        # get the noise mean from the first minimum (inclusive) to the end
        noise_amp = sum(fft_modulus[first_min_idx:]) / len(fft_modulus[first_min_idx:])

        # calculate and bin the snr
        snr_float = signal_amp / noise_amp
        # snr = snr_round * round(snr_float / snr_round)

        # fft_moduli.append(fft_modulus)
        # peak_indices.append(first_peak_idx)
        # peak_bases.append(first_min_idx)
        # signal_amps.append(signal_amp)
        # noise_amps.append(noise_amp)
        snrs_float.append(snr_float)
        # snrs_rounded.append(snr)
        
    # calculate heteroscedasticisy as a ratio of snrs start to finish
    # hsc_float = snrs_float[-1] / snrs_float[0]
    # hsc_rounded = hsc_round * round(hsc_float / hsc_round)
    
    result_dictionary = {
        # 'fft_moduli': fft_moduli, 
        # 'peak_indices': peak_indices, 
        # 'noise_start_indices': peak_bases,
        # 'signal_amps': signal_amps,
        # 'noise_amps': noise_amps,
        'snrs_float': snrs_float,
        # 'snrs_rounded': snrs_rounded,
        # 'hsc_float': hsc_float,
        # 'hsc_rounded': hsc_rounded
    }

    return result_dictionary

#################################################################################
# Estimate the diffusion coefficient for a scan using ordinary least squares
def diffusion_ols_fit(time_axis, gaussfit_sigmas):

    # get delta of variances of Gaussian fits
    fit_vars = np.array([np.power(sigma, 2.) for sigma in gaussfit_sigmas])
    delta_vars = np.array([var - fit_vars[0] for var in fit_vars])

    # alias time values
    times_b0 = time_axis

    # add a constant column for nonzero intercept (b1) fitting
    times_b1 = sm.add_constant(times_b0)
    
    # do the fit with zero intercept (b0)
    # ols_b0_model = sm.OLS(delta_vars, times_b0).fit()
    # ols_b0_slope = ols_b0_model.params[0]

    # do the fit with nonzero intercept (b1)
    ols_b1_model = sm.OLS(delta_vars, times_b1).fit()
    ols_b1_slope = ols_b1_model.params[1]

    # return ols_b0_slope
    return ols_b1_slope

#################################################################################
# Estimate the diffusion coefficient for a scan using weighted least squares
# Weights are square of SNR, nomralized as SNR_0 = 1
def diffusion_wls_fit(time_axis, gaussfit_sigmas, weights):

    # weight fits 
    these_weights = weights

    # get delta of variances of Gaussian fits
    fit_vars = np.array([np.power(sigma, 2.) for sigma in gaussfit_sigmas])
    delta_vars = np.array([var - fit_vars[0] for var in fit_vars])

    # get time values
    times_b0 = time_axis

    # add a constant column for nonzero intercept (b1) fitting
    times_b1 = sm.add_constant(times_b0)

    # do the fit with zero intercept (b0)
    # wls_b0_model = sm.WLS(delta_vars, times_b0, weights=these_weights).fit()
    # wls_b0_slope = wls_b0_model.params[0]
  
    # do the fit with nonzero intercept (b1)
    wls_b1_model = sm.WLS(delta_vars, times_b1, weights=these_weights).fit()
    wls_b1_slope = wls_b1_model.params[1]

    return wls_b1_slope
    # return wls_b0_slope

#################################################################################
# Get ratio of fit to nominal diffusion value
def estimates_precision(df, precision_level):

    total_sims = len(df)
    p_low = 1 - precision_level
    p_high = 1 + precision_level

    d_nom = df['nominal diffusion in cm2/s']
    d_wls = df['WLS fit diffusion in cm2/s']
    d_ols = df['OLS fit diffusion in cm2/s']

    d_wls_over_d_nom = [d_wls[i] / d_nom[i] for i in range(total_sims)]
    d_ols_over_d_nom = [d_ols[i] / d_nom[i] for i in range(total_sims)]

    wls_within = [d for d in d_wls_over_d_nom if p_low <= d <= p_high]
    ols_within = [d for d in d_ols_over_d_nom if p_low <= d <= p_high]

    wls_portion_pct = 100 * len(wls_within) / total_sims
    ols_portion_pct = 100 * len(ols_within) / total_sims

    result = {
        'WLS % fits within proximity': wls_portion_pct,
        'OLS % fits within proximity': ols_portion_pct,
    }

    return result
    
#################################################################################
# Load a file of results and analyze it

def loadfile(filename, snr_low, snr_high, precision_levels, num_bins):

    precision_levels = [round(pl * 100) for pl in precision_levels]

    # load the file as a data frame
    df_this_file = pd.read_csv(filename)
  
    # make columns of true vs estimate diffusion coefficients
    true_vs_wlsfit = np.asarray(list(zip(df_this_file['true diff cm2/s'], df_this_file['wls diff cm2/s'])))
    true_vs_olsfit = np.asarray(list(zip(df_this_file['true diff cm2/s'], df_this_file['ols diff cm2/s'])))

    # relative proximity of fit to actual diffusion constant, as percent of true value
    df_this_file['wls proximity'] = [100 * np.abs(true - fit) / true for true, fit in true_vs_wlsfit]
    df_this_file['ols proximity'] = [100 * np.abs(true - fit) / true for true, fit in true_vs_olsfit]

    # relative proximity as quotient of fit to actual diffusion constant
    df_this_file['wls D/D0'] = [fit / true for true, fit in true_vs_wlsfit]
    df_this_file['ols D/D0'] = [fit / true for true, fit in true_vs_olsfit]

    # put SNR estimates outside the range of bins in the extreme bins
    # set all snrs > 100 to 100 and snrs < 3.333334 to 3.333334
    snrs_est_groomed = []
    for snr in df_this_file['snr est']:
        if snr_low <= snr <= snr_high:
            snrs_est_groomed.append(snr)
        elif snr < snr_low:
            snrs_est_groomed.append(snr_low + snr_low/1000000)
        else:
            snrs_est_groomed.append(snr_high - snr_high/1000000)
    df_this_file['snrs_est_groomed'] = snrs_est_groomed
    
    snrs_true_groomed = []
    for snr in df_this_file['snr true']:
        if snr_low <= snr <= snr_high:
            snrs_true_groomed.append(snr)
        elif snr < snr_low:
            snrs_true_groomed.append(snr_low)
        else:
            snrs_true_groomed.append(snr_high)
    df_this_file['snrs_true_groomed'] = snrs_true_groomed

    # make SNR bins that are uniformly distributed in log scale
    snr_bins = [np.power(10, this_bin) for this_bin in np.linspace(np.log10(snr_low),np.log10(snr_high),num_bins)]

    # put the data in intervals defined by the bins
    df_this_file['snrs_est_bin'] = pd.cut(
        df_this_file['snrs_est_groomed'], 
        bins = snr_bins, 
        include_lowest=True)
    
    df_this_file['snrs_true_bin'] = pd.cut(
        df_this_file['snrs_true_groomed'], 
        bins = snr_bins, 
        include_lowest=True)

    # filtering out data that did not fit in a bin (thus gave NAN for bin)
    print('before filtering out NANs: ' + str(df_this_file.shape))
    df_est_bin_filtered = df_this_file.dropna()
    print('after filtering: ' + str(df_est_bin_filtered.shape))

    # get the mid values of the bin intervals
    df_this_file['snrs_est_bin_mid'] = [ival.mid for ival in df_this_file['snrs_est_bin']]
    df_this_file['snrs_true_bin_mid'] = [ival.mid for ival in df_this_file['snrs_true_bin']]

    # collate for precision counts
    snr_est_bin_mids_unique = np.unique(df_this_file['snrs_est_bin_mid'])
    snr_true_bin_mids_unique = np.unique(df_this_file['snrs_true_bin_mid'])
    dl_list = np.unique(df_this_file['true dl'])
    
    results_snr_est = df_this_file[['snrs_est_bin_mid', 'true dl', 'wls proximity', 'ols proximity']]
    results_snr_true = df_this_file[['snrs_true_bin_mid', 'true dl', 'wls proximity', 'ols proximity']]

    # print([results_snr_est, snr_bin_mids_unique, dl_list, precision_levels])

    # precision counts for all DLs
    precision_counts_data = {
        'snr_est precision counts': precision_counts(results_snr_est, snr_est_bin_mids_unique, dl_list, precision_levels),
        'snr_true precision counts': precision_counts(results_snr_true, snr_true_bin_mids_unique, dl_list, precision_levels)
    }

    # initialize a diffusion length dictionary
    dl_dict = {}

    # sort data into subdictionaries by diffusion length
    for this_dl in dl_list:
        # make a string to label this subdictionary
        dlstr = str(np.round(this_dl, 4))
        this_dltxt_str = 'dl = ' + dlstr

        # get the results for matching DLs
        these_dls = df_this_file[(df_this_file['true dl'] == this_dl)]

        # each subdictionary should have its own index starting with 0
        these_dls.reset_index(inplace = True)

        # collate results for estimated and true snrs
        these_dls_snr_est = these_dls[['snrs_est_bin_mid', 'true dl', 'wls proximity', 'ols proximity']]
        these_dls_snr_true = these_dls[['snrs_true_bin_mid', 'true dl', 'wls proximity', 'ols proximity']]

        # get the precision counts for these dls
        dl_dict.update({
            this_dltxt_str: {
                'data': these_dls, 
                'snr_est precision counts': precision_counts(these_dls_snr_est, snr_est_bin_mids_unique, [this_dl], precision_levels),
                'snr_true precision counts': precision_counts(these_dls_snr_true, snr_true_bin_mids_unique, [this_dl], precision_levels)
            }
        })

    # initialize result dictionary
    result = {'all': df_this_file, 'all precision counts': precision_counts_data, 'by dl': dl_dict}

    return result
 
def precision_counts(snr_dl_prox, snrs_bin_mid_unique, dls_unique, precision_levels):

    # initialize counts dictionary
    snr_dl_prox = snr_dl_prox.values.tolist()
    counts = {
        'wls counts': {},
        'ols counts': {}
    }

    # step through the precision levels 
    for pl in precision_levels:
        plstr = str(pl)     # string to label the precision level
        these_wls_counts = []   # initialize array of counts for wls fit
        these_ols_counts = []   # initialize array of counts for ols fit
        for snr in snrs_bin_mid_unique:
            for dl in dls_unique:
                wls_proxcount = 0
                ols_proxcount = 0
                snr_dl_match = 0
                for snr1, dl1, wls_prox, ols_prox in snr_dl_prox:
                    if snr1 == snr and dl1 == dl:
                        snr_dl_match += 1
                        if wls_prox <= pl:
                            wls_proxcount += 1
                        if ols_prox <= pl:
                            ols_proxcount += 1
                if snr_dl_match != 0:
                    these_wls_counts.append({'snr': snr, 'dl': dl, 'wls proximity portion': 100 * wls_proxcount/snr_dl_match})
                    these_ols_counts.append({'snr': snr, 'dl': dl, 'ols proximity portion': 100 * ols_proxcount/snr_dl_match})
                else:
                    these_wls_counts.append({'snr': snr, 'dl': dl, 'wls proximity portion': -1})
                    these_ols_counts.append({'snr': snr, 'dl': dl, 'ols proximity portion': -1})

        counts['wls counts'][plstr] = pd.DataFrame(these_wls_counts)
        counts['ols counts'][plstr] = pd.DataFrame(these_ols_counts)

    return counts

##########################
# plotting functions #
##########################

def colordefs():
    dictionary = {
    'msu_blue': '#003f7f',
    'msu_gold': '#f7941e',
    }
    return dictionary

def diffusion_plot(sim_result, filename, filetype):
    cd = colordefs()
    msu_blue = cd['msu_blue']
    msu_gold = cd['msu_gold']

    # aliases for brevity
    time_axis = sim_result['indices']['time_axis']
    total_runs = sim_result['indices']['total_runs']

    # select a random run to show the fit
    import random
    run_select = random.randint(0, total_runs-1)

    # get fitted sigma parameters and standard errors
    sigma_unc = list(zip(
        sim_result['run_results']['run_' + str(run_select)]['noisy_profile_fits']['parameter_estimates']['sigmas_t'],
        sim_result['run_results']['run_0']['noisy_profile_fits']['parameter_stdevs']['sigma_stdevs']
    ))
    # calculate relative uncertainty
    sigma_rel_err = [unc/sigma for sigma, unc in sigma_unc]
    # calculate variances of Gaussians (the variance parameter, not the uncertainty variance)
    fit_vars = np.array([np.power(sigma, 2.) for sigma in simulation_result['run_results']['run_' + str(run_select)]['noisy_profile_fits']['parameter_estimates']['sigmas_t']])
    true_vars = np.array([np.power(sigma, 2.) for sigma in simulation_result['run_results']['run_' + str(run_select)]['nominal_profiles']['parameters_t']['sigmas']])
    # collate the fit variances with the relative error of the sigma parameter
    fitvars_sigre = list(zip(fit_vars, sigma_rel_err))
    # propagate the relative error to the variance of the variance and apply multiplier for 90% confidence
    vars_abs_err = [1.64485 * 2 * sig_rel_err * var for var, sig_rel_err in fitvars_sigre]

    # get the y values based on the slopes
    y_values = np.array([var - fit_vars[0] for var in fit_vars])
    y_true = np.array([var - true_vars[0] for var in true_vars])

    # get the slope and the y-values for the selected example fit
    fit_slope0 = sim_result['run_results']['run_' + str(run_select)]['diffusion']['wls']
    y_fit0 = np.array([time * fit_slope0 for time in time_axis])

    fig, axes = plt.subplots(
        nrows=1, ncols=3, figsize=(10,6), 
        gridspec_kw={'width_ratios': [3, 3, 1]},
        sharey = 'all', constrained_layout=True)

    axes[0].errorbar(time_axis, y_values, yerr=vars_abs_err, 
                    color=msu_blue, ecolor=msu_blue, 
                    elinewidth=2, alpha = 0.5, fmt='o', zorder=1, 
                    label='_nolegend_')
    axes[0].plot(time_axis, y_fit0, color=msu_blue, linewidth=3, label='fit')
    axes[0].plot(time_axis, y_true, color=msu_gold, linewidth=3, label='nominal')

    # axes[0].set_ylim([-1,1])
    # axes[1].set_ylim(top=0.026)

    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['0', '$\\tau$'])

    axes[0].tick_params(
        left=False,
        direction='in', length=6, width=2,
        which='both', labelsize=16,
        labelleft=False)

    axes[0].legend(fontsize=16, loc='upper left') # fancybox=True,
    # axes[0].text(1, 0, '(a)', horizontalalignment='right', verticalalignment='center', fontsize=12)

    for i in range(total_runs):
        fit_slope = sim_result['run_results']['run_' + str(i)]['diffusion']['wls']
        y_fit = np.array([time * fit_slope for time in time_axis])
        axes[1].plot(time_axis, y_fit, color=msu_blue, linewidth=1, alpha = 0.1, label='_nolegend_') # 'diffusion fits' if i == 0 else '_nolegend_'

    axes[1].plot(time_axis, y_true, color=msu_gold, linewidth=3, label='_nolegend')

    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['0', '$\\tau$'])

    axes[1].tick_params(
        left=False,
        direction='in', length=6, width=2,
        which='both', labelsize=16)

    # axes[1].text(1, 0, '(b)', horizontalalignment='right', verticalalignment='center', fontsize=12)

    # number of bins for histogram of D fits
    bins = 12

    # get slopes of linear fits in given units
    d_slope = [diff for diff in sim_result['collated results']['WLS fit diffusion']]

    # get position of final value in each fitted slope
    dwls_pos = [d_slope[i] * time_axis[-1] for i in range(total_runs)]

    n, bins, patches = axes[2].hist(dwls_pos, bins=bins, density = True, color=msu_blue, orientation='horizontal')
    binspace = np.linspace(bins[0], bins[-1], 100)

    # axes[2].set_yticks([y_true_0p9, y_true_1p1])
    # axes[2].set_yticklabels([str(0.9) + '$D_0$', str(1.1) + '$D_0$'])
    axes[2].tick_params(
        left= False, bottom = False,
        direction='in', length=6, width=2,
        which='both', labelsize=16, color='w',
        labelleft = False, labelbottom = False,
        )

    # axes[2].text(np.max(n), 0, '(c)', horizontalalignment='right', verticalalignment='center', fontsize=12)
    # axes[2].axis('off')

    (mu, sigma) = norm.fit(dwls_pos)
    y = norm.pdf(binspace, mu, sigma)
    axes[2].plot(y, binspace, color = msu_gold, linewidth=3)

    # fig.supylabel('$\\sigma_t^2 - \\sigma_0^2$ (arb. units)', fontsize=12)
    axes[0].set_ylabel('$\\sigma_t^2 - \\sigma_0^2$ (arb. units)', fontsize=20)
    fig.supxlabel('t (arb. units)', fontsize=20)

    if filetype == 'none':
        pass
    else:
        fig.savefig(filename + '.' + filetype, format=filetype)

    fig.show()

###############################################################
# The following lines run the simulations and produce a plot. # 
# You could remove these lines if you prefer to define the    #
# functions without running the simulations immediately.      #
###############################################################

# run it
simulation_result = nds_runner('parameters.txt')
# plot the results
diffusion_plot(simulation_result, simulation_result['filename slug'], simulation_result['image type'])
