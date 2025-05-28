'''
#########################################################################
# Diffusion Insight Computation Engine (DICE) simulates optical         #
# measures of diffusion in optoelectronic semiconducting materials      #
# using experimental parameters, and evaluates the precision of         #
# composite fitting methods of estimating the diffusion coefficient.    #
#                                                                       #
# Copyright (C) 2023-2024 Joseph J. Thiebes                             #
#                                                                       #
# This software may be cited as follows:                                #
# Joseph J. Thiebes. (2024). DICE. Zenodo. DOI:10.5281/zenodo.10258192  #
#                                                                       #
# The concepts and methodologies underpinning this software were        #
# developed concurrently with the research findings presented in the    #
# paper referenced below. We strongly encourage users to consult the    #
# paper to gain comprehensive insights into the scientific and          #
# statistical principles that inform the functionality and application  #
# of this software.                                                     #
#                                                                       #
# Joseph J. Thiebes, Erik M. Grumstrup; Quantifying noise effects in    #
# optical measures of excited state transport. J. Chem. Phys. 28 March  #
# 2024; 160 (12): 124201. https://doi.org/10.1063/5.0190347             #
#                                                                       #
# This material is based upon work supported by the National Science    #
# Foundation under Grant No. 2154448. Any opinions, findings, and       #
# conclusions or recommendations expressed in this material are those   #
# of the author(s) and do not necessarily reflect the views of the      #
# National Science Foundation.                                          #
#                                                                       #
# This work is licensed under the Creative Commons Attribution 4.0      #
# International License. To view a copy of this license, visit          #
# http://creativecommons.org/licenses/by/4.0/ or send a letter to       #
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.          #
#                                                                       #
# This program is distributed in the hope that it will be useful,       #
# but WITHOUT ANY WARRANTY; without even the implied warranty of        #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                  #
#                                                                       #
# You should include a copy of the license or a link to it with         #
# every copy of the work you distribute. You can do this by             #
# including a link to the license in your README.md file or             #
# documentation.                                                        #
#########################################################################
# See the README.md file for information about how to use this tool.    #
#########################################################################
'''

import os
import re

# Library imports
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from dice.utils import (
    print_and_append,      slope_to_diffusion_constant,       make_diffusion_decay,
    add_noise,             gauss_fitting,                     fft_cnr,
    diffusion_ols_fit,     diffusion_wls_fit,                 slope_to_diffusion_constant,
)
from dice.parameters import open_parameters
from dice.analysis import (
    estimates_precision, 
)
from dice.reporting import (
    plot_accuracy_histogram,        summarize_results,        export_results,
)

def run_simulation(parameters_dictionary, progress_callback=None, message_callback=None):
    """
    Run DICE simulations using parsed parameters, with optional GUI callbacks.

    Parameters:
    - parameters_dictionary (dict): Dictionary containing all simulation parameters.
    - progress_callback (callable, optional): Function to call for progress updates.
                                              Expected signature: progress_callback(completed_runs, total_runs).
    - message_callback (callable, optional): Function to call for message/log updates.
                                             Expected signature: message_callback(message_string).
    """

    # Extract parameters
    image_type = parameters_dictionary['image type']
    l_unit = parameters_dictionary['length unit']
    t_unit = parameters_dictionary['time unit']
    numruns = parameters_dictionary['number of runs']
    wid = parameters_dictionary['spatial width']
    pix = parameters_dictionary['pixel width']
    x_axis = parameters_dictionary['x array']
    t_axis = parameters_dictionary['time series']
    tix = len(t_axis)
    sig2_0, amp_0, mu_0 = parameters_dictionary['t0 Gaussian sigma^2, amplitude, mean']
    ld = parameters_dictionary['nominal diffusion length']
    diff = parameters_dictionary['nominal diffusion coefficient']
    tau = parameters_dictionary['nominal lifetime (tau)']
    noise_series = parameters_dictionary['noise series']
    noise_num = len(noise_series)
    runs_total = numruns * noise_num
    retain_profile_data = parameters_dictionary['retain profile data']
    proximity_level = parameters_dictionary['proximity level']

    result_dictionary = {
        'indices': {
            'time axis': t_axis,
            'x axis': x_axis,
            'noise sigmas': noise_series,
            'total runs': runs_total
        },
        'parameters': {
            'sigma^2_0': sig2_0,
            'amplitude_0': amp_0,
            'mu_0': mu_0,
            'scan width': wid,
            'scan pixels': pix,
            'nominal diffusion length': ld,
            'nominal diffusion coeff': diff,
            'nominal lifetime': tau,
            'length units': l_unit,
            'time units': t_unit,
            'proximity level': proximity_level,
            'image width': parameters_dictionary['image width'],
            'image height': parameters_dictionary['image height'],
            'image dpi': parameters_dictionary['image dpi'],
            'image font size': parameters_dictionary['image font size'],
            'image tick length': parameters_dictionary['image tick length'],
            'image tick width': parameters_dictionary['image tick width'],
            'image numbins': parameters_dictionary['image numbins'],
            'image x_lim': parameters_dictionary['image x_lim'],
        },
        'run results': {},
    }

    # Aliases
    indices = result_dictionary['indices']
    parameters = result_dictionary['parameters']

    # Build filename slug and attach to result dict
    file_prefix = parameters_dictionary['filename slug']
    ld_txt = str(round(ld, 3))
    cnr_txt = str(round(1 / noise_series[0], 3))
    pix_txt = str(pix)
    tix_txt = str(tix)
    runs_txt = str(numruns)
    filename_slug = f"{file_prefix}_LD-{ld_txt}_CNR-{cnr_txt}_px-{pix_txt}_tx-{tix_txt}_runs-{runs_txt}"
    result_dictionary['filename slug'] = filename_slug

    summary_filename = f"{filename_slug}_summary.txt"
    result_filename = f"{filename_slug}_results.csv"
    image_filename = f"{filename_slug}_histogram.{image_type}"

    result_dictionary['parameters']['summary filename'] = summary_filename
    result_dictionary['parameters']['result filename'] = result_filename
    result_dictionary['parameters']['image type'] = image_type
    result_dictionary['parameters']['image filename'] = image_filename

    # Clear existing summary file for this run
    if os.path.exists(summary_filename):
        try:
            os.remove(summary_filename)
        except OSError as e:
            # Use print_and_append to ensure the message goes to console and potentially callback
            error_message = f"Error: Could not remove existing summary file {summary_filename}: {e}"
            print_and_append(summary_filename, error_message, gui_message_callback=message_callback)


    # Create simulation parameter sets
    run_numbers = list(range(runs_total))
    noise_list = np.concatenate([np.repeat(noise, numruns) for noise in noise_series])
    parameter_sets = zip(run_numbers, noise_list)

    # Run simulations
    collected_results_list = []
    if parameters_dictionary['multiprocessing']:
        if progress_callback:
            try:
                progress_callback(0, runs_total) # Before starting
            except Exception as e:
                print(f"Error in progress_callback (pre-parallel): {e}")

        collected_results_list = Parallel(n_jobs=-1)(
            delayed(scan_runner)(indices, parameters, ld, diff, tau, this_noise_sigma, this_run, retain_profile_data)
            for this_run, this_noise_sigma in parameter_sets
        )
        if progress_callback:
            try:
                progress_callback(runs_total, runs_total) # After completion
            except Exception as e:
                print(f"Error in progress_callback (post-parallel): {e}")
    else:
        for i, (this_run, this_noise_sigma) in enumerate(parameter_sets):
            # scan_runner returns a dictionary like {'run_0': {...}}, we want the inner dict
            scan_result_dict_outer = scan_runner(indices, parameters, ld, diff, tau, this_noise_sigma, this_run, retain_profile_data)
            collected_results_list.append(scan_result_dict_outer)
            if progress_callback:
                try:
                    progress_callback(i + 1, runs_total)
                except Exception as e:
                    print(f"Error in progress_callback (sequential): {e}") # Avoid callback errors stopping simulation
    
    # Store scan results
    for res_dict_outer in collected_results_list:
        result_dictionary['run results'].update(res_dict_outer)

    # Collate results
    collated_results = pd.DataFrame([
        [
            run_data['run'],
            run_data['run parameters']['nominal diffusion coefficient'],
            run_data['run parameters']['nominal lifetime'],
            run_data['run parameters']['nominal diffusion length'],
            1 / run_data['run parameters']['noise stdev'],
            run_data['cnr_0 estimate'],
            run_data['nominal profiles']['parameters_t']['sigma^2_t'][0],
            run_data['noisy profile fits']['sigma^2_t estimates'][0],
            run_data['diffusion']['unweighted fit']['MSD_t slope estimate'],
            run_data['diffusion']['unweighted fit']['MSD_t slope std error'],
            run_data['diffusion']['unweighted fit']['intercept estimate'],
            run_data['diffusion']['unweighted fit']['intercept standard error'],
            run_data['diffusion']['weighted fit']['MSD_t slope estimate'],
            run_data['diffusion']['weighted fit']['MSD_t slope std error'],
            run_data['diffusion']['weighted fit']['intercept estimate'],
            run_data['diffusion']['weighted fit']['intercept standard error'],
        ]
        for run_data in result_dictionary['run results'].values()
    ], columns=[
        'run number',
        'nominal diffusion coeff', 'nominal lifetime', 'nominal diffusion length',
        'nominal CNR', 'estimated CNR',
        'nominal sigma^2_0', 'estimated sigma^2_0',
        'unweighted fit diffusion slope', 'unweighted fit diffusion slope stderr',
        'unweighted fit intercept', 'unweighted fit intercept stderr',
        'weighted fit diffusion slope', 'weighted fit diffusion slope stderr',
        'weighted fit intercept', 'weighted fit intercept stderr',
    ])

    # If multiple time points, compute derived diffusion constants and precision stats
    if len(t_axis) > 1:
        fit_wls_slopes = collated_results['weighted fit diffusion slope']
        fit_ols_slopes = collated_results['unweighted fit diffusion slope']
        fit_wls_stderr = collated_results['weighted fit diffusion slope stderr']
        fit_ols_stderr = collated_results['unweighted fit diffusion slope stderr']

        nom_slopes = [d * 2 for d in collated_results['nominal diffusion coeff']]
        collated_results['nominal diffusion coeff [cm^2/s]'] = [
            slope_to_diffusion_constant(s, l_unit, t_unit) for s in nom_slopes
        ]
        collated_results['unweighted fit diffusion coeff [cm^2/s]'] = [
            slope_to_diffusion_constant(s, l_unit, t_unit) for s in fit_ols_slopes
        ]
        collated_results['unweighted fit diffusion stderr [cm^2/s]'] = [
            slope_to_diffusion_constant(s, l_unit, t_unit) for s in fit_ols_stderr
        ]
        collated_results['weighted fit diffusion coeff [cm^2/s]'] = [
            slope_to_diffusion_constant(s, l_unit, t_unit) for s in fit_wls_slopes
        ]
        collated_results['weighted fit diffusion stderr [cm^2/s]'] = [
            slope_to_diffusion_constant(s, l_unit, t_unit) for s in fit_wls_stderr
        ]

        result_dictionary['analysis'] = estimates_precision(collated_results, proximity_level)

    result_dictionary['collated results'] = collated_results
    return result_dictionary

def run_simulation_cli(parameters_filename: str):
    """
    Command Line Interfce (CLI)-compatible wrapper: 
    reads parameter file, runs simulation, saves output.
    """

    # Load parameter dictionary
    parameters_dict = open_parameters(parameters_filename)

    # Run the simulation
    result = run_simulation(parameters_dict)

    # Get summary lines for print and save
    # Ensure result_dictionary is complete before summarizing
    summary_lines = summarize_results(result_dictionary)
    summary_file = result_dictionary['parameters']['summary filename']
    for line in summary_lines:
        print_and_append(summary_file, line, gui_message_callback=message_callback)

    # Write CSV + histogram image
    # TODO: Consider if export_results also needs message_callback for its print statements.
    # For now, its print statements will go to console only.
    export_results(result_dictionary)

    return result_dictionary

def dice_runner(parameters_filename):
    '''Generate simulations using parameters'''
    """
    Execute a series of simulations based on parameters provided in a file.

    This function reads simulation parameters from a file, initializes the simulation
    environment, and executes multiple simulation runs. It collates results from 
    individual simulations, performs analysis, and exports summary data.

    Parameters:
    - parameters_filename (str): Path to a text file containing simulation parameters.

    Returns:
    - result_dictionary (dict): A dictionary containing indices, parameters, run results,
                                collated results, analysis, and a filename slug for output.

    Raises:
    - FileNotFoundError: If the parameters file does not exist or is unreadable.
    - KeyError: If the expected keys are not present in the parameters file.
    - ValueError: If parameter values are not of the expected type or out of expected range.
    - IOError: If there is an error during file writing operations.

    Note:
    - This function uses multiprocessing if enabled in the parameters to speed up simulations.
    """

    # open and process the parameters
    parameters_dictionary = open_parameters(parameters_filename)

    # image parameters
    image_type = parameters_dictionary['image type']
    image_width = parameters_dictionary['image width']
    image_height = parameters_dictionary['image height']
    image_dpi = parameters_dictionary['image dpi']
    image_font_size = parameters_dictionary['image font size']
    image_tick_l = parameters_dictionary['image tick length']
    image_tick_w = parameters_dictionary['image tick width']
    image_numbins = parameters_dictionary['image numbins']
    image_x_lim = parameters_dictionary['image x_lim']

    # Units
    l_unit = parameters_dictionary['length unit']
    t_unit= parameters_dictionary['time unit']

    # Number of simulation runs per parameter set
    numruns = parameters_dictionary['number of runs']

    # Spatial parameters
    wid = parameters_dictionary['spatial width']
    pix = parameters_dictionary['pixel width']
    x_axis = parameters_dictionary['x array']

    # time axis
    t_axis = parameters_dictionary['time series']
    tix = len(t_axis)

    # Gaussian parameters for t0
    sig2_0, amp_0, mu_0 = parameters_dictionary['t0 Gaussian sigma^2, amplitude, mean']
    ld = parameters_dictionary['nominal diffusion length']
    diff = parameters_dictionary['nominal diffusion coefficient']
    tau = parameters_dictionary['nominal lifetime (tau)']

    # series of standard deviations of noise to be added
    noise_series = parameters_dictionary['noise series']
    noise_num = len(noise_series)

    # calculate the total number of runs
    runs_total = numruns * noise_num

    # do we keep all the profile data in memory or just the analysis results
    retain_profile_data = parameters_dictionary['retain profile data']

    # proximity level
    proximity_level = parameters_dictionary['proximity level']

    # initialize the result dictionary
    result_dictionary = {
        'indices': {
            'time axis': t_axis, 
            'x axis': x_axis,
            'noise sigmas': noise_series,
            'total runs': runs_total
        },
        'parameters':{
            'sigma^2_0': sig2_0,
            'amplitude_0': amp_0,
            'mu_0': mu_0,
            'scan width': wid,
            'scan pixels': pix,
            'nominal diffusion length': ld,
            'nominal diffusion coeff': diff,
            'nominal lifetime': tau,
            'length units': l_unit,
            'time units': t_unit,
        },
        'run results': {},
    }

    # aliases for subdictionaries
    indices = result_dictionary['indices']
    parameters = result_dictionary['parameters']

    # parameter text slugs for file naming
    file_prefix = parameters_dictionary['filename slug']
    ld_txt = str(round(parameters_dictionary['nominal diffusion length'], 3))
    cnr_txt = str(round(1/parameters_dictionary['noise series'][0], 3))
    pix_txt =  str(parameters_dictionary['pixel width'])
    tix_txt = str(len(parameters_dictionary['time series']))
    runs_txt = str(parameters_dictionary['number of runs'])

    # the filename incorporates user-specified text along with several parameters for reference
    filename_slug = file_prefix + '_LD-' + ld_txt + '_CNR-' + cnr_txt + "_px-" + pix_txt + "_tx-" + tix_txt + '_runs-' + runs_txt
    result_dictionary['filename slug'] = filename_slug
    
    summary_filename = filename_slug + '_summary.txt'
    result_dictionary['parameters']['summary filename'] = summary_filename
    
    result_filename = filename_slug + '_results.csv'
    result_dictionary['parameters']['result filename'] = result_filename

    image_type = parameters_dictionary['image type']
    image_filename = filename_slug + '_histogram.' + image_type
    result_dictionary['parameters']['image type'] = image_type
    
    # Make array of parameters for iterative simulations, indexed by run number.
    # For future use handling multiple noise values.
    run_numbers = list(range(runs_total))
    noise_list = np.concatenate([np.repeat(noise, numruns) for noise in noise_series])
    parameter_sets = zip(run_numbers, noise_list)
    
    # create summary text file and record parameters
    print_and_append(summary_filename, 'Running ' + str(indices['total runs']) + ' simulations with the following parameters (rounded):')
    print_and_append(summary_filename, '')
    print_and_append(summary_filename, 'Spatial width: ' + str(wid) + ' ' + l_unit)
    print_and_append(summary_filename, 'Pixel width: ' + str(pix) + ' pixels')
    print_and_append(summary_filename, 'Number of time frames: ' + str(tix) + ' frames')
    print_and_append(summary_filename, 'Noise stdev: ' + str(round(noise_series[0], 3)))
    print_and_append(summary_filename, 'Initial contrast-to-noise ratio (CNR): ' + str(round(1/noise_series[0], 3)))
    print_and_append(summary_filename, 'Initial profile sigma^2: ' + str(round(sig2_0, 3)) + ' ' + l_unit + '^2')
    print_and_append(summary_filename, 'Nominal diffusion length: ' + str(round(ld, 3)) + ' ' + l_unit)
    print_and_append(summary_filename, 'Nominal diffusion coeff: ' + str(round(diff, 5)) + ' ' + l_unit + '^2 per ' + t_unit )
    print_and_append(summary_filename, 'Nominal lifetime: ' + str(tau) + ' ' + t_unit)
    print_and_append(summary_filename, '')

    if parameters_dictionary['multiprocessing'] == 1:
        print_and_append(summary_filename, 'Starting simulation with multiprocessing')
        print_and_append(summary_filename, '')
        result = Parallel(n_jobs=-1)(delayed(scan_runner)(indices, parameters, ld, diff, tau, this_noise_sigma, this_run, retain_profile_data) for this_run, this_noise_sigma in parameter_sets)
    else:
        print_and_append(summary_filename, 'Starting simulation without multiprocessing')
        print_and_append(summary_filename, '')
        result = [scan_runner(indices, parameters, ld, diff, tau, this_noise_sigma, this_run, retain_profile_data) for this_run, this_noise_sigma in parameter_sets]
 
    # update the result dictionary
    [result_dictionary['run results'].update(this_result) for this_result in result]

    # Collate data from runs
    print_and_append(summary_filename, 'Simulation completed. Collating results.')
    print_and_append(summary_filename, '')
    collated_results = pd.DataFrame(
        np.asarray(
            [
            [result_dictionary['run results'][run]['run'], 
            result_dictionary['run results'][run]['run parameters']['nominal diffusion coefficient'],
            result_dictionary['run results'][run]['run parameters']['nominal lifetime'],
            result_dictionary['run results'][run]['run parameters']['nominal diffusion length'],
            1/result_dictionary['run results'][run]['run parameters']['noise stdev'], 
            result_dictionary['run results'][run]['cnr_0 estimate'],
            result_dictionary['run results'][run]['nominal profiles']['parameters_t']['sigma^2_t'][0],
            result_dictionary['run results'][run]['noisy profile fits']['sigma^2_t estimates'][0],
            result_dictionary['run results'][run]['diffusion']['unweighted fit']['MSD_t slope estimate'],
            result_dictionary['run results'][run]['diffusion']['unweighted fit']['MSD_t slope std error'],
            result_dictionary['run results'][run]['diffusion']['unweighted fit']['intercept estimate'],
            result_dictionary['run results'][run]['diffusion']['unweighted fit']['intercept standard error'],
            result_dictionary['run results'][run]['diffusion']['weighted fit']['MSD_t slope estimate'],
            result_dictionary['run results'][run]['diffusion']['weighted fit']['MSD_t slope std error'],
            result_dictionary['run results'][run]['diffusion']['weighted fit']['intercept estimate'],
            result_dictionary['run results'][run]['diffusion']['weighted fit']['intercept standard error'],
            ] 
            for run in result_dictionary['run results'].keys()
            ]),
            columns=['run number', 
                    'nominal diffusion coeff', 'nominal lifetime', 'nominal diffusion length', 
                    'nominal CNR', 'estimated CNR', 
                    'nominal sigma^2_0', 'estimated sigma^2_0',
                    'unweighted fit diffusion slope', 'unweighted fit diffusion slope stderr', 
                    'unweighted fit intercept', 'unweighted fit intercept stderr', 
                    'weighted fit diffusion slope', 'weighted fit diffusion slope stderr',
                    'weighted fit intercept', 'weighted fit intercept stderr', 
                    ]
        )

    if len(t_axis) > 1:
        print_and_append(summary_filename, 'Converting diffusion constants to conventional units')
        print_and_append(summary_filename,'')

        # get the fitted slopes with user defined units
        fit_wls_slopes = [slope for slope in collated_results['weighted fit diffusion slope']]
        fit_wls_slope_stderrs = [stderr for stderr in collated_results['weighted fit diffusion slope stderr']]
        fit_ols_slopes = [slope for slope in collated_results['unweighted fit diffusion slope']]
        fit_ols_slope_stderrs = [stderr for stderr in collated_results['unweighted fit diffusion slope stderr']]

        # convert the nominal diffusion coefficient to a MSD(t) slope, and then
        # convert to diffusion constant in conventional units of cm^2 s^-1
        nom_slopes = [diff * 2. for diff in collated_results['nominal diffusion coeff']]
        collated_results['nominal diffusion coeff [cm^2/s]'] = [
            slope_to_diffusion_constant(slope, l_unit, t_unit) for slope in nom_slopes
        ]
        # convert fitted slopes and std errors to cm^2/s
        collated_results['unweighted fit diffusion coeff [cm^2/s]'] = [
            slope_to_diffusion_constant(slope, l_unit, t_unit) for slope in fit_ols_slopes
        ]
        collated_results['unweighted fit diffusion stderr [cm^2/s]'] = [
            slope_to_diffusion_constant(slope, l_unit, t_unit) for slope in fit_ols_slope_stderrs
        ]
        collated_results['weighted fit diffusion coeff [cm^2/s]'] = [
            slope_to_diffusion_constant(slope, l_unit, t_unit) for slope in fit_wls_slopes
        ]
        collated_results['weighted fit diffusion stderr [cm^2/s]'] = [
            slope_to_diffusion_constant(stderr, l_unit, t_unit) for stderr in fit_wls_slope_stderrs
        ]

        print_and_append(summary_filename, 'Analyzing precision and accuracy')
        print_and_append(summary_filename,'')
        result_dictionary['analysis'] = estimates_precision(collated_results, proximity_level)
        ols_proxpct = result_dictionary['analysis']['% fits within proximity']['unweighted fit']
        wls_proxpct = result_dictionary['analysis']['% fits within proximity']['weighted fit']
        print_and_append(summary_filename,'Portion of fits where D_estimate / D_nominal = 1 ± ' + str(proximity_level) + ':')
        print_and_append(summary_filename, '-- Unweighted fit: ' + str(round(ols_proxpct, 2)))
        print_and_append(summary_filename, '-- Weighted fit: ' + str(round(wls_proxpct, 2)))
        print_and_append(summary_filename,'')

    else:
        print_and_append(summary_filename, 'No diffusion estimates: only one time frame')

    #store collated results
    result_dictionary['collated results'] = collated_results

    # export file of collated results
    # filename includes several parameters for identification
    print_and_append(summary_filename, 'Exporting result data and histogram.')

    collated_results.to_csv(result_filename, index = False)
    plot_accuracy_histogram(
        result_dictionary,
        proximity = proximity_level,
        filename = image_filename,
        image_type = image_type,
        width = image_width,
        height = image_height,
        dpi = image_dpi,
        font_size = image_font_size,
        tick_length = image_tick_l,
        tick_width = image_tick_w,
        num_bins = image_numbins,
        x_lim = image_x_lim
    )

    print_and_append(summary_filename, '-- Summary file: ' + summary_filename)
    print_and_append(summary_filename, '-- Collated CSV file: ' + result_filename)
    print_and_append(summary_filename, '-- Histogram image file ' + image_filename)
    print_and_append(summary_filename,'')

    print_and_append(summary_filename, 'Done!')
    return result_dictionary

def scan_runner(indices, parameters, ld, this_diff, this_tau, this_noise, this_run, retain_profile_data):
    '''
    #################################################################################
    # Create one temporal series of Gaussian profiles, 
    # add noise to each profile in the series,
    # perform Gaussian fits and extract the sigma^2 parameter and its stderr,
    # and perform a linear fit for the series of fitted sigma^2 parameters.
    '''
    """
    Execute a simulation scan and generate a dictionary of results including Gaussian sigma^2 fits
    and linear MSD(t) fits for a given set of parameters.

    A "scan" or "run" refers to a single run of the model, which produces a 
    temporally evolved series of Gaussian distributions with decay, diffusion, and noise.

    Parameters:
    - indices (dict): A dictionary containing 'x axis' and 'time axis' as keys with corresponding values.
    - parameters (dict): A dictionary with the initial simulation parameters such as 't0_sigma^2', 't0_amplitude', etc.
    - ld (float): Nominal diffusion length for the current run.
    - this_diff (float): Nominal diffusion coefficient for the current run.
    - this_tau (float): Nominal lifetime for the current run.
    - this_noise (float): Stdev of normally-distributed noise to be added to the simulated profiles.
    - this_run (int): Identifier for the current simulation run.
    - retain_profile_data (bool): Flag indicating whether to retain profile data in the results.

    Returns:
    - result_dictionary (dict): A comprehensive dictionary containing the results of the simulation
                                run including profile fits, CNR estimates, and diffusion fits.

    Raises:
    - ValueError: If input parameters are out of the expected ranges or in incorrect formats.
    - RuntimeError: If the computation fails due to external library errors or internal logic errors.

    Notes:
    - The function is part of a larger simulation suite and expects specific input formats.
    - If 'retain_profile_data' is False, profile data will be removed from the result to save memory.
    """
    # run a single scan and generate a comprehensive dictionary of results
    # result dictionary is produced for each scan and placed as a subdictionary for the run

    # alias for brevity
    x_axis = indices['x axis']
    time_axis = indices['time axis']

    # initialize the run subdictionary and store the parameters for this run
    result_dictionary = {'run_' + str(this_run): {
                            'run': this_run,
                            'run parameters': {
                                'nominal diffusion coefficient': this_diff,
                                'nominal lifetime': this_tau,
                                'nominal diffusion length': ld, 
                                'noise stdev': this_noise
                            }
                            }
                        }
    
    # make a dictionary of parameters to pass for diffusion and decay generation
    diff_decay_parameters = {
        'x axis': x_axis,
        'time axis': time_axis,
        'sigma^2_0': parameters['sigma^2_0'],
        'amplitude_0': parameters['amplitude_0'],
        'mu_0': parameters['mu_0'],
        'nominal diffusion coefficient': this_diff,
        'nominal lifetime': this_tau,
    }
    # generate pure Gaussian PSF profiles
    nominal_profiles = make_diffusion_decay(diff_decay_parameters)

    # add noise to the profiles
    noisy_profiles = add_noise(this_noise, nominal_profiles['y_values_t'])
    # estimate the CNR at t0
    CNR_0_est = fft_cnr(noisy_profiles['y_values_t'][0])

    # fit the noisy profiles
    noisy_profile_fits = gauss_fitting(x_axis, noisy_profiles['y_values_t'])

    # get the fittted sigmas and stderrs of sigmas of the Gaussians
    gaussfit_sigma2_t = noisy_profile_fits['sigma^2_t estimates']
    gaussfit_sigma2_t_stderrs = list(zip(
        noisy_profile_fits['sigma^2_t estimates'], 
        noisy_profile_fits['sigma^2_t standard errors']
    ))

    if len(time_axis) > 1:
        # calculate the weights for diffusion fitting
        # -- if sigma or stdev are zero, that is bad, so give zero weight
        # -- otherwise, weight is calculated as:
        # -- normalized reciprocal of the relative variance of the fitted MSD
        weights = [np.power(stderr / sigma2, -2.) if (sigma2 !=0 and stderr != 0) else 0 for sigma2,stderr in gaussfit_sigma2_t_stderrs]
        # normalize so the sum of the weights is unity
        weights = [weight/np.sum(weights) for weight in weights]

        # get the unweighted (OLS) and weighted (WLS) fits of the MSD (change in sigma^2)
        ols_fit = diffusion_ols_fit(time_axis, gaussfit_sigma2_t)
        ols_slope_est = ols_fit['MSD_t slope estimate']
        ols_intercept_est = ols_fit['intercept estimate']
        ols_slope_stderr = ols_fit['MSD_t slope std error']
        ols_intercept_stderr = ols_fit['intercept standard error']

        wls_fit = diffusion_wls_fit(time_axis, gaussfit_sigma2_t, weights)
        wls_slope_est = wls_fit['MSD_t slope estimate']
        wls_intercept_est = wls_fit['intercept estimate']
        wls_slope_stderr = wls_fit['MSD_t slope std error']
        wls_intercept_stderr = wls_fit['intercept standard error']

    else:
        weights = 'N/A'
        weights = 'N/A'
        ols_fit = 'N/A'
        ols_slope_est = 'N/A'
        ols_intercept_est = 'N/A'
        ols_slope_stderr = 'N/A'
        ols_intercept_stderr = 'N/A'

        wls_fit = 'N/A'
        wls_slope_est = 'N/A'
        wls_intercept_est = 'N/A'
        wls_slope_stderr = 'N/A'
        wls_intercept_stderr = 'N/A'

    # delete the profile data if not retaining
    if retain_profile_data != 1:
        del noisy_profiles['y_values_t']
        del nominal_profiles['y_values_t']

    result_dictionary['run_' + str(this_run)].update({
        'nominal profiles': nominal_profiles, 
        'noisy profiles': noisy_profiles,
        'noisy profile fits': noisy_profile_fits,
        'cnr_0 estimate': CNR_0_est,
        'diffusion': {
            'unweighted fit': {
                'MSD_t slope estimate': ols_slope_est,
                'MSD_t slope std error': ols_slope_stderr,
                'intercept estimate': ols_intercept_est,
                'intercept standard error': ols_intercept_stderr,
                },
            'weighted fit': {
                'MSD_t slope estimate': wls_slope_est,
                'MSD_t slope std error': wls_slope_stderr,
                'intercept estimate': wls_intercept_est,
                'intercept standard error': wls_intercept_stderr,
                'weights': weights,
                },
            }
        })
    return result_dictionary
