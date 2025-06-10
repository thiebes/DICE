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

# Library imports
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from typing import Any, Dict, List, Tuple # Keep for type hints if used by runners

# Imports from the new local modules
from .utils import (open_parameters, print_and_append, slope_to_diffusion_constant,
                    fft_cnr, parameter_parser) 
                    # make_x_axis, make_time_axis, sigma_to_fwhm, fwhm_to_sigma2 are not directly used by runners
from .profiles import make_diffusion_decay, add_noise 
                    # gaussian, integrated_intensity, kinetic_decay_intensities, diffusion_sigma2_t are not directly used by runners
from .fitting import gauss_fitting, diffusion_ols_fit, diffusion_wls_fit
from .analysis import estimates_precision 
                    # load_files, precision_counts are not directly used by runners
from .plotting import plot_accuracy_histogram 
                    # colordefs is not directly used by runners

# Standard Python libraries (if still needed directly by dice_runner/scan_runner)
# import ast # No longer needed directly, handled by utils.open_parameters
# import os # No longer needed directly
# import re # No longer needed directly


##################################
# model run management functions #
##################################

###########################################################
# Generate simulations using parameters
def dice_runner(parameters_filename: str) -> Dict[str, Any]:
    """
    Execute a series of simulations based on parameters provided in a file.
    (Docstring adapted from original)
    """

    parameters_dictionary = open_parameters(parameters_filename)

    image_type = parameters_dictionary['image type']
    image_width = parameters_dictionary['image width']
    image_height = parameters_dictionary['image height']
    image_dpi = parameters_dictionary['image dpi']
    image_font_size = parameters_dictionary['image font size']
    image_tick_l = parameters_dictionary['image tick length']
    image_tick_w = parameters_dictionary['image tick width']
    image_numbins = parameters_dictionary['image numbins']
    image_x_lim = parameters_dictionary.get('image x_lim', None) # Use .get for optional param

    l_unit = parameters_dictionary['length unit']
    t_unit= parameters_dictionary['time unit']
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
    retain_profile_data = parameters_dictionary.get('retain profile data', False) # Default to False
    proximity_level = parameters_dictionary.get('proximity level', 0.1) # Default to 0.1

    result_dictionary: Dict[str, Any] = {
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
        'collated results': pd.DataFrame(), # Initialize as empty DataFrame
        'analysis': {}, # Initialize as empty dict
    }

    indices = result_dictionary['indices']
    parameters = result_dictionary['parameters']

    file_prefix = parameters_dictionary.get('filename slug', 'dice_sim') # Default slug
    ld_txt = str(round(ld, 3))
    # Handle case where noise_series might be empty or have unexpected structure
    cnr_val_for_slug = 1.0 / noise_series[0] if noise_series and noise_series[0] != 0 else np.nan
    cnr_txt = str(round(cnr_val_for_slug, 3)) if pd.notna(cnr_val_for_slug) else "NaN"

    pix_txt =  str(pix)
    tix_txt = str(tix)
    runs_txt = str(numruns)

    filename_slug = f"{file_prefix}_LD-{ld_txt}_CNR-{cnr_txt}_px-{pix_txt}_tx-{tix_txt}_runs-{runs_txt}"
    result_dictionary['filename slug'] = filename_slug
    
    summary_filename = filename_slug + '_summary.txt'
    parameters['summary filename'] = summary_filename # Store in parameters sub-dict
    
    result_filename = filename_slug + '_results.csv'
    parameters['result filename'] = result_filename # Store in parameters sub-dict

    image_filename = filename_slug + '_histogram.' + image_type
    parameters['image filename'] = image_filename # Store in parameters sub-dict
    parameters['image type'] = image_type


    run_numbers = list(range(runs_total))
    noise_list = np.concatenate([np.repeat(noise, numruns) for noise in noise_series]) if noise_series else []
    if not noise_list.size and runs_total > 0 : # If noise_series was empty but runs expected
        print_and_append(summary_filename, "Warning: Noise series is empty, but number of runs > 0. Cannot proceed with simulations.")
        return result_dictionary # Or raise error

    # Each element in parameter_sets: (run_number, noise_sigma, diffusion_coeff, lifetime)
    parameter_sets = list(zip(run_numbers, noise_list, [diff]*runs_total, [tau]*runs_total)) # Ensure it's a list for iteration
    
    print_and_append(summary_filename, f"Running {indices['total runs']} simulations with the following parameters (rounded):")
    print_and_append(summary_filename, "")
    print_and_append(summary_filename, f"Spatial width: {wid} {l_unit}")
    print_and_append(summary_filename, f"Pixel width: {pix} pixels")
    print_and_append(summary_filename, f"Number of time frames: {tix} frames")
    current_noise_sigma_for_print = noise_series[0] if noise_series else np.nan
    print_and_append(summary_filename, f"Noise stdev: {round(current_noise_sigma_for_print, 3) if pd.notna(current_noise_sigma_for_print) else 'N/A'}")
    current_cnr_for_print = 1.0/current_noise_sigma_for_print if pd.notna(current_noise_sigma_for_print) and current_noise_sigma_for_print != 0 else np.nan
    print_and_append(summary_filename, f"Initial contrast-to-noise ratio (CNR): {round(current_cnr_for_print, 3) if pd.notna(current_cnr_for_print) else 'N/A'}")
    print_and_append(summary_filename, f"Initial profile sigma^2: {round(sig2_0, 3)} {l_unit}^2")
    print_and_append(summary_filename, f"Nominal diffusion length: {round(ld, 3)} {l_unit}")
    print_and_append(summary_filename, f"Nominal diffusion coeff: {round(diff, 5)} {l_unit}^2 per {t_unit}")
    print_and_append(summary_filename, f"Nominal lifetime: {round(tau, 3)} {t_unit}") # Rounded tau
    print_and_append(summary_filename, "")

    if parameters_dictionary.get('multiprocessing', False): # Default to False
        print_and_append(summary_filename, 'Starting simulation with multiprocessing')
        print_and_append(summary_filename, "")
        # Pass only necessary parts of result_dictionary to avoid large object pickling if possible
        # scan_runner needs indices['x axis'], indices['time axis'], parameters['sigma^2_0'], parameters['amplitude_0'], parameters['mu_0']
        # For simplicity here, passing full sub-dictionaries. Could be optimized.
        scan_runner_args = {
            'x_axis': indices['x axis'],
            'time_axis': indices['time axis'],
            'sig2_0': parameters['sigma^2_0'],
            'amp_0': parameters['amplitude_0'],
            'mu_0': parameters['mu_0']
        }
        # p_set: (run_number, noise_sigma, diffusion_coeff, lifetime)
        results_list = Parallel(n_jobs=-1)(delayed(scan_runner)(scan_runner_args, ld, p_set[2], p_set[3], p_set[1], p_set[0], retain_profile_data) for p_set in parameter_sets)
    else:
        print_and_append(summary_filename, 'Starting simulation without multiprocessing')
        print_and_append(summary_filename, "")
        scan_runner_args = { # Define args for non-parallel case too
            'x_axis': indices['x axis'],
            'time_axis': indices['time axis'],
            'sig2_0': parameters['sigma^2_0'],
            'amp_0': parameters['amplitude_0'],
            'mu_0': parameters['mu_0']
        }
        # p_set: (run_number, noise_sigma, diffusion_coeff, lifetime)
        results_list = [scan_runner(scan_runner_args, ld, p_set[2], p_set[3], p_set[1], p_set[0], retain_profile_data) for p_set in parameter_sets]

    for res_dict_item in results_list:
        result_dictionary['run results'].update(res_dict_item)

    print_and_append(summary_filename, 'Simulation completed. Collating results.')
    print_and_append(summary_filename, "")
    
    # Check if 'run results' is populated
    if not result_dictionary['run results']:
        print_and_append(summary_filename, "No run results to collate. Exiting.")
        return result_dictionary

    collated_data_list = []
    for run_key in result_dictionary['run results']:
        run_data = result_dictionary['run results'][run_key]
        # Ensure all keys exist before trying to access them
        collated_data_list.append([
            run_data.get('run', np.nan),
            run_data.get('run parameters', {}).get('nominal diffusion coefficient', np.nan),
            run_data.get('run parameters', {}).get('nominal lifetime', np.nan),
            run_data.get('run parameters', {}).get('nominal diffusion length', np.nan),
            1.0 / run_data.get('run parameters', {}).get('noise stdev', np.nan) if run_data.get('run parameters', {}).get('noise stdev', 0) != 0 else np.nan,
            run_data.get('cnr_0 estimate', np.nan),
            run_data.get('nominal profiles', {}).get('parameters_t', {}).get('sigma^2_t', [np.nan])[0],
            run_data.get('noisy profile fits', {}).get('sigma^2_t estimates', [np.nan])[0],
            run_data.get('diffusion', {}).get('unweighted fit', {}).get('MSD_t slope estimate', np.nan),
            run_data.get('diffusion', {}).get('unweighted fit', {}).get('MSD_t slope std error', np.nan),
            run_data.get('diffusion', {}).get('unweighted fit', {}).get('intercept estimate', np.nan),
            run_data.get('diffusion', {}).get('unweighted fit', {}).get('intercept standard error', np.nan),
            run_data.get('diffusion', {}).get('weighted fit', {}).get('MSD_t slope estimate', np.nan),
            run_data.get('diffusion', {}).get('weighted fit', {}).get('MSD_t slope std error', np.nan),
            run_data.get('diffusion', {}).get('weighted fit', {}).get('intercept estimate', np.nan),
            run_data.get('diffusion', {}).get('weighted fit', {}).get('intercept standard error', np.nan),
        ])

    collated_results_df = pd.DataFrame(
        collated_data_list,
        columns=[
            'run number', 'nominal diffusion coeff', 'nominal lifetime', 'nominal diffusion length', 
            'nominal CNR', 'estimated CNR', 'nominal sigma^2_0', 'estimated sigma^2_0',
            'unweighted fit diffusion slope', 'unweighted fit diffusion slope stderr', 
            'unweighted fit intercept', 'unweighted fit intercept stderr', 
            'weighted fit diffusion slope', 'weighted fit diffusion slope stderr',
            'weighted fit intercept', 'weighted fit intercept stderr', 
        ]
    )

    if tix > 1 and not collated_results_df.empty:
        print_and_append(summary_filename, 'Converting diffusion constants to conventional units')
        print_and_append(summary_filename,"")

        nom_slopes = collated_results_df['nominal diffusion coeff'] * 2.0
        collated_results_df['nominal diffusion coeff [cm^2/s]'] = nom_slopes.apply(
            lambda slope: slope_to_diffusion_constant(slope, l_unit, t_unit) if pd.notna(slope) else np.nan
        )
        collated_results_df['unweighted fit diffusion coeff [cm^2/s]'] = collated_results_df['unweighted fit diffusion slope'].apply(
            lambda slope: slope_to_diffusion_constant(slope, l_unit, t_unit) if pd.notna(slope) else np.nan
        )
        collated_results_df['unweighted fit diffusion stderr [cm^2/s]'] = collated_results_df['unweighted fit diffusion slope stderr'].apply(
            lambda slope_err: slope_to_diffusion_constant(slope_err, l_unit, t_unit) if pd.notna(slope_err) else np.nan
        )
        collated_results_df['weighted fit diffusion coeff [cm^2/s]'] = collated_results_df['weighted fit diffusion slope'].apply(
            lambda slope: slope_to_diffusion_constant(slope, l_unit, t_unit) if pd.notna(slope) else np.nan
        )
        collated_results_df['weighted fit diffusion stderr [cm^2/s]'] = collated_results_df['weighted fit diffusion slope stderr'].apply(
            lambda slope_err: slope_to_diffusion_constant(slope_err, l_unit, t_unit) if pd.notna(slope_err) else np.nan
        )
        
        print_and_append(summary_filename, 'Analyzing precision and accuracy')
        print_and_append(summary_filename,"")
        analysis_results, d_wls_series, d_ols_series = estimates_precision(collated_results_df, float(proximity_level))
        result_dictionary['analysis'] = analysis_results # Store the dict part
        
        if d_wls_series is not None:
            collated_results_df['d_wls_over_d_nom'] = d_wls_series
        else:
            print("DEBUG: d_wls_series is None, 'd_wls_over_d_nom' column not added.")

        if d_ols_series is not None: 
            collated_results_df['d_ols_over_d_nom'] = d_ols_series
        else:
            print("DEBUG: d_ols_series is None, 'd_ols_over_d_nom' column not added.")

        print(f"DEBUG: collated_results_df columns after estimates_precision call and assignments: {collated_results_df.columns.tolist()}")
        print(f"DEBUG: collated_results_df d_wls_over_d_nom head (if exists):\n{collated_results_df['d_wls_over_d_nom'].head() if 'd_wls_over_d_nom' in collated_results_df else 'Column not present'}")
        
        ols_proxpct = analysis_results.get('% fits within proximity', {}).get('unweighted fit', np.nan)
        wls_proxpct = analysis_results.get('% fits within proximity', {}).get('weighted fit', np.nan)

        print_and_append(summary_filename,f"Portion of fits where D_estimate / D_nominal = 1 ± {proximity_level}:")
        print_and_append(summary_filename,f"-- Unweighted fit: {ols_proxpct:.2f}" if pd.notna(ols_proxpct) else "-- Unweighted fit: N/A")
        print_and_append(summary_filename,f"-- Weighted fit: {wls_proxpct:.2f}" if pd.notna(wls_proxpct) else "-- Weighted fit: N/A")
        print_and_append(summary_filename,"")

    else:
        print_and_append(summary_filename, 'No diffusion estimates: only one time frame or no results.')
        # Still save CSV if there's only one time frame, as it contains t0 data
    
    result_dictionary['collated results'] = collated_results_df

    if not collated_results_df.empty:
      print_and_append(summary_filename, 'Exporting result data.')
      collated_results_df.to_csv(result_filename, index = False)
      print_and_append(summary_filename, f"-- Collated CSV file: {result_filename}")
    else:
        print_and_append(summary_filename, 'No collated results to save.')

    # Conditionally plot histogram and print its log message
    if tix > 1 and not collated_results_df.empty:
        print_and_append(summary_filename, 'Exporting histogram.')
        plot_accuracy_histogram(
            result_dictionary, # Pass the main dict which contains 'collated results'
            proximity=float(proximity_level), filename=image_filename, image_type=image_type,
            width=image_width, height=image_height, dpi=image_dpi,
            font_size=image_font_size, tick_length=image_tick_l, tick_width=image_tick_w,
            num_bins=image_numbins, x_lim=image_x_lim
        )
        print_and_append(summary_filename, f"-- Histogram image file {image_filename}")
    elif tix <= 1 : # Explicitly state why histogram is not generated for single/no time frame
        print_and_append(summary_filename, "Histogram not generated: requires multiple time frames for diffusion analysis.")

    print_and_append(summary_filename, f"-- Summary file: {summary_filename}")
    print_and_append(summary_filename,"")
    print_and_append(summary_filename, 'Done!')
    return result_dictionary

def scan_runner(scan_runner_args: Dict[str, Any], ld: float, this_diff: float, this_tau: float, 
                this_noise: float, this_run: int, retain_profile_data: bool) -> Dict[str, Any]:
    """
    Execute a simulation scan and generate a dictionary of results.
    (Docstring adapted from original)
    """
    x_axis = scan_runner_args['x_axis']
    time_axis = scan_runner_args['time_axis']
    sig2_0 = scan_runner_args['sig2_0']
    amp_0 = scan_runner_args['amp_0']
    mu_0 = scan_runner_args['mu_0']

    result_dictionary_scan: Dict[str, Any] = {f'run_{this_run}': {
                            'run': this_run,
                            'run parameters': {
                                'nominal diffusion coefficient': this_diff,
                                'nominal lifetime': this_tau,
                                'nominal diffusion length': ld, 
                                'noise stdev': this_noise
                            }
                        }
                    }
    
    current_run_results = result_dictionary_scan[f'run_{this_run}']

    diff_decay_parameters = {
        'x axis': x_axis, 'time axis': time_axis,
        'sigma^2_0': sig2_0, 'amplitude_0': amp_0, 'mu_0': mu_0,
        'nominal diffusion coefficient': this_diff, 'nominal lifetime': this_tau,
    }
    nominal_profiles = make_diffusion_decay(diff_decay_parameters)
    current_run_results['nominal profiles'] = nominal_profiles

    # Ensure nominal_profiles['y_values_t'] is a list of np.ndarray or compatible
    # add_noise expects np.ndarray or list of them.
    noisy_profiles_dict = add_noise(this_noise, np.array(nominal_profiles['y_values_t']))
    current_run_results['noisy profiles'] = noisy_profiles_dict # Store the dict returned by add_noise

    # fft_cnr expects a single profile (1D array)
    cnr_0_est = fft_cnr(noisy_profiles_dict['y_values_t'][0]) if noisy_profiles_dict['y_values_t'].any() else np.nan
    current_run_results['cnr_0 estimate'] = cnr_0_est
    
    # gauss_fitting expects a list of profiles
    noisy_profile_fits = gauss_fitting(x_axis, list(noisy_profiles_dict['y_values_t']))
    current_run_results['noisy profile fits'] = noisy_profile_fits

    gaussfit_sigma2_t = noisy_profile_fits['sigma^2_t estimates']
    gaussfit_sigma2_t_stderrs = noisy_profile_fits['sigma^2_t standard errors'] # This is already a list of stderrs

    weights = []
    if len(time_axis) > 1:
        # Ensure gaussfit_sigma2_t and gaussfit_sigma2_t_stderrs are same length and not empty
        if gaussfit_sigma2_t and gaussfit_sigma2_t_stderrs and len(gaussfit_sigma2_t) == len(gaussfit_sigma2_t_stderrs):
            for sigma2, stderr in zip(gaussfit_sigma2_t, gaussfit_sigma2_t_stderrs):
                if pd.notna(sigma2) and pd.notna(stderr) and sigma2 != 0 and stderr != 0:
                    weights.append(np.power(stderr / sigma2, -2.))
                else:
                    weights.append(0) # Assign 0 weight if sigma2 or stderr is 0, NaN
            
            if np.sum(weights) != 0 : # Avoid division by zero if all weights are zero
                 weights = [w / np.sum(weights) for w in weights] # Normalize
            else: # if all weights are zero, then WLS is not meaningful
                 weights = [0] * len(gaussfit_sigma2_t) # Keep as list of zeros

        ols_fit_results = diffusion_ols_fit(time_axis, gaussfit_sigma2_t)
        wls_fit_results = diffusion_wls_fit(time_axis, gaussfit_sigma2_t, weights)
    else:
        ols_fit_results = {'MSD_t slope estimate': np.nan, 'MSD_t slope std error': np.nan, 'intercept estimate': np.nan, 'intercept standard error': np.nan}
        wls_fit_results = {'MSD_t slope estimate': np.nan, 'MSD_t slope std error': np.nan, 'intercept estimate': np.nan, 'intercept standard error': np.nan}
        weights = [np.nan] * len(time_axis)


    current_run_results['diffusion'] = {
        'unweighted fit': ols_fit_results,
        'weighted fit': {**wls_fit_results, 'weights': weights}, # Add weights to WLS results
    }
    
    if not retain_profile_data:
        if 'y_values_t' in current_run_results['noisy profiles']:
            del current_run_results['noisy profiles']['y_values_t']
        if 'y_values_t' in current_run_results['nominal profiles']:
            del current_run_results['nominal profiles']['y_values_t']

    return result_dictionary_scan
