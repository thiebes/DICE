import os
import re

import numpy as np
import pandas as pd

def precision_counts(cnr_ld_prox, cnr_bin_mid_unique, ld_unique, precision_levels):
    """
    Analyzes the precision of weighted and unweighted fits in diffusion studies by 
    calculating the proximity of estimated diffusion coefficients to their nominal values.

    Parameters:
    - cnr_ld_prox: DataFrame containing data for analysis. It should have four columns:
                  'CNR bin mid', 'nominal diffusion length', 'd_wls_over_d_nom', and 
                  'd_ols_over_d_nom', where the last two columns represent the proximity 
                  values (D_est/D_nom) for weighted (WLS) and unweighted (OLS) fits.
    - cnr_bin_mid_unique: List of unique middle values of CNR bins. These values are used 
                          to categorize the data into different CNR ranges.
    - ld_unique: List of unique diffusion lengths. These are used to categorize the data 
                 based on the diffusion length.
    - precision_levels: List of precision levels to analyze. These are fractions representing 
                        how close the estimated diffusion coefficient needs to be to the nominal 
                        value to be considered precise. For example, 0.1 (or 10%) means the 
                        estimated value should be within 10% of the nominal value.

    Returns:
    - A dictionary with keys formatted as 'precision proximity < X%', where X is the precision 
      level percentage. Each key maps to a DataFrame containing the analysis results for that 
      precision level. The DataFrames include counts and statistics (mean, standard deviation, 
      and difference in standard deviations between WLS and OLS fits) of the fits that fall within 
      the specified precision proximity for each combination of CNR bin middle value and diffusion length.
    """

    # initialize counts dictionary and subdictionaries to store results
    counts = {}

    # step through the precision levels to collate results
    # by precision level, cnr, ld, pix, and tix values
    for p in precision_levels:
        p_str = str(np.round(100*p))          # string to label the precision level
        print('Counting for precision within ' + p_str + '% of nominal value')
        these_counts = []   # initialize array of counts

        for cnr in cnr_bin_mid_unique:
            for ld in ld_unique:
                # store the records matching precision level, cnr, ld, pix, and tix value
                these_matches = cnr_ld_prox[
                    (cnr_ld_prox['CNR bin mid'] == cnr) & 
                    (cnr_ld_prox['nominal diffusion length'] == ld)
                    ]

                this_total = len(these_matches) # total number of matches
                ols_proxcount = len(these_matches[np.abs(these_matches['d_ols_over_d_nom'] - 1) < p])
                wls_proxcount = len(these_matches[np.abs(these_matches['d_wls_over_d_nom'] - 1) < p])

                # Calculations for mean and standard deviation of proximity ratios
                wls_prox_mean = np.mean(these_matches['d_wls_over_d_nom'])
                wls_prox_std  = np.std(these_matches['d_wls_over_d_nom'])
                ols_prox_mean = np.mean(these_matches['d_ols_over_d_nom'])
                ols_prox_std  = np.std(these_matches['d_ols_over_d_nom'])

                # Calculate the difference between the stdevs of wls and ols D/D0 values.
                # Higher values indicate that weighted fits are more precise than unweighted.
                # Negative values would indicate that unweighted fit is more precise than weighted.
                # - Note that better precision doesn't necessarily mean better accuracy:
                # - If there is bias, a more precise distribution may nevertheless be
                # - far from the nominal value, and may evben have fewer fits within proximity
                # - of the nominal value than a wider distribution would. Evaluation of the 
                # - spread of fits along with the mean D_est/D_nom is thus warranted.
                difference_ols_wls_std = ols_prox_std - wls_prox_std

                if this_total == 0:
                    # if none matched within precision proximity, report error
                    print('No results found for CNR = ' + str(cnr) + ', LD = ' + str(ld))
                    # flag the record for no matches
                    these_counts.append(
                        {'nominal CNR': cnr, 
                            'nominal diffusion length': ld, 
                            'weighted fits percent in proximity': -1, 
                            'unweighted fits percent in proximity': -1, 
                            'total in bin': -1,
                            'weighted fits mean D_est/D_nom': -1,
                            'unweighted fits mean D_est/D_nom': -1,
                            'weighted fits stdev D_est/D_nom':  -1,
                            'unweighted fits stdev D_est/D_nom':  -1,
                            'difference in weighted and unweighted stdev': -1,
                            })
                else:
                    these_counts.append(
                        {'nominal CNR': cnr, 
                            'nominal diffusion length': ld, 
                            'weighted fits percent in proximity': 100 * wls_proxcount/this_total, 
                            'unweighted fits percent in proximity': 100 * ols_proxcount/this_total, 
                            'total in bin': this_total,
                            'weighted fits mean D_est/D_nom': wls_prox_mean,
                            'unweighted fits mean D_est/D_nom': ols_prox_mean,
                            'weighted fits stdev D_est/D_nom':  wls_prox_std,
                            'unweighted fits stdev D_est/D_nom':  ols_prox_std,
                            'difference in weighted and unweighted stdev': difference_ols_wls_std,
                            })

        counts[f'precision proximity < {p_str}%'] = pd.DataFrame(these_counts)

    return counts

def estimates_precision(df, proximity_level):
    """
    Calculate the percentage of simulations where the estimated diffusion fits
    are within a specified proximity to the nominal diffusion value.

    Parameters:
    - df: (DataFrame) DataFrame containing diffusion data.
    - proximity_level: (float) The acceptable proximity level around the nominal value.

    Returns:
    - result: (dict) Dictionary with the percentage of fits within the specified proximity.
    """
    if len(df) == 0:
        raise ValueError("The DataFrame is empty.")

    p_low = 1 - proximity_level
    p_high = 1 + proximity_level

    # Vectorized operations to calculate ratios
    df['d_wls_over_d_nom'] = df['weighted fit diffusion coeff [cm^2/s]'] / df['nominal diffusion coeff [cm^2/s]']
    df['d_ols_over_d_nom'] = df['unweighted fit diffusion coeff [cm^2/s]'] / df['nominal diffusion coeff [cm^2/s]']

    # Conditions to check if values are within the proximity level
    wls_within = df['d_wls_over_d_nom'].between(p_low, p_high)
    ols_within = df['d_ols_over_d_nom'].between(p_low, p_high)

    # Calculate percentages
    wls_portion_pct = 100 * wls_within.sum() / len(df)
    ols_portion_pct = 100 * ols_within.sum() / len(df)

    result = {
        '% fits within proximity': {
            'weighted fit': wls_portion_pct,
            'unweighted fit': ols_portion_pct,
        }
    }

    return result

def load_files(file_path, file_match, cnr_low, cnr_high, precision_levels, num_bins):
    """
    Loads, collates, and analyzes precision from diffusion study result files.

    Parameters:
    - file_path: Path to the directory containing files.
    - file_match: String to match the beginning of filenames.
    - cnr_low: Low end of CNR range for these files.
    - cnr_high: High end of CNR range for these files.
    - precision_levels: Array of proximities to nominal values as a fraction.
    - num_bins: Number of CNR bins for sorting.

    Returns:
    - Dictionary with keys 'all' for full DataFrame and 'precision counts' for precision analysis.
    """

    # get the directory listing
    try:
        dir_list = os.listdir(file_path)
    except FileNotFoundError:
        return {'error': 'Directory not found'}

    # find matching files and report how many were found
    these_files = [re.findall('(?:^'+file_match+'.+)', s) for s in dir_list]
    these_files = [x[0] for x in these_files if x != []]
    print(f'Found {len(these_files)} matching files.')

    # initialize result dataframe
    df = pd.DataFrame()

    # load the file(s) and label pixels and time frames
    for filename in these_files:
        # load each file as a data frame
        try:
            df_this_file = pd.read_csv(os.path.join(file_path, filename))
        except Exception as e:
            print(f'Error reading {filename}: {e}')
            continue
        # get the number of pixels and time frames from the filename
        this_pix = re.findall(r"px-(\d+)", filename)
        this_tix = re.findall(r"tx-(\d+)", filename)
        if this_pix and this_tix:
            df_this_file['number of pixels'] = int(this_pix[0])
            df_this_file['number of time frames'] = int(this_tix[0])
        else:
            print(f'Filename format incorrect: {filename}')
            continue
        # append the result dataframe with this file data
        df = pd.concat([df, df_this_file], ignore_index=True)

    # older format has columns:
    # ['run num', 'diff nom', 'tau nom', 'ld nom', 'cnr', 'sigma2_0 nom',
    #    'sigma2_0 est', 'ols fit', 'wls fit', 'diff nom cm2/s',
    #    'wls diff cm2/s', 'ols diff cm2/s'],

    # fix older format
    if 'diff nom cm2/s' in df.columns:
       df = df.rename(
           columns={
               'run num': 'run number',
               'diff nom': 'nominal diffusion coeff',
               'tau nom': 'nominal lifetime',
               'ld nom': 'nominal diffusion length',
               'cnr': 'nominal CNR',
               'sigma2_0 nom': 'nominal sigma^2_0',
               'sigma2_0 est': 'estimated sigma^2_0',
               'ols fit': 'unweighted fit diffusion slope',
               'wls fit': 'weighted fit diffusion slope',
               'diff nom cm2/s': 'nominal diffusion coeff [cm^2/s]', 
               'wls diff cm2/s': 'weighted fit diffusion coeff [cm^2/s]',
               'ols diff cm2/s': 'unweighted fit diffusion coeff [cm^2/s]',
               }
            )

    # Calculations for relative proximity
    df['d_wls_over_d_nom'] = df['weighted fit diffusion coeff [cm^2/s]'] / df['nominal diffusion coeff [cm^2/s]']
    df['d_ols_over_d_nom'] = df['unweighted fit diffusion coeff [cm^2/s]'] / df['nominal diffusion coeff [cm^2/s]']

    # Verify CNR range
    if df['nominal CNR'].min() < cnr_low or df['nominal CNR'].max() > cnr_high:
        return {'error': 'nominal CNR values exist outside the expected range'}

    # CNR bin calculations
    cnr_bins = np.power(10, np.linspace(np.log10(cnr_low), np.log10(cnr_high), num_bins))
    df['CNR bins'] = pd.cut(df['nominal CNR'], bins=cnr_bins, include_lowest=True)
    df['CNR bin mid'] = [ival.mid for ival in df['CNR bins']]

    # Unique values for precision counts
    cnr_bin_mid_unique = df['CNR bin mid'].unique()
    ld_unique = df['nominal diffusion length'].unique()

    # Abbreviated results for speed
    results_brief = df[['CNR bin mid', 'nominal diffusion length', 'd_wls_over_d_nom', 'd_ols_over_d_nom']]

    # Get counts of results within precision proximity using precision_counts function
    precision_counts_data = precision_counts(
        results_brief, 
        cnr_bin_mid_unique, 
        ld_unique,
        precision_levels)

    # Return full results and precision counts
    result = {
        'all': df, 
        'precision counts': precision_counts_data
    }

    return result