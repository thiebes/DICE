import numpy as np
import pandas as pd
import os
import re
from .utils import slope_to_diffusion_constant # For estimates_precision

def estimates_precision(df: pd.DataFrame, proximity_level: float) -> dict:
    """
    Calculate the percentage of simulations where the estimated diffusion fits
    are within a specified proximity to the nominal diffusion value.

    Parameters:
    - df: (DataFrame) DataFrame containing diffusion data. Must include columns:
          'weighted fit diffusion coeff [cm^2/s]', 'unweighted fit diffusion coeff [cm^2/s]',
          and 'nominal diffusion coeff [cm^2/s]'.
    - proximity_level: (float) The acceptable proximity level around the nominal value (e.g., 0.1 for 10%).

    Returns:
    - result: (dict) Dictionary with the percentage of fits within the specified proximity.
    - d_wls_over_d_nom_series (pd.Series or None): Series of WLS D_est/D_nom ratios.
    - d_ols_over_d_nom_series (pd.Series or None): Series of OLS D_est/D_nom ratios.
    """
    print(f"DEBUG: Entering estimates_precision. DataFrame shape: {df.shape}, Columns: {df.columns.tolist()}")

    required_cols = [
        'weighted fit diffusion coeff [cm^2/s]',
        'unweighted fit diffusion coeff [cm^2/s]',
        'nominal diffusion coeff [cm^2/s]'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: estimates_precision missing required columns: {missing_cols}. Cannot calculate D_est/D_nom ratios.")
        analysis_result = {
            '% fits within proximity': {
                'weighted fit': 0, 
                'unweighted fit': 0,
            }
        }
        return analysis_result

    if df.empty:
        print("Warning: estimates_precision received an empty DataFrame.")
        analysis_result = {'% fits within proximity': {'weighted fit': 0, 'unweighted fit': 0}}
        return analysis_result

    # Drop rows where nominal diffusion coefficient is zero or NaN to avoid division by zero or invalid ratios
    # Create a copy to avoid SettingWithCopyWarning if df is a slice
    df_cleaned = df.copy() 
    df_cleaned.dropna(subset=['nominal diffusion coeff [cm^2/s]'], inplace=True)
    df_cleaned = df_cleaned[df_cleaned['nominal diffusion coeff [cm^2/s]'] != 0]

    if df_cleaned.empty:
        print("Warning: DataFrame has no valid data after cleaning for precision estimation.")
        analysis_result = {'% fits within proximity': {'weighted fit': 0, 'unweighted fit': 0}}
        return analysis_result

    d_wls_over_d_nom_series = df_cleaned['weighted fit diffusion coeff [cm^2/s]'] / df_cleaned['nominal diffusion coeff [cm^2/s]']
    d_ols_over_d_nom_series = df_cleaned['unweighted fit diffusion coeff [cm^2/s]'] / df_cleaned['nominal diffusion coeff [cm^2/s]']
    
    print(f"DEBUG: d_wls_over_d_nom_series head:\n{d_wls_over_d_nom_series.head()}")

    p_low = 1 - proximity_level
    p_high = 1 + proximity_level

    wls_within = d_wls_over_d_nom_series.between(p_low, p_high)
    ols_within = d_ols_over_d_nom_series.between(p_low, p_high)

    wls_portion_pct = 100 * wls_within.sum() / len(d_wls_over_d_nom_series) if len(d_wls_over_d_nom_series) > 0 else 0
    ols_portion_pct = 100 * ols_within.sum() / len(d_ols_over_d_nom_series) if len(d_ols_over_d_nom_series) > 0 else 0
    
    analysis_result = {
        '% fits within proximity': {
            'weighted fit': wls_portion_pct,
            'unweighted fit': ols_portion_pct,
        },
        'counts within proximity': { 
            'weighted fit': wls_within.sum(),
            'unweighted fit': ols_within.sum(),
        },
        'total valid simulations for precision': len(df_cleaned)
    }

    return analysis_result

def load_files(file_path: str, file_match: str, cnr_low: float, cnr_high: float, precision_levels: list, num_bins: int) -> dict:
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
      Returns {'error': 'message'} if critical errors occur.
    """

    try:
        dir_list = os.listdir(file_path)
    except FileNotFoundError:
        return {'error': f'Directory not found: {file_path}'}

    these_files = [f for f in dir_list if f.startswith(file_match) and f.endswith('.csv')]
    # print(f'Found {len(these_files)} matching files.') # Commented out print

    if not these_files:
        return {'error': f'No files matching "{file_match}*.csv" found in {file_path}'}

    df_list = []
    for filename in these_files:
        try:
            df_this_file = pd.read_csv(os.path.join(file_path, filename))
            this_pix_match = re.search(r"px-(\d+)", filename)
            this_tix_match = re.search(r"tx-(\d+)", filename)
            
            if this_pix_match and this_tix_match:
                df_this_file['number of pixels'] = int(this_pix_match.group(1))
                df_this_file['number of time frames'] = int(this_tix_match.group(1))
            else:
                # print(f'Warning: Could not parse pixels/time frames from filename: {filename}') # Commented out print
                df_this_file['number of pixels'] = np.nan
                df_this_file['number of time frames'] = np.nan
            df_list.append(df_this_file)
        except Exception as e:
            # print(f'Error reading or processing {filename}: {e}') # Commented out print
            continue
    
    if not df_list:
        return {'error': 'No files could be successfully read or processed.'}

    df = pd.concat(df_list, ignore_index=True)

    # Standardize column names (example for older format)
    rename_map = {
        'run num': 'run number', 'diff nom': 'nominal diffusion coeff',
        'tau nom': 'nominal lifetime', 'ld nom': 'nominal diffusion length',
        'cnr': 'nominal CNR', 'sigma2_0 nom': 'nominal sigma^2_0',
        'sigma2_0 est': 'estimated sigma^2_0', 'ols fit': 'unweighted fit diffusion slope',
        'wls fit': 'weighted fit diffusion slope', 'diff nom cm2/s': 'nominal diffusion coeff [cm^2/s]', 
        'wls diff cm2/s': 'weighted fit diffusion coeff [cm^2/s]',
        'ols diff cm2/s': 'unweighted fit diffusion coeff [cm^2/s]',
    }
    df = df.rename(columns=lambda c: rename_map.get(c, c))


    # Ensure required columns for ratio calculation exist
    if not all(col in df.columns for col in ['weighted fit diffusion coeff [cm^2/s]', 
                                             'unweighted fit diffusion coeff [cm^2/s]', 
                                             'nominal diffusion coeff [cm^2/s]']):
        return {'error': 'DataFrame is missing one or more required columns for D_est/D_nom calculation after potential rename.'}


    # Calculations for relative proximity, handling potential division by zero or NaN
    df['d_wls_over_d_nom'] = np.where(
        (df['nominal diffusion coeff [cm^2/s]'] != 0) & ~df['nominal diffusion coeff [cm^2/s]'].isnull(),
        df['weighted fit diffusion coeff [cm^2/s]'] / df['nominal diffusion coeff [cm^2/s]'],
        np.nan
    )
    df['d_ols_over_d_nom'] = np.where(
        (df['nominal diffusion coeff [cm^2/s]'] != 0) & ~df['nominal diffusion coeff [cm^2/s]'].isnull(),
        df['unweighted fit diffusion coeff [cm^2/s]'] / df['nominal diffusion coeff [cm^2/s]'],
        np.nan
    )
    
    if 'nominal CNR' not in df.columns:
        return {'error': "Column 'nominal CNR' not found, cannot perform CNR binning."}

    # CNR bin calculations
    # Filter out NaN or inf CNR values before binning
    df_cnr_valid = df[np.isfinite(df['nominal CNR'])]
    if df_cnr_valid.empty:
        return {'error': "No valid 'nominal CNR' values for binning."}
        
    # Check if cnr_low and cnr_high are valid for logspace
    if cnr_low <= 0 or cnr_high <=0:
         return {'error': "cnr_low and cnr_high must be positive for logspace binning."}

    try:
        cnr_bins = np.power(10, np.linspace(np.log10(cnr_low), np.log10(cnr_high), num_bins + 1))
    except ValueError as e:
        return {'error': f"Error creating CNR bins: {e}. Check cnr_low, cnr_high, num_bins."}

    df['CNR bins'] = pd.cut(df['nominal CNR'], bins=cnr_bins, include_lowest=True, right=True)
    df['CNR bin mid'] = df['CNR bins'].apply(lambda x: x.mid if pd.notnull(x) else np.nan)


    cnr_bin_mid_unique = sorted(df['CNR bin mid'].dropna().unique())
    ld_unique = sorted(df['nominal diffusion length'].dropna().unique()) if 'nominal diffusion length' in df.columns else []

    # Abbreviated results for speed
    results_brief = df[['CNR bin mid', 'nominal diffusion length', 'd_wls_over_d_nom', 'd_ols_over_d_nom']].copy()
    results_brief.dropna(subset=['d_wls_over_d_nom', 'd_ols_over_d_nom'], how='all', inplace=True)


    precision_counts_data = precision_counts(
        results_brief, 
        cnr_bin_mid_unique, 
        ld_unique,
        precision_levels)

    result = {
        'all': df, 
        'precision counts': precision_counts_data
    }

    return result

def precision_counts(cnr_ld_prox_df: pd.DataFrame, cnr_bin_mid_unique: list, ld_unique: list, precision_levels: list) -> dict:
    """
    Analyzes the precision of weighted and unweighted fits in diffusion studies.
    (Docstring adapted from original)
    """
    counts = {}
    if cnr_ld_prox_df.empty: # Handle empty input DataFrame
        # print("Warning: cnr_ld_prox_df is empty in precision_counts.") # Commented out print
        for p in precision_levels:
            p_str = str(np.round(100 * p))
            counts[f'precision proximity < {p_str}%'] = pd.DataFrame() # Return empty DataFrame
        return counts

    for p in precision_levels:
        p_str = str(np.round(100 * p))
        # print(f'Counting for precision within {p_str}% of nominal value') # Commented out print
        these_counts_list = []

        for cnr in cnr_bin_mid_unique:
            for ld in ld_unique:
                these_matches = cnr_ld_prox_df[
                    (cnr_ld_prox_df['CNR bin mid'] == cnr) & 
                    (cnr_ld_prox_df['nominal diffusion length'] == ld)
                ]

                this_total = len(these_matches)
                
                # Ensure 'd_ols_over_d_nom' and 'd_wls_over_d_nom' exist and are numeric before filtering
                ols_proxcount = 0
                wls_proxcount = 0
                wls_prox_mean, wls_prox_std = np.nan, np.nan
                ols_prox_mean, ols_prox_std = np.nan, np.nan
                difference_ols_wls_std = np.nan

                if this_total > 0:
                    if 'd_ols_over_d_nom' in these_matches.columns:
                        ols_proxcount = len(these_matches[np.abs(these_matches['d_ols_over_d_nom'] - 1) < p])
                        ols_prox_mean = np.mean(these_matches['d_ols_over_d_nom'].dropna())
                        ols_prox_std  = np.std(these_matches['d_ols_over_d_nom'].dropna())
                    
                    if 'd_wls_over_d_nom' in these_matches.columns:
                        wls_proxcount = len(these_matches[np.abs(these_matches['d_wls_over_d_nom'] - 1) < p])
                        wls_prox_mean = np.mean(these_matches['d_wls_over_d_nom'].dropna())
                        wls_prox_std  = np.std(these_matches['d_wls_over_d_nom'].dropna())

                    if pd.notna(ols_prox_std) and pd.notna(wls_prox_std):
                         difference_ols_wls_std = ols_prox_std - wls_prox_std
                
                current_count_dict = {
                    'nominal CNR': cnr, 
                    'nominal diffusion length': ld, 
                    'weighted fits percent in proximity': 100 * wls_proxcount / this_total if this_total > 0 else 0, 
                    'unweighted fits percent in proximity': 100 * ols_proxcount / this_total if this_total > 0 else 0, 
                    'total in bin': this_total,
                    'weighted fits mean D_est/D_nom': wls_prox_mean,
                    'unweighted fits mean D_est/D_nom': ols_prox_mean,
                    'weighted fits stdev D_est/D_nom':  wls_prox_std,
                    'unweighted fits stdev D_est/D_nom':  ols_prox_std,
                    'difference in weighted and unweighted stdev': difference_ols_wls_std,
                }
                these_counts_list.append(current_count_dict)
        
        counts[f'precision proximity < {p_str}%'] = pd.DataFrame(these_counts_list)

    return counts
