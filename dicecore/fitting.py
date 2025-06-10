from typing import cast

import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import statsmodels.api as sm

from .profiles import gaussian # For gauss_fitting

def gauss_fitting(x_axis: np.ndarray, noisy_profiles: list) -> dict:
    """
    Fits Gaussian function to noisy data at each time point.

    This function takes a set of noisy Gaussian profiles and fits a Gaussian
    function to each profile using non-linear least squares optimization. The
    parameter estimates and their standard errors of the variance of the
    Gaussian are stored and returned in a dictionary.

    Parameters:
    - x_axis: (array-like) The x-values over which the profiles are defined.
    - noisy_profiles: (list of array-like) The y-values of the noisy profiles
      for each time point to be fitted.

    Returns:
    - fit_dictionary: (dict) A dictionary containing the fitted parameter
      estimates and the standard deviations for each time point.

    Raises:
    - ValueError: If any parameter bounds are invalid or inverted.
    - RuntimeError: If curve_fit fails.
    """

    # initialize results dictionary
    fit_dictionary = {
        'sigma^2_t estimates': [], 
        'sigma^2_t standard errors': [], 
        } 
    
    for this_profile in noisy_profiles:
        # this fitting algorithm is ignorant of the input parameters
        
        xpix = len(x_axis)
        xmin,xmax = np.min(x_axis), np.max(x_axis)
        xwid = np.abs(xmax - xmin)

        max_amp_idx = np.argmax(this_profile)

        mu0 = x_axis[max_amp_idx] # Guess mu at the peak of the noisy data
        sigma2_0 = np.power(xwid / 4,2) 
        a0 = this_profile[max_amp_idx]

        # Ensure mu0 guess is within bounds if x_axis is very small
        mu0 = np.clip(mu0, xmin, xmax)


        # Bounds for mu: can be anywhere on the x_axis
        mu0_min = xmin
        mu0_max = xmax
        
        sigma2_min = np.power(xwid / (2*xpix),2) # sigma^2 min is half pixel width squared
        sigma2_max = np.power(xwid,2)      

        a0_max = np.max(this_profile) * 1.5 if np.max(this_profile) > 0 else 1.0 # Max amplitude
        a0_min = np.min(this_profile) if np.min(this_profile) < 0 else 0 # Min amplitude can be negative if noise is high

        # Check and correct bounds if min > max
        if mu0_min > mu0_max: mu0_min, mu0_max = mu0_max, mu0_min
        if sigma2_min > sigma2_max: sigma2_min, sigma2_max = sigma2_max, sigma2_min
        if a0_min > a0_max: a0_min, a0_max = a0_max, a0_min # Should not happen with logic above

        p0 = [mu0, sigma2_0, a0]                   
        bounds_min = [mu0_min, sigma2_min, a0_min] 
        bounds_max = [mu0_max, sigma2_max, a0_max] 

        try:
            parms, covars = curve_fit(
                gaussian, x_axis, this_profile, 
                p0 = p0,
                bounds=(bounds_min, bounds_max),
                maxfev=5000,
                ftol=1e-3, xtol=1e-3 # Looser tolerances for faster, possibly less precise, fits
            )
            fit_dictionary['sigma^2_t estimates'].append(parms[1])
            coefficient_variance_table = np.diag(covars) 
            coefficient_stdev_table = np.sqrt(np.abs(coefficient_variance_table)) # abs for safety if variance is tiny negative
            fit_dictionary['sigma^2_t standard errors'].append(coefficient_stdev_table[1])
        except RuntimeError:
            # If fit fails, append NaN or a placeholder
            fit_dictionary['sigma^2_t estimates'].append(np.nan)
            fit_dictionary['sigma^2_t standard errors'].append(np.nan)
            # print(f"RuntimeError in curve_fit for a profile. Appending NaN. P0:{p0}, Bounds: {bounds_min}, {bounds_max}")

    return fit_dictionary

def diffusion_ols_fit(time_axis: np.ndarray, gaussfit_sigma2_t: list) -> dict:
    """
    Estimates the Mean Squared Displacement (MSD) over time for a scan using 
    Ordinary Least Squares (OLS). This is useful in diffusion studies, where 
    the MSD is expected to linearly increase with time for a diffusive process.

    Parameters:
    - time_axis: (array-like) The time points for each measurement. These should 
                 be evenly spaced for accurate OLS fitting.
    - gaussfit_sigma2_t: (array-like) Squared width (variance) from Gaussian fits 
                         at each time point. This represents the displacement data 
                         for OLS fitting.

    Returns:
    - result: (dict) A dictionary containing key results from the OLS fit.
              Returns dict with NaN values if fit fails or inputs are inadequate.
    """
    # Convert to numpy array and remove any NaNs from fitting failures
    valid_indices = ~np.isnan(gaussfit_sigma2_t)
    time_axis_clean = np.asarray(time_axis)[valid_indices]
    gaussfit_sigma2_t_clean = np.asarray(gaussfit_sigma2_t)[valid_indices]

    if len(gaussfit_sigma2_t_clean) < 2:
        # Not enough data points to perform a fit
        return {
            'MSD_t slope estimate': np.nan, 'intercept estimate': np.nan,
            'MSD_t slope std error': np.nan, 'intercept standard error': np.nan,
        }
    
    delta_vars = gaussfit_sigma2_t_clean - gaussfit_sigma2_t_clean[0]
    design_matrix = sm.add_constant(time_axis_clean)

    try:
        ols_model = sm.OLS(delta_vars, design_matrix).fit()
        return {
            'MSD_t slope estimate': ols_model.params[1],
            'intercept estimate': ols_model.params[0],
            'MSD_t slope std error': ols_model.bse[1],
            'intercept standard error': ols_model.bse[0],
        }
    except Exception: # Catch any error during fitting (e.g. singular matrix)
        return {
            'MSD_t slope estimate': np.nan, 'intercept estimate': np.nan,
            'MSD_t slope std error': np.nan, 'intercept standard error': np.nan,
        }


def diffusion_wls_fit(
    time_axis: np.ndarray,
    gaussfit_sigma2_t: np.ndarray,
    weights: np.ndarray
) -> dict:
    """
    Estimates the diffusion coefficient for a scan using Weighted Least Squares (WLS).

    Parameters:
    - time_axis: (array-like) The time points for each measurement.
    - gaussfit_sigma2_t: (array-like) Variances of Gaussian fits at each time point.
    - weights: (array-like) Weights to apply to each measurement.

    Returns:
    - result: (dict) A dictionary containing the slope ('slope') and the standard error
                     of the slope ('std error') as the estimation of the diffusion coefficient.
              Returns dict with NaN values if fit fails or inputs are inadequate.
    """
    # Convert to numpy array and remove any NaNs from fitting failures or invalid weights
    valid_indices = ~np.isnan(gaussfit_sigma2_t) & ~np.isnan(weights) & (np.asarray(weights) > 0) # Ensure weights are positive
    time_axis_clean = np.asarray(time_axis)[valid_indices]
    gaussfit_sigma2_t_clean = np.asarray(gaussfit_sigma2_t)[valid_indices]
    weights_clean = np.asarray(weights)[valid_indices].astype(np.float64)
    
    if len(gaussfit_sigma2_t_clean) < 2 or len(weights_clean) < 2:
        # Not enough data points to perform a fit
        return {
            'MSD_t slope estimate': np.nan, 'intercept estimate': np.nan,
            'MSD_t slope std error': np.nan, 'intercept standard error': np.nan,
        }
        
    delta_vars = gaussfit_sigma2_t_clean - gaussfit_sigma2_t_clean[0]
    design_matrix = sm.add_constant(time_axis_clean)

    try:
        model = sm.WLS(delta_vars, design_matrix, weights=weights_clean)  # type: ignore[arg-type]
        wls_model = model.fit()
        return {
            'MSD_t slope estimate': wls_model.params[1],
            'intercept estimate': wls_model.params[0],
            'MSD_t slope std error': wls_model.bse[1],
            'intercept standard error': wls_model.bse[0],
        }
    except Exception: # Catch any error during fitting
        return {
            'MSD_t slope estimate': np.nan, 'intercept estimate': np.nan,
            'MSD_t slope std error': np.nan, 'intercept standard error': np.nan,
        }
