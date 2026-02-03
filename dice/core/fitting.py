"""
Fitting algorithms for Gaussian profiles and diffusion analysis.

This module provides functions for fitting Gaussian profiles to noisy data
and performing linear regression on MSD data to extract diffusion coefficients.
"""

import numpy as np
from scipy.optimize import curve_fit
import statsmodels.api as sm
from typing import Dict, List, Tuple, Optional, Union
from ..utils.validators import validate_array_like
from .profiles import gaussian


def gauss_fitting(x_axis: np.ndarray, noisy_profiles: Union[np.ndarray, List]) -> Dict:
    """
    Fit Gaussian functions to noisy profiles at each time point.
    
    This function fits a Gaussian to each profile using non-linear least squares
    optimization. The variance (sigma^2) estimates and their standard errors
    are returned.
    
    Parameters
    ----------
    x_axis : np.ndarray
        The x-values over which the profiles are defined.
    noisy_profiles : np.ndarray or list
        The y-values of the noisy profiles for each time point.
        Can be 2D array (time x space) or list of 1D arrays.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'sigma^2_t estimates': List of fitted variance values
        - 'sigma^2_t standard errors': List of standard errors
    
    Raises
    ------
    ValueError
        If fitting fails or bounds are invalid.
    """
    x_axis = validate_array_like(x_axis, "x_axis", ndim=1)
    
    # Convert to list of profiles if needed
    if isinstance(noisy_profiles, np.ndarray):
        if noisy_profiles.ndim == 1:
            profiles = [noisy_profiles]
        else:
            profiles = list(noisy_profiles)
    else:
        profiles = list(noisy_profiles)
    
    # Initialize results dictionary
    fit_dictionary = {
        'sigma^2_t estimates': [],
        'sigma^2_t standard errors': [],
    }
    
    # Get x-axis properties
    xpix = len(x_axis)
    xmin, xmax = np.min(x_axis), np.max(x_axis)
    xwid = np.abs(xmax - xmin)
    
    for this_profile in profiles:
        this_profile = validate_array_like(this_profile, "profile", ndim=1)
        
        if len(this_profile) != len(x_axis):
            raise ValueError("Profile and x_axis must have the same length")
        
        # Set bounds for mu (central region of the window)
        mu_margin = xwid * 0.3  # Allow mu within central 60% of window
        mu0_min = xmin + mu_margin
        mu0_max = xmax - mu_margin

        # Ensure valid mu bounds
        if mu0_min >= mu0_max:
            # Fallback: use full range
            mu0_min = xmin
            mu0_max = xmax

        # Find maximum amplitude within the mu bounds
        valid_mask = (x_axis >= mu0_min) & (x_axis <= mu0_max)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) > 0:
            # Find maximum within valid region
            valid_profile = this_profile[valid_mask]
            max_amp_idx_in_valid = np.argmax(np.abs(valid_profile))
            max_amp_idx = valid_indices[max_amp_idx_in_valid]
        else:
            # Fallback: use global maximum
            max_amp_idx = np.argmax(np.abs(this_profile))

        # Initial guesses
        mu0 = x_axis[max_amp_idx]  # Mean at maximum within bounds
        sigma2_0 = (xwid / 4) ** 2  # Sigma^2 guess: 1/16 of squared scan width
        a0 = np.abs(this_profile[max_amp_idx])  # Amplitude guess: absolute max value within bounds
        
        # Set bounds for sigma^2
        sigma2_min = (xwid / xpix) ** 2  # Minimum: ~1 pixel width
        sigma2_max = xwid ** 2  # Maximum: entire window width
        
        # Set bounds for amplitude
        a0_min = 0  # Non-negative amplitude
        a0_max = 2 * np.max(np.abs(this_profile))  # Maximum: 2x the max value
        
        # Ensure amplitude bounds are valid
        if a0_max <= 0:
            a0_max = 1.0  # Fallback for zero profile
        
        # Initial parameters and bounds
        p0 = [mu0, sigma2_0, a0]
        bounds_min = [mu0_min, sigma2_min, a0_min]
        bounds_max = [mu0_max, sigma2_max, a0_max]
        
        try:
            # Perform the fit
            parms, covars = curve_fit(
                gaussian, x_axis, this_profile,
                p0=p0,
                bounds=(bounds_min, bounds_max),
                maxfev=5000
            )
            
            # Extract sigma^2 and its standard error
            sigma2_estimate = parms[1]
            
            # Get standard errors from covariance matrix
            if covars is not None and covars.shape == (3, 3):
                variances = np.diag(covars)
                stdev_table = np.sqrt(np.abs(variances))
                sigma2_stderr = stdev_table[1]
            else:
                # Fallback if covariance matrix is invalid
                sigma2_stderr = np.nan
            
        except (RuntimeError, ValueError) as e:
            # Fitting failed - use fallback values
            print(f"Warning: Gaussian fitting failed: {e}")
            sigma2_estimate = sigma2_0
            sigma2_stderr = np.nan
        
        fit_dictionary['sigma^2_t estimates'].append(sigma2_estimate)
        fit_dictionary['sigma^2_t standard errors'].append(sigma2_stderr)
    
    return fit_dictionary


def fit_gaussian_profile(x: np.ndarray, y: np.ndarray, 
                        initial_guess: Optional[Tuple[float, float, float]] = None) -> Dict:
    """
    Fit a single Gaussian profile.
    
    Parameters
    ----------
    x : np.ndarray
        X-axis values.
    y : np.ndarray
        Y-axis values (profile data).
    initial_guess : tuple, optional
        Initial guess for (mu, sigma2, amplitude).
    
    Returns
    -------
    dict
        Fitted parameters and errors.
    """
    x = validate_array_like(x, "x", ndim=1)
    y = validate_array_like(y, "y", ndim=1)
    
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    
    if initial_guess is None:
        # Auto-generate initial guess
        mu_guess = x[np.argmax(np.abs(y))]
        sigma2_guess = ((np.max(x) - np.min(x)) / 4) ** 2
        amp_guess = np.max(np.abs(y))
        initial_guess = (mu_guess, sigma2_guess, amp_guess)
    
    try:
        parms, covars = curve_fit(gaussian, x, y, p0=initial_guess)
        
        # Calculate standard errors
        if covars is not None:
            errors = np.sqrt(np.diag(np.abs(covars)))
        else:
            errors = [np.nan, np.nan, np.nan]
        
        return {
            'mu': parms[0],
            'sigma2': parms[1],
            'amplitude': parms[2],
            'mu_error': errors[0],
            'sigma2_error': errors[1],
            'amplitude_error': errors[2],
            'success': True
        }
    except Exception as e:
        return {
            'mu': initial_guess[0],
            'sigma2': initial_guess[1],
            'amplitude': initial_guess[2],
            'mu_error': np.nan,
            'sigma2_error': np.nan,
            'amplitude_error': np.nan,
            'success': False,
            'error': str(e)
        }


def diffusion_ols_fit(time_axis: np.ndarray, gaussfit_sigma2_t: np.ndarray) -> Dict:
    """
    Estimate diffusion coefficient using Ordinary Least Squares (OLS).
    
    Fits MSD vs. time to extract the diffusion coefficient from the slope.
    MSD(t) = sigma^2(t) - sigma^2(0) = 2*D*t (in 1D)
    
    Parameters
    ----------
    time_axis : np.ndarray
        Time points for each measurement.
    gaussfit_sigma2_t : np.ndarray
        Variance (sigma^2) from Gaussian fits at each time point.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'MSD_t slope estimate': Slope of MSD vs. time (2*D in 1D)
        - 'intercept estimate': Y-intercept of the fit
        - 'MSD_t slope std error': Standard error of the slope
        - 'intercept standard error': Standard error of the intercept
        - 'r_squared': Coefficient of determination
    
    Raises
    ------
    ValueError
        If arrays have incompatible lengths or insufficient data.
    """
    time_axis = validate_array_like(time_axis, "time_axis")
    gaussfit_sigma2_t = validate_array_like(gaussfit_sigma2_t, "gaussfit_sigma2_t")
    
    if len(gaussfit_sigma2_t) < 2:
        raise ValueError("Need at least 2 points for linear regression")
    
    if len(time_axis) != len(gaussfit_sigma2_t):
        raise ValueError("time_axis and gaussfit_sigma2_t must have the same length")
    
    # Calculate MSD: change in variance from t=0
    msd = gaussfit_sigma2_t - gaussfit_sigma2_t[0]
    
    # Prepare design matrix for OLS (add constant for intercept)
    X = sm.add_constant(time_axis)
    
    # Fit the model
    ols_model = sm.OLS(msd, X).fit()
    
    return {
        'MSD_t slope estimate': ols_model.params[1],
        'intercept estimate': ols_model.params[0],
        'MSD_t slope std error': ols_model.bse[1],
        'intercept standard error': ols_model.bse[0],
        'r_squared': ols_model.rsquared
    }


def diffusion_wls_fit(time_axis: np.ndarray, gaussfit_sigma2_t: np.ndarray, 
                     weights: Optional[np.ndarray] = None) -> Dict:
    """
    Estimate diffusion coefficient using Weighted Least Squares (WLS).
    
    WLS accounts for heteroscedasticity in the MSD data by applying
    weights to each measurement.
    
    Parameters
    ----------
    time_axis : np.ndarray
        Time points for each measurement.
    gaussfit_sigma2_t : np.ndarray
        Variance (sigma^2) from Gaussian fits at each time point.
    weights : np.ndarray, optional
        Weights for each measurement. If None, uses inverse variance weighting.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'MSD_t slope estimate': Slope of MSD vs. time (2*D in 1D)
        - 'intercept estimate': Y-intercept of the fit
        - 'MSD_t slope std error': Standard error of the slope
        - 'intercept standard error': Standard error of the intercept
        - 'r_squared': Coefficient of determination
    
    Raises
    ------
    ValueError
        If arrays have incompatible lengths.
    """
    time_axis = validate_array_like(time_axis, "time_axis")
    gaussfit_sigma2_t = validate_array_like(gaussfit_sigma2_t, "gaussfit_sigma2_t")
    
    if len(gaussfit_sigma2_t) < 2:
        raise ValueError("Need at least 2 points for linear regression")
    
    if len(time_axis) != len(gaussfit_sigma2_t):
        raise ValueError("time_axis and gaussfit_sigma2_t must have the same length")
    
    # Calculate MSD
    msd = gaussfit_sigma2_t - gaussfit_sigma2_t[0]
    
    # Generate weights if not provided
    if weights is None:
        # Use inverse variance weighting
        # Weight inversely proportional to the variance at each time
        # Avoid division by zero
        min_sigma2 = np.maximum(gaussfit_sigma2_t, 1e-10)
        weights = 1.0 / min_sigma2
        # Normalize weights
        weights = weights / np.sum(weights) * len(weights)
    else:
        weights = validate_array_like(weights, "weights")
        if len(weights) != len(time_axis):
            raise ValueError("weights must have the same length as time_axis")
    
    # Prepare design matrix for WLS
    X = sm.add_constant(time_axis)
    
    # Fit the model using WLS
    wls_model = sm.WLS(msd, X, weights=weights).fit()
    
    return {
        'MSD_t slope estimate': wls_model.params[1],
        'intercept estimate': wls_model.params[0],
        'MSD_t slope std error': wls_model.bse[1],
        'intercept standard error': wls_model.bse[0],
        'r_squared': wls_model.rsquared
    }


def calculate_fit_weights(sigma2_errors: np.ndarray, 
                         method: str = 'inverse_variance') -> np.ndarray:
    """
    Calculate weights for weighted least squares fitting.
    
    Parameters
    ----------
    sigma2_errors : np.ndarray
        Standard errors of variance estimates.
    method : str
        Weighting method ('inverse_variance' or 'uniform').
    
    Returns
    -------
    np.ndarray
        Weights for each data point.
    """
    sigma2_errors = validate_array_like(sigma2_errors, "sigma2_errors")
    
    if method == 'inverse_variance':
        # Weight by inverse of variance (1/sigma^2)
        # Avoid division by zero
        safe_errors = np.maximum(sigma2_errors, 1e-10)
        weights = 1.0 / (safe_errors ** 2)
        # Normalize
        weights = weights / np.mean(weights)
    elif method == 'uniform':
        # Equal weights
        weights = np.ones_like(sigma2_errors)
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    return weights