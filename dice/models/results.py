"""
Result dataclasses for DICE simulations.

This module defines dataclasses for organizing simulation results: the
result of each run and the collected output of a Monte Carlo simulation.
"""

from dataclasses import dataclass, field, fields
from typing import Optional, List, Any, Tuple
import numpy as np

from .estimators import DEFAULT_ESTIMATOR_KEYS, SlopeFit
from .parameters import NoiseRange


@dataclass
class RunResult:
    """
    Results from a single Monte Carlo simulation run.

    Each estimator in dice.models.estimators stores its SlopeFit in four
    fields named after its key: ``{key}_slope``, ``{key}_slope_stderr``,
    ``{key}_intercept`` and ``{key}_intercept_stderr``, all None when it
    produced no fit. Code that handles any estimator reads and writes them
    through slope_fit() and set_slope_fit().

    Attributes
    ----------
    run_id : int
        Identifier for this run.
    nominal_diffusion_coefficient : float
        Input diffusion coefficient.
    nominal_lifetime : float
        Input lifetime.
    nominal_diffusion_length : float
        Input diffusion length.
    nominal_sigma2_0 : float
        Input initial variance.
    noise_sigma : float
        Noise standard deviation used.
    cnr_0_estimate : float
        Estimated contrast-to-noise ratio at t=0.
    estimated_sigma2_0 : float, optional
        Estimated initial variance from fitting.
    ols_slope : float, optional
        OLS fit slope (2*D).
    ols_slope_stderr : float, optional
        Standard error of OLS slope.
    ols_intercept : float, optional
        OLS fit intercept.
    ols_intercept_stderr : float, optional
        Standard error of OLS intercept.
    wls_slope : float, optional
        WLS fit slope (2*D).
    wls_slope_stderr : float, optional
        Standard error of WLS slope.
    wls_intercept : float, optional
        WLS fit intercept.
    wls_intercept_stderr : float, optional
        Standard error of WLS intercept.
    nominal_profiles : np.ndarray, optional
        Nominal profile data if retained.
    noisy_profiles : np.ndarray, optional
        Noisy profile data if retained.
    fitted_sigma2_t : np.ndarray, optional
        Fitted variance values at each time point.
    fitted_sigma2_stderrs : np.ndarray, optional
        Standard errors of fitted variances.
    weights : np.ndarray, optional
        Weights used in WLS fitting.
    """
    # Core identifier
    run_id: int

    # Nominal physics parameters
    nominal_diffusion_coefficient: float
    nominal_lifetime: float
    nominal_diffusion_length: float
    nominal_sigma2_0: float

    # Noise and CNR
    noise_sigma: float
    cnr_0_estimate: float

    # Estimated values
    estimated_sigma2_0: Optional[float] = None

    # OLS fit results
    ols_slope: Optional[float] = None
    ols_slope_stderr: Optional[float] = None
    ols_intercept: Optional[float] = None
    ols_intercept_stderr: Optional[float] = None

    # WLS fit results
    wls_slope: Optional[float] = None
    wls_slope_stderr: Optional[float] = None
    wls_intercept: Optional[float] = None
    wls_intercept_stderr: Optional[float] = None

    # Optional profile data
    nominal_profiles: Optional[np.ndarray] = None
    noisy_profiles: Optional[np.ndarray] = None
    fitted_sigma2_t: Optional[np.ndarray] = None
    fitted_sigma2_stderrs: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None

    def slope_fit(self, key: str) -> Optional[SlopeFit]:
        """Estimator `key`'s fit for this run, or None if it produced none."""
        names = _slope_fit_field_names(key)
        if getattr(self, names[0]) is None:
            return None
        return SlopeFit(*(getattr(self, name) for name in names))

    def set_slope_fit(self, key: str, fit: Optional[SlopeFit]) -> None:
        """Store `fit` in estimator `key`'s fields; None clears them."""
        for name, slope_fit_field in zip(_slope_fit_field_names(key), fields(SlopeFit)):
            setattr(self, name, None if fit is None else getattr(fit, slope_fit_field.name))

    def accuracy(self, key: str) -> float:
        """Estimator `key`'s D estimate (slope / 2) over the nominal D; NaN when absent."""
        fit = self.slope_fit(key)
        if fit is not None and self.nominal_diffusion_coefficient > 0:
            return (fit.slope / 2) / self.nominal_diffusion_coefficient
        return float('nan')

    @property
    def ols_accuracy(self) -> float:
        """Calculate OLS estimate accuracy relative to nominal."""
        return self.accuracy('ols')

    @property
    def wls_accuracy(self) -> float:
        """Calculate WLS estimate accuracy relative to nominal."""
        return self.accuracy('wls')


def _slope_fit_field_names(key: str) -> Tuple[str, ...]:
    """
    The RunResult fields that hold estimator `key`'s SlopeFit, in SlopeFit field order.

    Raises
    ------
    AttributeError
        If RunResult has no such fields, so that a misspelled key cannot
        create new attributes or read as "no fit".
    """
    names = tuple(f'{key}_{slope_fit_field.name}' for slope_fit_field in fields(SlopeFit))
    missing = [name for name in names if name not in RunResult.__dataclass_fields__]
    if missing:
        raise AttributeError(f"RunResult has no fields {missing} for estimator {key!r}")
    return names


@dataclass
class MonteCarloOutput:
    """
    Raw output from Monte Carlo simulation runs.

    Contains the direct results from run_monte_carlo_simulation() before
    any statistical analysis is performed.

    Attributes
    ----------
    parameters : Any
        Simulation parameters object (LegacySimulationParameters, or None for CSV-loaded data).
    run_results : List[RunResult]
        Individual results from each Monte Carlo run.
    num_runs : int
        Total number of simulation runs completed.
    noise_values : List[float]
        Noise sigma values used in the simulation: the fixed noise levels,
        each used for the same number of runs, or, with noise_range, one
        value for each run, in run order.
    estimator_keys : Tuple[str, ...]
        Keys of the estimators that ran, as in dice.models.estimators. The
        analysis reports each of them, even one whose every fit failed.
    noise_range : NoiseRange, optional
        The range from which each run drew its own noise sigma; None when
        the runs used fixed noise levels.
    seed : int, optional
        The seed the simulation ran with, which reproduces it; None when it
        drew fresh entropy, or for results loaded from a file.
    """
    parameters: Any
    run_results: List[RunResult]
    num_runs: int
    noise_values: List[float] = field(default_factory=list)
    estimator_keys: Tuple[str, ...] = DEFAULT_ESTIMATOR_KEYS
    noise_range: Optional[NoiseRange] = None
    seed: Optional[int] = None

    def d_estimates(self, key: str) -> np.ndarray:
        """Estimator `key`'s D estimates (slope / 2) of the runs that have a fit, in run order."""
        _slope_fit_field_names(key)  # reject an unknown key even when there are no runs
        return np.array([fit.slope / 2 for run in self.run_results
                         if (fit := run.slope_fit(key)) is not None])
