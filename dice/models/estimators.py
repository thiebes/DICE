"""
Diffusion-coefficient estimators: the per-run slope fit and its output labels.

An estimator turns one run's per-frame fitted variances into a straight-line
fit of MSD against time; the slope is 2 D in one dimension. The registry in
this module is the one place that names an estimator. Results, statistics,
export, and the GUI enumerate it.
"""

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional, Set, Tuple

import numpy as np

# Keys of the dict returned by dice.core.fitting.diffusion_ols_fit and diffusion_wls_fit.
_FIT_DICT_KEYS = ('MSD_t slope estimate', 'MSD_t slope std error',
                  'intercept estimate', 'intercept standard error')


@dataclass(frozen=True)
class SlopeFit:
    """
    One estimator's straight-line fit of MSD against time, in simulation units.

    Attributes
    ----------
    slope : float
        MSD slope (2*D in 1D).
    slope_stderr : float, optional
        Standard error of the slope.
    intercept : float, optional
        Fit intercept.
    intercept_stderr : float, optional
        Standard error of the intercept.
    """
    slope: float
    slope_stderr: Optional[float] = None
    intercept: Optional[float] = None
    intercept_stderr: Optional[float] = None

    @classmethod
    def from_fit_dict(cls, fit: Mapping[str, float]) -> 'SlopeFit':
        """Build from a diffusion_ols_fit or diffusion_wls_fit result, rejecting a wrong schema."""
        missing = [key for key in _FIT_DICT_KEYS if key not in fit]
        if missing:
            raise KeyError(f"slope-fit result is missing {missing}; got {sorted(fit)}")
        return cls(slope=fit['MSD_t slope estimate'],
                   slope_stderr=fit['MSD_t slope std error'],
                   intercept=fit['intercept estimate'],
                   intercept_stderr=fit['intercept standard error'])


@dataclass(frozen=True)
class FrameFits:
    """
    One run's per-frame Gaussian-fit output: the input every estimator reads.

    Attributes
    ----------
    time_axis : np.ndarray
        Time of each frame, in the simulation time unit.
    sigma2 : np.ndarray
        Fitted variance of each frame, in the simulation length unit squared.
    sigma2_stderr : np.ndarray
        Standard error of each fitted variance, from the fit covariance.
    """
    time_axis: np.ndarray
    sigma2: np.ndarray
    sigma2_stderr: np.ndarray

    def __post_init__(self):
        lengths = {len(self.time_axis), len(self.sigma2), len(self.sigma2_stderr)}
        if len(lengths) != 1:
            raise ValueError(
                f"time_axis, sigma2 and sigma2_stderr must have the same length; got "
                f"{len(self.time_axis)}, {len(self.sigma2)} and {len(self.sigma2_stderr)}"
            )


@dataclass(frozen=True)
class EstimatorSpec:
    """
    A diffusion-coefficient estimator: its per-run fit and the labels every output uses.

    Attributes
    ----------
    key : str
        RunResult field prefix ('ols' -> ols_slope, ...) and analysis-dict key.
    csv_label : str
        CSV column prefix ('<csv_label> diffusion slope', ...). No label may
        begin with another, because CSV readers match columns by prefix.
    summary_name : str
        Name in the notes of the summary file, and, up to a comma, in its
        result when the estimator comes first in display order.
    short_name : str
        Column name in the tables of the summary file, and name in GUI messages.
    estimate : callable
        Fits one run's FrameFits; returns None when it produces no fit.
    """
    key: str
    csv_label: str
    summary_name: str
    short_name: str
    estimate: Callable[[FrameFits], Optional[SlopeFit]]


def _estimate_ols(fits: FrameFits) -> SlopeFit:
    # Imported here so that importing dice.models does not load statsmodels.
    from ..core.fitting import diffusion_ols_fit
    return SlopeFit.from_fit_dict(diffusion_ols_fit(fits.time_axis, fits.sigma2))


def _estimate_wls(fits: FrameFits) -> SlopeFit:
    from ..core.fitting import calculate_wls_weights, diffusion_wls_fit
    weights = calculate_wls_weights(fits.sigma2, fits.sigma2_stderr)
    return SlopeFit.from_fit_dict(diffusion_wls_fit(fits.time_axis, fits.sigma2, weights))


OLS = EstimatorSpec(key='ols', csv_label='unweighted fit',
                    summary_name='Ordinary (unweighted) Least Squares, for comparison',
                    short_name='OLS', estimate=_estimate_ols)
WLS = EstimatorSpec(key='wls', csv_label='weighted fit',
                    summary_name='Weighted Least Squares',
                    short_name='WLS', estimate=_estimate_wls)

# Column order of the CSV and the analysis dict (OLS block first).
ESTIMATORS: Tuple[EstimatorSpec, ...] = (OLS, WLS)
# Order of the summary text and the GUI (WLS, the default estimator, first).
# OLS is reported for comparison with it; the plots show the WLS estimates.
DISPLAY_ORDER: Tuple[str, ...] = ('wls', 'ols')
# Estimators every simulation runs; any further estimator is opt-in.
DEFAULT_ESTIMATOR_KEYS: Tuple[str, ...] = ('ols', 'wls')


def get_estimator(key: str) -> EstimatorSpec:
    """Return the registered estimator with this key; KeyError if there is none."""
    for spec in ESTIMATORS:
        if spec.key == key:
            return spec
    raise KeyError(f"unknown estimator {key!r}; known: {[spec.key for spec in ESTIMATORS]}")


def _known_keys(keys: Iterable[str]) -> Set[str]:
    """The set of `keys`, after checking that each names a registered estimator."""
    wanted = set(keys)
    for key in wanted:
        get_estimator(key)
    return wanted


def in_column_order(keys: Iterable[str]) -> Tuple[EstimatorSpec, ...]:
    """The estimators named in `keys`, in CSV and analysis-dict order; KeyError on an unknown key."""
    wanted = _known_keys(keys)
    return tuple(spec for spec in ESTIMATORS if spec.key in wanted)


def in_display_order(keys: Iterable[str]) -> Tuple[EstimatorSpec, ...]:
    """The estimators named in `keys`, in summary and GUI order; KeyError on an unknown key."""
    wanted = _known_keys(keys)
    return tuple(get_estimator(key) for key in DISPLAY_ORDER if key in wanted)
