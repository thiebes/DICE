"""
A simulation run's estimates equal the fitting primitives composed by hand.

run_single_simulation is called with a fixed seed, and the same profiles are
rebuilt from make_diffusion_decay and add_noise with that seed. Every slope-fit
field must then equal gauss_fitting, calculate_wls_weights, diffusion_ols_fit
and diffusion_wls_fit applied directly: exact equality, with NaN equal to NaN.
This pins the per-run chain so that rerouting it changes no number.
"""

import numpy as np
import pytest
from fft_cnr import fft_cnr

from dice.analysis.simulation import run_single_simulation
from dice.core.fitting import (
    calculate_wls_weights,
    diffusion_ols_fit,
    diffusion_wls_fit,
    gauss_fitting,
)
from dice.core.noise import add_noise
from dice.core.profiles import make_diffusion_decay
from dice.utils.legacy_compatibility import create_parameters_from_legacy

X_AXIS = np.linspace(-2.5, 2.5, 60)
TIME_AXIS = np.linspace(0.0, 1.0, 5)
NOISE_SIGMA = 0.03

# Keys of the dicts returned by diffusion_ols_fit and diffusion_wls_fit, by RunResult field suffix.
FIT_FIELDS = {
    'slope': 'MSD_t slope estimate',
    'slope_stderr': 'MSD_t slope std error',
    'intercept': 'intercept estimate',
    'intercept_stderr': 'intercept standard error',
}


@pytest.fixture(scope='module')
def parameters():
    return create_parameters_from_legacy(
        {'sigma^2_0': 0.18033688, 'amplitude_0': 1.0, 'mu_0': 0.0},
        diffusion_coefficient=0.01, lifetime=1.0, diffusion_length=0.1)


def _noisy_profiles(parameters, time_axis, seed):
    nominal = make_diffusion_decay({
        'x axis': X_AXIS,
        'time axis': time_axis,
        'sigma^2_0': parameters.profile.sigma2_0,
        'amplitude_0': parameters.profile.amplitude_0,
        'mu_0': parameters.profile.mu_0,
        'nominal diffusion coefficient': parameters.physics.diffusion_coefficient,
        'nominal lifetime': parameters.physics.lifetime,
    })
    return add_noise(nominal['y_values_t'], noise_sigma=NOISE_SIGMA, seed=seed)['y_values_t']


def _same(a, b) -> bool:
    """Exact equality of two scalars, with NaN equal to NaN and the type preserved."""
    if type(a) is not type(b):
        return False
    return bool(a == b or (np.isnan(a) and np.isnan(b)))


@pytest.mark.parametrize('seed', [3, 31, 2024])
def test_run_fields_equal_the_fitting_primitives(parameters, seed):
    result = run_single_simulation(
        0, X_AXIS, TIME_AXIS, parameters, NOISE_SIGMA, retain_profile_data=True, seed=seed)

    noisy = _noisy_profiles(parameters, TIME_AXIS, seed)
    fits = gauss_fitting(X_AXIS, noisy)
    sigma2 = np.array(fits['sigma^2_t estimates'])
    sigma2_stderr = np.array(fits['sigma^2_t standard errors'])
    weights = calculate_wls_weights(sigma2, sigma2_stderr)
    expected = {
        'ols': diffusion_ols_fit(TIME_AXIS, sigma2),
        'wls': diffusion_wls_fit(TIME_AXIS, sigma2, weights),
    }

    for key, fit in expected.items():
        for suffix, fit_key in FIT_FIELDS.items():
            value = getattr(result, f'{key}_{suffix}')
            assert _same(value, fit[fit_key]), (key, suffix, value, fit[fit_key])
    assert result.cnr_0_estimate == fft_cnr(noisy[0]).cnr
    assert result.estimated_sigma2_0 == sigma2[0]
    np.testing.assert_array_equal(result.noisy_profiles, noisy)
    np.testing.assert_array_equal(result.fitted_sigma2_t, sigma2)
    np.testing.assert_array_equal(result.fitted_sigma2_stderrs, sigma2_stderr)
    np.testing.assert_array_equal(result.weights, weights)


def test_single_frame_run_has_no_slope_fits(parameters):
    result = run_single_simulation(
        0, X_AXIS, np.array([0.0]), parameters, NOISE_SIGMA, retain_profile_data=True, seed=5)

    for key in ('ols', 'wls'):
        for suffix in FIT_FIELDS:
            assert getattr(result, f'{key}_{suffix}') is None
    assert result.weights is None
    assert result.estimated_sigma2_0 == result.fitted_sigma2_t[0]


def test_profile_data_is_dropped_unless_retained(parameters):
    result = run_single_simulation(0, X_AXIS, TIME_AXIS, parameters, NOISE_SIGMA, seed=3)

    assert result.wls_slope is not None and result.ols_slope is not None
    assert result.weights is None
    assert result.fitted_sigma2_t is None


def test_simulation_records_the_estimators_that_ran(parameters):
    from dice.analysis.simulation import run_monte_carlo_simulation
    output = run_monte_carlo_simulation(parameters, X_AXIS, TIME_AXIS, noise_values=[NOISE_SIGMA],
                                        num_runs=2, multiprocessing=False, seed=8)

    assert output.estimator_keys == ('ols', 'wls')
    for key in output.estimator_keys:
        assert all(run.slope_fit(key) is not None for run in output.run_results)
