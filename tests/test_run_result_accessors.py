"""
RunResult and MonteCarloOutput accessors that address an estimator by its registry key.

slope_fit, set_slope_fit, accuracy, and d_estimates must agree exactly with the
flat ols_* and wls_* fields and with the comprehensions they replace, and must
reject a key that has no fields instead of reading it as "no fit".
"""

import math
import pickle

import numpy as np
import pytest

from dice.models import DEFAULT_ESTIMATOR_KEYS, ESTIMATORS, SlopeFit
from dice.models.results import MonteCarloOutput, RunResult

NOMINAL_D = 0.01


def _run(run_id=0, nominal_d=NOMINAL_D, **fit_fields) -> RunResult:
    return RunResult(run_id=run_id, nominal_diffusion_coefficient=nominal_d, nominal_lifetime=1.0,
                     nominal_diffusion_length=0.1, nominal_sigma2_0=0.18, noise_sigma=0.02,
                     cnr_0_estimate=50.0, **fit_fields)


def _output(runs) -> MonteCarloOutput:
    return MonteCarloOutput(parameters=None, run_results=runs, num_runs=len(runs))


RUNS = [
    _run(0, ols_slope=np.float64(0.0213), ols_slope_stderr=np.float64(0.0011),
         ols_intercept=np.float64(0.0004), ols_intercept_stderr=np.float64(0.0002),
         wls_slope=np.float64(0.0208), wls_slope_stderr=np.float64(0.0009),
         wls_intercept=np.float64(0.0003), wls_intercept_stderr=np.float64(0.0001)),
    _run(1),
    _run(2, ols_slope=0.0191, ols_slope_stderr=0.0, wls_slope=0.0199,
         wls_slope_stderr=float('nan')),
    _run(3, wls_slope=0.0261),
]


class TestSlopeFit:

    @pytest.mark.parametrize('run', RUNS)
    @pytest.mark.parametrize('key', ['ols', 'wls'])
    def test_matches_the_flat_fields(self, run, key):
        fit = run.slope_fit(key)
        if getattr(run, f'{key}_slope') is None:
            assert fit is None
        else:
            assert (fit.slope, fit.slope_stderr, fit.intercept, fit.intercept_stderr) == tuple(
                getattr(run, f'{key}_{suffix}')
                for suffix in ('slope', 'slope_stderr', 'intercept', 'intercept_stderr'))
            assert type(fit.slope) is type(getattr(run, f'{key}_slope'))

    def test_every_registered_estimator_has_fields(self):
        for spec in ESTIMATORS:
            assert _run().slope_fit(spec.key) is None

    def test_set_then_get_round_trips(self):
        run = _run()
        fit = SlopeFit(slope=0.02, slope_stderr=0.001, intercept=0.0003, intercept_stderr=0.0001)
        run.set_slope_fit('wls', fit)
        assert run.slope_fit('wls') == fit
        assert (run.wls_slope, run.wls_slope_stderr, run.wls_intercept,
                run.wls_intercept_stderr) == (0.02, 0.001, 0.0003, 0.0001)
        assert run.slope_fit('ols') is None

    def test_set_none_clears_every_field(self):
        run = _run(ols_slope=0.02, ols_slope_stderr=0.001, ols_intercept=0.1,
                   ols_intercept_stderr=0.01)
        run.set_slope_fit('ols', None)
        assert (run.ols_slope, run.ols_slope_stderr, run.ols_intercept,
                run.ols_intercept_stderr) == (None, None, None, None)

    def test_slope_only_fit_leaves_the_rest_none(self):
        run = _run()
        run.set_slope_fit('ols', SlopeFit(slope=0.02))
        assert run.slope_fit('ols') == SlopeFit(0.02, None, None, None)

    @pytest.mark.parametrize('key', ['nope', 'OLS', 'weighted fit'])
    def test_unknown_key_is_rejected(self, key):
        run = _run()
        with pytest.raises(AttributeError, match=f"for estimator {key!r}"):
            run.slope_fit(key)
        with pytest.raises(AttributeError, match=f"for estimator {key!r}"):
            run.set_slope_fit(key, SlopeFit(slope=0.02))
        assert not hasattr(run, f'{key}_slope')


class TestAccuracy:

    @pytest.mark.parametrize('nominal_d', [NOMINAL_D, 0.0, -1.0])
    @pytest.mark.parametrize('run_index', range(len(RUNS)))
    def test_matches_the_properties(self, run_index, nominal_d):
        run = RUNS[run_index]
        run = _run(nominal_d=nominal_d, ols_slope=run.ols_slope, wls_slope=run.wls_slope)
        for key, value in (('ols', run.ols_accuracy), ('wls', run.wls_accuracy)):
            slope = getattr(run, f'{key}_slope')
            expected = ((slope / 2) / nominal_d
                        if slope is not None and nominal_d > 0 else float('nan'))
            assert run.accuracy(key) == value or (math.isnan(value) and math.isnan(expected))
            assert value == expected or (math.isnan(value) and math.isnan(expected))


class TestDEstimates:

    @pytest.mark.parametrize('runs', [RUNS, [], [_run(0), _run(1)], RUNS[:1]])
    @pytest.mark.parametrize('key', ['ols', 'wls'])
    def test_equals_the_legacy_comprehension(self, runs, key):
        legacy = np.array([getattr(r, f'{key}_slope') for r in runs
                           if getattr(r, f'{key}_slope') is not None]) / 2
        estimates = _output(runs).d_estimates(key)
        assert estimates.dtype == legacy.dtype == np.float64
        np.testing.assert_array_equal(estimates, legacy)

    def test_unknown_key_is_rejected_without_runs(self):
        with pytest.raises(AttributeError, match="for estimator 'nope'"):
            _output([]).d_estimates('nope')

    def test_estimator_keys_default(self):
        assert _output(RUNS).estimator_keys == DEFAULT_ESTIMATOR_KEYS == ('ols', 'wls')


def test_pickle_round_trip():
    output = MonteCarloOutput(parameters=None, run_results=list(RUNS), num_runs=len(RUNS),
                              noise_values=[0.02], estimator_keys=('ols',))
    restored = pickle.loads(pickle.dumps(output))
    assert restored.estimator_keys == ('ols',)
    for key in ('ols', 'wls'):
        np.testing.assert_array_equal(restored.d_estimates(key), output.d_estimates(key))
        assert [r.slope_fit(key) for r in restored.run_results[:2]] == [
            r.slope_fit(key) for r in output.run_results[:2]]


def test_output_pickled_before_estimator_keys_existed_reads_the_default():
    output = _output(list(RUNS))
    del output.__dict__['estimator_keys']  # the state an older MonteCarloOutput pickles
    restored = pickle.loads(pickle.dumps(output))
    assert restored.estimator_keys == DEFAULT_ESTIMATOR_KEYS
