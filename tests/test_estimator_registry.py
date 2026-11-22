"""
The estimator registry: keys, output labels, orders, lookup, and the slope-fit schema.

The labels are the literals the CSV, the summary, and the GUI messages used
before the registry existed, so these tests pin the output strings too.
"""

import subprocess
import sys

import numpy as np
import pytest

from dice.core.fitting import calculate_wls_weights, diffusion_ols_fit, diffusion_wls_fit
from dice.models import (
    DEFAULT_ESTIMATOR_KEYS,
    DISPLAY_ORDER,
    ESTIMATORS,
    FrameFits,
    SlopeFit,
    get_estimator,
)
from dice.models.estimators import in_column_order, in_display_order

FIT_DICT = {
    'MSD_t slope estimate': 0.02,
    'MSD_t slope std error': 0.001,
    'intercept estimate': 0.0003,
    'intercept standard error': 0.0001,
    'r_squared': 0.97,
}


class TestRegistry:

    def test_keys_in_column_order(self):
        assert [spec.key for spec in ESTIMATORS] == ['ols', 'wls']

    def test_labels_are_the_output_literals(self):
        assert [(s.key, s.csv_label, s.summary_name, s.short_name) for s in ESTIMATORS] == [
            ('ols', 'unweighted fit', 'Ordinary (unweighted) Least Squares, for comparison', 'OLS'),
            ('wls', 'weighted fit', 'Weighted Least Squares', 'WLS'),
        ]

    def test_orders(self):
        assert DISPLAY_ORDER == ('wls', 'ols')
        assert DEFAULT_ESTIMATOR_KEYS == ('ols', 'wls')
        assert set(DISPLAY_ORDER) == set(DEFAULT_ESTIMATOR_KEYS) == {s.key for s in ESTIMATORS}

    def test_no_csv_label_is_a_prefix_of_another(self):
        # CSV readers find columns by prefix, so 'weighted fit' must not match 'weighted fit, ...'.
        labels = [spec.csv_label for spec in ESTIMATORS]
        for label in labels:
            for other in labels:
                if other != label:
                    assert not other.startswith(label), (label, other)

    def test_lookup(self):
        assert get_estimator('wls').short_name == 'WLS'
        with pytest.raises(KeyError, match="unknown estimator 'x'"):
            get_estimator('x')

    def test_column_and_display_order_ignore_the_order_given(self):
        assert [s.key for s in in_column_order(['wls', 'ols'])] == ['ols', 'wls']
        assert [s.key for s in in_display_order({'ols': 1, 'wls': 2})] == ['wls', 'ols']
        assert [s.key for s in in_display_order(['ols'])] == ['ols']
        assert in_column_order([]) == ()

    @pytest.mark.parametrize('order', [in_column_order, in_display_order])
    def test_orders_reject_an_unknown_key(self, order):
        with pytest.raises(KeyError, match="unknown estimator 'wsl'"):
            order(['ols', 'wsl'])

    def test_importing_models_does_not_load_the_fitting_stack(self):
        code = ("import sys, dice.models; "
                "print('statsmodels' in sys.modules, 'dice.core.fitting' in sys.modules)")
        output = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                                check=True).stdout
        assert output.split() == ['False', 'False']


class TestSlopeFitSchema:

    def test_from_fit_dict_maps_each_key(self):
        assert SlopeFit.from_fit_dict(FIT_DICT) == SlopeFit(
            slope=0.02, slope_stderr=0.001, intercept=0.0003, intercept_stderr=0.0001)

    @pytest.mark.parametrize('missing', ['MSD_t slope estimate', 'intercept standard error'])
    def test_from_fit_dict_names_a_missing_key(self, missing):
        malformed = {key: value for key, value in FIT_DICT.items() if key != missing}
        with pytest.raises(KeyError, match=missing):
            SlopeFit.from_fit_dict(malformed)

    def test_from_fit_dict_rejects_a_renamed_key(self):
        renamed = dict(FIT_DICT)
        renamed['MSD slope estimate'] = renamed.pop('MSD_t slope estimate')
        with pytest.raises(KeyError, match='MSD_t slope estimate'):
            SlopeFit.from_fit_dict(renamed)


class TestEstimates:
    """Each registered estimate equals its fitting function applied directly."""

    TIME = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    SIGMA2 = np.array([0.1803, 0.1861, 0.1898, 0.1957, 0.2001])
    SIGMA2_STDERR = np.array([0.0021, 0.0024, 0.0022, 0.0027, 0.0026])

    def _frame_fits(self) -> FrameFits:
        return FrameFits(time_axis=self.TIME, sigma2=self.SIGMA2, sigma2_stderr=self.SIGMA2_STDERR)

    def test_ols(self):
        expected = SlopeFit.from_fit_dict(diffusion_ols_fit(self.TIME, self.SIGMA2))
        assert get_estimator('ols').estimate(self._frame_fits()) == expected

    def test_wls(self):
        weights = calculate_wls_weights(self.SIGMA2, self.SIGMA2_STDERR)
        expected = SlopeFit.from_fit_dict(diffusion_wls_fit(self.TIME, self.SIGMA2, weights))
        assert get_estimator('wls').estimate(self._frame_fits()) == expected

    def test_frame_fits_reject_mismatched_lengths(self):
        with pytest.raises(ValueError, match='same length; got 5, 4 and 5'):
            FrameFits(time_axis=self.TIME, sigma2=self.SIGMA2[:4], sigma2_stderr=self.SIGMA2_STDERR)
