"""
Tests for dice.models.results module, focusing on filter_by_cnr.
"""

import pytest
import numpy as np
import pandas as pd
from dice.models.results import (
    RunResult,
    StatisticalAnalysis,
    ProcessedSimulationResult,
)


def _make_run_result(run_id, cnr, ols_slope=1.0, wls_slope=1.0, noise_sigma=0.05):
    """Create a RunResult with the given CNR and slope values."""
    return RunResult(
        run_id=run_id,
        nominal_diffusion_coefficient=0.5,
        nominal_lifetime=2.0,
        nominal_diffusion_length=1.0,
        nominal_sigma2_0=0.18,
        noise_sigma=noise_sigma,
        cnr_0_estimate=cnr,
        estimated_sigma2_0=0.18,
        ols_slope=ols_slope,
        ols_slope_stderr=0.01,
        ols_intercept=0.18,
        ols_intercept_stderr=0.01,
        wls_slope=wls_slope,
        wls_slope_stderr=0.01,
        wls_intercept=0.18,
        wls_intercept_stderr=0.01,
    )


def _make_processed_result(run_results, proximity_level=0.1):
    """Create a ProcessedSimulationResult from a list of RunResults."""
    wls_d = np.array([r.wls_slope / 2 for r in run_results if r.wls_slope is not None])
    nominal_d = run_results[0].nominal_diffusion_coefficient

    ratios = wls_d / nominal_d
    within = np.logical_and(ratios >= 1 - proximity_level, ratios <= 1 + proximity_level)
    fraction = np.mean(within) if len(within) > 0 else 0.0

    analysis = StatisticalAnalysis(
        proximity_level=proximity_level,
        fraction_within_proximity=float(fraction),
        mean_estimate=float(np.mean(wls_d)) if len(wls_d) > 0 else np.nan,
        median_estimate=float(np.median(wls_d)) if len(wls_d) > 0 else np.nan,
        std_estimate=float(np.std(wls_d)) if len(wls_d) > 0 else np.nan,
        mean_relative_error=float(np.mean(wls_d) / nominal_d) if len(wls_d) > 0 else np.nan,
    )

    data = []
    for run in run_results:
        data.append({
            'run_id': run.run_id,
            'nominal_diffusion_coefficient': run.nominal_diffusion_coefficient,
            'nominal_lifetime': run.nominal_lifetime,
            'nominal_diffusion_length': run.nominal_diffusion_length,
            'nominal_sigma2_0': run.nominal_sigma2_0,
            'noise_sigma': run.noise_sigma,
            'cnr_0_estimate': run.cnr_0_estimate,
            'estimated_sigma2_0': run.estimated_sigma2_0,
            'ols_slope': run.ols_slope,
            'wls_slope': run.wls_slope,
        })
    df = pd.DataFrame(data)

    return ProcessedSimulationResult(
        parameters={'nominal diffusion coefficient': nominal_d},
        run_results=run_results,
        analysis=analysis,
        dataframe=df,
        filename_slug='test',
    )


class TestFilterByCnr:
    """Tests for ProcessedSimulationResult.filter_by_cnr."""

    def _make_runs_with_varying_cnr(self):
        """Create 10 runs with CNR values from 5 to 50."""
        return [
            _make_run_result(i, cnr=5.0 + i * 5, wls_slope=1.0)
            for i in range(10)
        ]

    def test_filter_min_cnr(self):
        """Filtering by min_cnr excludes runs below threshold."""
        runs = self._make_runs_with_varying_cnr()
        result = _make_processed_result(runs)

        filtered = result.filter_by_cnr(min_cnr=25.0)

        assert len(filtered.run_results) == 6
        assert all(r.cnr_0_estimate >= 25.0 for r in filtered.run_results)

    def test_filter_max_cnr(self):
        """Filtering by max_cnr excludes runs above threshold."""
        runs = self._make_runs_with_varying_cnr()
        result = _make_processed_result(runs)

        filtered = result.filter_by_cnr(max_cnr=30.0)

        assert len(filtered.run_results) == 6
        assert all(r.cnr_0_estimate <= 30.0 for r in filtered.run_results)

    def test_filter_cnr_range(self):
        """Filtering by both min and max CNR selects the correct subset."""
        runs = self._make_runs_with_varying_cnr()
        result = _make_processed_result(runs)

        filtered = result.filter_by_cnr(min_cnr=15.0, max_cnr=35.0)

        assert len(filtered.run_results) == 5
        assert all(15.0 <= r.cnr_0_estimate <= 35.0 for r in filtered.run_results)

    def test_filter_no_args_returns_all(self):
        """Calling filter_by_cnr with no arguments returns all runs."""
        runs = self._make_runs_with_varying_cnr()
        result = _make_processed_result(runs)

        filtered = result.filter_by_cnr()

        assert len(filtered.run_results) == len(runs)

    def test_filter_recalculates_analysis(self):
        """Analysis is recalculated for the filtered subset, not stale."""
        # Create runs with different slopes: low-CNR runs get slope far from nominal,
        # high-CNR runs get slope exactly at nominal (D=0.5, so slope=1.0).
        runs = []
        for i in range(10):
            cnr = 5.0 + i * 5
            # Low-CNR runs: slope=2.0 (D_est=1.0, 100% off from nominal 0.5)
            # High-CNR runs: slope=1.0 (D_est=0.5, exactly at nominal)
            wls_slope = 2.0 if cnr < 25.0 else 1.0
            runs.append(_make_run_result(i, cnr=cnr, wls_slope=wls_slope))

        result = _make_processed_result(runs, proximity_level=0.1)

        # Filter to only high-CNR runs (slope=1.0 => D=0.5 => perfect match)
        filtered = result.filter_by_cnr(min_cnr=25.0)

        # All filtered runs have D_est = 0.5 = nominal, so fraction within 10% = 1.0
        assert filtered.analysis.fraction_within_proximity == 1.0
        assert np.isclose(filtered.analysis.mean_estimate, 0.5)

        # The original result should have a lower precision (mixed good + bad runs)
        assert result.analysis.fraction_within_proximity < 1.0

    def test_filter_empty_result(self):
        """Filtering that excludes all runs returns empty result with nan analysis."""
        runs = self._make_runs_with_varying_cnr()
        result = _make_processed_result(runs)

        with pytest.raises(ValueError, match="No run results provided"):
            result.filter_by_cnr(min_cnr=1000.0)

    def test_filter_dataframe_matches_runs(self):
        """Filtered DataFrame has same number of rows as filtered run_results."""
        runs = self._make_runs_with_varying_cnr()
        result = _make_processed_result(runs)

        filtered = result.filter_by_cnr(min_cnr=20.0)

        assert len(filtered.dataframe) == len(filtered.run_results)

    def test_filter_filename_slug_appended(self):
        """Filtered result has '_filtered' appended to filename slug."""
        runs = self._make_runs_with_varying_cnr()
        result = _make_processed_result(runs)

        filtered = result.filter_by_cnr(min_cnr=20.0)

        assert filtered.filename_slug == 'test_filtered'

    def test_filter_preserves_parameters(self):
        """Filtered result preserves the original parameters dict."""
        runs = self._make_runs_with_varying_cnr()
        result = _make_processed_result(runs)

        filtered = result.filter_by_cnr(min_cnr=20.0)

        assert filtered.parameters == result.parameters
