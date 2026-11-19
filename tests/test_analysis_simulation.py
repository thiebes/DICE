"""
Tests for dice.analysis.simulation module.
"""

import pytest
import warnings
import numpy as np
from dice.analysis.simulation import (
    run_single_simulation,
    calculate_wls_weights,
    run_monte_carlo_simulation,
)


def get_legacy_simulation_params():
    """
    Get legacy simulation parameters for testing.

    Uses MockSimulationParameters with deprecation warning suppressed.
    For new tests, prefer using the default_simulation_params fixture.
    """
    from dice.utils.legacy_compatibility import MockSimulationParameters
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return MockSimulationParameters()


class TestCalculateWLSWeights:
    """Test weight calculation for WLS fitting."""
    
    def test_basic_weights(self):
        """Test basic weight calculation."""
        sigma2_values = np.array([1.0, 2.0, 3.0, 4.0])
        # Different relative errors: 0.05, 0.1, 0.15, 0.2
        sigma2_errors = np.array([0.05, 0.2, 0.45, 0.8])
        
        weights = calculate_wls_weights(sigma2_values, sigma2_errors)
        
        assert len(weights) == 4
        # Weights should be normalized to sum to number of points
        assert np.sum(weights) == pytest.approx(4.0, rel=0.01)
        # Smaller relative errors should have larger weights
        assert weights[0] > weights[1] > weights[2] > weights[3]
    
    def test_zero_error_handling(self):
        """Test handling of zero errors."""
        sigma2_values = np.array([1.0, 2.0, 3.0])
        sigma2_errors = np.array([0.1, 0.0, 0.3])  # Zero error
        
        weights = calculate_wls_weights(sigma2_values, sigma2_errors)
        
        assert weights[1] == 0  # Zero weight for zero error
        assert weights[0] > 0
        assert weights[2] > 0
    
    def test_zero_value_handling(self):
        """Test handling of zero values."""
        sigma2_values = np.array([0.0, 2.0, 3.0])  # Zero value
        sigma2_errors = np.array([0.1, 0.2, 0.3])
        
        weights = calculate_wls_weights(sigma2_values, sigma2_errors)
        
        assert weights[0] == 0  # Zero weight for zero value
        assert weights[1] > 0
        assert weights[2] > 0
    
    def test_nan_handling(self):
        """Test handling of NaN errors."""
        sigma2_values = np.array([1.0, 2.0, 3.0])
        sigma2_errors = np.array([0.1, np.nan, 0.3])  # NaN error
        
        weights = calculate_wls_weights(sigma2_values, sigma2_errors)
        
        assert weights[1] == 0  # Zero weight for NaN
        assert weights[0] > 0
        assert weights[2] > 0
    
    def test_all_zero_fallback(self):
        """Test fallback when all weights would be zero."""
        sigma2_values = np.array([0.0, 0.0, 0.0])
        sigma2_errors = np.array([0.1, 0.2, 0.3])
        
        weights = calculate_wls_weights(sigma2_values, sigma2_errors)
        
        # Should fall back to uniform weights
        assert np.all(weights == 1.0)


class TestRunSingleSimulation:
    """Test single simulation runs."""
    
    def test_basic_simulation(self):
        """Test basic simulation run."""
        # Set up parameters
        params = get_legacy_simulation_params()
        params.profile.sigma2_0 = 1.0
        params.profile.amplitude_0 = 1.0
        params.profile.mu_0 = 0.0
        params.physics.diffusion_coefficient = 0.5
        params.physics.lifetime = 10.0
        params.physics.diffusion_length = np.sqrt(0.5 * 10.0)
        
        x_axis = np.linspace(-10, 10, 201)
        time_axis = np.array([0, 1, 2, 3, 4])
        
        result = run_single_simulation(
            run_id=0,
            x_axis=x_axis,
            time_axis=time_axis,
            parameters=params,
            noise_sigma=0.01,
            retain_profile_data=False
        )
        
        assert result.run_id == 0
        assert result.nominal_diffusion_coefficient == 0.5
        assert result.nominal_lifetime == 10.0
        assert result.noise_sigma == 0.01
        assert result.cnr_0_estimate > 0
        assert result.estimated_sigma2_0 > 0
        
        # Should have OLS and WLS results
        assert result.ols_slope is not None
        assert result.wls_slope is not None
    
    def test_single_timepoint(self):
        """Test with single time point (no diffusion fit)."""
        params = get_legacy_simulation_params()
        params.profile.sigma2_0 = 1.0
        params.profile.amplitude_0 = 1.0
        params.profile.mu_0 = 0.0
        params.physics.diffusion_coefficient = 0.5
        params.physics.lifetime = 10.0
        params.physics.diffusion_length = np.sqrt(5.0)
        
        x_axis = np.linspace(-10, 10, 201)
        time_axis = np.array([0])  # Single time point
        
        result = run_single_simulation(
            run_id=1,
            x_axis=x_axis,
            time_axis=time_axis,
            parameters=params,
            noise_sigma=0.05,
            retain_profile_data=False
        )
        
        assert result.run_id == 1
        # No diffusion fit possible with single time point
        assert result.ols_slope is None
        assert result.wls_slope is None
        assert result.cnr_0_estimate > 0
    
    def test_retain_profile_data(self):
        """Test retaining profile data."""
        params = get_legacy_simulation_params()
        params.profile.sigma2_0 = 1.0
        params.profile.amplitude_0 = 1.0
        params.profile.mu_0 = 0.0
        params.physics.diffusion_coefficient = 0.5
        params.physics.lifetime = 10.0
        params.physics.diffusion_length = np.sqrt(5.0)
        
        x_axis = np.linspace(-10, 10, 101)
        time_axis = np.array([0, 1, 2])
        
        result = run_single_simulation(
            run_id=2,
            x_axis=x_axis,
            time_axis=time_axis,
            parameters=params,
            noise_sigma=0.02,
            retain_profile_data=True
        )
        
        # Profile data should be retained
        assert hasattr(result, 'nominal_profiles')
        assert hasattr(result, 'noisy_profiles')
        assert hasattr(result, 'fitted_sigma2_t')
        assert hasattr(result, 'fitted_sigma2_stderrs')
        assert hasattr(result, 'weights')
        
        # Check shapes
        assert result.nominal_profiles.shape == (3, 101)
        assert result.noisy_profiles.shape == (3, 101)
        assert len(result.fitted_sigma2_t) == 3
    
    def test_high_noise(self):
        """Test with high noise level."""
        params = get_legacy_simulation_params()
        params.profile.sigma2_0 = 1.0
        params.profile.amplitude_0 = 1.0
        params.profile.mu_0 = 0.0
        params.physics.diffusion_coefficient = 0.5
        params.physics.lifetime = 10.0
        params.physics.diffusion_length = np.sqrt(5.0)
        
        x_axis = np.linspace(-10, 10, 201)
        time_axis = np.array([0, 1, 2])
        
        result = run_single_simulation(
            run_id=3,
            x_axis=x_axis,
            time_axis=time_axis,
            parameters=params,
            noise_sigma=0.5,  # High noise
            retain_profile_data=False
        )
        
        # Should still complete, but CNR will be low
        assert result.cnr_0_estimate < 10  # Low CNR due to high noise
        assert result.ols_slope is not None  # Should still attempt fitting


class TestRunMonteCarloSimulation:
    """Test Monte Carlo simulation runs."""
    
    def test_basic_monte_carlo(self):
        """Test basic Monte Carlo simulation."""
        params = get_legacy_simulation_params()
        params.profile.sigma2_0 = 1.0
        params.profile.amplitude_0 = 1.0
        params.profile.mu_0 = 0.0
        params.physics.diffusion_coefficient = 0.5
        params.physics.lifetime = 10.0
        params.physics.diffusion_length = np.sqrt(5.0)
        
        x_axis = np.linspace(-10, 10, 101)
        time_axis = np.array([0, 1, 2])
        noise_values = [0.01, 0.02]
        num_runs = 5
        
        result = run_monte_carlo_simulation(
            parameters=params,
            x_axis=x_axis,
            time_axis=time_axis,
            noise_values=noise_values,
            num_runs=num_runs,
            multiprocessing=False,  # Easier to test
            retain_profile_data=False
        )
        
        # Should have correct number of runs
        assert result.num_runs == 10  # 5 runs × 2 noise values
        assert len(result.run_results) == 10
        assert result.noise_values == noise_values
        
        # Check run IDs are sequential
        run_ids = [r.run_id for r in result.run_results]
        assert run_ids == list(range(10))
        
        # Check noise values are distributed correctly
        noise_sigmas = [r.noise_sigma for r in result.run_results]
        assert noise_sigmas.count(0.01) == 5
        assert noise_sigmas.count(0.02) == 5
    
    def test_single_noise_value(self):
        """Test with single noise value."""
        params = get_legacy_simulation_params()
        params.profile.sigma2_0 = 1.0
        params.profile.amplitude_0 = 1.0
        params.profile.mu_0 = 0.0
        params.physics.diffusion_coefficient = 0.5
        params.physics.lifetime = 10.0
        params.physics.diffusion_length = np.sqrt(5.0)
        
        x_axis = np.linspace(-10, 10, 101)
        time_axis = np.array([0, 1])
        
        result = run_monte_carlo_simulation(
            parameters=params,
            x_axis=x_axis,
            time_axis=time_axis,
            noise_values=[0.05],
            num_runs=3,
            multiprocessing=False
        )
        
        assert result.num_runs == 3
        assert len(result.run_results) == 3
        assert all(r.noise_sigma == 0.05 for r in result.run_results)
    
    def test_progress_callback(self):
        """Test progress callback functionality."""
        params = get_legacy_simulation_params()
        params.profile.sigma2_0 = 1.0
        params.profile.amplitude_0 = 1.0
        params.profile.mu_0 = 0.0
        params.physics.diffusion_coefficient = 0.5
        params.physics.lifetime = 10.0
        params.physics.diffusion_length = np.sqrt(5.0)
        
        x_axis = np.linspace(-10, 10, 51)
        time_axis = np.array([0, 1])
        
        progress_calls = []
        
        def progress_callback(current, total):
            progress_calls.append((current, total))
        
        result = run_monte_carlo_simulation(
            parameters=params,
            x_axis=x_axis,
            time_axis=time_axis,
            noise_values=[0.05],
            num_runs=3,
            multiprocessing=False,
            progress_callback=progress_callback
        )
        
        # Should have been called 3 times
        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3)


class TestSeededSimulation:
    """A seed reproduces a simulation exactly; without one every run draws fresh entropy."""

    def _simulate(self, seed, noise_values=(0.02, 0.05), num_runs=3, **parallel_options):
        return run_monte_carlo_simulation(
            parameters=get_legacy_simulation_params(),
            x_axis=np.linspace(-10, 10, 51),
            time_axis=np.array([0.0, 1.0, 2.0]),
            noise_values=list(noise_values),
            num_runs=num_runs,
            multiprocessing=parallel_options.pop('multiprocessing', False),
            seed=seed,
            **parallel_options,
        )

    @staticmethod
    def _estimates(result) -> np.ndarray:
        return np.array([
            (r.cnr_0_estimate, r.estimated_sigma2_0, r.ols_slope, r.wls_slope)
            for r in result.run_results
        ])

    def test_same_seed_identical(self):
        np.testing.assert_array_equal(
            self._estimates(self._simulate(11)), self._estimates(self._simulate(11)))

    def test_different_seeds_differ(self):
        assert not np.array_equal(
            self._estimates(self._simulate(11)), self._estimates(self._simulate(12)))

    def test_unseeded_simulations_differ(self):
        assert not np.array_equal(
            self._estimates(self._simulate(None)), self._estimates(self._simulate(None)))

    def test_runs_draw_distinct_noise(self):
        estimates = self._estimates(self._simulate(11, noise_values=(0.02,), num_runs=4))
        assert len(np.unique(estimates[:, 0])) == 4

    def test_seed_sequence_reused_without_advancing(self):
        from numpy.random import SeedSequence
        root = SeedSequence(11)
        first, second = self._simulate(root), self._simulate(root)
        np.testing.assert_array_equal(self._estimates(first), self._estimates(second))
        np.testing.assert_array_equal(self._estimates(first), self._estimates(self._simulate(11)))

    def test_same_seed_identical_for_any_worker_count(self):
        # Starts joblib worker pools of two sizes, which costs tens of seconds.
        from joblib import effective_n_jobs
        run_counts = (3, effective_n_jobs(-1) + 3)  # below and above the full pool size
        run_count_to_serial = {
            num_runs: self._estimates(self._simulate(21, noise_values=(0.02,), num_runs=num_runs))
            for num_runs in run_counts
        }
        for n_jobs in (1, 2, -1):
            for num_runs, serial in run_count_to_serial.items():
                progress_calls = []
                parallel = self._simulate(
                    21, noise_values=(0.02,), num_runs=num_runs, multiprocessing=True,
                    n_jobs=n_jobs, progress_callback=lambda done, total: progress_calls.append((done, total)),
                )
                np.testing.assert_array_equal(self._estimates(parallel), serial)
                assert [r.run_id for r in parallel.run_results] == list(range(num_runs))
                assert progress_calls[-1] == (num_runs, num_runs)

    def test_run_seeds_are_children_of_one_root(self):
        from dice.analysis.simulation import spawn_run_seeds
        assert spawn_run_seeds(None, 3) == [None, None, None]
        children = spawn_run_seeds(11, 3)
        assert [child.spawn_key for child in children] == [(0,), (1,), (2,)]
        assert {child.entropy for child in children} == {11}


class TestProcessPoolBackend:
    """The standard library's process pool, which a standalone build uses, gives the serial results."""

    def _simulate(self, num_runs=7, noise_values=(0.02,), **options):
        return run_monte_carlo_simulation(
            parameters=get_legacy_simulation_params(),
            x_axis=np.linspace(-10, 10, 51),
            time_axis=np.array([0.0, 1.0, 2.0]),
            noise_values=list(noise_values),
            num_runs=num_runs,
            seed=11,
            **options,
        )

    @staticmethod
    def _estimates(result) -> np.ndarray:
        return np.array([
            (r.run_id, r.cnr_0_estimate, r.estimated_sigma2_0, r.ols_slope, r.wls_slope)
            for r in result.run_results
        ])

    @pytest.fixture
    def new_children(self):
        """Child processes started during the test; joblib's reusable workers from earlier tests are not counted."""
        import multiprocessing

        earlier = set(multiprocessing.active_children())
        return lambda: set(multiprocessing.active_children()) - earlier

    class RecordingExecutor:
        """Stands in for ProcessPoolExecutor: records how it was made and runs the tasks in-process."""
        created: list = []

        def __init__(self, max_workers, mp_context):
            type(self).created.append((max_workers, mp_context.get_start_method()))

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def map(self, function, iterable, chunksize=1):
            return [function(item) for item in iterable]

    def test_same_results_as_serial_with_progress_after_each_batch(self, new_children):
        # Starts a pool of two worker processes, which costs a few seconds.
        from dice.analysis.simulation import PROCESS_POOL_BACKEND, split_into_batches

        serial = self._simulate(multiprocessing=False)
        progress = []
        pooled = self._simulate(
            multiprocessing=True, n_jobs=2, parallel_backend=PROCESS_POOL_BACKEND,
            progress_callback=lambda done, total: progress.append((done, total)),
        )

        np.testing.assert_array_equal(self._estimates(pooled), self._estimates(serial))
        assert [r.run_id for r in pooled.run_results] == list(range(7))
        assert progress == [(batch.stop, 7) for batch in split_into_batches(7, 2)]
        assert len(progress) > 1
        assert new_children() == set()

    def test_stop_request_ends_the_simulation_and_its_workers(self, new_children):
        # The GUI's Stop makes the progress callback raise InterruptedError.
        from dice.analysis.simulation import PROCESS_POOL_BACKEND

        progress = []

        def stop_after_first_batch(done, total):
            progress.append((done, total))
            raise InterruptedError("Simulation cancelled by user")

        with pytest.raises(InterruptedError):
            self._simulate(
                multiprocessing=True, n_jobs=2, parallel_backend=PROCESS_POOL_BACKEND,
                progress_callback=stop_after_first_batch,
            )

        assert len(progress) == 1
        assert new_children() == set()

    def test_failed_run_raises_in_the_caller_and_ends_the_workers(self, new_children):
        from dice.analysis.simulation import PROCESS_POOL_BACKEND

        with pytest.raises(ValueError, match="noise_sigma"):
            self._simulate(
                noise_values=(-0.02,), multiprocessing=True, n_jobs=2,
                parallel_backend=PROCESS_POOL_BACKEND,
            )

        assert new_children() == set()

    @pytest.mark.parametrize('n_jobs', [1, 3, -1])
    def test_workers_follow_n_jobs_up_to_the_run_count(self, monkeypatch, n_jobs):
        import dice.analysis.simulation as simulation
        from joblib import effective_n_jobs

        monkeypatch.setattr(self.RecordingExecutor, 'created', [])
        monkeypatch.setattr(simulation, 'ProcessPoolExecutor', self.RecordingExecutor)

        pooled = self._simulate(
            num_runs=5, multiprocessing=True, n_jobs=n_jobs,
            parallel_backend=simulation.PROCESS_POOL_BACKEND,
        )

        assert self.RecordingExecutor.created == [(min(effective_n_jobs(n_jobs), 5), 'spawn')]
        np.testing.assert_array_equal(
            self._estimates(pooled), self._estimates(self._simulate(num_runs=5, multiprocessing=False)))

    def test_windows_pool_is_capped_at_its_limit(self, monkeypatch):
        import dice.analysis.simulation as simulation

        monkeypatch.setattr(self.RecordingExecutor, 'created', [])
        monkeypatch.setattr(simulation, 'ProcessPoolExecutor', self.RecordingExecutor)
        monkeypatch.setattr(simulation.sys, 'platform', 'win32')
        monkeypatch.setattr(simulation, 'run_single_simulation', lambda *arguments: arguments[0])

        results = simulation.run_in_process_pool([(run_id,) for run_id in range(200)], worker_count=100)

        assert self.RecordingExecutor.created == [(simulation.MAX_WINDOWS_POOL_WORKERS, 'spawn')]
        assert results == list(range(200))

    def test_standalone_build_runs_parallel_simulations_in_the_pool(self, monkeypatch):
        import dice.analysis.simulation as simulation
        from joblib import effective_n_jobs

        def no_joblib(*args, **kwargs):
            raise AssertionError("joblib.Parallel cannot start workers in a standalone build")

        pool_calls = []

        def recording_pool(run_parameters, worker_count, progress_callback=None):
            pool_calls.append(worker_count)
            return [simulation.run_single_simulation(*arguments) for arguments in run_parameters]

        serial = self._simulate(multiprocessing=False)
        monkeypatch.setattr(simulation, 'joblib_workers_supported', lambda: False)
        monkeypatch.setattr(simulation, 'Parallel', no_joblib)
        monkeypatch.setattr(simulation, 'run_in_process_pool', recording_pool)

        pooled = self._simulate(multiprocessing=True, n_jobs=3)

        assert simulation.default_parallel_backend() == simulation.PROCESS_POOL_BACKEND
        assert pool_calls == [effective_n_jobs(3)]
        np.testing.assert_array_equal(self._estimates(pooled), self._estimates(serial))

    def test_regular_install_runs_parallel_simulations_with_joblib(self, monkeypatch):
        import dice.analysis.simulation as simulation

        monkeypatch.setattr(simulation, 'joblib_workers_supported', lambda: True)
        assert simulation.default_parallel_backend() == simulation.JOBLIB_BACKEND

    def test_serial_simulation_starts_no_pool(self, monkeypatch):
        import dice.analysis.simulation as simulation

        monkeypatch.setattr(self.RecordingExecutor, 'created', [])
        monkeypatch.setattr(simulation, 'ProcessPoolExecutor', self.RecordingExecutor)

        self._simulate(multiprocessing=False, parallel_backend=simulation.PROCESS_POOL_BACKEND)

        assert self.RecordingExecutor.created == []

    def test_unknown_backend_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown parallel backend 'threads'"):
            self._simulate(multiprocessing=True, parallel_backend='threads')


class TestWorkerCountReachesJoblib:
    """The n_jobs argument is the worker count joblib receives for every batch."""

    class RecordingParallel:
        """Stands in for joblib.Parallel: records n_jobs and runs the tasks in-process."""
        n_jobs_seen: list = []

        def __init__(self, n_jobs=None, **kwargs):
            type(self).n_jobs_seen.append(n_jobs)

        def __call__(self, tasks):
            return [function(*args, **kwargs) for function, args, kwargs in tasks]

    def _simulate(self, **options):
        return run_monte_carlo_simulation(
            parameters=get_legacy_simulation_params(),
            x_axis=np.linspace(-10, 10, 51),
            time_axis=np.array([0.0, 1.0, 2.0]),
            noise_values=[0.02],
            num_runs=5,
            seed=13,
            **options,
        )

    @pytest.mark.parametrize('n_jobs', [1, 3, -1])
    def test_n_jobs_is_passed_to_every_parallel_call(self, monkeypatch, n_jobs):
        import dice.analysis.simulation as simulation

        serial = self._simulate(multiprocessing=False)
        monkeypatch.setattr(self.RecordingParallel, 'n_jobs_seen', [])
        monkeypatch.setattr(simulation, 'Parallel', self.RecordingParallel)

        parallel = self._simulate(multiprocessing=True, n_jobs=n_jobs)

        assert self.RecordingParallel.n_jobs_seen
        assert set(self.RecordingParallel.n_jobs_seen) == {n_jobs}
        np.testing.assert_array_equal(
            [(r.ols_slope, r.wls_slope) for r in parallel.run_results],
            [(r.ols_slope, r.wls_slope) for r in serial.run_results])


class TestSplitIntoBatches:
    """Parallel batches cover every run once and keep every worker busy."""

    @pytest.mark.parametrize('worker_count', [1, 2, 8])
    @pytest.mark.parametrize('run_count', [1, 7, 8, 9, 70, 71, 100])
    def test_batches_cover_each_run_once(self, run_count, worker_count):
        from dice.analysis.simulation import MAX_PROGRESS_BATCHES, split_into_batches
        batches = split_into_batches(run_count, worker_count)
        assert [i for batch in batches for i in batch] == list(range(run_count))
        assert 1 <= len(batches) <= MAX_PROGRESS_BATCHES
        sizes = [len(batch) for batch in batches]
        assert max(sizes) - min(sizes) <= 1
        assert min(sizes) >= min(run_count, worker_count)

    def test_fewer_runs_than_workers_is_one_batch(self):
        from dice.analysis.simulation import split_into_batches
        assert split_into_batches(8, 8) == [range(0, 8)]
        assert split_into_batches(3, 8) == [range(0, 3)]

    def test_no_runs_no_batches(self):
        from dice.analysis.simulation import split_into_batches
        assert split_into_batches(0, 8) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])