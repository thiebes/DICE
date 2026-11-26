"""
Tests for dice.analysis.simulation module.
"""

import warnings

import pytest
import numpy as np
from seeded_results import assert_seeded_results_agree
from dice.analysis.simulation import (
    run_single_simulation,
    calculate_wls_weights,
    run_monte_carlo_simulation,
)


def get_legacy_simulation_params():
    """Simulation parameters with fixed values, in the structure the engine reads."""
    from dice.utils.legacy_compatibility import MockSimulationParameters
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
    """A seed reproduces a simulation to within floating-point rounding; without one every run draws fresh entropy."""

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
        assert_seeded_results_agree(
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
        assert_seeded_results_agree(self._estimates(first), self._estimates(second))
        assert_seeded_results_agree(self._estimates(first), self._estimates(self._simulate(11)))

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
                assert_seeded_results_agree(self._estimates(parallel), serial)
                assert [r.run_id for r in parallel.run_results] == list(range(num_runs))
                assert progress_calls[-1] == (num_runs, num_runs)

    def test_run_seeds_are_children_of_one_root(self):
        from dice.analysis.simulation import spawn_run_seeds
        assert spawn_run_seeds(None, 3) == [None, None, None]
        children = spawn_run_seeds(11, 3)
        assert [child.spawn_key for child in children] == [(0,), (1,), (2,)]
        assert {child.entropy for child in children} == {11}


class TestUnitAndAmplitudeInvariance:
    """The same physical simulation with the same seed gives the same estimates in any unit and at any amplitude.

    quick_test.txt (FWHM 1 micrometer, diffusion length 0.1 micrometer, CNR 50)
    restated in another global length unit, or with amplitude_0 and the noise
    scaled together so the CNR stays 50. Each run's D_est/D_nom, which has no
    unit, must match the micrometer, amplitude-1 simulation.
    """

    RUNS = 4

    @classmethod
    def _accuracy(cls, length_unit: str, amplitude_0: float) -> dict:
        import ast
        from pathlib import Path
        from dice.io.parameters import parameter_parser
        from dice.utils.legacy_compatibility import create_parameters_from_legacy
        from dice.utils.units import convert_length

        quick_test = Path(__file__).parent.parent / 'data' / 'examples' / 'quick_test.txt'
        params = ast.literal_eval(quick_test.read_text())
        scale = convert_length(1.0, 'micrometer', length_unit)
        params.update({
            'length unit': length_unit,
            'FWHM_0': params['FWHM_0'] * scale,
            'spatial width': params['spatial width'] * scale,
            'nominal diffusion length': params['nominal diffusion length'] * scale,
            'amplitude_0': amplitude_0,
            'noise value': params['noise value'] * amplitude_0,
            'number of runs': cls.RUNS,
        })
        parsed = parameter_parser(params)
        sim_params = create_parameters_from_legacy(
            parsed, parsed['nominal diffusion coefficient'], parsed['nominal lifetime (tau)'],
            parsed['nominal diffusion length'])
        result = run_monte_carlo_simulation(
            sim_params, parsed['x array'], parsed['time series'], parsed['noise series'],
            num_runs=cls.RUNS, multiprocessing=False, seed=31)
        nominal_d = sim_params.physics.diffusion_coefficient
        return {key: result.d_estimates(key) / nominal_d for key in ('wls', 'ols')}

    @pytest.mark.parametrize('length_unit, amplitude_0', [
        ('nanometer', 1.0),
        ('millimeter', 1.0),
        ('centimeter', 1.0),
        ('meter', 1.0),
        ('micrometer', 1e-5),
        ('micrometer', 1e-3),
        ('micrometer', 1e3),
        ('meter', 1e-5),
    ])
    def test_estimates_match_the_micrometer_simulation(self, length_unit, amplitude_0):
        reference = self._accuracy('micrometer', 1.0)
        restated = self._accuracy(length_unit, amplitude_0)
        for key in ('wls', 'ols'):
            assert len(reference[key]) == len(restated[key]) == self.RUNS, key
            np.testing.assert_allclose(restated[key], reference[key], rtol=1e-6, err_msg=key)


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
        from dice.analysis.simulation import PROCESS_POOL_BACKEND

        serial = self._simulate(multiprocessing=False)
        progress = []
        pooled = self._simulate(
            multiprocessing=True, n_jobs=2, parallel_backend=PROCESS_POOL_BACKEND,
            progress_callback=lambda done, total: progress.append((done, total)),
        )

        assert_seeded_results_agree(self._estimates(pooled), self._estimates(serial))
        assert [r.run_id for r in pooled.run_results] == list(range(7))
        # The first batch has one run per worker; the sizes of the later ones follow the clock.
        assert progress[0] == (2, 7)
        assert progress[-1] == (7, 7)
        assert [done for done, _ in progress] == sorted({done for done, _ in progress})
        assert {total for _, total in progress} == {7}
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
        assert_seeded_results_agree(
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
        assert_seeded_results_agree(self._estimates(pooled), self._estimates(serial))

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
    """The n_jobs argument is the worker count joblib receives for the one call of a simulation."""

    class RecordingParallel:
        """Stands in for joblib.Parallel: records its options and runs the tasks in-process, one as each is asked for."""
        n_jobs_seen: list = []
        options_seen: list = []

        def __init__(self, n_jobs=None, **kwargs):
            type(self).n_jobs_seen.append(n_jobs)
            type(self).options_seen.append(kwargs)

        def __call__(self, tasks):
            return (function(*args, **kwargs) for function, args, kwargs in tasks)

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
    def test_n_jobs_reaches_the_one_parallel_call(self, monkeypatch, n_jobs):
        import dice.analysis.simulation as simulation

        serial = self._simulate(multiprocessing=False)
        monkeypatch.setattr(self.RecordingParallel, 'n_jobs_seen', [])
        monkeypatch.setattr(self.RecordingParallel, 'options_seen', [])
        monkeypatch.setattr(simulation, 'Parallel', self.RecordingParallel)

        parallel = self._simulate(multiprocessing=True, n_jobs=n_jobs)

        # One call for the whole simulation, whose results come back as they finish.
        assert self.RecordingParallel.n_jobs_seen == [n_jobs]
        assert self.RecordingParallel.options_seen == [{'return_as': 'generator'}]
        assert_seeded_results_agree(
            [(r.ols_slope, r.wls_slope) for r in parallel.run_results],
            [(r.ols_slope, r.wls_slope) for r in serial.run_results])


class TestJoblibProgressAndStop:
    """A simulation in joblib's workers reports by the clock and ends when the progress callback raises."""

    class CountingClock:
        """A clock that advances by a fixed step each time it is read."""

        def __init__(self, step: float):
            self.now = 0.0
            self.step = step

        def __call__(self) -> float:
            self.now += self.step
            return self.now

    class InProcessParallel:
        """Stands in for joblib.Parallel: runs each task when its result is asked for, and records a close.

        At the close it issues the warnings of warnings_at_close, each a pair
        of the message and the name of the module that issues it.
        """
        closed: list = []
        warnings_at_close: list = []

        def __init__(self, n_jobs=None, **kwargs):
            pass

        def __call__(self, tasks):
            def results():
                try:
                    for function, args, kwargs in tasks:
                        yield function(*args, **kwargs)
                finally:
                    type(self).closed.append(True)
                    for message, module in type(self).warnings_at_close:
                        warnings.warn_explicit(
                            message, UserWarning, filename="parallel.py", lineno=1, module=module, registry={})
            return results()

    @pytest.fixture
    def run_parameters(self, monkeypatch):
        """Arguments of 40 runs that only return their index, and joblib replaced by an in-process stand-in."""
        import dice.analysis.simulation as simulation
        monkeypatch.setattr(self.InProcessParallel, 'closed', [])
        monkeypatch.setattr(self.InProcessParallel, 'warnings_at_close', [])
        monkeypatch.setattr(simulation, 'Parallel', self.InProcessParallel)
        monkeypatch.setattr(simulation, 'run_single_simulation', lambda index: index)
        return [(index,) for index in range(40)]

    def test_reports_follow_the_clock_and_end_with_the_last_run(self, run_parameters):
        from dice.analysis.simulation import PROGRESS_REPORT_SECONDS, run_in_joblib_workers
        reports = []
        # The clock is read once at the start and once for each result, so
        # the report interval passes every fourth result.
        clock = self.CountingClock(step=PROGRESS_REPORT_SECONDS / 4)

        results = run_in_joblib_workers(
            run_parameters, 2, lambda done, total: reports.append((done, total)), clock=clock)

        assert results == list(range(40))
        assert reports[-1] == (40, 40)
        assert 5 < len(reports) < 40
        assert [done for done, _ in reports] == sorted({done for done, _ in reports})
        assert {total for _, total in reports} == {40}

    def test_fast_runs_report_only_at_the_end(self, run_parameters):
        from dice.analysis.simulation import run_in_joblib_workers
        reports = []

        run_in_joblib_workers(
            run_parameters, 2, lambda done, total: reports.append((done, total)), clock=lambda: 0.0)

        assert reports == [(40, 40)]

    def test_no_callback_returns_all_results_and_closes(self, run_parameters):
        from dice.analysis.simulation import run_in_joblib_workers
        assert run_in_joblib_workers(run_parameters, 2) == list(range(40))
        assert self.InProcessParallel.closed == [True]

    def test_exception_of_the_callback_ends_the_runs_that_have_not_finished(self, run_parameters):
        from dice.analysis.simulation import PROGRESS_REPORT_SECONDS, run_in_joblib_workers
        reports = []

        def stop_at_first_report(done, total):
            reports.append((done, total))
            raise InterruptedError("Simulation cancelled by user")

        with pytest.raises(InterruptedError):
            run_in_joblib_workers(
                run_parameters, 2, stop_at_first_report, clock=self.CountingClock(step=PROGRESS_REPORT_SECONDS))

        assert len(reports) == 1
        assert reports[0][0] < 40
        # The generator of results was closed, which is how joblib learns to stop its workers.
        assert self.InProcessParallel.closed == [True]

    def test_only_the_early_exit_warnings_of_joblib_are_ignored_at_a_stop(self, run_parameters, monkeypatch):
        """joblib's two messages about runs it cancelled or did not use are expected; any other warning is shown."""
        from dice.analysis.simulation import PROGRESS_REPORT_SECONDS, run_in_joblib_workers
        # The three messages of joblib's Parallel._warn_exit_early, word for word.
        advice = " You could benefit from adjusting the input task iterator to limit unnecessary computation time."
        cancelled = "7 tasks which were still being processed by the workers have been cancelled."
        not_used = "2 tasks have been successfully executed  but not used."
        monkeypatch.setattr(self.InProcessParallel, 'warnings_at_close', [
            (cancelled + advice, "joblib.parallel"),
            (not_used + advice, "joblib.parallel"),
            (not_used + " Additionally, " + cancelled + advice, "joblib.parallel"),
            ("A worker stopped while some jobs were given to the executor.", "joblib.parallel"),
            (cancelled + advice, "another.package"),
        ])

        def stop_at_first_report(done, total):
            raise InterruptedError("Simulation cancelled by user")

        with pytest.warns(UserWarning) as shown:
            with pytest.raises(InterruptedError):
                run_in_joblib_workers(
                    run_parameters, 2, stop_at_first_report, clock=self.CountingClock(step=PROGRESS_REPORT_SECONDS))

        assert [str(warning.message) for warning in shown] == [
            "A worker stopped while some jobs were given to the executor.", cancelled + advice]

    def test_stop_ends_real_workers_and_the_next_simulation_runs(self):
        """With joblib itself: a stop replaces the workers, raises no warning, and leaves the pool usable.

        Costs a few seconds.
        """
        import os
        import time
        from joblib import Parallel, delayed
        from dice.analysis.simulation import run_in_joblib_workers

        def worker_processes() -> set:
            """Process identifiers of the workers that joblib uses for two jobs."""
            return set(Parallel(n_jobs=2)(delayed(os.getpid)() for _ in range(8)))

        parameters = get_legacy_simulation_params()
        x_axis, time_axis = np.linspace(-10, 10, 51), np.array([0.0, 1.0, 2.0])
        arguments = [(run_id, x_axis, time_axis, parameters, 0.02, False, None) for run_id in range(3000)]
        reports = []

        def stop_at_first_report(done, total):
            reports.append((done, total))
            raise InterruptedError("Simulation cancelled by user")

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            workers_before = worker_processes()
            with pytest.raises(InterruptedError):
                run_in_joblib_workers(arguments, 2, stop_at_first_report, clock=time.perf_counter)
            workers_after = worker_processes()
            results = run_in_joblib_workers(arguments[:4], 2)

        assert len(reports) == 1 and reports[0][0] < 3000
        # joblib ended the workers that held the runs not yet finished, and
        # started others. Workers that were left to finish those runs would
        # answer here under the same identifiers.
        assert workers_before and workers_after
        assert workers_before.isdisjoint(workers_after)
        assert [result.run_id for result in results] == [0, 1, 2, 3]


class TestTimedBatches:
    """Parallel batches cover every run once, keep every worker busy, and take about a second each."""

    class Clock:
        """A clock that the test advances by the duration of each batch."""

        def __init__(self):
            self.now = 0.0

        def __call__(self) -> float:
            return self.now

    def _batches(self, run_count: int, worker_count: int, seconds_per_run: float) -> list:
        """The batches of a simulation whose workers each take seconds_per_run for a run."""
        from dice.analysis.simulation import timed_batches
        clock = self.Clock()
        batches = []
        for batch in timed_batches(run_count, worker_count, clock=clock):
            batches.append(batch)
            clock.now += len(batch) * seconds_per_run / worker_count
        return batches

    @pytest.mark.parametrize('seconds_per_run', [0.0, 0.001, 0.05, 3.0])
    @pytest.mark.parametrize('worker_count', [1, 2, 8])
    @pytest.mark.parametrize('run_count', [1, 7, 8, 9, 70, 71, 100, 5000])
    def test_batches_cover_each_run_once(self, run_count, worker_count, seconds_per_run):
        batches = self._batches(run_count, worker_count, seconds_per_run)
        assert [i for batch in batches for i in batch] == list(range(run_count))
        assert min(len(batch) for batch in batches) >= min(run_count, worker_count)

    def test_first_batch_has_one_run_per_worker(self):
        assert self._batches(1000, 8, 0.01)[0] == range(0, 8)

    def test_fewer_runs_than_workers_is_one_batch(self):
        assert self._batches(8, 8, 0.01) == [range(0, 8)]
        assert self._batches(3, 8, 0.01) == [range(0, 3)]

    def test_no_runs_no_batches(self):
        assert self._batches(0, 8, 0.01) == []

    def test_batches_grow_to_the_target_duration(self):
        """Eight workers at 10 ms a run finish 800 runs in the target second."""
        from dice.analysis.simulation import MAX_BATCH_GROWTH, TARGET_BATCH_SECONDS
        assert (MAX_BATCH_GROWTH, TARGET_BATCH_SECONDS) == (8, 1.0)

        sizes = [len(batch) for batch in self._batches(100_000, 8, 0.01)]

        assert sizes[:3] == [8, 64, 512]
        # The clock is a sum of floating-point durations, so a size can be one run off.
        assert all(abs(size - 800) <= 1 for size in sizes[3:-1])

    def test_long_simulation_reports_often(self):
        """A simulation of 200,000 runs reports about once a second."""
        seconds_per_run, worker_count = 0.015, 8
        batches = self._batches(200_000, worker_count, seconds_per_run)
        durations = [len(batch) * seconds_per_run / worker_count for batch in batches]

        assert len(batches) > 300
        assert max(durations) <= 2 * 1.0

    def test_slow_runs_keep_one_run_per_worker(self):
        """A run longer than the target cannot be split: each batch has one run per worker."""
        assert {len(batch) for batch in self._batches(64, 8, 30.0)} == {8}

    def test_runs_too_few_for_a_last_batch_join_the_batch_before(self):
        batches = self._batches(20, 8, 30.0)
        assert batches == [range(0, 8), range(8, 20)]

    def test_batch_that_took_no_time_grows_by_the_largest_factor(self):
        from dice.analysis.simulation import MAX_BATCH_GROWTH, next_batch_size
        assert next_batch_size(8, 0.0, 8) == 8 * MAX_BATCH_GROWTH
        assert next_batch_size(8, 1e-9, 8) == 8 * MAX_BATCH_GROWTH


class TestNoiseLevelForEachRun:
    """With a noise range, run i uses noise value i; the seeds and the result stay consistent."""

    NOISE_RANGE_SIGMAS = [0.04, 0.011, 0.025, 0.018, 0.05, 0.013, 0.031]

    def _simulate(self, noise_values, num_runs, noise_range=None, **options):
        from dice.models.parameters import NoiseRange
        return run_monte_carlo_simulation(
            parameters=get_legacy_simulation_params(),
            x_axis=np.linspace(-10, 10, 51),
            time_axis=np.array([0.0, 1.0, 2.0]),
            noise_values=list(noise_values),
            num_runs=num_runs,
            seed=17,
            noise_range=noise_range if noise_range is not None else NoiseRange(0.01, 0.05),
            multiprocessing=options.pop('multiprocessing', False),
            **options,
        )

    @staticmethod
    def _run_values(result) -> np.ndarray:
        return np.array([
            (r.run_id, r.noise_sigma, r.cnr_0_estimate, r.estimated_sigma2_0, r.ols_slope, r.wls_slope)
            for r in result.run_results
        ])

    def test_each_run_uses_its_own_noise_value(self):
        from dice.models.parameters import NoiseRange
        noise_range = NoiseRange(0.01, 0.05, logarithmic=True)
        result = self._simulate(self.NOISE_RANGE_SIGMAS, len(self.NOISE_RANGE_SIGMAS), noise_range)

        assert [r.noise_sigma for r in result.run_results] == self.NOISE_RANGE_SIGMAS
        assert [r.run_id for r in result.run_results] == list(range(7))
        assert result.num_runs == len(result.run_results) == 7
        assert result.noise_values == self.NOISE_RANGE_SIGMAS
        assert result.noise_range == noise_range

    def test_fixed_levels_record_no_range(self):
        result = run_monte_carlo_simulation(
            parameters=get_legacy_simulation_params(), x_axis=np.linspace(-10, 10, 51),
            time_axis=np.array([0.0, 1.0]), noise_values=[0.02, 0.05], num_runs=2,
            multiprocessing=False, seed=17,
        )
        assert result.noise_range is None
        assert [r.noise_sigma for r in result.run_results] == [0.02, 0.02, 0.05, 0.05]

    def test_same_streams_as_fixed_noise(self):
        # Run i draws its noise from the same seed stream whether its sigma is
        # a fixed level or its own value, so equal sigmas give equal runs.
        per_run = self._simulate([0.02] * 4, 4)
        fixed = run_monte_carlo_simulation(
            parameters=get_legacy_simulation_params(), x_axis=np.linspace(-10, 10, 51),
            time_axis=np.array([0.0, 1.0, 2.0]), noise_values=[0.02], num_runs=4,
            multiprocessing=False, seed=17,
        )
        assert_seeded_results_agree(self._run_values(per_run), self._run_values(fixed))

    @pytest.mark.parametrize('noise_values', [[0.02] * 6, [0.02] * 8, []])
    def test_one_value_per_run_required(self, noise_values):
        with pytest.raises(ValueError, match="one noise value for each of the 7 runs; got"):
            self._simulate(noise_values, 7)

    def test_serial_joblib_and_process_pool_identical(self):
        # Starts a joblib pool and a process pool of two workers each, which costs some seconds.
        from dice.analysis.simulation import JOBLIB_BACKEND, PROCESS_POOL_BACKEND
        runs = len(self.NOISE_RANGE_SIGMAS)
        serial = self._simulate(self.NOISE_RANGE_SIGMAS, runs)
        for backend in (JOBLIB_BACKEND, PROCESS_POOL_BACKEND):
            parallel = self._simulate(self.NOISE_RANGE_SIGMAS, runs, multiprocessing=True,
                                      n_jobs=2, parallel_backend=backend)
            assert_seeded_results_agree(self._run_values(parallel), self._run_values(serial),
                                          err_msg=backend)
            assert parallel.noise_values == serial.noise_values
            assert parallel.noise_range == serial.noise_range


class TestNoiseRangeModel:
    """NoiseRange accepts only positive, finite bounds with low < high."""

    def test_valid_range(self):
        from dice.models.parameters import NoiseRange
        noise_range = NoiseRange(0.01, 0.05, logarithmic=True)
        assert (noise_range.low, noise_range.high, noise_range.logarithmic) == (0.01, 0.05, True)
        assert NoiseRange(0.01, 0.05).logarithmic is False

    @pytest.mark.parametrize('low, high, message', [
        (0.0, 0.05, "Both values must be positive and finite"),
        (-0.01, 0.05, "Both values must be positive and finite"),
        (0.01, float('inf'), "Both values must be positive and finite"),
        (float('nan'), 0.05, "Both values must be positive and finite"),
        (0.05, 0.01, r"The first value \(low\) must be less than the second \(high\)"),
        (0.05, 0.05, r"The first value \(low\) must be less than the second \(high\)"),
    ])
    def test_invalid_range_rejected(self, low, high, message):
        from dice.models.parameters import NoiseRange
        with pytest.raises(ValueError, match=r"Invalid noise range \[.*\]\. " + message):
            NoiseRange(low, high)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

class TestSeededResultsAgree:
    """Seeded results agree to within floating-point rounding, per quantity, independent of units."""

    RESULTS = np.array([[45.108568, 0.969032, 2.198109e-3, 2.1e-18],
                        [46.546530, 0.991401, 2.063041e-3, -3.0e-3]])

    def test_identical_results_agree(self):
        from dice.analysis.simulation import seeded_results_agree
        assert seeded_results_agree(self.RESULTS, self.RESULTS.copy())

    def test_last_digit_differences_agree(self):
        # The size of difference seen on Intel Macs: one or two units in the last place.
        from dice.analysis.simulation import seeded_results_agree
        nudged = np.nextafter(np.nextafter(self.RESULTS, np.inf), np.inf)
        assert seeded_results_agree(nudged, self.RESULTS)

    def test_a_relative_difference_of_1e_9_does_not_agree(self):
        from dice.analysis.simulation import seeded_results_agree
        changed = self.RESULTS.copy()
        changed[1, 2] *= 1 + 1e-9
        assert not seeded_results_agree(changed, self.RESULTS)

    def test_a_value_near_zero_is_judged_against_its_column(self):
        # 2.1e-18 changing by 1e-18 is a large relative change but rounding for a column of size 3e-3.
        from dice.analysis.simulation import seeded_results_agree
        changed = self.RESULTS.copy()
        changed[0, 3] += 1e-18
        assert seeded_results_agree(changed, self.RESULTS)

    @pytest.mark.parametrize('scale', [1e-12, 1.0, 1e9])
    def test_the_comparison_does_not_depend_on_units(self, scale):
        from dice.analysis.simulation import seeded_results_agree
        nudged = np.nextafter(self.RESULTS * scale, np.inf)
        assert seeded_results_agree(nudged, self.RESULTS * scale)
        assert not seeded_results_agree(self.RESULTS * scale * (1 + 1e-9), self.RESULTS * scale)

    @pytest.mark.parametrize('actual, desired, expected', [
        ([1.0, np.nan], [1.0, np.nan], True),
        ([1.0, np.nan], [1.0, 2.0], False),
        ([np.inf, 1.0], [np.inf, 1.0], True),
        ([np.inf, 1.0], [-np.inf, 1.0], False),
        ([np.inf, 1.0], [1e300, 1.0], False),
        ([1.0, 2.0], [1.0, 2.0, 3.0], False),
        (0.0, 0.0, True),
        (0.0, 1e-300, False),
    ])
    def test_nan_infinity_shape_and_zero(self, actual, desired, expected):
        from dice.analysis.simulation import seeded_results_agree
        assert seeded_results_agree(actual, desired) is expected
