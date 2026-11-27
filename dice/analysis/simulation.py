"""
Monte Carlo simulation runner for DICE.

This module provides the core simulation functionality for running Monte Carlo
simulations of diffusion measurements with noise.
"""

import multiprocessing as mp
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from typing import Callable, Iterator, List, Optional, Any, Sequence, Union
from joblib import Parallel, delayed, effective_n_jobs

from ..core.profiles import make_diffusion_decay
from fft_cnr import fft_cnr
from ..core.noise import Seed, add_noise
from ..core.fitting import gauss_fitting, calculate_wls_weights
from ..models.estimators import DEFAULT_ESTIMATOR_KEYS, FrameFits, get_estimator
from ..models.parameters import NoiseRange
from ..models.results import RunResult, MonteCarloOutput
from ..utils.runtime import joblib_workers_supported
from ..models.parameters import SimulationParameters

# The shortest time between two progress reports of a parallel simulation in
# joblib's workers. Its results come back as they finish; a report for each
# one would be more than a reader or a progress bar can use.
PROGRESS_REPORT_SECONDS = 0.25
# What joblib warns of when the generator of its results is closed early.
JOBLIB_EARLY_EXIT_MESSAGE = r".*(executed\s+but not used|have been cancelled)"
# A parallel simulation in the process pool runs in batches. The progress
# callback reports after each batch, and a stop that it asks for takes effect
# there, so the duration of a batch is how soon a long simulation shows
# progress and how long a stop can take. Each batch is sized from the
# duration of the batch before it.
TARGET_BATCH_SECONDS = 1.0
# The most a batch grows from one to the next. The first batches are small,
# and their duration holds the start of the worker pool.
MAX_BATCH_GROWTH = 8

# Worker pools for a parallel simulation: joblib's default (loky) backend, or
# the standard library's ProcessPoolExecutor, which a standalone build uses.
JOBLIB_BACKEND = 'joblib'
PROCESS_POOL_BACKEND = 'process_pool'
PARALLEL_BACKENDS = (JOBLIB_BACKEND, PROCESS_POOL_BACKEND)

# ProcessPoolExecutor rejects more workers than this on Windows.
MAX_WINDOWS_POOL_WORKERS = 61

# Two simulations with the same seed draw the same random numbers, and on
# most platforms their results repeat bit for bit. The macOS wheels of NumPy
# and SciPy for Intel processors use Apple's Accelerate library, whose
# least-squares results can differ in the last digit from one call to the
# next: relative differences up to about 1e-15 were seen in continuous
# integration. Seeded results are therefore compared with this relative
# tolerance, far below the change a different seed or different physics makes.
SEEDED_RESULTS_RTOL = 1e-12


def run_single_simulation(
    run_id: int,
    x_axis: np.ndarray,
    time_axis: np.ndarray,
    parameters: SimulationParameters,
    noise_sigma: float,
    retain_profile_data: bool = False,
    seed: Seed = None,
) -> RunResult:
    """
    Execute a single Monte Carlo simulation run.
    
    This function simulates a single measurement of diffusion with noise,
    performing Gaussian fits and extracting diffusion coefficients.
    
    Parameters
    ----------
    run_id : int
        Identifier for this simulation run.
    x_axis : np.ndarray
        Spatial axis values.
    time_axis : np.ndarray
        Time axis values.
    parameters : SimulationParameters
        Simulation parameters including initial conditions and physics.
    noise_sigma : float
        Standard deviation of noise to add.
    retain_profile_data : bool
        Whether to retain full profile data in results.
    seed : int or SeedSequence, optional
        Seed for this run's noise; None draws fresh entropy.

    Returns
    -------
    RunResult
        Results from this simulation run.
    """
    # Generate nominal profiles with diffusion and decay
    profile_params = {
        'x axis': x_axis,
        'time axis': time_axis,
        'sigma^2_0': parameters.profile.sigma2_0,
        'amplitude_0': parameters.profile.amplitude_0,
        'mu_0': parameters.profile.mu_0,
        'nominal diffusion coefficient': parameters.physics.diffusion_coefficient,
        'nominal lifetime': parameters.physics.lifetime
    }
    nominal_profiles = make_diffusion_decay(profile_params)
    
    # Add noise to the profiles
    noisy_result = add_noise(
        nominal_profiles['y_values_t'],
        noise_sigma=noise_sigma,
        seed=seed,
    )
    noisy_profiles = noisy_result['y_values_t']
    
    # Estimate CNR at t=0
    cnr_0_estimate = fft_cnr(noisy_profiles[0]).cnr
    
    # Fit Gaussians to noisy profiles
    fit_results = gauss_fitting(x_axis, noisy_profiles)
    gaussfit_sigma2_t = np.array(fit_results['sigma^2_t estimates'])
    gaussfit_sigma2_stderrs = np.array(fit_results['sigma^2_t standard errors'])
    
    # Create result object; the estimators fill in their slope fits below
    result = RunResult(
        run_id=run_id,
        nominal_diffusion_coefficient=parameters.physics.diffusion_coefficient,
        nominal_lifetime=parameters.physics.lifetime,
        nominal_diffusion_length=parameters.physics.diffusion_length,
        noise_sigma=noise_sigma,
        cnr_0_estimate=cnr_0_estimate,
        nominal_sigma2_0=parameters.profile.sigma2_0,
        estimated_sigma2_0=gaussfit_sigma2_t[0] if len(gaussfit_sigma2_t) > 0 else None,
    )

    weights = None
    if len(time_axis) > 1:
        # WLS weights, inversely proportional to the squared relative error
        # (kept for retained profile data; the WLS estimator computes its own)
        weights = calculate_wls_weights(gaussfit_sigma2_t, gaussfit_sigma2_stderrs)

        # Fit MSD against time with each estimator, from the same per-frame fits
        frame_fits = FrameFits(time_axis=time_axis, sigma2=gaussfit_sigma2_t,
                               sigma2_stderr=gaussfit_sigma2_stderrs)
        for key in DEFAULT_ESTIMATOR_KEYS:
            result.set_slope_fit(key, get_estimator(key).estimate(frame_fits))

    # Add profile data if requested
    if retain_profile_data:
        # Convert lists to numpy arrays for consistency
        result.nominal_profiles = np.array(nominal_profiles['y_values_t'])
        result.noisy_profiles = np.array(noisy_profiles) if isinstance(noisy_profiles, list) else noisy_profiles
        result.fitted_sigma2_t = gaussfit_sigma2_t
        result.fitted_sigma2_stderrs = gaussfit_sigma2_stderrs
        result.weights = weights
    
    return result


def run_monte_carlo_simulation(
    parameters: SimulationParameters,
    x_axis: np.ndarray,
    time_axis: np.ndarray,
    noise_values: List[float],
    num_runs: int,
    multiprocessing: bool = True,
    retain_profile_data: bool = False,
    progress_callback: Optional[callable] = None,
    seed: Seed = None,
    n_jobs: int = -1,
    parallel_backend: Optional[str] = None,
    noise_range: Optional[NoiseRange] = None,
) -> MonteCarloOutput:
    """
    Run Monte Carlo simulation at fixed noise levels, or with a noise level for each run.

    Parameters
    ----------
    parameters : SimulationParameters
        Simulation parameters, as simulation_parameters_from_dict builds them.
    x_axis : np.ndarray
        Spatial axis values.
    time_axis : np.ndarray
        Time axis values.
    noise_values : List[float]
        Noise sigma values to simulate: the fixed noise levels, or, with
        noise_range, one value for each run, in run order.
    num_runs : int
        Number of runs per noise level; with noise_range, the total number
        of runs, which must equal the number of noise values.
    multiprocessing : bool
        Whether to use parallel processing.
    retain_profile_data : bool
        Whether to retain full profile data.
    progress_callback : callable, optional
        Function to call with progress updates.
    seed : int or SeedSequence, optional
        Root seed. Each run gets its own child stream, so a seeded simulation
        gives the same results serially or in parallel, to within the rounding
        that SEEDED_RESULTS_RTOL describes. None draws fresh entropy for every run.
    n_jobs : int
        Number of worker processes when multiprocessing; -1 uses all cores.
    parallel_backend : str, optional
        Worker pool when multiprocessing: 'joblib' or 'process_pool' (the
        standard library's ProcessPoolExecutor). None selects joblib, or the
        process pool in a standalone build, where joblib cannot start workers.
        Both give the same results as a serial run with the same seed.
    noise_range : NoiseRange, optional
        The range from which noise_values were drawn, one for each run. The
        result records it, so that the analysis and the outputs treat the
        noise as varying from run to run. None for fixed noise levels.

    Returns
    -------
    MonteCarloOutput
        Aggregated simulation results.

    Raises
    ------
    ValueError
        If the parallel backend is unknown, or with noise_range, if the
        number of noise values differs from num_runs.
    """
    if parallel_backend is None:
        parallel_backend = default_parallel_backend()
    elif parallel_backend not in PARALLEL_BACKENDS:
        raise ValueError(
            f"Unknown parallel backend {parallel_backend!r}; expected one of {PARALLEL_BACKENDS}"
        )

    run_noise_sigmas = noise_sigma_of_each_run(noise_values, num_runs, noise_range)
    run_seeds = spawn_run_seeds(seed, len(run_noise_sigmas))

    # Create parameter sets for all runs
    run_parameters = [
        (run_id, x_axis, time_axis, parameters, noise_sigma, retain_profile_data, run_seeds[run_id])
        for run_id, noise_sigma in enumerate(run_noise_sigmas)
    ]

    total_runs = len(run_parameters)

    # Run simulations
    if multiprocessing and parallel_backend == PROCESS_POOL_BACKEND:
        results = run_in_process_pool(run_parameters, effective_n_jobs(n_jobs), progress_callback)
    elif multiprocessing:
        results = run_in_joblib_workers(run_parameters, n_jobs, progress_callback)
    else:
        # Sequential execution with optional progress updates
        results = []
        for i, params in enumerate(run_parameters):
            result = run_single_simulation(*params)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total_runs)
    
    # Create aggregated result
    simulation_result = MonteCarloOutput(
        parameters=parameters,
        run_results=results,
        num_runs=total_runs,
        noise_values=noise_values,
        estimator_keys=DEFAULT_ESTIMATOR_KEYS,
        noise_range=noise_range,
        seed=seed_record(seed),
        analysis_seed=analysis_seed(seed, len(run_seeds)),
    )

    return simulation_result


def noise_sigma_of_each_run(
    noise_values: Sequence[float], num_runs: int, noise_range: Optional[NoiseRange] = None,
) -> List[float]:
    """The noise sigma of each run, in run order.

    Fixed noise levels give num_runs runs each, level by level. With a noise
    range, the noise values are already one per run.

    Raises
    ------
    ValueError
        With a noise range, if the number of noise values differs from num_runs.
    """
    if noise_range is None:
        return [noise_sigma for noise_sigma in noise_values for _ in range(num_runs)]
    if len(noise_values) != num_runs:
        raise ValueError(
            f"A noise range needs one noise value for each of the {num_runs} runs; "
            f"got {len(noise_values)}."
        )
    return list(noise_values)


def seed_record(seed: Seed) -> Optional[Union[int, str]]:
    """
    What reproduces a seeded simulation, for its outputs; None when it is unseeded.

    An integer seed is recorded as the integer, and so is a SeedSequence
    made from one integer that is not a child of another. Any other
    SeedSequence is recorded as text with its entropy and its spawn key: the
    entropy alone names its parent, whose noise is different.
    """
    if seed is None:
        return None
    if not isinstance(seed, np.random.SeedSequence):
        return int(seed)
    if seed.spawn_key == () and isinstance(seed.entropy, (int, np.integer)):
        return int(seed.entropy)
    spawn_key = tuple(int(key) for key in seed.spawn_key)
    return f"SeedSequence(entropy={seed.entropy!r}, spawn_key={spawn_key!r})"


def spawn_run_seeds(seed: Seed, run_count: int) -> List[Optional[np.random.SeedSequence]]:
    """One independent child seed per run from a single root; all None when unseeded.

    A SeedSequence argument is copied before spawning, so the caller's object
    is not advanced and the same seed always yields the same children.
    """
    if seed is None:
        return [None] * run_count
    if isinstance(seed, np.random.SeedSequence):
        root = np.random.SeedSequence(seed.entropy, spawn_key=seed.spawn_key, pool_size=seed.pool_size)
    else:
        root = np.random.SeedSequence(seed)
    return root.spawn(run_count)


def analysis_seed(seed: Seed, run_count: int) -> Optional[np.random.SeedSequence]:
    """
    The seed of the resampling in the analysis of a simulation; None when unseeded.

    It is the child of the simulation's root seed after the run_count run
    seeds that spawn_run_seeds gives, so a seeded simulation repeats its
    intervals, and the run seeds are those they would be without it.
    """
    if seed is None:
        return None
    if isinstance(seed, np.random.SeedSequence):
        entropy, spawn_key, pool_size = seed.entropy, tuple(seed.spawn_key), seed.pool_size
    else:
        entropy, spawn_key, pool_size = seed, (), np.random.SeedSequence(seed).pool_size
    return np.random.SeedSequence(entropy, spawn_key=spawn_key + (run_count,), pool_size=pool_size)


def seeded_results_agree(actual: Any, desired: Any, rtol: float = SEEDED_RESULTS_RTOL) -> bool:
    """
    Whether two sets of seeded simulation results agree to within floating-point rounding.

    Each quantity is compared relative to its largest magnitude: for a 2-D
    array (runs by quantities) that is the largest magnitude in its column,
    so a value near zero, such as an intercept, is held to the same standard
    as the rest of its column, and the comparison does not depend on the units.
    NaN equals NaN, and infinities must match exactly.

    Parameters
    ----------
    actual, desired : array_like
        Results of the two simulations, of the same shape.
    rtol : float, optional
        Largest relative difference that counts as rounding.

    Returns
    -------
    bool
        True if the shapes match and every value agrees.
    """
    actual = np.asarray(actual, dtype=float)
    desired = np.asarray(desired, dtype=float)
    if actual.shape != desired.shape:
        return False
    equal = (actual == desired) | (np.isnan(actual) & np.isnan(desired))
    finite = np.isfinite(actual) & np.isfinite(desired)
    if not np.all(equal | finite):
        return False
    # Compare only the finite pairs; equal infinities and NaNs were settled above.
    actual_finite = np.where(finite, actual, 0.0)
    desired_finite = np.where(finite, desired, 0.0)
    magnitude = np.maximum(np.abs(actual_finite), np.abs(desired_finite))
    scale = magnitude.max(axis=0) if actual.ndim >= 2 else magnitude.max(initial=0.0)
    difference = np.abs(actual_finite - desired_finite)
    return bool(np.all(equal | (difference <= rtol * scale)))


def run_in_joblib_workers(
    run_parameters: List[tuple],
    n_jobs: int,
    progress_callback: Optional[callable] = None,
    clock: Callable[[], float] = time.perf_counter,
) -> List[RunResult]:
    """Run each set of run_single_simulation arguments in joblib's worker pool, in one call.

    The results come back in the order of the arguments, each as soon as it
    and the ones before it have finished, and the progress callback reports
    at most once in PROGRESS_REPORT_SECONDS and after the last run. The runs
    are one call to joblib and not a series of batches, because joblib
    adapts the number of runs it sends to a worker in one message during a
    call, and starts that again with every call.

    An exception that the callback raises (the GUI's Stop raises
    InterruptedError) ends the runs that have not finished: joblib stops its
    workers, and starts new ones for the next simulation.

    Parameters
    ----------
    run_parameters : list of tuple
        The arguments of run_single_simulation for each run.
    n_jobs : int
        Number of workers, as joblib reads it (-1 for one per processor).
    progress_callback : callable, optional
        Called with the number of finished runs and the number of runs.
    clock : callable
        Returns the time in seconds. Tests replace it.

    Returns
    -------
    List[RunResult]
        One result per set of arguments, in the same order.
    """
    total_runs = len(run_parameters)
    finished_runs = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(run_single_simulation)(*arguments) for arguments in run_parameters
    )
    results = []
    last_report = clock()
    try:
        for result in finished_runs:
            results.append(result)
            if progress_callback is None:
                continue
            now = clock()
            if len(results) == total_runs or now - last_report >= PROGRESS_REPORT_SECONDS:
                last_report = now
                progress_callback(len(results), total_runs)
    finally:
        # After the last result this does nothing. After an exception it ends
        # the runs that have not finished; joblib then warns that it
        # cancelled them or did not use their results, which is what the
        # exception asked for. Only these two messages are ignored.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=JOBLIB_EARLY_EXIT_MESSAGE, category=UserWarning, module="joblib")
            finished_runs.close()
    return results


def timed_batches(
    run_count: int,
    worker_count: int,
    clock: Callable[[], float] = time.perf_counter,
) -> Iterator[range]:
    """Consecutive run-index ranges, each sized so that its runs take about TARGET_BATCH_SECONDS.

    The caller runs a batch before it asks for the next one, and the time
    between the two requests is taken as the duration of the batch. The first
    batch has one run per worker. Every batch has at least one run per
    worker, because a smaller batch would leave workers idle; runs too few to
    fill a last batch join the batch before them.

    Parameters
    ----------
    run_count : int
        Number of runs of the simulation.
    worker_count : int
        Number of worker processes.
    clock : callable
        Returns the time in seconds. Tests replace it.

    Yields
    ------
    range
        The run indices of the next batch.
    """
    worker_count = max(1, worker_count)
    size = worker_count
    start = 0
    while start < run_count:
        stop = start + size
        if run_count - stop < worker_count:
            stop = run_count
        began = clock()
        yield range(start, stop)
        size = next_batch_size(stop - start, clock() - began, worker_count)
        start = stop


def next_batch_size(batch_size: int, batch_seconds: float, worker_count: int) -> int:
    """Number of runs that take about TARGET_BATCH_SECONDS, from the duration of the batch before.

    At least one run per worker, and at most MAX_BATCH_GROWTH times the batch before.
    """
    largest = batch_size * MAX_BATCH_GROWTH
    if batch_seconds <= 0:
        return largest
    size = int(batch_size * TARGET_BATCH_SECONDS / batch_seconds)
    return max(worker_count, min(size, largest))


def default_parallel_backend() -> str:
    """The worker pool of a parallel simulation that names none.

    joblib, except in a standalone build, where joblib cannot start workers
    and the standard library's process pool takes its place.
    """
    return JOBLIB_BACKEND if joblib_workers_supported() else PROCESS_POOL_BACKEND


def run_in_process_pool(
    run_parameters: List[tuple],
    worker_count: int,
    progress_callback: Optional[callable] = None,
) -> List[RunResult]:
    """Run each set of run_single_simulation arguments in a pool of worker processes.

    The workers start with the 'spawn' method on every platform, because a
    forked copy of the GUI process would inherit its threads. A standalone
    build calls multiprocessing.freeze_support() first thing in its entry
    script, so that each worker it spawns runs simulations instead of the
    application. The runs are sent in batches of about TARGET_BATCH_SECONDS
    (timed_batches), and the progress callback reports after each batch; an
    exception it raises (the GUI's Stop raises InterruptedError) propagates
    once the workers have exited.

    Returns
    -------
    List[RunResult]
        One result per set of arguments, in the same order.
    """
    total_runs = len(run_parameters)
    if sys.platform == 'win32':
        worker_count = min(worker_count, MAX_WINDOWS_POOL_WORKERS)
    worker_count = max(1, min(worker_count, total_runs))

    results = []
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp.get_context('spawn')) as executor:
        for batch in timed_batches(total_runs, worker_count):
            # Several runs per message to a worker, and still several
            # messages per worker, so that workers that finish early take more.
            chunk_size = max(1, len(batch) // (4 * worker_count))
            results.extend(executor.map(
                _run_single_simulation_from_arguments,
                (run_parameters[run_index] for run_index in batch),
                chunksize=chunk_size,
            ))

            if progress_callback:
                progress_callback(batch.stop, total_runs)

    return results


def _run_single_simulation_from_arguments(arguments: tuple) -> RunResult:
    """run_single_simulation for one argument tuple; a pool worker can unpickle it by name."""
    return run_single_simulation(*arguments)

