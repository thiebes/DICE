"""
Monte Carlo simulation runner for DICE.

This module provides the core simulation functionality for running Monte Carlo
simulations of diffusion measurements with noise.
"""

import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from typing import List, Tuple, Optional, Any
from joblib import Parallel, delayed, effective_n_jobs
import pandas as pd

from ..core.profiles import gaussian, make_diffusion_decay
from fft_cnr import fft_cnr
from ..core.noise import Seed, add_noise
from ..core.fitting import gauss_fitting, calculate_wls_weights
from ..models.estimators import DEFAULT_ESTIMATOR_KEYS, FrameFits, get_estimator
from ..models.parameters import SimulationParameters
from ..models.results import RunResult, MonteCarloOutput
from ..utils.runtime import joblib_workers_supported
from ..utils.legacy_compatibility import LegacySimulationParameters

# Most batches a parallel simulation is split into; the progress callback reports after each.
MAX_PROGRESS_BATCHES = 10

# Worker pools for a parallel simulation: joblib's default (loky) backend, or
# the standard library's ProcessPoolExecutor, which a standalone build uses.
JOBLIB_BACKEND = 'joblib'
PROCESS_POOL_BACKEND = 'process_pool'
PARALLEL_BACKENDS = (JOBLIB_BACKEND, PROCESS_POOL_BACKEND)

# ProcessPoolExecutor rejects more workers than this on Windows.
MAX_WINDOWS_POOL_WORKERS = 61


def run_single_simulation(
    run_id: int,
    x_axis: np.ndarray,
    time_axis: np.ndarray,
    parameters: LegacySimulationParameters,
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
    parameters : LegacySimulationParameters
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
    parameters: LegacySimulationParameters,
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
) -> MonteCarloOutput:
    """
    Run Monte Carlo simulation with multiple noise values.
    
    Parameters
    ----------
    parameters : SimulationParameters
        Simulation parameters.
    x_axis : np.ndarray
        Spatial axis values.
    time_axis : np.ndarray
        Time axis values.
    noise_values : List[float]
        List of noise sigma values to simulate.
    num_runs : int
        Number of runs per noise value.
    multiprocessing : bool
        Whether to use parallel processing.
    retain_profile_data : bool
        Whether to retain full profile data.
    progress_callback : callable, optional
        Function to call with progress updates.
    seed : int or SeedSequence, optional
        Root seed. Each run gets its own child stream, so a seeded simulation
        gives identical results serially or in parallel. None draws fresh
        entropy for every run.
    n_jobs : int
        Number of worker processes when multiprocessing; -1 uses all cores.
    parallel_backend : str, optional
        Worker pool when multiprocessing: 'joblib' or 'process_pool' (the
        standard library's ProcessPoolExecutor). None selects joblib, or the
        process pool in a standalone build, where joblib cannot start workers.
        Both give the same results as a serial run with the same seed.

    Returns
    -------
    MonteCarloOutput
        Aggregated simulation results.
    """
    if parallel_backend is None:
        parallel_backend = default_parallel_backend()
    elif parallel_backend not in PARALLEL_BACKENDS:
        raise ValueError(
            f"Unknown parallel backend {parallel_backend!r}; expected one of {PARALLEL_BACKENDS}"
        )

    run_seeds = spawn_run_seeds(seed, len(noise_values) * num_runs)

    # Create parameter sets for all runs
    run_parameters = []
    run_id = 0

    for noise_sigma in noise_values:
        for _ in range(num_runs):
            run_parameters.append((
                run_id,
                x_axis,
                time_axis,
                parameters,
                noise_sigma,
                retain_profile_data,
                run_seeds[run_id],
            ))
            run_id += 1
    
    total_runs = len(run_parameters)

    # Run simulations
    if multiprocessing and parallel_backend == PROCESS_POOL_BACKEND:
        results = run_in_process_pool(run_parameters, effective_n_jobs(n_jobs), progress_callback)
    elif multiprocessing:
        # Run in batches so the progress callback can report between them;
        # joblib reuses its worker pool from one batch to the next.
        results = []

        for batch in split_into_batches(total_runs, effective_n_jobs(n_jobs)):
            batch_results = Parallel(n_jobs=n_jobs)(
                delayed(run_single_simulation)(*run_parameters[run_index])
                for run_index in batch
            )
            results.extend(batch_results)

            if progress_callback:
                progress_callback(batch.stop, total_runs)
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
    )

    return simulation_result


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


def split_into_batches(run_count: int, worker_count: int) -> List[range]:
    """Consecutive run-index ranges for progress reporting, each with at least one run per worker.

    At most MAX_PROGRESS_BATCHES batches, with sizes that differ by at most one.
    A batch smaller than the worker pool would leave workers idle and make
    them start one batch at a time instead of together.
    """
    if run_count <= 0:
        return []
    batch_count = max(1, min(MAX_PROGRESS_BATCHES, run_count // max(1, worker_count)))
    base_size, larger_batches = divmod(run_count, batch_count)
    batches = []
    start = 0
    for batch_index in range(batch_count):
        stop = start + base_size + (1 if batch_index < larger_batches else 0)
        batches.append(range(start, stop))
        start = stop
    return batches


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
    application. As in the joblib path, the progress callback reports after
    each batch; an exception it raises (the GUI's Stop raises
    InterruptedError) propagates once the workers have exited.

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
        for batch in split_into_batches(total_runs, worker_count):
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

