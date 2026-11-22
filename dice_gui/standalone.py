"""
Entry point of the standalone (PyInstaller) DICE application.

dice-gui.spec builds the application from this script, which decides what
the process is for before it imports Qt or the GUI:

- A worker process of a parallel simulation starts the application again;
  multiprocessing.freeze_support() runs the worker and exits.
- With --self-test, the application simulates the quick_test example serially
  and in the process pool with one seed, prints whether the results agree to
  within floating-point rounding, and exits with status 0 if they do and 1
  otherwise. It opens no window.
- Otherwise it starts the GUI.
"""

import multiprocessing
import sys
import traceback
from typing import List, Optional, TextIO, Tuple

SELF_TEST_OPTION = '--self-test'

# Root seed of both self-test simulations.
SELF_TEST_SEED = 1

SELF_TEST_EXAMPLE = 'quick_test.txt'


def main() -> None:
    """Run a worker, the self-test, or the GUI."""
    multiprocessing.freeze_support()
    if SELF_TEST_OPTION in sys.argv[1:]:
        sys.exit(self_test_exit_status())

    from dice_gui.dice_gui import main as start_gui
    start_gui()


def self_test_exit_status() -> int:
    """Run the self-test on standard output; 0 if it passes, 1 if it fails or raises.

    The exception is printed rather than raised, because an unhandled
    exception in the windowed Windows build opens a dialog and waits.
    """
    try:
        passed = run_self_test(sys.stdout)
    except Exception:
        traceback.print_exc()
        return 1
    return 0 if passed else 1


def run_self_test(output: Optional[TextIO]) -> bool:
    """
    Simulate the bundled quick_test example serially and in the process pool with one seed.

    Parameters
    ----------
    output : TextIO or None
        Stream for the report. None (the standard output of a windowed
        build started without redirection) discards it.

    Returns
    -------
    bool
        True if the two simulations agree to within floating-point rounding
        (dice.analysis.simulation.seeded_results_agree), the progress
        reached every run, no worker process remained afterwards, and, in a
        standalone build, the application's parallel simulations use the
        process pool and the application carries its license notice.
    """
    import time

    import numpy as np
    from joblib import cpu_count

    from dice import __version__
    from dice.analysis.simulation import (
        PROCESS_POOL_BACKEND,
        SEEDED_RESULTS_RTOL,
        default_parallel_backend,
        run_monte_carlo_simulation,
        seeded_results_agree,
    )
    from dice.io.parameters import open_parameters
    from dice.utils.legacy_compatibility import create_parameters_from_legacy
    from dice.utils.runtime import bundled_license_notice, examples_directory, is_standalone_build

    example = examples_directory() / SELF_TEST_EXAMPLE
    parameters = open_parameters(example)
    simulation_parameters = create_parameters_from_legacy(
        parameters_dict=parameters,
        diffusion_coefficient=parameters['nominal diffusion coefficient'],
        lifetime=parameters['nominal lifetime (tau)'],
        diffusion_length=parameters['nominal diffusion length'],
    )

    def simulate(parallel: bool) -> Tuple[np.ndarray, List[Tuple[int, int]], float]:
        progress = []
        start = time.perf_counter()
        result = run_monte_carlo_simulation(
            parameters=simulation_parameters,
            x_axis=parameters['x array'],
            time_axis=parameters['time series'],
            noise_values=parameters['noise series'],
            num_runs=parameters['number of runs'],
            multiprocessing=parallel,
            progress_callback=lambda done, total: progress.append((done, total)),
            seed=SELF_TEST_SEED,
            parallel_backend=PROCESS_POOL_BACKEND,
        )
        return run_values(result.run_results), progress, time.perf_counter() - start

    print(f"DICE {__version__} self-test", file=output)
    print(f"Example: {example}, seed {SELF_TEST_SEED}", file=output)
    serial, _, serial_seconds = simulate(parallel=False)
    print(f"Serial: {len(serial)} runs in {serial_seconds:.1f} s", file=output)
    # Child processes that existed before, such as joblib's reusable workers
    # in a regular installation, are not the pool's.
    earlier_children = set(multiprocessing.active_children())
    parallel, progress, parallel_seconds = simulate(parallel=True)
    print(f"Process pool on {cpu_count()} CPU cores: {len(parallel)} runs in {parallel_seconds:.1f} s",
          file=output)

    total_runs = len(serial)
    agree = seeded_results_agree(parallel, serial)
    progress_complete = bool(progress) and progress[-1] == (total_runs, total_runs)
    remaining_workers = set(multiprocessing.active_children()) - earlier_children
    application_backend = default_parallel_backend()
    backend_tested = application_backend == PROCESS_POOL_BACKEND or not is_standalone_build()
    # A standalone build bundles PyQt6 and must carry the license texts.
    license_notice = bundled_license_notice()
    notice_carried = license_notice is not None or not is_standalone_build()

    print(f"Results agree to within rounding (relative {SEEDED_RESULTS_RTOL:g}): {'yes' if agree else 'no'}",
          file=output)
    print(f"Progress reached {total_runs} of {total_runs} runs: {'yes' if progress_complete else 'no'}",
          file=output)
    print(f"Worker processes still running: {len(remaining_workers)}", file=output)
    print(f"Parallel simulations of this application use: {application_backend}", file=output)
    if is_standalone_build():
        print(f"License notice bundled: {'yes' if license_notice is not None else 'no'}", file=output)

    passed = agree and progress_complete and not remaining_workers and backend_tested and notice_carried
    print(f"Self-test {'passed' if passed else 'failed'}", file=output)
    return passed


def run_values(run_results: list) -> 'np.ndarray':
    """Every field of every run as one float array, NaN where a field is None.

    The self-test keeps no profile data, so every field is a number or None.
    """
    import dataclasses

    import numpy as np

    return np.array(
        [[np.nan if value is None else value for value in dataclasses.astuple(run)] for run in run_results],
        dtype=float,
    )


if __name__ == '__main__':
    main()
