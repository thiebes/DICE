"""
Assertions for seeded simulation results, which repeat to within floating-point rounding.

See dice.analysis.simulation.SEEDED_RESULTS_RTOL for why the comparison allows
a relative difference of 1e-12 instead of requiring bit-identical values.
"""

from typing import Any

import numpy as np
import pandas as pd

from dice.analysis.simulation import SEEDED_RESULTS_RTOL, seeded_results_agree


def assert_seeded_results_agree(actual: Any, desired: Any, err_msg: str = '') -> None:
    """Assert that two arrays of seeded results agree to within floating-point rounding."""
    actual_array = np.asarray(actual, dtype=float)
    desired_array = np.asarray(desired, dtype=float)
    if seeded_results_agree(actual_array, desired_array):
        return
    if actual_array.shape != desired_array.shape:
        detail = f"shapes differ: {actual_array.shape} != {desired_array.shape}"
    else:
        with np.errstate(invalid='ignore', divide='ignore'):
            relative = np.abs(actual_array - desired_array) / np.abs(desired_array)
        detail = (f"largest relative difference {np.nanmax(relative):.3g} "
                  f"exceeds {SEEDED_RESULTS_RTOL:g}, or NaN or infinity differs")
    raise AssertionError(f"{err_msg} seeded results disagree: {detail}".strip())


def assert_seeded_frames_agree(actual: pd.DataFrame, desired: pd.DataFrame) -> None:
    """Same columns in the same order; numeric columns agree to within rounding, the rest exactly."""
    assert list(actual.columns) == list(desired.columns)
    numeric = [column for column in desired.columns if pd.api.types.is_numeric_dtype(desired[column])]
    other = [column for column in desired.columns if column not in numeric]
    pd.testing.assert_frame_equal(actual[other], desired[other], check_exact=True)
    assert_seeded_results_agree(actual[numeric].to_numpy(dtype=float), desired[numeric].to_numpy(dtype=float))
