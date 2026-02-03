"""
Input validation functions for DICE GUI.

This module provides validators for all parameter types used in DICE simulations.
GUI validators wrap core validators from dice.utils.validators, converting
exceptions to ValidationResult objects and handling string input parsing.
"""

from typing import Optional, Tuple, List, Union

from dice.utils.validators import (
    validate_numeric as _validate_numeric,
    validate_integer as _validate_integer,
    validate_file_path as _validate_file_path,
    validate_filename_slug as _validate_filename_slug,
    validate_time_values as _validate_time_values,
    validate_range as _validate_range,
)
from dice.utils.converters import (
    fwhm_to_sigma as _fwhm_to_sigma,
    sigma_to_fwhm as _sigma_to_fwhm,
    calculate_pixel_size as _calculate_pixel_size,
)
from dice.core.diffusion import calculate_diffusion_length as _calculate_diffusion_length


class ValidationResult:
    """Result of a validation check."""

    def __init__(self, is_valid: bool, error_message: str = "", value=None):
        self.is_valid = is_valid
        self.error_message = error_message
        self.value = value

    def __bool__(self):
        return self.is_valid


def validate_positive_integer(value: str, param_name: str = "Value") -> ValidationResult:
    """Validate that a string represents a positive integer."""
    if not value or value.strip() == "":
        return ValidationResult(False, f"{param_name} is required")

    try:
        parsed = int(value.strip())
    except ValueError:
        return ValidationResult(False, f"{param_name} must be a valid integer")

    try:
        validated = _validate_integer(parsed, name=param_name, min_val=1)
        return ValidationResult(True, value=validated)
    except (TypeError, ValueError) as e:
        return ValidationResult(False, str(e))


def validate_positive_float(value: str, param_name: str = "Value", allow_zero: bool = False) -> ValidationResult:
    """Validate that a string represents a positive float."""
    if not value or value.strip() == "":
        return ValidationResult(False, f"{param_name} is required")

    try:
        parsed = float(value.strip())
    except ValueError:
        return ValidationResult(False, f"{param_name} must be a valid number")

    try:
        min_val = 0.0 if allow_zero else None
        validated = _validate_numeric(parsed, name=param_name, min_val=min_val)
        if not allow_zero and validated <= 0:
            return ValidationResult(False, f"{param_name} must be greater than 0")
        return ValidationResult(True, value=validated)
    except (TypeError, ValueError) as e:
        return ValidationResult(False, str(e))


def validate_float(value: str, param_name: str = "Value") -> ValidationResult:
    """Validate that a string represents any float value."""
    if not value or value.strip() == "":
        return ValidationResult(False, f"{param_name} is required")

    try:
        parsed = float(value.strip())
    except ValueError:
        return ValidationResult(False, f"{param_name} must be a valid number")

    try:
        validated = _validate_numeric(parsed, name=param_name)
        return ValidationResult(True, value=validated)
    except (TypeError, ValueError) as e:
        return ValidationResult(False, str(e))


def validate_proximity_level(value: str) -> ValidationResult:
    """Validate proximity level (must be between 0 and 1, exclusive)."""
    if not value or value.strip() == "":
        return ValidationResult(False, "Proximity level is required")

    try:
        parsed = float(value.strip())
    except ValueError:
        return ValidationResult(False, "Proximity level must be a valid number")

    try:
        validated = _validate_numeric(parsed, name="Proximity level", min_val=0.0, max_val=1.0)
        if validated <= 0 or validated >= 1:
            return ValidationResult(False, "Proximity level must be between 0 and 1")
        return ValidationResult(True, value=validated)
    except (TypeError, ValueError) as e:
        return ValidationResult(False, str(e))


def validate_time_range(start: str, stop: str, steps: str) -> ValidationResult:
    """Validate time range parameters."""
    start_result = validate_float(start, "Start time")
    if not start_result:
        return start_result

    stop_result = validate_float(stop, "Stop time")
    if not stop_result:
        return stop_result

    steps_result = validate_positive_integer(steps, "Number of steps")
    if not steps_result:
        return steps_result

    try:
        validated = _validate_range(
            start_result.value,
            stop_result.value,
            steps_result.value,
            name="Time range"
        )
        return ValidationResult(True, value=validated)
    except (TypeError, ValueError) as e:
        return ValidationResult(False, str(e))


def validate_time_series(value: str) -> ValidationResult:
    """Validate comma-separated time series."""
    if not value or value.strip() == "":
        return ValidationResult(False, "Time series is required")

    try:
        parsed = [float(v.strip()) for v in value.split(",") if v.strip()]
    except ValueError:
        return ValidationResult(False, "Time series must be comma-separated numbers")

    if len(parsed) == 0:
        return ValidationResult(False, "Time series must contain at least one value")

    try:
        validated = _validate_time_values(parsed, name="Time series")
        return ValidationResult(True, value=validated)
    except (TypeError, ValueError) as e:
        return ValidationResult(False, str(e))


def validate_file_path(path: str) -> ValidationResult:
    """Validate that a file path exists."""
    if not path or path.strip() == "":
        return ValidationResult(False, "File path is required")

    import os

    try:
        validated = _validate_file_path(path.strip(), must_exist=True, name="File")
    except (TypeError, ValueError) as e:
        return ValidationResult(False, str(e))

    if not os.path.isfile(validated):
        return ValidationResult(False, f"Path is not a file: {path}")

    return ValidationResult(True, value=validated)


def validate_filename_slug(value: str) -> ValidationResult:
    """Validate filename slug (basic validation, no special characters)."""
    if not value or value.strip() == "":
        return ValidationResult(False, "Filename slug is required")

    try:
        validated = _validate_filename_slug(value.strip(), name="Filename slug")
        return ValidationResult(True, value=validated)
    except (TypeError, ValueError) as e:
        return ValidationResult(False, str(e))


def convert_fwhm_to_sigma(fwhm_str: str) -> ValidationResult:
    """Convert FWHM string to sigma, returning ValidationResult."""
    result = validate_positive_float(fwhm_str, "FWHM")
    if not result:
        return result

    try:
        sigma = _fwhm_to_sigma(result.value)
        return ValidationResult(True, value=sigma)
    except ValueError as e:
        return ValidationResult(False, str(e))


def convert_sigma_to_fwhm(sigma_str: str) -> ValidationResult:
    """Convert sigma string to FWHM, returning ValidationResult."""
    result = validate_positive_float(sigma_str, "Sigma")
    if not result:
        return result

    try:
        fwhm = _sigma_to_fwhm(result.value)
        return ValidationResult(True, value=fwhm)
    except ValueError as e:
        return ValidationResult(False, str(e))


def calculate_diffusion_length(d_str: str, tau_str: str) -> ValidationResult:
    """Calculate diffusion length from D and tau strings, returning ValidationResult."""
    d_result = validate_positive_float(d_str, "D", allow_zero=True)
    if not d_result:
        return d_result

    tau_result = validate_positive_float(tau_str, "tau", allow_zero=True)
    if not tau_result:
        return tau_result

    try:
        length = _calculate_diffusion_length(d_result.value, tau_result.value)
        return ValidationResult(True, value=length)
    except ValueError as e:
        return ValidationResult(False, str(e))


def calculate_pixel_size(spatial_str: str, pixel_count: int) -> ValidationResult:
    """Calculate pixel size from spatial width string and pixel count, returning ValidationResult."""
    result = validate_positive_float(spatial_str, "Spatial width")
    if not result:
        return result

    try:
        pixel_size = _calculate_pixel_size(result.value, pixel_count)
        return ValidationResult(True, value=pixel_size)
    except ValueError as e:
        return ValidationResult(False, str(e))
