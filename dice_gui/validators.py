"""
Input validation functions for DICE GUI.

This module provides validators for all parameter types used in DICE simulations.
"""

from typing import Optional, Tuple, List, Union


class ValidationResult:
    """Result of a validation check."""

    def __init__(self, is_valid: bool, error_message: str = ""):
        self.is_valid = is_valid
        self.error_message = error_message

    def __bool__(self):
        return self.is_valid


def validate_positive_integer(value: str, param_name: str = "Value") -> ValidationResult:
    """Validate that a string represents a positive integer."""
    if not value or value.strip() == "":
        return ValidationResult(False, f"{param_name} is required")

    try:
        int_value = int(value)
        if int_value <= 0:
            return ValidationResult(False, f"{param_name} must be greater than 0")
        return ValidationResult(True)
    except ValueError:
        return ValidationResult(False, f"{param_name} must be a valid integer")


def validate_positive_float(value: str, param_name: str = "Value", allow_zero: bool = False) -> ValidationResult:
    """Validate that a string represents a positive float."""
    if not value or value.strip() == "":
        return ValidationResult(False, f"{param_name} is required")

    try:
        float_value = float(value)
        if allow_zero:
            if float_value < 0:
                return ValidationResult(False, f"{param_name} must be greater than or equal to 0")
        else:
            if float_value <= 0:
                return ValidationResult(False, f"{param_name} must be greater than 0")
        return ValidationResult(True)
    except ValueError:
        return ValidationResult(False, f"{param_name} must be a valid number")


def validate_float(value: str, param_name: str = "Value") -> ValidationResult:
    """Validate that a string represents any float value."""
    if not value or value.strip() == "":
        return ValidationResult(False, f"{param_name} is required")

    try:
        float(value)
        return ValidationResult(True)
    except ValueError:
        return ValidationResult(False, f"{param_name} must be a valid number")


def validate_proximity_level(value: str) -> ValidationResult:
    """Validate proximity level (must be between 0 and 1, exclusive)."""
    if not value or value.strip() == "":
        return ValidationResult(False, "Proximity level is required")

    try:
        float_value = float(value)
        if float_value <= 0 or float_value >= 1:
            return ValidationResult(False, "Proximity level must be between 0 and 1")
        return ValidationResult(True)
    except ValueError:
        return ValidationResult(False, "Proximity level must be a valid number")


def validate_time_range(start: str, stop: str, steps: str) -> ValidationResult:
    """Validate time range parameters."""
    # Validate each field individually first
    start_result = validate_float(start, "Start time")
    if not start_result:
        return start_result

    stop_result = validate_float(stop, "Stop time")
    if not stop_result:
        return stop_result

    steps_result = validate_positive_integer(steps, "Number of steps")
    if not steps_result:
        return steps_result

    # Validate relationship between start and stop
    start_val = float(start)
    stop_val = float(stop)
    if start_val >= stop_val:
        return ValidationResult(False, "Start time must be less than stop time")

    return ValidationResult(True)


def validate_time_series(value: str) -> ValidationResult:
    """Validate comma-separated time series."""
    if not value or value.strip() == "":
        return ValidationResult(False, "Time series is required")

    try:
        values = [float(v.strip()) for v in value.split(",") if v.strip()]
        if len(values) == 0:
            return ValidationResult(False, "Time series must contain at least one value")

        # Check for finite values
        if not all(float('-inf') < v < float('inf') for v in values):
            return ValidationResult(False, "All time values must be finite")

        # Check for duplicates
        if len(values) != len(set(values)):
            return ValidationResult(False, "Time series values must be unique")

        return ValidationResult(True)
    except ValueError:
        return ValidationResult(False, "Time series must be comma-separated numbers")


def validate_file_path(path: str) -> ValidationResult:
    """Validate that a file path exists."""
    if not path or path.strip() == "":
        return ValidationResult(False, "File path is required")

    import os
    if not os.path.exists(path):
        return ValidationResult(False, f"File not found: {path}")

    if not os.path.isfile(path):
        return ValidationResult(False, f"Path is not a file: {path}")

    return ValidationResult(True)


def validate_filename_slug(value: str) -> ValidationResult:
    """Validate filename slug (basic validation, no special characters)."""
    if not value or value.strip() == "":
        return ValidationResult(False, "Filename slug is required")

    # Check for invalid characters
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in invalid_chars:
        if char in value:
            return ValidationResult(False, f"Filename slug cannot contain: {', '.join(invalid_chars)}")

    return ValidationResult(True)


def convert_fwhm_to_sigma(fwhm: float) -> float:
    """Convert FWHM to sigma."""
    import math
    return fwhm / (2 * math.sqrt(2 * math.log(2)))


def convert_sigma_to_fwhm(sigma: float) -> float:
    """Convert sigma to FWHM."""
    import math
    return sigma * 2 * math.sqrt(2 * math.log(2))


def calculate_diffusion_length(D: float, tau: float) -> float:
    """Calculate diffusion length from D and tau."""
    import math
    return math.sqrt(D * tau)


def calculate_pixel_size(spatial_width: float, pixel_width: int) -> float:
    """Calculate pixel size from spatial width and number of pixels."""
    return spatial_width / pixel_width
