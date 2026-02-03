"""
Input validation functions for DICE package.

This module provides common validation functions used throughout the package
to ensure data integrity and provide meaningful error messages.
"""

import numpy as np
from typing import Any, Union, List, Tuple, Optional


def validate_numeric(value: Any, name: str = "value", 
                    min_val: Optional[float] = None,
                    max_val: Optional[float] = None,
                    allow_inf: bool = False) -> Union[int, float]:
    """
    Validate that a value is numeric and optionally within bounds.
    
    Parameters
    ----------
    value : Any
        The value to validate.
    name : str, optional
        The name of the parameter for error messages.
    min_val : float, optional
        Minimum allowed value (inclusive).
    max_val : float, optional
        Maximum allowed value (inclusive).
    allow_inf : bool, optional
        Whether to allow infinite values.
    
    Returns
    -------
    int or float
        The validated numeric value.
    
    Raises
    ------
    TypeError
        If the value is not numeric.
    ValueError
        If the value is outside bounds or infinite when not allowed.
    """
    if not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    
    value = float(value)
    
    if not allow_inf and not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    
    if min_val is not None and value < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {value}")
    
    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be <= {max_val}, got {value}")
    
    return value


def validate_array_like(value: Any, name: str = "array",
                       ndim: Optional[int] = None,
                       shape: Optional[Tuple] = None,
                       non_empty: bool = True) -> np.ndarray:
    """
    Validate and convert a value to a numpy array with optional shape checks.
    
    Parameters
    ----------
    value : array_like
        The value to validate and convert.
    name : str, optional
        The name of the parameter for error messages.
    ndim : int, optional
        Expected number of dimensions.
    shape : tuple, optional
        Expected exact shape.
    non_empty : bool, optional
        Whether to require non-empty array.
    
    Returns
    -------
    np.ndarray
        The input as a numpy array.
    
    Raises
    ------
    ValueError
        If the value doesn't meet requirements.
    """
    try:
        array = np.asarray(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cannot convert {name} to array: {e}")
    
    if non_empty and array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {array.ndim}")
    
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    
    return array


def validate_integer(value: Any, name: str = "value",
                    min_val: Optional[int] = None,
                    max_val: Optional[int] = None) -> int:
    """
    Validate that a value is an integer, optionally within bounds.
    
    Parameters
    ----------
    value : Any
        The value to validate.
    name : str, optional
        The name of the parameter for error messages.
    min_val : int, optional
        Minimum allowed value (inclusive).
    max_val : int, optional
        Maximum allowed value (inclusive).
    
    Returns
    -------
    int
        The validated integer value.
    
    Raises
    ------
    TypeError
        If the value is not an integer.
    ValueError
        If the value is outside bounds.
    """
    if isinstance(value, (np.integer, np.int_)):
        value = int(value)
    elif isinstance(value, float):
        if not value.is_integer():
            raise TypeError(f"{name} must be an integer, got float {value}")
        value = int(value)
    elif not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    
    if min_val is not None and value < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {value}")
    
    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be <= {max_val}, got {value}")
    
    return value


def validate_string_choice(value: str, choices: List[str], 
                          name: str = "value",
                          case_sensitive: bool = True) -> str:
    """
    Validate that a string value is one of the allowed choices.
    
    Parameters
    ----------
    value : str
        The value to validate.
    choices : list of str
        List of allowed values.
    name : str, optional
        The name of the parameter for error messages.
    case_sensitive : bool, optional
        Whether to perform case-sensitive comparison.
    
    Returns
    -------
    str
        The validated string value (potentially normalized case).
    
    Raises
    ------
    ValueError
        If the value is not in the allowed choices.
    TypeError
        If the value is not a string.
    """
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    
    if not case_sensitive:
        value_lower = value.lower()
        choices_lower = [c.lower() for c in choices]
        if value_lower in choices_lower:
            # Return the original case from choices
            return choices[choices_lower.index(value_lower)]
        else:
            raise ValueError(f"{name} must be one of {choices}, got '{value}'")
    else:
        if value not in choices:
            raise ValueError(f"{name} must be one of {choices}, got '{value}'")
        return value


def validate_file_path(path: str, must_exist: bool = False,
                      extension: Optional[str] = None,
                      name: str = "path") -> str:
    """
    Validate a file path.
    
    Parameters
    ----------
    path : str
        The file path to validate.
    must_exist : bool, optional
        Whether the file must already exist.
    extension : str, optional
        Required file extension (e.g., '.txt').
    name : str, optional
        The name of the parameter for error messages.
    
    Returns
    -------
    str
        The validated file path.
    
    Raises
    ------
    ValueError
        If validation fails.
    TypeError
        If path is not a string.
    """
    import os
    
    if not isinstance(path, str):
        raise TypeError(f"{name} must be a string, got {type(path).__name__}")
    
    if must_exist and not os.path.exists(path):
        raise ValueError(f"{name} does not exist: {path}")
    
    if extension is not None:
        if not path.endswith(extension):
            raise ValueError(f"{name} must have extension {extension}, got {path}")

    return path


def validate_filename_slug(value: str, name: str = "filename") -> str:
    """
    Validate that a filename slug contains no forbidden characters.

    Parameters
    ----------
    value : str
        The filename slug to validate.
    name : str, optional
        The name of the parameter for error messages.

    Returns
    -------
    str
        The validated filename slug.

    Raises
    ------
    TypeError
        If the value is not a string.
    ValueError
        If the value is empty or contains forbidden characters.
    """
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")

    if not value or not value.strip():
        raise ValueError(f"{name} cannot be empty")

    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in invalid_chars:
        if char in value:
            raise ValueError(f"{name} cannot contain: {', '.join(invalid_chars)}")

    return value


def validate_time_values(values: List[float], name: str = "time values") -> List[float]:
    """
    Validate a list of time values (must be finite, unique, and non-empty).

    Parameters
    ----------
    values : list of float
        The time values to validate.
    name : str, optional
        The name of the parameter for error messages.

    Returns
    -------
    list of float
        The validated time values.

    Raises
    ------
    TypeError
        If values is not a list or contains non-numeric types.
    ValueError
        If values is empty, contains non-finite numbers, or has duplicates.
    """
    if not isinstance(values, list):
        raise TypeError(f"{name} must be a list, got {type(values).__name__}")

    if len(values) == 0:
        raise ValueError(f"{name} must contain at least one value")

    validated = []
    for i, v in enumerate(values):
        if not isinstance(v, (int, float, np.number)):
            raise TypeError(f"{name}[{i}] must be numeric, got {type(v).__name__}")
        float_v = float(v)
        if not np.isfinite(float_v):
            raise ValueError(f"All {name} must be finite, got {float_v}")
        validated.append(float_v)

    if len(validated) != len(set(validated)):
        raise ValueError(f"{name} must be unique")

    return validated


def validate_range(start: float, stop: float, steps: int,
                   name: str = "range") -> Tuple[float, float, int]:
    """
    Validate range parameters (start < stop, steps > 0).

    Parameters
    ----------
    start : float
        The start value of the range.
    stop : float
        The stop value of the range.
    steps : int
        The number of steps in the range.
    name : str, optional
        The name of the parameter for error messages.

    Returns
    -------
    tuple
        A tuple of (start, stop, steps).

    Raises
    ------
    TypeError
        If parameters have wrong types.
    ValueError
        If start >= stop or steps <= 0.
    """
    start = validate_numeric(start, name=f"{name} start")
    stop = validate_numeric(stop, name=f"{name} stop")
    steps = validate_integer(steps, name=f"{name} steps", min_val=1)

    if start >= stop:
        raise ValueError(f"{name} start must be less than stop, got {start} >= {stop}")

    return (start, stop, steps)