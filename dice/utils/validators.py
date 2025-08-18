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