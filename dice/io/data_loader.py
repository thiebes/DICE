"""
Data loading utilities for DICE.

This module provides functions for loading experimental data files
for CNR estimation and analysis.
"""

import warnings

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Optional, Tuple


def load_profile_data(
    filename: Union[str, Path],
    delimiter: str = ',',
    header: Optional[int] = None
) -> np.ndarray:
    """
    Load profile data from CSV or text file.
    
    Parameters
    ----------
    filename : str or Path
        Path to data file.
    delimiter : str
        Column delimiter.
    header : int, optional
        Row number to use as header.
    
    Returns
    -------
    np.ndarray
        Profile data as numpy array.
    """
    try:
        data = np.loadtxt(filename, delimiter=delimiter, skiprows=header if header else 0)
        return data
    except Exception as e:
        raise IOError(f"Error loading profile data from {filename}: {e}")


def load_time_series_profiles(
    filename: Union[str, Path],
    delimiter: str = ',',
    transpose: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load time series of profiles from file.
    
    Parameters
    ----------
    filename : str or Path
        Path to data file.
    delimiter : str
        Column delimiter.
    transpose : bool
        Whether to transpose the data.
    
    Returns
    -------
    tuple
        (profiles, time_axis) where profiles is 2D array (time x space)
        and time_axis is 1D array if available in first column.
    """
    try:
        data = np.loadtxt(filename, delimiter=delimiter)
        
        if transpose:
            data = data.T
        
        # Check if first column looks like time axis
        if data.shape[1] > 1:
            first_col = data[:, 0]
            if np.all(np.diff(first_col) > 0):  # Monotonically increasing
                time_axis = first_col
                profiles = data[:, 1:]
            else:
                time_axis = None
                profiles = data
        else:
            time_axis = None
            profiles = data
        
        return profiles, time_axis
        
    except Exception as e:
        raise IOError(f"Error loading time series from {filename}: {e}")


def _first_line_is_numeric(filepath: Path, delimiter: str = ',') -> bool:
    """Whether every field of the file's first non-blank line is a number, so the line is data, not a header."""
    # utf-8-sig drops the byte-order mark that spreadsheet programs write before the first value
    with open(filepath, encoding='utf-8-sig') as f:
        first_line = next((line for line in f if line.strip()), '')
    fields = first_line.split(delimiter)
    try:
        for field in fields:
            float(field)
    except ValueError:
        return False
    return True


def load_experimental_data(
    filename: Union[str, Path],
    format: str = 'auto'
) -> dict:
    """
    Load experimental data with automatic format detection.

    A CSV file has a header row only when its first non-blank line contains
    a field that is not a number. A CSV without one, such as a measured
    profile on a single line, is read with np.loadtxt as the noise estimate
    reads it: one line loads as a 1D array. A CSV with a header row is read
    with pandas.

    Parameters
    ----------
    filename : str or Path
        Path to data file.
    format : str
        File format ('csv', 'txt', 'npy', or 'auto' for detection).

    Returns
    -------
    dict
        'filename', 'format', 'data', 'shape' and 'ndim'; a CSV with a header
        row also has 'columns', the header's column names.
    """
    filepath = Path(filename)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filename}")
    
    # Auto-detect format
    if format == 'auto':
        suffix = filepath.suffix.lower()
        if suffix == '.csv':
            format = 'csv'
        elif suffix in ['.txt', '.dat']:
            format = 'txt'
        elif suffix == '.npy':
            format = 'npy'
        else:
            format = 'txt'  # Default
    
    result = {
        'filename': str(filepath),
        'format': format,
    }
    
    try:
        if format == 'csv':
            if _first_line_is_numeric(filepath):
                result['data'] = np.loadtxt(filepath, delimiter=',', encoding='utf-8-sig')
            else:
                df = pd.read_csv(filepath)
                result['data'] = df.values
                result['columns'] = df.columns.tolist()
            
        elif format == 'npy':
            result['data'] = np.load(filepath)
            
        else:  # txt or other
            result['data'] = np.loadtxt(filepath)
        
        # Add data shape info
        result['shape'] = result['data'].shape
        result['ndim'] = result['data'].ndim
        
    except Exception as e:
        raise IOError(f"Error loading {format} file {filename}: {e}")
    
    return result


def save_profile_data(
    data: np.ndarray,
    filename: Union[str, Path],
    delimiter: str = ',',
    header: Optional[str] = None
) -> None:
    """
    Save profile data to file.
    
    Parameters
    ----------
    data : np.ndarray
        Data to save.
    filename : str or Path
        Output filename.
    delimiter : str
        Column delimiter.
    header : str, optional
        Header line to prepend.
    """
    try:
        np.savetxt(filename, data, delimiter=delimiter, header=header if header else '')
    except Exception as e:
        raise IOError(f"Error saving profile data to {filename}: {e}")


def batch_load_profiles(
    directory: Union[str, Path],
    pattern: str = "*.csv",
    **load_kwargs
) -> List[dict]:
    """
    Load multiple profile files from a directory.
    
    Parameters
    ----------
    directory : str or Path
        Directory containing files.
    pattern : str
        Glob pattern for file matching.
    **load_kwargs
        Additional arguments for load_experimental_data.
    
    Returns
    -------
    list
        List of loaded data dictionaries.
    """
    directory = Path(directory)
    files = sorted(directory.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {directory}")
    
    results = []
    for file in files:
        try:
            data = load_experimental_data(file, **load_kwargs)
            results.append(data)
        except Exception as e:
            warnings.warn(f"Failed to load {file}: {e}", stacklevel=2)
    
    return results