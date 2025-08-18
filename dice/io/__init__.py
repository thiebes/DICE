"""
Input/Output modules for DICE.

This package contains modules for loading parameters, saving results,
and handling experimental data files.
"""

from .parameters import (
    open_parameters,
    parameter_parser,
    handle_time_parameters,
    handle_noise_parameters,
    handle_profile_width_parameters,
    handle_diffusion_parameters,
    create_simulation_parameters,
    validate_parameters,
)

from .results import (
    print_and_append,
    export_collated_results,
    write_summary_file,
    export_to_json,
    load_results_csv,
    load_multiple_results,
    export_legacy_format,
)

from .data_loader import (
    load_profile_data,
    load_time_series_profiles,
    load_experimental_data,
    save_profile_data,
    batch_load_profiles,
)

__all__ = [
    # Parameters functions
    'open_parameters',
    'parameter_parser',
    'handle_time_parameters',
    'handle_noise_parameters',
    'handle_profile_width_parameters',
    'handle_diffusion_parameters',
    'create_simulation_parameters',
    'validate_parameters',
    # Results functions
    'print_and_append',
    'export_collated_results',
    'write_summary_file',
    'export_to_json',
    'load_results_csv',
    'load_multiple_results',
    'export_legacy_format',
    # Data loader functions
    'load_profile_data',
    'load_time_series_profiles',
    'load_experimental_data',
    'save_profile_data',
    'batch_load_profiles',
]