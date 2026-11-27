"""
DICE - Diffusion Insight Computation Engine

A Python package for quantifying noise effects in optical measures of
excited state transport in optoelectronic semiconducting materials.
"""

__version__ = "2.0.0.dev0"
__author__ = "Joseph J. Thiebes"
__license__ = "MIT"

# Lazy imports (PEP 562) so that importing the package for metadata
# (e.g. __version__) does not pull in heavy dependencies like scipy.
_LAZY_IMPORTS = {
    "open_parameters": ".io.parameters",
    "run_monte_carlo_simulation": ".analysis.simulation",
    "analyze_simulation_results": ".analysis.statistics",
    "simulation_parameters_from_dict": ".models.parameters",
    "make_x_axis": ".utils.axes",
    "make_time_axis": ".utils.axes",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def dice_runner(parameters_filename, allow_network_paths: bool = False):
    """
    Run a DICE simulation from a parameters file.

    Parameters
    ----------
    parameters_filename : str
        Path to parameters file.
    allow_network_paths : bool
        Open an 'estimate noise from data' file on a network path; see
        dice.io.parameters.open_parameters.

    Returns
    -------
    MonteCarloOutput
        The simulation results.
    """
    from .io.parameters import noise_range_from_parameters, open_parameters, validate_parameters
    from .analysis.simulation import run_monte_carlo_simulation
    from .analysis.statistics import analyze_simulation_results
    from .models.parameters import simulation_parameters_from_dict

    # Load parameters
    parameters = open_parameters(parameters_filename, allow_network_paths=allow_network_paths)
    validate_parameters(parameters)

    # Create axes
    x_axis = parameters['x array']
    time_axis = parameters['time series']

    # Create simulation parameters
    sim_params = simulation_parameters_from_dict(
        parameters_dict=parameters,
        diffusion_coefficient=parameters['nominal diffusion coefficient'],
        lifetime=parameters['nominal lifetime (tau)'],
        diffusion_length=parameters['nominal diffusion length']
    )

    # Run simulation
    result = run_monte_carlo_simulation(
        parameters=sim_params,
        x_axis=x_axis,
        time_axis=time_axis,
        noise_values=parameters['noise series'],
        num_runs=parameters['number of runs'],
        multiprocessing=parameters.get('multiprocessing', 1) != 0,
        retain_profile_data=parameters.get('retain profile data', 0) != 0,
        noise_range=noise_range_from_parameters(parameters),
    )

    # Analyze results
    analyze_simulation_results(result)

    return result

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "dice_runner",
    "open_parameters",
    "run_monte_carlo_simulation",
    "analyze_simulation_results",
    "simulation_parameters_from_dict",
    "make_x_axis",
    "make_time_axis",
    "core",
    "analysis",
    "io",
    "visualization",
    "cli",
    "utils",
    "models",
]
