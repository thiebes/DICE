"""
DICE - Diffusion Insight Computation Engine

A Python package for quantifying noise effects in optical measures of
excited state transport in optoelectronic semiconducting materials.
"""

__version__ = "1.3.0"
__author__ = "Joseph J. Thiebes"
__license__ = "CC BY 4.0"

# Lazy imports (PEP 562) so that importing the package for metadata
# (e.g. __version__) does not pull in heavy dependencies like scipy.
_LAZY_IMPORTS = {
    "open_parameters": ".io.parameters",
    "run_monte_carlo_simulation": ".analysis.simulation",
    "analyze_simulation_results": ".analysis.statistics",
    "create_parameters_from_legacy": ".utils.legacy_compatibility",
    "make_x_axis": ".utils.axes",
    "make_time_axis": ".utils.axes",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def dice_runner(parameters_filename):
    """
    Run a DICE simulation from a parameters file.

    Parameters
    ----------
    parameters_filename : str
        Path to parameters file.

    Returns
    -------
    MonteCarloOutput
        The simulation results.
    """
    from .io.parameters import open_parameters
    from .analysis.simulation import run_monte_carlo_simulation
    from .analysis.statistics import analyze_simulation_results
    from .utils.legacy_compatibility import create_parameters_from_legacy

    # Load parameters
    parameters = open_parameters(parameters_filename)

    # Create axes
    x_axis = parameters['x array']
    time_axis = parameters['time series']

    # Create simulation parameters
    sim_params = create_parameters_from_legacy(
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
        retain_profile_data=parameters.get('retain profile data', 0) != 0
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
    "create_parameters_from_legacy",
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
