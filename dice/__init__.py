from dice.simulation import run_simulation, run_simulation_cli
from dice.parameters import open_parameters, parameter_parser
from dice.reporting import plot_accuracy_histogram, summarize_results, export_results
from dice.analysis import estimates_precision

__version__ = "1.3.0"
__all__ = [
    "run_simulation",
    "run_simulation_cli",
    "open_parameters",
    "parameter_parser",
    "plot_accuracy_histogram",
    "summarize_results",
    "export_results",
    "estimates_precision",
]
