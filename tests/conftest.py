"""
Test fixtures for DICE simulation testing.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile


@pytest.fixture
def sample_parameters_dict():
    """Basic parameter dictionary for testing."""
    return {
        'filename slug': 'test_simulation',
        'number of runs': 10,
        'length unit': 'um',
        'time unit': 'ms',
        'spatial width': 10.0,
        'pixel width': 100,
        'mean_0': 0.0,
        'amplitude_0': 1.0,
        'sigma_0': 1.0,
        'time series': np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        'noise value': 0.01,
        'proximity level': 0.1,
        'nominal diffusion length': 1.0,
        'image type': 'png',
        'image width': 8.5,
        'image height': 5.0,
        'image dpi': 300,
        'image font size': 8,
        'image tick length': 6,
        'image tick width': 2,
        'image numbins': 35,
        'image x_lim': None,
        'retain profile data': False,
        'multiprocessing': False,
    }


@pytest.fixture
def sample_x_array():
    """X-axis data for testing."""
    return np.linspace(-5.0, 5.0, 100)


@pytest.fixture
def sample_time_series():
    """Time series data for testing."""
    return np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def sample_gaussian_data():
    """Gaussian profile data for testing."""
    x = np.linspace(-5.0, 5.0, 100)
    mu = 0.0
    sig2 = 1.0
    amp = 1.0
    y = amp * np.exp(-1 * np.power(x - mu, 2) / (2 * sig2))
    return x, y


@pytest.fixture
def sample_noisy_profile():
    """Noisy Gaussian profile for testing."""
    x = np.linspace(-5.0, 5.0, 100)
    mu = 0.0
    sig2 = 1.0
    amp = 1.0
    y_clean = amp * np.exp(-1 * np.power(x - mu, 2) / (2 * sig2))
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.01, size=y_clean.shape)
    y_noisy = y_clean + noise
    return x, y_noisy


@pytest.fixture
def mock_csv_file(tmp_path):
    """Create a temporary CSV file for testing."""
    csv_path = tmp_path / "test_data.csv"
    data = pd.DataFrame({
        'x': np.linspace(-5, 5, 10),
        'y': np.linspace(0, 1, 10)
    })
    data.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def mock_parameters_file(tmp_path):
    """Create a temporary parameter file for testing."""
    param_path = tmp_path / "test_parameters.txt"
    params = {
        'filename slug': 'test',
        'number of runs': 5,
        'spatial width': 10.0,
        'pixel width': 100,
        'mean_0': 0.0,
        'amplitude_0': 1.0,
        'sigma_0': 1.0,
        'time range': (0.0, 5.0, 6),
        'noise value': 0.01,
        'proximity level': 0.1,
        'nominal diffusion length': 1.0,
    }
    param_path.write_text(str(params))
    return str(param_path)


# Modern dataclass-based fixtures (replacements for MockSimulationParameters)

@pytest.fixture
def default_gaussian_params():
    """Default GaussianParameters for testing."""
    from dice.models.parameters import GaussianParameters
    return GaussianParameters(amplitude=1.0, sigma2=1.0, mu=0.0)


@pytest.fixture
def default_noise_params():
    """Default NoiseParameters for testing."""
    from dice.models.parameters import NoiseParameters
    return NoiseParameters(mode='single', value=0.05)


@pytest.fixture
def default_spatial_params():
    """Default SpatialParameters for testing."""
    from dice.models.parameters import SpatialParameters
    return SpatialParameters(width=10.0, pixels=100, center=0.0)


@pytest.fixture
def default_temporal_params():
    """Default TemporalParameters for testing."""
    from dice.models.parameters import TemporalParameters
    return TemporalParameters(mode='range', start=0.0, end=5.0, frames=6)


@pytest.fixture
def default_output_params():
    """Default OutputParameters for testing."""
    from dice.models.parameters import OutputParameters
    return OutputParameters()


@pytest.fixture
def default_simulation_params(
    default_gaussian_params,
    default_noise_params,
    default_spatial_params,
    default_temporal_params,
    default_output_params
):
    """
    Default SimulationParameters for testing.

    This fixture replaces MockSimulationParameters with modern dataclasses.
    Values match the original mock defaults: D=1.0, tau=10.0, num_runs=100.
    """
    from dice.models.parameters import SimulationParameters
    return SimulationParameters(
        num_runs=100,
        diffusion_coefficient=1.0,
        lifetime=10.0,
        gaussian=default_gaussian_params,
        noise=default_noise_params,
        spatial=default_spatial_params,
        temporal=default_temporal_params,
        output=default_output_params,
        multiprocessing=False  # Disable for tests
    )


@pytest.fixture
def legacy_simulation_params():
    """
    Parameters with legacy .profile/.physics structure for compatibility testing.

    Use this fixture when testing code that expects the legacy parameter format.
    """
    from dice.utils.legacy_compatibility import MockSimulationParameters
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return MockSimulationParameters()
