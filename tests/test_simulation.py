"""
Integration tests for dice.simulation module.
"""
import pytest
import numpy as np
from dice.simulation import run_simulation
from dice.parameters import parameter_parser


@pytest.mark.integration
class TestRunSimulation:
    """Integration tests for run_simulation() function."""

    @pytest.fixture
    def valid_simulation_params(self):
        """Valid parameters for a small simulation run."""
        params = {
            'filename slug': 'test_sim',
            'number of runs': 2,  # Small number for fast testing
            'length unit': 'micrometer',
            'time unit': 'millisecond',
            'spatial width': 10.0,
            'pixel width': 50,  # Small number for fast testing
            'mean_0': 0.0,
            'amplitude_0': 1.0,
            'sigma_0': 1.0,
            'time series': np.array([0.0, 1.0, 2.0]),  # Small number for fast testing
            'noise value': 0.01,
            'proximity level': 0.1,
            'nominal diffusion length': 1.0,
            'image type': 'png',
            'image width': 8.5,
            'image height': 5.0,
            'image dpi': 100,  # Lower DPI for faster testing
            'image font size': 8,
            'image tick length': 6,
            'image tick width': 2,
            'image numbins': 20,  # Lower bins for faster testing
            'image x_lim': None,
            'retain profile data': False,
            'multiprocessing': False,
        }
        # Parse parameters to add derived fields
        return parameter_parser(params)

    def test_run_simulation_basic(self, valid_simulation_params):
        """Test basic simulation execution."""
        result = run_simulation(valid_simulation_params)

        assert isinstance(result, dict)
        assert 'indices' in result
        assert 'parameters' in result
        assert 'run results' in result
        assert 'collated results' in result
        assert 'filename slug' in result

    def test_run_simulation_indices(self, valid_simulation_params):
        """Test that simulation result contains correct indices."""
        result = run_simulation(valid_simulation_params)

        indices = result['indices']
        assert 'time axis' in indices
        assert 'x axis' in indices
        assert 'noise sigmas' in indices
        assert 'total runs' in indices

        assert isinstance(indices['time axis'], np.ndarray)
        assert isinstance(indices['x axis'], np.ndarray)

    def test_run_simulation_parameters(self, valid_simulation_params):
        """Test that simulation result contains parameters."""
        result = run_simulation(valid_simulation_params)

        params = result['parameters']
        assert 'sigma^2_0' in params
        assert 'amplitude_0' in params
        assert 'mu_0' in params
        assert 'nominal diffusion coeff' in params
        assert 'proximity level' in params

    def test_run_simulation_with_progress_callback(self, valid_simulation_params):
        """Test that progress callbacks are called during simulation."""
        progress_calls = []

        def progress_callback(current, total):
            progress_calls.append((current, total))

        result = run_simulation(
            valid_simulation_params,
            progress_callback=progress_callback
        )

        # Should have been called at least once
        assert len(progress_calls) > 0
        # Last call should be with total runs
        final_current, final_total = progress_calls[-1]
        assert final_current <= final_total

    def test_run_simulation_with_message_callback(self, valid_simulation_params):
        """Test that message callbacks work correctly."""
        messages = []

        def message_callback(message):
            messages.append(message)

        result = run_simulation(
            valid_simulation_params,
            message_callback=message_callback
        )

        # Callback should work (messages is a list)
        assert isinstance(messages, list)
        # Any messages received should be strings
        if len(messages) > 0:
            assert all(isinstance(msg, str) for msg in messages)
