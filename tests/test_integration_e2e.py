"""
End-to-end integration tests for DICE simulation pipeline.

These tests verify that the full simulation pipeline works correctly
after the modular refactoring.
"""

import pytest
import tempfile
from pathlib import Path
import numpy as np


class TestEndToEndSimulation:
    """Full pipeline integration tests."""

    def test_parameter_loading(self, tmp_path):
        """Verify parameter file can be loaded and parsed."""
        from dice.io.parameters import open_parameters

        # Create test parameter file
        params_file = tmp_path / "params.txt"
        params_file.write_text(str({
            'filename slug': 'e2e_test',
            'number of runs': 5,
            'spatial width': 5.0,
            'pixel width': 50,
            'mean_0': 0.0,
            'amplitude_0': 1.0,
            'sigma_0': 0.5,
            'time range': [0, 1, 5],
            'noise value': 0.02,
            'proximity level': 0.1,
            'nominal diffusion length': 0.1,
        }))

        # Load parameters
        params = open_parameters(str(params_file))

        assert params['filename slug'] == 'e2e_test'
        assert params['number of runs'] == 5
        assert 'time series' in params  # Derived from time range
        assert 'x array' in params  # Derived from spatial/pixel width

    def test_monte_carlo_simulation_runs(self):
        """Verify Monte Carlo simulation executes without errors."""
        from dice.analysis.simulation import run_monte_carlo_simulation
        from dice.utils.legacy_compatibility import create_parameters_from_legacy
        from dice.utils.axes import make_x_axis

        # Create minimal parameters
        x_axis = np.linspace(-5, 5, 50)
        time_axis = np.array([0.0, 0.5, 1.0])
        noise_values = [0.02, 0.02, 0.02]

        sim_params = create_parameters_from_legacy(
            parameters_dict={
                'sigma^2_0': 0.25,
                'amplitude_0': 1.0,
                'mu_0': 0.0,
            },
            diffusion_coefficient=0.01,
            lifetime=1.0,
            diffusion_length=0.1
        )

        # Run simulation with minimal iterations
        result = run_monte_carlo_simulation(
            parameters=sim_params,
            x_axis=x_axis,
            time_axis=time_axis,
            noise_values=noise_values,
            num_runs=3,
            multiprocessing=False,
            retain_profile_data=False
        )

        assert result is not None
        assert len(result) == 3  # 3 runs

    def test_statistics_analysis(self):
        """Verify statistical analysis functions work correctly."""
        from dice.analysis.statistics import (
            calculate_proximity_percentage,
            calculate_mean_and_std
        )

        # Test proximity calculation
        estimates = np.array([0.95, 1.0, 1.05, 1.1, 0.9])
        nominal = 1.0
        proximity = 0.1

        pct = calculate_proximity_percentage(estimates, nominal, proximity)
        assert 0 <= pct <= 100

        # Test mean/std calculation
        mean, std = calculate_mean_and_std(estimates)
        assert np.isclose(mean, 1.0, atol=0.1)
        assert std > 0


class TestGUIInterface:
    """Tests for GUI interface parameter building."""

    def test_gui_interface_builds_parameters(self):
        """Verify GUI interface correctly builds parameter dictionaries."""
        from dice_gui.dice_interface import DiceInterface

        interface = DiceInterface()
        gui_params = {
            'number_of_runs': 10,
            'filename_slug': 'gui_test',
            'amplitude_0': 1.0,
            'mean_0': 0.0,
            'profile_width_type': 'sigma',
            'profile_width_value': 0.5,
            'diffusion_type': 'length',
            'diffusion_length': 0.1,
            'noise_type': 'fixed',
            'noise_value': 0.02,
            'spatial_width': 5.0,
            'pixel_width': 50,
            'time_type': 'range',
            'time_start': 0,
            'time_stop': 1,
            'time_steps': 5,
            'proximity_level': 0.1,
        }

        params_dict = interface.build_parameters_dict(gui_params)

        assert params_dict['number of runs'] == 10
        assert params_dict['sigma_0'] == 0.5
        assert params_dict['nominal diffusion length'] == 0.1
        assert params_dict['noise value'] == 0.02

    def test_gui_interface_handles_fwhm(self):
        """Verify GUI interface handles FWHM profile width."""
        from dice_gui.dice_interface import DiceInterface

        interface = DiceInterface()
        gui_params = {
            'number_of_runs': 5,
            'filename_slug': 'fwhm_test',
            'amplitude_0': 1.0,
            'mean_0': 0.0,
            'profile_width_type': 'fwhm',
            'profile_width_value': 1.0,
            'diffusion_type': 'coefficient',
            'diffusion_coefficient': 0.01,
            'lifetime': 1.0,
            'noise_type': 'fixed',
            'noise_value': 0.02,
            'spatial_width': 5.0,
            'pixel_width': 50,
            'time_type': 'series',
            'time_series': '0.0, 0.5, 1.0, 2.0',
            'proximity_level': 0.1,
        }

        params_dict = interface.build_parameters_dict(gui_params)

        assert 'FWHM_0' in params_dict
        assert params_dict['FWHM_0'] == 1.0
        assert 'sigma_0' not in params_dict
        assert params_dict['nominal diffusion coefficient'] == 0.01
        assert params_dict['time series'] == [0.0, 0.5, 1.0, 2.0]


class TestLegacyCompatibility:
    """Tests for backward compatibility with legacy format."""

    def test_legacy_parameter_conversion(self):
        """Verify legacy parameters can be converted to new format."""
        from dice.utils.legacy_compatibility import create_parameters_from_legacy

        legacy_params = {
            'sigma^2_0': 1.0,
            'amplitude_0': 1.0,
            'mu_0': 0.0,
        }

        sim_params = create_parameters_from_legacy(
            parameters_dict=legacy_params,
            diffusion_coefficient=0.01,
            lifetime=10.0,
            diffusion_length=0.316
        )

        assert sim_params.profile.sigma2_0 == 1.0
        assert sim_params.physics.diffusion_coefficient == 0.01
        assert sim_params.physics.lifetime == 10.0


class TestCoreModules:
    """Tests for core physics modules."""

    def test_gaussian_profile_generation(self):
        """Verify Gaussian profile generation."""
        from dice.core.profiles import gaussian

        x = np.linspace(-5, 5, 100)
        y = gaussian(x, mu=0.0, sig2=1.0, amp=1.0)

        assert len(y) == len(x)
        assert np.max(y) <= 1.0
        assert y[50] == pytest.approx(1.0, rel=0.01)  # Peak at center

    def test_noise_addition(self):
        """Verify noise addition preserves signal shape."""
        from dice.core.noise import add_noise

        x = np.linspace(-5, 5, 100)
        signal = np.exp(-x**2 / 2)

        noisy = add_noise(signal, noise_std=0.01, seed=42)

        assert noisy.shape == signal.shape
        assert not np.allclose(noisy, signal)  # Noise was added
        # Signal correlation should be high
        correlation = np.corrcoef(signal, noisy)[0, 1]
        assert correlation > 0.9


class TestVersionConsistency:
    """Test version information is accessible and consistent."""

    def test_version_accessible(self):
        """Verify version can be imported."""
        import dice
        assert hasattr(dice, '__version__')
        assert dice.__version__ == '1.3.0'

    def test_author_accessible(self):
        """Verify author info is accessible."""
        import dice
        assert hasattr(dice, '__author__')
        assert 'Thiebes' in dice.__author__
