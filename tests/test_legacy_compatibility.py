"""
Tests for dice.utils.legacy_compatibility module.
"""

import pytest
import warnings
import numpy as np
from dice.utils.legacy_compatibility import (
    LegacyProfileParameters,
    LegacyPhysicsParameters,
    LegacySimulationParameters,
    create_parameters_from_legacy,
    MockSimulationParameters,
)


class TestLegacyParameterClasses:
    """Test legacy parameter classes."""

    def test_legacy_profile_parameters_deprecation_warning(self):
        """Test that LegacyProfileParameters raises deprecation warning."""
        with pytest.warns(DeprecationWarning, match="LegacyProfileParameters is deprecated"):
            LegacyProfileParameters(sigma2_0=1.5, amplitude_0=2.0, mu_0=0.5)

    def test_legacy_profile_parameters(self):
        """Test LegacyProfileParameters creation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            profile = LegacyProfileParameters(
                sigma2_0=1.5,
                amplitude_0=2.0,
                mu_0=0.5
            )

        assert profile.sigma2_0 == 1.5
        assert profile.amplitude_0 == 2.0
        assert profile.mu_0 == 0.5

    def test_legacy_physics_parameters_deprecation_warning(self):
        """Test that LegacyPhysicsParameters raises deprecation warning."""
        with pytest.warns(DeprecationWarning, match="LegacyPhysicsParameters is deprecated"):
            LegacyPhysicsParameters(
                diffusion_coefficient=0.5,
                lifetime=10.0,
                diffusion_length=np.sqrt(5.0)
            )

    def test_legacy_physics_parameters(self):
        """Test LegacyPhysicsParameters creation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            physics = LegacyPhysicsParameters(
                diffusion_coefficient=0.5,
                lifetime=10.0,
                diffusion_length=np.sqrt(5.0)
            )

        assert physics.diffusion_coefficient == 0.5
        assert physics.lifetime == 10.0
        assert physics.diffusion_length == pytest.approx(np.sqrt(5.0))

    def test_legacy_simulation_parameters_deprecation_warning(self):
        """Test that LegacySimulationParameters raises deprecation warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            profile = LegacyProfileParameters(1.0, 1.0, 0.0)
            physics = LegacyPhysicsParameters(0.5, 10.0, np.sqrt(5.0))

        with pytest.warns(DeprecationWarning, match="LegacySimulationParameters is deprecated"):
            LegacySimulationParameters(profile, physics)

    def test_legacy_simulation_parameters(self):
        """Test LegacySimulationParameters creation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            profile = LegacyProfileParameters(1.0, 1.0, 0.0)
            physics = LegacyPhysicsParameters(0.5, 10.0, np.sqrt(5.0))
            params = LegacySimulationParameters(profile, physics)

        assert params.profile is profile
        assert params.physics is physics


class TestCreateParametersFromLegacy:
    """Test legacy parameter creation function."""

    def test_basic_creation(self):
        """Test basic parameter creation from legacy format."""
        params_dict = {
            'sigma^2_0': 1.0,
            'amplitude_0': 1.5,
            'mu_0': 0.0
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            params = create_parameters_from_legacy(
                parameters_dict=params_dict,
                diffusion_coefficient=0.5,
                lifetime=10.0,
                diffusion_length=np.sqrt(5.0)
            )

        assert params.profile.sigma2_0 == 1.0
        assert params.profile.amplitude_0 == 1.5
        assert params.profile.mu_0 == 0.0
        assert params.physics.diffusion_coefficient == 0.5
        assert params.physics.lifetime == 10.0
        assert params.physics.diffusion_length == pytest.approx(np.sqrt(5.0))

    def test_missing_keys(self):
        """Test error handling for missing keys."""
        params_dict = {
            'amplitude_0': 1.0,
            # Missing sigma^2_0 and mu_0
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(KeyError):
                create_parameters_from_legacy(
                    parameters_dict=params_dict,
                    diffusion_coefficient=0.5,
                    lifetime=10.0,
                    diffusion_length=np.sqrt(5.0)
                )


class TestMockSimulationParameters:
    """Test mock simulation parameters for testing."""

    def test_mock_deprecation_warning(self):
        """Test that MockSimulationParameters raises deprecation warning."""
        with pytest.warns(DeprecationWarning, match="MockSimulationParameters is deprecated"):
            MockSimulationParameters()

    def test_mock_creation(self):
        """Test mock parameter creation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            params = MockSimulationParameters()

        assert hasattr(params, 'profile')
        assert hasattr(params, 'physics')
        assert hasattr(params, 'num_runs')
        assert hasattr(params, 'noise_values')

        assert params.profile.sigma2_0 == 1.0
        assert params.physics.diffusion_coefficient == 1.0
        assert params.physics.lifetime == 10.0
        assert params.num_runs == 100
        assert params.noise_values == [0.05]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])