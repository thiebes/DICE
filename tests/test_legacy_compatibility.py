"""
Tests for dice.utils.legacy_compatibility module.
"""

import pytest
import numpy as np
from dice.utils.legacy_compatibility import (
    LegacyProfileParameters,
    LegacyPhysicsParameters,
    LegacySimulationParameters,
    create_parameters_from_legacy,
    convert_legacy_result_to_dict,
    MockSimulationParameters,
)


class TestLegacyParameterClasses:
    """Test legacy parameter classes."""
    
    def test_legacy_profile_parameters(self):
        """Test LegacyProfileParameters creation."""
        profile = LegacyProfileParameters(
            sigma2_0=1.5,
            amplitude_0=2.0,
            mu_0=0.5
        )
        
        assert profile.sigma2_0 == 1.5
        assert profile.amplitude_0 == 2.0
        assert profile.mu_0 == 0.5
    
    def test_legacy_physics_parameters(self):
        """Test LegacyPhysicsParameters creation."""
        physics = LegacyPhysicsParameters(
            diffusion_coefficient=0.5,
            lifetime=10.0,
            diffusion_length=np.sqrt(5.0)
        )
        
        assert physics.diffusion_coefficient == 0.5
        assert physics.lifetime == 10.0
        assert physics.diffusion_length == pytest.approx(np.sqrt(5.0))
    
    def test_legacy_simulation_parameters(self):
        """Test LegacySimulationParameters creation."""
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
        
        with pytest.raises(KeyError):
            create_parameters_from_legacy(
                parameters_dict=params_dict,
                diffusion_coefficient=0.5,
                lifetime=10.0,
                diffusion_length=np.sqrt(5.0)
            )


class TestConvertLegacyResultToDict:
    """Test legacy result conversion function."""
    
    def test_basic_conversion(self):
        """Test basic result conversion."""
        # Create a mock RunResult
        class MockRunResult:
            def __init__(self):
                self.nominal_diffusion_coefficient = 0.5
                self.nominal_lifetime = 10.0
                self.nominal_diffusion_length = np.sqrt(5.0)
                self.noise_sigma = 0.05
                self.cnr_0_estimate = 20.0
                self.nominal_sigma2_0 = 1.0
                self.estimated_sigma2_0 = 1.05
                self.ols_slope = 1.0
                self.ols_slope_stderr = 0.1
                self.ols_intercept = 0.0
                self.ols_intercept_stderr = 0.01
                self.wls_slope = 1.02
                self.wls_slope_stderr = 0.08
                self.wls_intercept = 0.01
                self.wls_intercept_stderr = 0.008
        
        result = MockRunResult()
        run_id = 5
        
        legacy_dict = convert_legacy_result_to_dict(result, run_id)
        
        # Check structure
        assert f'run_{run_id}' in legacy_dict
        run_data = legacy_dict[f'run_{run_id}']
        
        assert run_data['run'] == run_id
        assert 'run parameters' in run_data
        assert 'cnr_0 estimate' in run_data
        assert 'nominal profiles' in run_data
        assert 'noisy profile fits' in run_data
        assert 'diffusion' in run_data
        
        # Check specific values
        assert run_data['run parameters']['nominal diffusion coefficient'] == 0.5
        assert run_data['cnr_0 estimate'] == 20.0
        assert run_data['diffusion']['unweighted fit']['MSD_t slope estimate'] == 1.0
        assert run_data['diffusion']['weighted fit']['MSD_t slope estimate'] == 1.02
    
    def test_conversion_with_none_values(self):
        """Test conversion when some values are None."""
        class MockRunResult:
            def __init__(self):
                self.nominal_diffusion_coefficient = 0.5
                self.nominal_lifetime = 10.0
                self.nominal_diffusion_length = np.sqrt(5.0)
                self.noise_sigma = 0.05
                self.cnr_0_estimate = 20.0
                self.nominal_sigma2_0 = 1.0
                self.estimated_sigma2_0 = None
                self.ols_slope = None
                self.ols_slope_stderr = None
                self.ols_intercept = None
                self.ols_intercept_stderr = None
                self.wls_slope = None
                self.wls_slope_stderr = None
                self.wls_intercept = None
                self.wls_intercept_stderr = None
        
        result = MockRunResult()
        run_id = 3
        
        legacy_dict = convert_legacy_result_to_dict(result, run_id)
        run_data = legacy_dict[f'run_{run_id}']
        
        # Check that None values become 'N/A'
        assert run_data['diffusion']['unweighted fit']['MSD_t slope estimate'] == 'N/A'
        assert run_data['diffusion']['weighted fit']['MSD_t slope estimate'] == 'N/A'
        assert run_data['noisy profile fits']['sigma^2_t estimates'] == []


class TestMockSimulationParameters:
    """Test mock simulation parameters for testing."""
    
    def test_mock_creation(self):
        """Test mock parameter creation."""
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