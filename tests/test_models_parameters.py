"""
Tests for the simulation engine's parameter classes in dice.models.parameters.
"""

import re
import warnings
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from dice.models.parameters import (
    PhysicsParameters,
    ProfileParameters,
    SimulationParameters,
    simulation_parameters_from_dict,
)


class TestParameterClasses:

    def test_profile_parameters_hold_the_initial_profile(self):
        profile = ProfileParameters(sigma2_0=1.5, amplitude_0=2.0, mu_0=0.5)

        assert (profile.sigma2_0, profile.amplitude_0, profile.mu_0) == (1.5, 2.0, 0.5)

    def test_physics_parameters_hold_the_nominal_transport_values(self):
        physics = PhysicsParameters(diffusion_coefficient=0.5, lifetime=10.0, diffusion_length=np.sqrt(5.0))

        assert physics.diffusion_coefficient == 0.5
        assert physics.lifetime == 10.0
        assert physics.diffusion_length == pytest.approx(np.sqrt(5.0))

    def test_simulation_parameters_hold_the_profile_and_the_physics(self):
        profile = ProfileParameters(1.0, 1.0, 0.0)
        physics = PhysicsParameters(0.5, 10.0, np.sqrt(5.0))

        params = SimulationParameters(profile, physics)

        assert params.profile is profile
        assert params.physics is physics

    def test_a_field_can_be_set_after_creation(self):
        params = SimulationParameters(ProfileParameters(1.0, 1.0, 0.0), PhysicsParameters(0.5, 10.0, np.sqrt(5.0)))

        params.profile.amplitude_0 = 3.0

        assert params.profile.amplitude_0 == 3.0

    def test_fields_are_those_the_engine_reads(self):
        """The engine reads these names. 1.2.0 had a dataclass named SimulationParameters with other fields."""
        assert [field.name for field in fields(SimulationParameters)] == ['profile', 'physics']
        assert [field.name for field in fields(ProfileParameters)] == ['sigma2_0', 'amplitude_0', 'mu_0']
        assert [field.name for field in fields(PhysicsParameters)] == [
            'diffusion_coefficient', 'lifetime', 'diffusion_length']


class TestSimulationParametersFromDict:

    ARGUMENTS = dict(diffusion_coefficient=0.5, lifetime=10.0, diffusion_length=np.sqrt(5.0))

    def test_values_come_from_the_dictionary_and_the_arguments(self):
        params = simulation_parameters_from_dict(
            parameters_dict={'sigma^2_0': 1.0, 'amplitude_0': 1.5, 'mu_0': 0.25}, **self.ARGUMENTS)

        assert params == SimulationParameters(
            profile=ProfileParameters(sigma2_0=1.0, amplitude_0=1.5, mu_0=0.25),
            physics=PhysicsParameters(diffusion_coefficient=0.5, lifetime=10.0, diffusion_length=np.sqrt(5.0)),
        )

    def test_mean_0_is_read_when_mu_0_is_absent(self):
        params = simulation_parameters_from_dict(
            parameters_dict={'sigma^2_0': 1.0, 'amplitude_0': 1.5, 'mean_0': 0.75}, **self.ARGUMENTS)

        assert params.profile.mu_0 == 0.75

    def test_center_is_zero_when_no_key_gives_it(self):
        params = simulation_parameters_from_dict(
            parameters_dict={'sigma^2_0': 1.0, 'amplitude_0': 1.5}, **self.ARGUMENTS)

        assert params.profile.mu_0 == 0.0

    @pytest.mark.parametrize('missing', ['sigma^2_0', 'amplitude_0'])
    def test_missing_width_or_amplitude_is_a_key_error(self, missing):
        parameters = {'sigma^2_0': 1.0, 'amplitude_0': 1.5, 'mu_0': 0.0}
        del parameters[missing]

        with pytest.raises(KeyError, match=re.escape(missing)):
            simulation_parameters_from_dict(parameters_dict=parameters, **self.ARGUMENTS)


class TestBuilderOutputIsTheEngineInput:
    """Every run path builds these objects, so building them raises no warning."""

    def test_builder_raises_no_warning_and_its_output_runs_in_the_engine(self):
        from dice.analysis.simulation import run_monte_carlo_simulation
        from dice.io.parameters import open_parameters

        params = open_parameters(Path(__file__).parent.parent / 'data' / 'examples' / 'quick_test.txt')
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sim_params = simulation_parameters_from_dict(
                params, params['nominal diffusion coefficient'],
                params['nominal lifetime (tau)'], params['nominal diffusion length'])

        result = run_monte_carlo_simulation(
            sim_params, params['x array'], params['time series'], params['noise series'],
            num_runs=2, multiprocessing=False, seed=1)

        assert result.parameters is sim_params
        assert result.num_runs == 2
        assert result.parameters.profile.sigma2_0 == params['sigma^2_0']
        assert result.parameters.physics.diffusion_coefficient == params['nominal diffusion coefficient']
