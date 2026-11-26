"""
Integration tests for the parameter and profile dataclasses in dice.models.

The inputs come from DICE itself: parameter dictionaries parsed from the real
files in data/examples, and profiles from make_diffusion_decay, add_noise, and
gauss_fitting. The tests check that the dictionaries those functions produce
carry the field names the dataclasses read.
"""

from pathlib import Path

import numpy as np
import pytest

from dice.core.fitting import gauss_fitting
from dice.core.noise import add_noise
from dice.core.profiles import make_diffusion_decay
from dice.io.parameters import open_parameters
from dice.models.parameters import (
    GaussianParameters,
    NoiseParameters,
    OutputParameters,
    SimulationParameters,
    SpatialParameters,
    TemporalParameters,
)
from dice.models.profiles import GaussianProfile, ProfileData, TimeSeriesProfile

EXAMPLES_DIR = Path(__file__).parent.parent / 'data' / 'examples'
EXAMPLE_FILES = ['parameters.txt', 'publication.txt', 'quick_test.txt', 'high_precision.txt']
GENERATOR_PARAMETER_KEYS = {'amplitude_t', 'sigma^2_t', 'fwhm_t', 'mu_t', 'integrated intensity_t'}


@pytest.fixture(scope='module')
def quick_test_parameters() -> dict:
    return open_parameters(EXAMPLES_DIR / 'quick_test.txt')


@pytest.fixture(scope='module')
def quick_test_generated(quick_test_parameters) -> dict:
    """Noise-free profiles that make_diffusion_decay generates for quick_test.txt."""
    params = quick_test_parameters
    return make_diffusion_decay({
        'x axis': params['x array'],
        'time axis': params['time series'],
        'sigma^2_0': params['sigma^2_0'],
        'amplitude_0': params['amplitude_0'],
        'mu_0': params['mu_0'],
        'nominal diffusion coefficient': params['nominal diffusion coefficient'],
        'nominal lifetime': params['nominal lifetime (tau)'],
    })


def nominal_profiles(params: dict, generated: dict) -> list[GaussianProfile]:
    parameters_t = generated['parameters_t']
    return [
        GaussianProfile(
            time=t,
            x_axis=params['x array'],
            y_values=np.asarray(y_values),
            amplitude=parameters_t['amplitude_t'][i],
            sigma2=parameters_t['sigma^2_t'][i],
            mu=parameters_t['mu_t'][i],
        )
        for i, (t, y_values) in enumerate(zip(params['time series'], generated['y_values_t']))
    ]


def simulation_fields(params: dict) -> dict:
    """The component dataclasses of SimulationParameters, built from a parsed dictionary."""
    return {
        'num_runs': params['number of runs'],
        'gaussian': GaussianParameters(
            amplitude_0=params['amplitude_0'], sigma2_0=params['sigma^2_0'], mu_0=params['mu_0']),
        'noise': NoiseParameters(mode='single', value=params['noise series'][0]),
        'spatial': SpatialParameters(width=params['spatial width'], pixels=params['pixel width']),
        'temporal': TemporalParameters(mode='series', series=list(params['time series'])),
    }


class TestDataclassesFromParsedFiles:
    """The dataclasses take the values of the dictionaries the parser produces."""

    @pytest.mark.parametrize('filename', EXAMPLE_FILES)
    def test_axes_match_parsed_axes(self, filename):
        params = open_parameters(EXAMPLES_DIR / filename)
        fields = simulation_fields(params)
        np.testing.assert_allclose(fields['temporal'].make_axis(), params['time series'])
        np.testing.assert_allclose(fields['spatial'].make_axis(), params['x array'])

    @pytest.mark.parametrize('filename', EXAMPLE_FILES)
    def test_pixel_size_is_the_spacing_of_the_parsed_axis(self, filename):
        params = open_parameters(EXAMPLES_DIR / filename)
        spatial = simulation_fields(params)['spatial']
        assert spatial.pixel_size == pytest.approx(np.diff(params['x array'])[0], rel=1e-12)

    def test_gaussian_field_names_after_rename(self, quick_test_parameters):
        gaussian = simulation_fields(quick_test_parameters)['gaussian']
        assert set(vars(gaussian)) == {'amplitude_0', 'sigma2_0', 'mu_0'}
        with pytest.raises(TypeError):
            GaussianParameters(amplitude=1.0, sigma2=1.0, mu=0.0)


class TestParameterValidation:
    """Malformed values are rejected with a message naming the problem."""

    @pytest.mark.parametrize('build, message', [
        (lambda: GaussianParameters(amplitude_0=-1.0), "Amplitude must be non-negative"),
        (lambda: GaussianParameters(sigma2_0=0.0), "Variance"),
        (lambda: NoiseParameters(mode='single'), "Value required"),
        (lambda: NoiseParameters(mode='range'), "Range required"),
        (lambda: NoiseParameters(mode='estimate'), "Data file required"),
        (lambda: NoiseParameters(value=-0.02), "non-negative"),
        (lambda: NoiseParameters(mode='range', range=(0.02,)), "tuple of"),
        (lambda: NoiseParameters(mode='range', range=(0.05, 0.02)), "min must be less than max"),
        (lambda: NoiseParameters(mode='range', range=(-0.02, 0.05)), "non-negative"),
        (lambda: SpatialParameters(width=0.0, pixels=50), "Spatial width must be positive"),
        (lambda: SpatialParameters(width=5.0, pixels=0), "pixels must be positive"),
        (lambda: TemporalParameters(mode='range', start=1.0, end=0.0, frames=5), "Start time"),
        (lambda: TemporalParameters(mode='range', start=0.0, end=1.0, frames=0), "frames must be positive"),
        (lambda: TemporalParameters(mode='range', start=0.0), "required for range mode"),
        (lambda: TemporalParameters(mode='series', series=[]), "cannot be empty"),
        (lambda: OutputParameters(image_type='bmp'), "Unknown image type 'bmp'"),
        (lambda: OutputParameters(image_width=0.0), "dimensions must be positive"),
        (lambda: OutputParameters(image_dpi=0), "DPI must be positive"),
        (lambda: OutputParameters(num_bins=0), "bins must be positive"),
    ])
    def test_invalid_component(self, build, message):
        with pytest.raises(ValueError, match=message):
            build()

    @pytest.mark.parametrize('overrides, message', [
        ({'num_runs': 0}, "Number of runs must be positive"),
        ({'diffusion_coefficient': -0.01}, "Diffusion coefficient must be non-negative"),
        ({'lifetime': -1.0}, "Lifetime must be non-negative"),
    ])
    def test_invalid_simulation_parameters(self, quick_test_parameters, overrides, message):
        fields = simulation_fields(quick_test_parameters) | {
            'diffusion_coefficient': quick_test_parameters['nominal diffusion coefficient'],
            'lifetime': quick_test_parameters['nominal lifetime (tau)'],
        }
        with pytest.raises(ValueError, match=message):
            SimulationParameters(**(fields | overrides))

    def test_zero_lifetime_has_infinite_diffusion_length(self, quick_test_parameters):
        no_decay = SimulationParameters(
            **simulation_fields(quick_test_parameters), diffusion_coefficient=0.01, lifetime=0.0)
        assert no_decay.diffusion_length == np.inf


class TestGaussianProfileFromGenerator:
    """GaussianProfile agrees with the profiles make_diffusion_decay generates."""

    def test_generator_parameter_schema(self, quick_test_generated, quick_test_parameters):
        assert set(quick_test_generated['parameters_t']) == GENERATOR_PARAMETER_KEYS
        frame_count = len(quick_test_parameters['time series'])
        assert all(len(values) == frame_count for values in quick_test_generated['parameters_t'].values())

    def test_derived_quantities_match_generator(self, quick_test_parameters, quick_test_generated):
        parameters_t = quick_test_generated['parameters_t']
        for i, profile in enumerate(nominal_profiles(quick_test_parameters, quick_test_generated)):
            assert profile.integrated_intensity == pytest.approx(parameters_t['integrated intensity_t'][i], rel=1e-12)
            assert profile.fwhm == pytest.approx(parameters_t['fwhm_t'][i], rel=1e-12)
            assert profile.sigma == pytest.approx(np.sqrt(parameters_t['sigma^2_t'][i]), rel=1e-12)
            assert profile.noise_free and profile.cnr is None

    def test_add_noise_keeps_nominal_parameters(self, quick_test_parameters, quick_test_generated):
        profile = nominal_profiles(quick_test_parameters, quick_test_generated)[0]
        noisy = profile.add_noise(quick_test_parameters['noise series'][0])
        assert not noisy.noise_free
        assert (noisy.amplitude, noisy.sigma2, noisy.mu) == (profile.amplitude, profile.sigma2, profile.mu)
        assert noisy.integrated_intensity == profile.integrated_intensity
        assert noisy.y_values.shape == profile.y_values.shape
        assert not np.array_equal(noisy.y_values, profile.y_values)

    def test_shape_mismatch_rejected(self, quick_test_parameters):
        x_axis = quick_test_parameters['x array']
        with pytest.raises(ValueError, match="same shape"):
            GaussianProfile(time=0.0, x_axis=x_axis, y_values=np.zeros(x_axis.size - 1),
                            amplitude=1.0, sigma2=1.0, mu=0.0)


class TestProfileDataFromSimulation:
    """ProfileData assembled from one simulated run: generator, noise, and Gaussian fits."""

    @pytest.fixture(scope='class')
    def run_profiles(self, quick_test_parameters, quick_test_generated) -> ProfileData:
        params = quick_test_parameters
        nominal = nominal_profiles(params, quick_test_generated)
        noise_sigma = params['noise series'][0]
        noisy_y = add_noise(np.asarray(quick_test_generated['y_values_t']), noise_sigma, seed=7)['y_values_t']
        fits = gauss_fitting(params['x array'], noisy_y)
        assert set(fits) == {'sigma^2_t estimates', 'sigma^2_t standard errors', 'fit_failures'}
        noisy = [
            GaussianProfile(p.time, p.x_axis, y, p.amplitude, p.sigma2, p.mu, noise_free=False)
            for p, y in zip(nominal, noisy_y)
        ]
        fitted = [
            GaussianProfile(p.time, p.x_axis, p.y_values, p.amplitude, sigma2, p.mu)
            for p, sigma2 in zip(nominal, fits['sigma^2_t estimates'])
        ]
        return ProfileData(nominal, noisy, fitted, noise_sigma)

    def test_time_and_variance_series(self, run_profiles, quick_test_parameters, quick_test_generated):
        np.testing.assert_allclose(run_profiles.time_axis, quick_test_parameters['time series'])
        np.testing.assert_allclose(run_profiles.nominal_sigma2_t, quick_test_generated['parameters_t']['sigma^2_t'])
        fitted = run_profiles.fitted_sigma2_t
        assert run_profiles.num_timepoints == len(quick_test_parameters['time series'])
        np.testing.assert_allclose(run_profiles.msd_t, fitted - fitted[0])

    def test_profile_lookup(self, run_profiles, quick_test_parameters):
        last_time = quick_test_parameters['time series'][-1]
        assert run_profiles.get_profile_at_time(last_time, 'fitted') is run_profiles.fitted_profiles[-1]
        assert run_profiles.get_profile_at_time(last_time, 'noisy') is run_profiles.noisy_profiles[-1]
        assert run_profiles.get_profile_at_time(last_time + 10.0) is None
        with pytest.raises(ValueError, match="Invalid profile_type"):
            run_profiles.get_profile_at_time(last_time, 'smoothed')

    def test_mismatched_lists_rejected(self, run_profiles):
        with pytest.raises(ValueError, match="same length"):
            ProfileData(run_profiles.nominal_profiles, run_profiles.noisy_profiles[:-1],
                        run_profiles.fitted_profiles, run_profiles.noise_sigma)


class TestTimeSeriesProfileFromGenerator:
    """TimeSeriesProfile reads make_diffusion_decay's output, including its 'sigma^2_t' key."""

    def test_profile_list_carries_generator_parameters(self, quick_test_parameters, quick_test_generated):
        params, parameters_t = quick_test_parameters, quick_test_generated['parameters_t']
        series = TimeSeriesProfile(
            x_axis=params['x array'],
            time_axis=params['time series'],
            y_values=np.asarray(quick_test_generated['y_values_t']),
            parameters=parameters_t,
        )
        assert (series.num_timepoints, series.num_spatial_points) == (
            len(params['time series']), len(params['x array']))
        np.testing.assert_array_equal(series.get_profile(1), quick_test_generated['y_values_t'][1])

        profiles = series.to_profile_list()
        np.testing.assert_allclose([p.sigma2 for p in profiles], parameters_t['sigma^2_t'], rtol=1e-12)
        np.testing.assert_allclose([p.amplitude for p in profiles], parameters_t['amplitude_t'], rtol=1e-12)
        np.testing.assert_allclose([p.integrated_intensity for p in profiles],
                                   parameters_t['integrated intensity_t'], rtol=1e-12)

    def test_shape_mismatch_rejected(self, quick_test_parameters, quick_test_generated):
        with pytest.raises(ValueError, match="doesn't match"):
            TimeSeriesProfile(
                x_axis=quick_test_parameters['x array'],
                time_axis=quick_test_parameters['time series'][:-1],
                y_values=np.asarray(quick_test_generated['y_values_t']),
            )


class TestOutputParameterConventions:
    """OutputParameters accepts the image types the run path accepts."""

    @pytest.mark.parametrize('image_type', ['png', 'eps', 'WEBP', 'tiff'])
    def test_output_parameters_accept_the_run_time_image_types(self, image_type):
        assert OutputParameters(image_type=image_type).image_type == image_type
