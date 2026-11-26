"""
Integration tests for dice.io.data_loader.

Inputs are the real data/examples/example_profile.csv, profiles that DICE
simulates (make_diffusion_decay with add_noise), a results CSV written by
export_collated_results, and malformed files derived from those.
"""

from pathlib import Path

import numpy as np
import pytest
from fft_cnr import fft_cnr

from dice.core.noise import add_noise
from dice.core.profiles import make_diffusion_decay
from dice.io.data_loader import (
    batch_load_profiles,
    load_experimental_data,
    load_profile_data,
    load_time_series_profiles,
    save_profile_data,
)
from dice.io.parameters import open_parameters

EXAMPLES_DIR = Path(__file__).parent.parent / 'data' / 'examples'
EXAMPLE_PROFILE = EXAMPLES_DIR / 'example_profile.csv'
LOADED_DATA_KEYS = {'filename', 'format', 'data', 'shape', 'ndim'}


@pytest.fixture(scope='module')
def quick_test_parameters() -> dict:
    return open_parameters(EXAMPLES_DIR / 'quick_test.txt')


@pytest.fixture(scope='module')
def simulated_profiles(quick_test_parameters) -> np.ndarray:
    """Noisy profiles (time x space) that DICE simulates for quick_test.txt."""
    params = quick_test_parameters
    generated = make_diffusion_decay({
        'x axis': params['x array'],
        'time axis': params['time series'],
        'sigma^2_0': params['sigma^2_0'],
        'amplitude_0': params['amplitude_0'],
        'mu_0': params['mu_0'],
        'nominal diffusion coefficient': params['nominal diffusion coefficient'],
        'nominal lifetime': params['nominal lifetime (tau)'],
    })
    noise_sigma = params['noise series'][0]
    return add_noise(np.asarray(generated['y_values_t']), noise_sigma, seed=11)['y_values_t']


class TestLoadProfileData:
    """Single profiles load as 1-D arrays, the format the noise estimate reads."""

    def test_example_profile(self):
        profile = load_profile_data(EXAMPLE_PROFILE)
        expected = [float(value) for value in EXAMPLE_PROFILE.read_text().split(',')]
        assert profile.shape == (41,)
        np.testing.assert_array_equal(profile, expected)

    def test_same_cnr_as_parameter_parser_path(self):
        # handle_noise_parameters reads the file with np.loadtxt(delimiter=',').
        parser_profile = np.loadtxt(EXAMPLE_PROFILE, delimiter=',')
        assert fft_cnr(load_profile_data(EXAMPLE_PROFILE)).cnr == fft_cnr(parser_profile).cnr

    def test_header_row_skipped(self, tmp_path):
        with_header = tmp_path / 'with_header.csv'
        with_header.write_text("intensity profile\n" + EXAMPLE_PROFILE.read_text())
        np.testing.assert_array_equal(load_profile_data(with_header, header=1),
                                      load_profile_data(EXAMPLE_PROFILE))

    @pytest.mark.parametrize('contents', [
        lambda text: text.replace('1530', 'peak', 1),
        lambda text: text + "\n1,2,3\n",
    ], ids=['non_numeric', 'ragged_rows'])
    def test_malformed_file_raises_ioerror(self, tmp_path, contents):
        bad_file = tmp_path / 'bad.csv'
        bad_file.write_text(contents(EXAMPLE_PROFILE.read_text()))
        with pytest.raises(IOError, match="Error loading profile data"):
            load_profile_data(bad_file)

    def test_missing_file_raises_ioerror(self, tmp_path):
        with pytest.raises(IOError, match="missing.csv"):
            load_profile_data(tmp_path / 'missing.csv')


class TestSaveProfileData:
    """Saved profiles load back unchanged; saving needs an existing directory."""

    def test_round_trip_simulated_profile(self, tmp_path, simulated_profiles):
        t0_profile = simulated_profiles[0]
        profile_file = tmp_path / 't0.csv'
        save_profile_data(t0_profile.reshape(1, -1), profile_file, header='simulated t = 0 profile')
        assert profile_file.read_text().startswith('# simulated t = 0 profile')
        np.testing.assert_array_equal(load_profile_data(profile_file), t0_profile)

    def test_missing_directory_raises_ioerror(self, tmp_path, simulated_profiles):
        with pytest.raises(IOError, match="Error saving profile data"):
            save_profile_data(simulated_profiles, tmp_path / 'absent' / 'profiles.csv')


class TestLoadTimeSeriesProfiles:
    """Profiles load as rows (time x space); the first column is time only when the caller says so."""

    def test_time_column_split_from_profiles(self, tmp_path, quick_test_parameters, simulated_profiles):
        time_axis = quick_test_parameters['time series']
        series_file = tmp_path / 'series.csv'
        np.savetxt(series_file, np.column_stack([time_axis, simulated_profiles]), delimiter=',')
        profiles, loaded_time = load_time_series_profiles(series_file, time_column=True)
        np.testing.assert_array_equal(loaded_time, time_axis)
        np.testing.assert_array_equal(profiles, simulated_profiles)

    def test_transposed_file(self, tmp_path, quick_test_parameters, simulated_profiles):
        time_axis = quick_test_parameters['time series']
        series_file = tmp_path / 'series_by_column.csv'
        np.savetxt(series_file, np.column_stack([time_axis, simulated_profiles]).T, delimiter=',')
        profiles, loaded_time = load_time_series_profiles(series_file, transpose=True, time_column=True)
        np.testing.assert_array_equal(loaded_time, time_axis)
        np.testing.assert_array_equal(profiles, simulated_profiles)

    def test_increasing_first_pixel_is_not_taken_as_time(self, tmp_path, simulated_profiles):
        # Frames ordered by their first pixel, so that column increases like a time axis would
        frames = simulated_profiles[np.argsort(simulated_profiles[:, 0])]
        assert np.all(np.diff(frames[:, 0]) > 0)
        series_file = tmp_path / 'series.csv'
        np.savetxt(series_file, frames, delimiter=',')
        profiles, loaded_time = load_time_series_profiles(series_file)
        assert loaded_time is None
        np.testing.assert_array_equal(profiles, frames)

    def test_one_row_file_is_one_profile(self):
        profiles, loaded_time = load_time_series_profiles(EXAMPLE_PROFILE)
        assert loaded_time is None
        assert profiles.shape == (1, 41)
        np.testing.assert_array_equal(profiles[0], load_profile_data(EXAMPLE_PROFILE))

    def test_one_row_file_with_time(self, tmp_path):
        series_file = tmp_path / 'one_frame.csv'
        series_file.write_text("0.0," + EXAMPLE_PROFILE.read_text())
        profiles, loaded_time = load_time_series_profiles(series_file, time_column=True)
        np.testing.assert_array_equal(loaded_time, [0.0])
        np.testing.assert_array_equal(profiles, [load_profile_data(EXAMPLE_PROFILE)])

    def test_one_profile_per_column_file(self, tmp_path):
        column_file = tmp_path / 'profile_by_column.csv'
        column_file.write_text("\n".join(EXAMPLE_PROFILE.read_text().split(',')))
        profiles, loaded_time = load_time_series_profiles(column_file, transpose=True)
        assert loaded_time is None
        np.testing.assert_array_equal(profiles, [load_profile_data(EXAMPLE_PROFILE)])

    def test_time_column_without_profile_columns_raises_ioerror(self, tmp_path):
        column_file = tmp_path / 'one_column.csv'
        column_file.write_text("\n".join(EXAMPLE_PROFILE.read_text().split(',')))
        with pytest.raises(IOError, match="time column needs at least one profile column"):
            load_time_series_profiles(column_file, time_column=True)

    def test_malformed_file_raises_ioerror(self, tmp_path):
        bad_file = tmp_path / 'bad_series.csv'
        bad_file.write_text("0,1,2\n1,x,3\n")
        with pytest.raises(IOError, match="Error loading time series"):
            load_time_series_profiles(bad_file)


class TestLoadExperimentalData:
    """Format detection, the returned dictionary's schema, and failure modes."""

    def test_npy_round_trip(self, tmp_path, simulated_profiles):
        npy_file = tmp_path / 'profiles.npy'
        np.save(npy_file, simulated_profiles)
        loaded = load_experimental_data(npy_file)
        assert set(loaded) == LOADED_DATA_KEYS
        assert loaded['format'] == 'npy'
        assert loaded['filename'] == str(npy_file)
        assert (loaded['shape'], loaded['ndim']) == (simulated_profiles.shape, 2)
        np.testing.assert_array_equal(loaded['data'], simulated_profiles)

    @pytest.mark.parametrize('suffix', ['.txt', '.dat', '.prof'])
    def test_whitespace_text_formats(self, tmp_path, simulated_profiles, suffix):
        text_file = tmp_path / f'profiles{suffix}'
        np.savetxt(text_file, simulated_profiles)
        loaded = load_experimental_data(text_file)
        assert loaded['format'] == 'txt'
        np.testing.assert_array_equal(loaded['data'], simulated_profiles)

    def test_csv_with_header_reads_columns(self, tmp_path, quick_test_parameters):
        from dice.analysis.simulation import run_monte_carlo_simulation
        from dice.io.results import export_collated_results
        from dice.models.parameters import simulation_parameters_from_dict
        params = quick_test_parameters
        sim_params = simulation_parameters_from_dict(
            params, params['nominal diffusion coefficient'],
            params['nominal lifetime (tau)'], params['nominal diffusion length'])
        result = run_monte_carlo_simulation(
            sim_params, params['x array'], params['time series'], params['noise series'],
            num_runs=3, multiprocessing=False, seed=2)
        results_csv = tmp_path / 'results.csv'
        exported = export_collated_results(result, results_csv, length_unit='micrometer', time_unit='nanosecond')

        loaded = load_experimental_data(results_csv)
        assert set(loaded) == LOADED_DATA_KEYS | {'columns'}
        assert loaded['columns'] == list(exported.columns)
        assert loaded['shape'] == exported.shape
        np.testing.assert_allclose(loaded['data'].astype(float), exported.to_numpy(dtype=float))

    def test_single_line_profile_csv(self):
        loaded = load_experimental_data(EXAMPLE_PROFILE)
        assert set(loaded) == LOADED_DATA_KEYS
        assert loaded['format'] == 'csv'
        assert (loaded['shape'], loaded['ndim']) == ((41,), 1)
        # The same array the noise estimate reads from the file
        np.testing.assert_array_equal(loaded['data'], np.loadtxt(EXAMPLE_PROFILE, delimiter=','))

    def test_headerless_csv_keeps_first_row(self, tmp_path, simulated_profiles):
        profiles_csv = tmp_path / 'profiles.csv'
        np.savetxt(profiles_csv, simulated_profiles, delimiter=',')
        loaded = load_experimental_data(profiles_csv)
        assert set(loaded) == LOADED_DATA_KEYS
        np.testing.assert_array_equal(loaded['data'], simulated_profiles)

    @pytest.mark.parametrize('prefix', ['﻿', '\n\n'], ids=['byte_order_mark', 'leading_blank_lines'])
    def test_single_line_profile_csv_with_prefix(self, tmp_path, prefix):
        profile_csv = tmp_path / 'profile.csv'
        profile_csv.write_text(prefix + EXAMPLE_PROFILE.read_text(), encoding='utf-8')
        loaded = load_experimental_data(profile_csv)
        assert 'columns' not in loaded
        np.testing.assert_array_equal(loaded['data'], load_profile_data(EXAMPLE_PROFILE))

    def test_malformed_headerless_csv_raises_ioerror(self, tmp_path):
        ragged = tmp_path / 'ragged.csv'
        ragged.write_text(EXAMPLE_PROFILE.read_text() + "\n1,2,3\n")
        with pytest.raises(IOError, match="Error loading csv file"):
            load_experimental_data(ragged)

    def test_empty_csv_raises_ioerror(self, tmp_path):
        empty = tmp_path / 'empty.csv'
        empty.write_text("")
        with pytest.raises(IOError, match="Error loading csv file"):
            load_experimental_data(empty)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="absent.csv"):
            load_experimental_data(tmp_path / 'absent.csv')

    def test_corrupt_npy_raises_ioerror(self, tmp_path):
        corrupt = tmp_path / 'corrupt.npy'
        corrupt.write_text(EXAMPLE_PROFILE.read_text())
        with pytest.raises(IOError, match="Error loading npy file"):
            load_experimental_data(corrupt)


class TestBatchLoadProfiles:
    """Batch loading keeps the readable files in name order and warns about the rest."""

    def test_skips_unreadable_file_with_warning(self, tmp_path, simulated_profiles):
        np.save(tmp_path / 'a_first.npy', simulated_profiles[:2])
        np.save(tmp_path / 'c_last.npy', simulated_profiles[2:])
        (tmp_path / 'b_corrupt.npy').write_text("not an array")

        with pytest.warns(UserWarning, match="Failed to load .*b_corrupt.npy"):
            loaded = batch_load_profiles(tmp_path, pattern='*.npy')

        assert [Path(item['filename']).name for item in loaded] == ['a_first.npy', 'c_last.npy']
        np.testing.assert_array_equal(
            np.vstack([item['data'] for item in loaded]), simulated_profiles)

    def test_no_matching_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=r"No files matching '\*.npy'"):
            batch_load_profiles(tmp_path, pattern='*.npy')
