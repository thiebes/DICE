"""
A measured profile on a network path in a parameters file.

Windows contacts the computer that such a path names, and may send it the
user's sign-in credentials, so reading a parameters file refuses the path
unless it is allowed. No test opens a network path: the host name ends in
.invalid, which never resolves, and the reader is replaced in every test that
lets the path through.
"""

import ast
import contextlib
import shutil
from pathlib import Path

import numpy as np
import pytest

import dice
from dice.cli.arguments import parse_arguments
from dice.cli.main import main
from dice.io.parameters import is_network_path, network_paths, open_parameters

EXAMPLES_DIR = Path(__file__).parent.parent / 'data' / 'examples'
QUICK_TEST = EXAMPLES_DIR / 'quick_test.txt'
# The forward-slash form is absolute on every platform, so a path that is let
# through reaches the reader unchanged. The backslash form is absolute only on
# Windows; on Linux and macOS it is a relative file name, which DICE resolves
# against the folder of the parameters file. Both are refused alike.
NETWORK_PROFILE = '//dice-test.invalid/share/profile.csv'
BACKSLASH_PROFILE = '\\\\dice-test.invalid\\share\\profile.csv'

# The example profile is narrower than fft-cnr needs, which the parser warns about.
pytestmark = pytest.mark.filterwarnings("ignore:The peak of the measured profile:UserWarning")


@pytest.mark.parametrize('path', [
    '\\\\server\\share\\profile.csv',
    '//server/share/profile.csv',
    '\\\\?\\UNC\\server\\share\\profile.csv',
    '\\\\.\\pipe\\name',
    '\\/server/share/profile.csv',
    '  //server/share/profile.csv',
    # The NT prefix, which Windows passes through without changing it.
    '\\??\\UNC\\server\\share\\profile.csv',
    '/??/UNC/server/share/profile.csv',
    # Local forms that the rule covers too, because it reads only the text.
    '\\\\?\\C:\\data\\profile.csv',
    '//home/user/profile.csv',
])
def test_paths_that_start_with_two_separators_or_the_nt_prefix_are_network_paths(path):
    assert is_network_path(path)


@pytest.mark.parametrize('path', [
    'C:\\data\\profile.csv',
    'Z:\\profile.csv',          # a mapped drive letter
    '\\data\\profile.csv',      # the root of the current drive
    '/home/user/profile.csv',
    'data/profile.csv',
    'profile.csv',
    '',
    None,
    5,
])
def test_local_paths_and_other_values_are_not_network_paths(path):
    assert not is_network_path(path)


def write_file_with_profile(directory: Path, profile: str, key: str = 'estimate noise from data') -> Path:
    """quick_test.txt with its noise taken from the profile at the given path."""
    params = ast.literal_eval(QUICK_TEST.read_text(encoding='utf-8'))
    del params['noise value']
    params[key] = profile
    path = directory / 'with_profile.txt'
    path.write_text(str(params), encoding='utf-8')
    return path


@pytest.fixture
def profile_reads(monkeypatch) -> list:
    """Record each network path the parser reads a profile from, and give it the example profile instead."""
    example = np.loadtxt(EXAMPLES_DIR / 'example_profile.csv', delimiter=',')
    real_loadtxt = np.loadtxt
    paths = []

    def loadtxt(fname, *args, **kwargs):
        if is_network_path(fname):
            paths.append(fname)
            return example
        return real_loadtxt(fname, *args, **kwargs)

    monkeypatch.setattr(np, 'loadtxt', loadtxt)
    return paths


def test_network_paths_names_each_profile_key_with_a_network_path():
    assert network_paths({'estimate noise from data': NETWORK_PROFILE, 'noise value': 0.02}) == [
        ('estimate noise from data', NETWORK_PROFILE)]
    assert network_paths({'estimate_noise_from_data': NETWORK_PROFILE}) == [
        ('estimate_noise_from_data', NETWORK_PROFILE)]
    assert network_paths({'estimate noise from data': 'profile.csv'}) == []


class TestOpenParameters:

    @pytest.mark.parametrize('profile', [NETWORK_PROFILE, BACKSLASH_PROFILE])
    @pytest.mark.parametrize('key', ['estimate noise from data', 'estimate_noise_from_data'])
    def test_network_path_is_refused_before_it_is_opened(self, tmp_path, profile_reads, key, profile):
        params_file = write_file_with_profile(tmp_path, profile, key)

        with pytest.raises(ValueError, match="names a network or device path") as raised:
            open_parameters(params_file)

        assert profile in str(raised.value)
        assert '--allow-network-paths' in str(raised.value)
        assert profile_reads == []

    def test_allowed_network_path_reaches_the_reader_unchanged(self, tmp_path, profile_reads):
        params_file = write_file_with_profile(tmp_path, NETWORK_PROFILE)

        parsed = open_parameters(params_file, allow_network_paths=True)

        assert profile_reads == [NETWORK_PROFILE]
        assert parsed['estimate noise from data'] == NETWORK_PROFILE
        assert parsed['noise series'][0] > 0

    def test_relative_path_is_not_affected(self, tmp_path):
        shutil.copy(EXAMPLES_DIR / 'example_profile.csv', tmp_path / 'profile.csv')

        parsed = open_parameters(write_file_with_profile(tmp_path, 'profile.csv'))

        assert Path(parsed['estimate noise from data']) == tmp_path / 'profile.csv'

    def test_dice_runner_passes_its_permission_to_open_parameters(self, monkeypatch):
        import dice.io.parameters as io_parameters
        received = []

        def stop_after_recording(filename, seed=None, allow_network_paths=False):
            received.append(allow_network_paths)
            raise RuntimeError("recorded")

        monkeypatch.setattr(io_parameters, 'open_parameters', stop_after_recording)
        for allowed in (True, False):
            with pytest.raises(RuntimeError, match="recorded"):
                dice.dice_runner('parameters.txt', allow_network_paths=allowed)
        with pytest.raises(RuntimeError, match="recorded"):
            dice.dice_runner('parameters.txt')

        assert received == [True, False, False]

    def test_dice_runner_refuses_a_network_path(self, tmp_path, profile_reads):
        with pytest.raises(ValueError, match="names a network or device path"):
            dice.dice_runner(str(write_file_with_profile(tmp_path, NETWORK_PROFILE)))
        assert profile_reads == []


class TestCommandLine:

    def test_option_is_off_unless_given(self):
        assert parse_arguments([str(QUICK_TEST)]).allow_network_paths is False
        assert parse_arguments([str(QUICK_TEST), '--allow-network-paths']).allow_network_paths is True

    def test_dry_run_refuses_a_network_path_and_names_the_option(self, tmp_path, capsys, profile_reads):
        params_file = write_file_with_profile(tmp_path, NETWORK_PROFILE)

        with contextlib.chdir(tmp_path):
            exit_code = main([str(params_file), '--dry-run'])

        error = capsys.readouterr().out.splitlines()[-1]
        assert exit_code == 1
        assert error.startswith("Error: ") and NETWORK_PROFILE in error and '--allow-network-paths' in error
        assert profile_reads == []

    def test_option_lets_the_path_through(self, tmp_path, capsys, profile_reads):
        params_file = write_file_with_profile(tmp_path, NETWORK_PROFILE)

        with contextlib.chdir(tmp_path):
            exit_code = main([str(params_file), '--dry-run', '--allow-network-paths'])

        assert exit_code == 0, capsys.readouterr().out
        assert profile_reads == [NETWORK_PROFILE]
