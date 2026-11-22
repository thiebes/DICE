"""
Characterization of the analysis dict, the collated CSV, and the summary text.

The inputs are MonteCarloOutput objects built from literal RunResult values,
with the edge cases the writers branch on: a run with no slope fits, a run with
only one estimator, a slope stderr of 0.0 (an empty CSV cell), a NaN stderr,
two noise values (so the per-noise analysis is filled), and a simulation in
which every fit failed. Each output must equal its golden file in
tests/golden/estimators/, which pins key order, column order, labels, and
every number. The last digit of a float differs between processors, so the
analysis dict is written with 12 significant digits (the file is then the
same wherever it is regenerated) and its numbers are compared with a
relative tolerance of 1e-9; keys, their order, and the text files are
compared exactly.

The golden files were written by the code before the estimator registry was
introduced. Regenerate them only deliberately, after a reviewed change to an
output format:

    python tests/test_output_characterization.py
"""

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pytest

import dice
from dice.analysis.statistics import analyze_simulation_results
from dice.io.results import export_collated_results, write_summary_file
from dice.models.parameters import OutputUnitPreferences
from dice.models.results import MonteCarloOutput, RunResult
from dice.utils.legacy_compatibility import create_parameters_from_legacy

GOLDEN_DIR = Path(__file__).parent / 'golden' / 'estimators'
SUMMARY_PARAMETERS = {'spatial width': 5, 'pixel width': 100, 'time series': [0.0, 0.5, 1.0]}

# (slope, slope stderr, intercept, intercept stderr)
Fit = Tuple[float, float, float, float]


def _run(run_id: int, noise: float, ols: Optional[Fit] = None,
         wls: Optional[Fit] = None) -> RunResult:
    fields = dict(run_id=run_id, nominal_diffusion_coefficient=0.01, nominal_lifetime=1.0,
                  nominal_diffusion_length=0.1, nominal_sigma2_0=0.18033688,
                  noise_sigma=noise, cnr_0_estimate=1.0 / noise + 0.25 * run_id,
                  estimated_sigma2_0=0.18 + 0.001 * run_id)
    for key, fit in (('ols', ols), ('wls', wls)):
        if fit is not None:
            fields.update({f'{key}_slope': fit[0], f'{key}_slope_stderr': fit[1],
                           f'{key}_intercept': fit[2], f'{key}_intercept_stderr': fit[3]})
    return RunResult(**fields)


def _output(runs, noise_values) -> MonteCarloOutput:
    parameters = create_parameters_from_legacy(
        {'sigma^2_0': 0.18033688, 'amplitude_0': 1.0, 'mu_0': 0.0},
        diffusion_coefficient=0.01, lifetime=1.0, diffusion_length=0.1)
    return MonteCarloOutput(parameters=parameters, run_results=runs, num_runs=len(runs),
                            noise_values=noise_values)


def edge_case_output() -> MonteCarloOutput:
    return _output([
        _run(0, 0.02),
        _run(1, 0.02, (0.0213, 0.0011, 0.0004, 0.0002), (0.0208, 0.0009, 0.0003, 0.0001)),
        _run(2, 0.02, (0.0191, 0.0, -0.0002, 0.0003), (0.0199, float('nan'), 0.0001, 0.0002)),
        _run(3, 0.05, None, (0.0261, 0.0033, 0.0009, 0.0006)),
        _run(4, 0.05, (0.0233, 0.0038, 0.0012, 0.0008), None),
        _run(5, 0.05, (0.0177, 0.0029, -0.0006, 0.0005), (0.0189, 0.0024, -0.0004, 0.0004)),
    ], noise_values=[0.02, 0.05])


def all_failed_output() -> MonteCarloOutput:
    return _output([_run(0, 0.02), _run(1, 0.02)], noise_values=[0.02])


def weighted_first_output() -> MonteCarloOutput:
    """The first run has only a WLS fit, so the CSV puts the WLS columns first."""
    return _output([
        _run(0, 0.02, None, (0.0208, 0.0009, 0.0003, 0.0001)),
        _run(1, 0.02, (0.0213, 0.0011, 0.0004, 0.0002), (0.0199, 0.0010, 0.0001, 0.0002)),
    ], noise_values=[0.02])


ANALYSIS_SIGNIFICANT_DIGITS = 12


def _rounded_floats(value: Any) -> Any:
    """`value` with each float, at any depth of dicts and lists, at 12 significant digits."""
    if isinstance(value, dict):
        return {key: _rounded_floats(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded_floats(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(f"{float(value):.{ANALYSIS_SIGNIFICANT_DIGITS}g}")
    return value


def _analysis_json(output: MonteCarloOutput, path: Path) -> None:
    analysis = _rounded_floats(analyze_simulation_results(output))
    path.write_text(json.dumps(analysis, indent=1, default=float), encoding='utf-8')


def _summary(output: MonteCarloOutput, path: Path, **options) -> None:
    write_summary_file(output, path, SUMMARY_PARAMETERS, analyze_simulation_results(output), **options)


CM2_PER_S = OutputUnitPreferences(diffusion_length='centimeter', diffusion_time='second')


def _csv(**options) -> Callable[[MonteCarloOutput, Path], None]:
    options = {'length_unit': 'micrometer', 'time_unit': 'nanosecond', **options}
    return lambda output, path: export_collated_results(output, path, **options)


# Golden file name -> (output builder, writer of that output to a path)
CASES: Dict[str, Tuple[Callable[[], MonteCarloOutput], Callable[[MonteCarloOutput, Path], None]]] = {
    'edge_cases_analysis.json': (edge_case_output, _analysis_json),
    'edge_cases_summary.txt': (edge_case_output, _summary),
    'edge_cases_output_units_summary.txt': (
        edge_case_output, lambda output, path: _summary(output, path, output_units=CM2_PER_S)),
    'edge_cases_default.csv': (edge_case_output, _csv()),
    'edge_cases_output_units.csv': (edge_case_output, _csv(output_units=OutputUnitPreferences(
        diffusion_length='micrometer', diffusion_time='nanosecond'))),
    'edge_cases_nanometer_picosecond.csv': (edge_case_output, _csv(
        length_unit='nanometer', time_unit='picosecond')),
    'all_failed_analysis.json': (all_failed_output, _analysis_json),
    'all_failed_summary.txt': (all_failed_output, _summary),
    'all_failed_default.csv': (all_failed_output, _csv()),
    'weighted_first_default.csv': (weighted_first_output, _csv()),
}


def written_text(name: str, directory: Path) -> str:
    """Write case `name` with the DICE writer and read it back with universal newlines.

    The summary names the DICE version. The golden files hold a placeholder
    in its place, so that a release does not change them.
    """
    build, write = CASES[name]
    path = directory / name
    write(build(), path)
    text = path.read_text(encoding='utf-8')
    return text.replace(f"DICE version:\n    {dice.__version__}\n", "DICE version:\n    <version>\n")


def assert_same_values(written: Any, golden: Any, where: str = 'analysis') -> None:
    """Equal structure and key order; floats agree to a relative 1e-9 (NaN equals NaN)."""
    if isinstance(golden, dict):
        assert isinstance(written, dict) and list(written) == list(golden), where
        for key, value in golden.items():
            assert_same_values(written[key], value, f"{where}[{key!r}]")
    elif isinstance(golden, list):
        assert isinstance(written, list) and len(written) == len(golden), where
        for index, value in enumerate(golden):
            assert_same_values(written[index], value, f"{where}[{index}]")
    elif isinstance(golden, float):
        assert written == pytest.approx(golden, rel=1e-9, nan_ok=True), where
    else:
        assert written == golden and type(written) is type(golden), where


@pytest.mark.parametrize('name', list(CASES))
def test_output_matches_golden(name, tmp_path):
    golden = (GOLDEN_DIR / name).read_text(encoding='utf-8')
    written = written_text(name, tmp_path)
    if name.endswith('.json'):
        assert_same_values(json.loads(written), json.loads(golden))
    else:
        assert written == golden


def test_comparison_of_the_analysis_tolerates_only_the_last_digits():
    golden = {'a': [0.300641842582, float('nan')], 'n': 4}
    assert_same_values({'a': [0.3006418425824019, float('nan')], 'n': 4}, golden)
    for different in ({'a': [0.30064185, float('nan')], 'n': 4},   # a number
                      {'a': [0.300641842582, float('nan')], 'n': 4.0},   # an int written as a float
                      {'n': 4, 'a': [0.300641842582, float('nan')]},   # key order
                      {'a': [0.300641842582], 'n': 4}):   # a missing value
        with pytest.raises(AssertionError):
            assert_same_values(different, golden)


def test_completion_message_says_when_no_run_gave_an_estimate():
    """The summary, the console and the GUI take their last line from this one function."""
    from dice.io.results import completion_message

    assert edge_case_output().has_estimates()
    assert completion_message(edge_case_output()) == "Simulation completed successfully."
    assert not all_failed_output().has_estimates()
    assert completion_message(all_failed_output()) == (
        "Simulation completed, but no run gave an estimate of the diffusion coefficient.")


def test_analysis_keeps_estimators_whose_every_fit_failed():
    analysis = analyze_simulation_results(all_failed_output())
    assert list(analysis['accuracy']) == ['ols', 'wls']
    assert analysis['statistics'] == {
        key: {'num_successful': 0, 'num_failed': 2, 'success_rate': 0.0}
        for key in ('ols', 'wls')
    }


def test_analysis_reports_only_the_estimators_that_ran():
    full = analyze_simulation_results(edge_case_output())
    wls_only = analyze_simulation_results(
        dataclasses.replace(edge_case_output(), estimator_keys=('wls',)))

    assert list(wls_only['accuracy']) == list(wls_only['statistics']) == ['wls']
    assert all(list(fractions) == ['wls'] for fractions in wls_only['precision'].values())
    for name, entry in wls_only['noise_analysis'].items():
        assert list(entry) == ['cnr', 'num_runs', 'wls_precision_10%', 'wls_bias']
        assert entry == {key: full['noise_analysis'][name][key] for key in entry}
    assert json.dumps(wls_only['accuracy']['wls']) == json.dumps(full['accuracy']['wls'])


def test_analysis_rejects_an_unknown_estimator_key():
    misspelled = dataclasses.replace(edge_case_output(), estimator_keys=('ols', 'wsl'))
    with pytest.raises(KeyError, match="unknown estimator 'wsl'"):
        analyze_simulation_results(misspelled)


def test_summary_reports_only_the_estimators_in_the_analysis(tmp_path):
    wls_only = dataclasses.replace(edge_case_output(), estimator_keys=('wls',))
    _summary(wls_only, tmp_path / 'summary.txt')
    text = (tmp_path / 'summary.txt').read_text(encoding='utf-8')
    golden = (GOLDEN_DIR / 'edge_cases_summary.txt').read_text(encoding='utf-8')

    assert 'Least Squares' not in text.replace('  Weighted Least Squares:', '')
    wls_block = golden[golden.index('  Weighted Least Squares:'):golden.index('  Ordinary')]
    assert wls_block in text
    precision_lines = [line for line in text.splitlines() if line.startswith('    WLS:')]
    assert precision_lines == [line.split('    OLS:')[0] for line in golden.splitlines()
                               if line.startswith('    WLS:')]


def test_summary_rejects_an_unknown_estimator_key(tmp_path):
    analysis = analyze_simulation_results(edge_case_output())
    analysis['accuracy']['wsl'] = analysis['accuracy'].pop('wls')
    with pytest.raises(KeyError, match="unknown estimator 'wsl'"):
        write_summary_file(edge_case_output(), tmp_path / 'summary.txt', SUMMARY_PARAMETERS,
                           analysis)


def test_summary_names_a_missing_accuracy_metric(tmp_path):
    analysis = analyze_simulation_results(edge_case_output())
    del analysis['accuracy']['wls']['rmse']
    with pytest.raises(KeyError, match="'rmse'"):
        write_summary_file(edge_case_output(), tmp_path / 'summary.txt', SUMMARY_PARAMETERS, analysis)


def test_summary_in_output_units_converts_every_diffusion_coefficient(tmp_path):
    """1 micrometer^2/ns is 10 cm^2/s: the nominal D and four statistics scale, nothing else changes."""
    output = edge_case_output()
    analysis = analyze_simulation_results(output)
    _summary(output, tmp_path / 'global.txt')
    _summary(output, tmp_path / 'cm2_per_s.txt', output_units=CM2_PER_S)
    global_lines = (tmp_path / 'global.txt').read_text(encoding='utf-8').splitlines()
    converted_lines = (tmp_path / 'cm2_per_s.txt').read_text(encoding='utf-8').splitlines()

    changed = [(a, b) for a, b in zip(global_lines, converted_lines) if a != b]
    assert len(global_lines) == len(converted_lines)
    assert all(a.endswith(' micrometer^2/nanosecond') and b.endswith(' centimeter^2/second')
               for a, b in changed)
    # The nominal value and std, mean, standard error, bias, RMSE of both estimators.
    assert len(changed) == 1 + 5 * 2
    assert "    0.10000 centimeter^2/second" in converted_lines
    wls = analysis['accuracy']['wls']
    for key in ('std', 'mean', 'standard_error', 'bias', 'rmse'):
        assert f"        {wls[key] * 10:.5f} centimeter^2/second" in converted_lines


def test_csv_columns_follow_first_appearance(tmp_path):
    columns = written_text('weighted_first_default.csv', tmp_path).splitlines()[0].split(',')
    assert columns.index('weighted fit diffusion slope') < columns.index(
        'nominal diffusion coeff [cm²/s]') < columns.index('unweighted fit diffusion slope')


def regenerate() -> None:
    import tempfile
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        for name in CASES:
            text = written_text(name, Path(scratch))
            (GOLDEN_DIR / name).write_text(text, encoding='utf-8', newline='\n')
            print(f"wrote {GOLDEN_DIR / name}")


if __name__ == '__main__':
    sys.exit(regenerate())
