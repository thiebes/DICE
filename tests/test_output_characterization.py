"""
Characterization of the analysis dict, the collated CSV, and the summary text.

The inputs are MonteCarloOutput objects built from literal RunResult values,
with the edge cases the writers branch on: a run with no slope fits, a run with
only one estimator, a slope stderr of 0.0 (an empty CSV cell), a NaN stderr,
two noise values (so the per-noise analysis is filled), and a simulation in
which every fit failed. Each output must equal its golden file in
tests/golden/estimators/, which pins key order, column order, labels, and
every number.

The golden files were written by the code before the estimator registry was
introduced. Regenerate them only deliberately, after a reviewed change to an
output format:

    python tests/test_output_characterization.py
"""

import json
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import pytest

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


def _analysis_json(output: MonteCarloOutput, path: Path) -> None:
    analysis = analyze_simulation_results(output)
    path.write_text(json.dumps(analysis, indent=1, default=float), encoding='utf-8')


def _summary(output: MonteCarloOutput, path: Path) -> None:
    write_summary_file(output, path, SUMMARY_PARAMETERS, analyze_simulation_results(output))


def _csv(**options) -> Callable[[MonteCarloOutput, Path], None]:
    return lambda output, path: export_collated_results(output, path, **options)


# Golden file name -> (output builder, writer of that output to a path)
CASES: Dict[str, Tuple[Callable[[], MonteCarloOutput], Callable[[MonteCarloOutput, Path], None]]] = {
    'edge_cases_analysis.json': (edge_case_output, _analysis_json),
    'edge_cases_summary.txt': (edge_case_output, _summary),
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
    """Write case `name` with the DICE writer and read it back with universal newlines."""
    build, write = CASES[name]
    path = directory / name
    write(build(), path)
    return path.read_text(encoding='utf-8')


@pytest.mark.parametrize('name', list(CASES))
def test_output_matches_golden(name, tmp_path):
    golden = (GOLDEN_DIR / name).read_text(encoding='utf-8')
    assert written_text(name, tmp_path) == golden


def test_analysis_keeps_estimators_whose_every_fit_failed():
    analysis = analyze_simulation_results(all_failed_output())
    assert list(analysis['accuracy']) == ['ols', 'wls']
    assert analysis['statistics'] == {
        key: {'num_successful': 0, 'num_failed': 2, 'success_rate': 0.0}
        for key in ('ols', 'wls')
    }


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
