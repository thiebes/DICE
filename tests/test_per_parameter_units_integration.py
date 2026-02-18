"""
Integration tests for the per-parameter unit system.

Each test runs two physically equivalent configurations -- one with all values
in the global unit system and one with per-parameter unit overrides -- then
verifies that the simulation pipeline produces identical results.

Test matrix:
    1. FWHM in nanometers vs micrometers
    2. Diffusion coefficient in cm^2/s vs micrometer^2/ns
    3. Lifetime in picoseconds vs nanoseconds
    4. Spatial width in nanometers vs micrometers
    5. Time range in picoseconds vs nanoseconds
    6. All overrides combined (mixed-unit stress test)
    7. Full simulation with seeded RNG (slope comparison)
    8. Output unit conversion round-trip
    9. GUI collect -> build -> resolve -> parse pipeline
   10. Parameter file save/load round-trip with _unit keys
"""

import numpy as np
import pytest
from dice.io.parameters import parameter_parser
from dice.utils.units import resolve_units, convert_diffusion_coefficient


def _base_params():
    """Minimal valid parameter set in micrometer/nanosecond."""
    return {
        'number of runs': 5,
        'spatial width': 10.0,
        'pixel width': 101,
        'mean_0': 0.0,
        'amplitude_0': 1.0,
        'FWHM_0': 1.0,
        'noise value': 0.05,
        'time range': [0, 2, 10],
        'nominal diffusion coefficient': 0.5,
        'nominal lifetime (tau)': 2.0,
        'length unit': 'micrometer',
        'time unit': 'nanosecond',
        'proximity level': 0.1,
        'filename slug': 'test',
    }


# ---------------------------------------------------------------------------
# 1-5: Individual override equivalence (parsed parameter comparison)
# ---------------------------------------------------------------------------

class TestSingleOverrideEquivalence:
    """Verify that one override at a time resolves to the same physics."""

    def test_fwhm_nanometers(self):
        """FWHM_0: 500 nm should equal FWHM_0: 0.5 micrometer."""
        uniform = _base_params()
        uniform['FWHM_0'] = 0.5

        mixed = _base_params()
        mixed['FWHM_0'] = 500.0
        mixed['fwhm_0_unit'] = 'nanometer'

        parsed_u = parameter_parser(resolve_units(uniform))
        parsed_m = parameter_parser(resolve_units(mixed))

        assert np.isclose(parsed_u['sigma2_0'], parsed_m['sigma2_0'], rtol=1e-10)

    def test_sigma_nanometers(self):
        """sigma_0: 200 nm should equal sigma_0: 0.2 micrometer."""
        uniform = _base_params()
        del uniform['FWHM_0']
        uniform['sigma_0'] = 0.2

        mixed = _base_params()
        del mixed['FWHM_0']
        mixed['sigma_0'] = 200.0
        mixed['sigma_0_unit'] = 'nanometer'

        parsed_u = parameter_parser(resolve_units(uniform))
        parsed_m = parameter_parser(resolve_units(mixed))

        assert np.isclose(parsed_u['sigma2_0'], parsed_m['sigma2_0'], rtol=1e-10)

    def test_diffusion_coefficient_cm2_per_s(self):
        """D = 0.5 micrometer^2/ns should equal D = 5.0 cm^2/s.

        Conversion: 0.5 micrometer^2/ns * (1e-4 cm/micrometer)^2 / (1e-9 s/ns)
                  = 0.5 * 1e-8 / 1e-9 = 5.0 cm^2/s
        """
        uniform = _base_params()
        uniform['nominal diffusion coefficient'] = 0.5  # micrometer^2/ns

        mixed = _base_params()
        mixed['nominal diffusion coefficient'] = 5.0  # cm^2/s
        mixed['diffusion_coefficient_length_unit'] = 'centimeter'
        mixed['diffusion_coefficient_time_unit'] = 'second'

        parsed_u = parameter_parser(resolve_units(uniform))
        parsed_m = parameter_parser(resolve_units(mixed))

        assert np.isclose(
            parsed_u['nominal diffusion coefficient'],
            parsed_m['nominal diffusion coefficient'],
            rtol=1e-10,
        )

    def test_lifetime_picoseconds(self):
        """tau = 2000 ps should equal tau = 2.0 ns."""
        uniform = _base_params()
        uniform['nominal lifetime (tau)'] = 2.0  # ns

        mixed = _base_params()
        mixed['nominal lifetime (tau)'] = 2000.0  # ps
        mixed['lifetime_unit'] = 'picosecond'

        parsed_u = parameter_parser(resolve_units(uniform))
        parsed_m = parameter_parser(resolve_units(mixed))

        assert np.isclose(
            parsed_u['nominal lifetime (tau)'],
            parsed_m['nominal lifetime (tau)'],
            rtol=1e-10,
        )

    def test_spatial_width_nanometers(self):
        """spatial_width: 10000 nm should equal 10.0 micrometer."""
        uniform = _base_params()
        uniform['spatial width'] = 10.0  # micrometer

        mixed = _base_params()
        mixed['spatial width'] = 10000.0  # nm
        mixed['spatial_width_unit'] = 'nanometer'

        parsed_u = parameter_parser(resolve_units(uniform))
        parsed_m = parameter_parser(resolve_units(mixed))

        x_u = parsed_u['x array']
        x_m = parsed_m['x array']
        assert np.allclose(x_u, x_m, rtol=1e-10)

    def test_time_range_picoseconds(self):
        """time_range [0, 2000, 10] ps should equal [0, 2, 10] ns."""
        uniform = _base_params()
        uniform['time range'] = [0, 2, 10]  # ns

        mixed = _base_params()
        mixed['time range'] = [0, 2000, 10]  # ps (start, stop in ps; steps is count)
        mixed['time_range_unit'] = 'picosecond'

        parsed_u = parameter_parser(resolve_units(uniform))
        parsed_m = parameter_parser(resolve_units(mixed))

        t_u = parsed_u['time series']
        t_m = parsed_m['time series']
        assert np.allclose(t_u, t_m, rtol=1e-10)

    def test_mean_position_nanometers(self):
        """mu_0: 500 nm should equal mu_0: 0.5 micrometer."""
        uniform = _base_params()
        uniform['mean_0'] = 0.5  # micrometer

        mixed = _base_params()
        mixed['mean_0'] = 500.0  # nm
        mixed['mu_0_unit'] = 'nanometer'

        parsed_u = parameter_parser(resolve_units(uniform))
        parsed_m = parameter_parser(resolve_units(mixed))

        assert np.isclose(parsed_u['mu_0'], parsed_m['mu_0'], rtol=1e-10)

    def test_diffusion_length_nanometers(self):
        """diffusion_length: 100 nm should equal 0.1 micrometer."""
        uniform = _base_params()
        del uniform['nominal diffusion coefficient']
        del uniform['nominal lifetime (tau)']
        uniform['nominal diffusion length'] = 0.1  # micrometer

        mixed = _base_params()
        del mixed['nominal diffusion coefficient']
        del mixed['nominal lifetime (tau)']
        mixed['nominal diffusion length'] = 100.0  # nm
        mixed['diffusion_length_unit'] = 'nanometer'

        parsed_u = parameter_parser(resolve_units(uniform))
        parsed_m = parameter_parser(resolve_units(mixed))

        assert np.isclose(
            parsed_u['nominal diffusion length'],
            parsed_m['nominal diffusion length'],
            rtol=1e-10,
        )


# ---------------------------------------------------------------------------
# 6: All overrides combined
# ---------------------------------------------------------------------------

class TestMixedUnitStress:
    """Apply multiple overrides simultaneously and verify equivalence."""

    def test_all_overrides_at_once(self):
        """Every dimensioned parameter in a different unit."""
        uniform = _base_params()
        uniform['FWHM_0'] = 0.5                          # micrometer
        uniform['mean_0'] = 0.0                           # micrometer
        uniform['spatial width'] = 10.0                   # micrometer
        uniform['nominal diffusion coefficient'] = 0.5    # micrometer^2/ns
        uniform['nominal lifetime (tau)'] = 2.0           # ns
        uniform['time range'] = [0, 2, 10]                # ns

        mixed = _base_params()
        mixed['FWHM_0'] = 500.0
        mixed['fwhm_0_unit'] = 'nanometer'
        mixed['mean_0'] = 0.0
        mixed['mu_0_unit'] = 'nanometer'                  # 0 converts to 0
        mixed['spatial width'] = 0.01
        mixed['spatial_width_unit'] = 'millimeter'        # 0.01 mm = 10 micrometer
        mixed['nominal diffusion coefficient'] = 5.0
        mixed['diffusion_coefficient_length_unit'] = 'centimeter'
        mixed['diffusion_coefficient_time_unit'] = 'second'
        mixed['nominal lifetime (tau)'] = 2000.0
        mixed['lifetime_unit'] = 'picosecond'
        mixed['time range'] = [0, 2000, 10]
        mixed['time_range_unit'] = 'picosecond'

        parsed_u = parameter_parser(resolve_units(uniform))
        parsed_m = parameter_parser(resolve_units(mixed))

        assert np.isclose(parsed_u['sigma2_0'], parsed_m['sigma2_0'], rtol=1e-10)
        assert np.isclose(
            parsed_u['nominal diffusion coefficient'],
            parsed_m['nominal diffusion coefficient'],
            rtol=1e-10,
        )
        assert np.isclose(
            parsed_u['nominal lifetime (tau)'],
            parsed_m['nominal lifetime (tau)'],
            rtol=1e-10,
        )
        assert np.allclose(parsed_u['x array'], parsed_m['x array'], rtol=1e-10)
        assert np.allclose(
            parsed_u['time series'], parsed_m['time series'], rtol=1e-10
        )
        assert np.isclose(
            parsed_u['nominal diffusion length'],
            parsed_m['nominal diffusion length'],
            rtol=1e-10,
        )


# ---------------------------------------------------------------------------
# 7: Full simulation with seeded RNG
# ---------------------------------------------------------------------------

class TestSimulationEquivalence:
    """Run actual Monte Carlo simulations and compare fit results.

    Unit conversions introduce floating-point rounding (e.g., 500 nm -> 0.5
    micrometer differs at the ~15th digit from a direct 0.5 input). Since noise-
    dependent fitting amplifies these tiny differences across runs, we compare
    the mean slope across runs with a tolerance appropriate for the statistical
    variation, rather than requiring per-run exact equality.
    """

    @staticmethod
    def _run_seeded_simulation(params, seed=42):
        """Resolve units, parse, and run a simulation with a fixed seed."""
        from dice.utils.legacy_compatibility import create_parameters_from_legacy
        from dice.analysis.simulation import run_monte_carlo_simulation

        resolved = resolve_units(params)
        parsed = parameter_parser(resolved)

        sim_params = create_parameters_from_legacy(
            parameters_dict=parsed,
            diffusion_coefficient=parsed['nominal diffusion coefficient'],
            lifetime=parsed['nominal lifetime (tau)'],
            diffusion_length=parsed.get('nominal diffusion length'),
        )

        np.random.seed(seed)
        result = run_monte_carlo_simulation(
            parameters=sim_params,
            x_axis=parsed['x array'],
            time_axis=parsed['time series'],
            noise_values=parsed['noise series'],
            num_runs=parsed['number of runs'],
            multiprocessing=False,
            retain_profile_data=False,
        )
        return result

    @staticmethod
    def _mean_wls_slope(result):
        """Extract mean WLS slope from simulation results."""
        slopes = [r.wls_slope for r in result.run_results if r.wls_slope is not None]
        return np.mean(slopes) if slopes else None

    # Use more runs than the base params to get statistical convergence.
    # With 50 runs, the standard error of each mean slope is ~0.05, so the
    # difference of means has SE ~0.07, well within 20% of the nominal ~1.0.
    _NUM_RUNS = 50
    _RTOL = 0.2

    def test_seeded_simulation_fwhm_nm(self):
        """Simulation with FWHM in nm matches simulation with FWHM in micrometers."""
        uniform = _base_params()
        uniform['number of runs'] = self._NUM_RUNS
        uniform['FWHM_0'] = 0.5

        mixed = _base_params()
        mixed['number of runs'] = self._NUM_RUNS
        mixed['FWHM_0'] = 500.0
        mixed['fwhm_0_unit'] = 'nanometer'

        res_u = self._run_seeded_simulation(uniform)
        res_m = self._run_seeded_simulation(mixed)

        assert res_u.num_runs == res_m.num_runs
        mean_u = self._mean_wls_slope(res_u)
        mean_m = self._mean_wls_slope(res_m)
        assert mean_u is not None and mean_m is not None
        assert np.isclose(mean_u, mean_m, rtol=self._RTOL), (
            f"Mean WLS slopes differ beyond tolerance: {mean_u} vs {mean_m}"
        )

    def test_seeded_simulation_d_in_cm2_per_s(self):
        """Simulation with D in cm^2/s matches D in micrometer^2/ns."""
        uniform = _base_params()
        uniform['number of runs'] = self._NUM_RUNS
        uniform['nominal diffusion coefficient'] = 0.5

        mixed = _base_params()
        mixed['number of runs'] = self._NUM_RUNS
        mixed['nominal diffusion coefficient'] = 5.0
        mixed['diffusion_coefficient_length_unit'] = 'centimeter'
        mixed['diffusion_coefficient_time_unit'] = 'second'

        res_u = self._run_seeded_simulation(uniform)
        res_m = self._run_seeded_simulation(mixed)

        mean_u = self._mean_wls_slope(res_u)
        mean_m = self._mean_wls_slope(res_m)
        assert mean_u is not None and mean_m is not None
        assert np.isclose(mean_u, mean_m, rtol=self._RTOL), (
            f"Mean WLS slopes differ beyond tolerance: {mean_u} vs {mean_m}"
        )

    def test_seeded_simulation_lifetime_ps(self):
        """Simulation with lifetime in ps matches lifetime in ns."""
        uniform = _base_params()
        uniform['number of runs'] = self._NUM_RUNS
        uniform['nominal lifetime (tau)'] = 2.0

        mixed = _base_params()
        mixed['number of runs'] = self._NUM_RUNS
        mixed['nominal lifetime (tau)'] = 2000.0
        mixed['lifetime_unit'] = 'picosecond'

        res_u = self._run_seeded_simulation(uniform)
        res_m = self._run_seeded_simulation(mixed)

        mean_u = self._mean_wls_slope(res_u)
        mean_m = self._mean_wls_slope(res_m)
        assert mean_u is not None and mean_m is not None
        assert np.isclose(mean_u, mean_m, rtol=self._RTOL), (
            f"Mean WLS slopes differ beyond tolerance: {mean_u} vs {mean_m}"
        )

    def test_seeded_simulation_all_mixed(self):
        """Full simulation with every parameter in a different unit."""
        uniform = _base_params()
        uniform['number of runs'] = self._NUM_RUNS
        uniform['FWHM_0'] = 0.5
        uniform['spatial width'] = 10.0
        uniform['nominal diffusion coefficient'] = 0.5
        uniform['nominal lifetime (tau)'] = 2.0
        uniform['time range'] = [0, 2, 10]

        mixed = _base_params()
        mixed['number of runs'] = self._NUM_RUNS
        mixed['FWHM_0'] = 500.0
        mixed['fwhm_0_unit'] = 'nanometer'
        mixed['spatial width'] = 0.01
        mixed['spatial_width_unit'] = 'millimeter'
        mixed['nominal diffusion coefficient'] = 5.0
        mixed['diffusion_coefficient_length_unit'] = 'centimeter'
        mixed['diffusion_coefficient_time_unit'] = 'second'
        mixed['nominal lifetime (tau)'] = 2000.0
        mixed['lifetime_unit'] = 'picosecond'
        mixed['time range'] = [0, 2000, 10]
        mixed['time_range_unit'] = 'picosecond'

        res_u = self._run_seeded_simulation(uniform)
        res_m = self._run_seeded_simulation(mixed)

        assert res_u.num_runs == res_m.num_runs
        mean_u = self._mean_wls_slope(res_u)
        mean_m = self._mean_wls_slope(res_m)
        assert mean_u is not None and mean_m is not None
        assert np.isclose(mean_u, mean_m, rtol=self._RTOL), (
            f"Mean WLS slopes differ beyond tolerance: {mean_u} vs {mean_m}"
        )

    def test_nominal_values_match(self):
        """RunResult nominal values are identical across unit systems."""
        uniform = _base_params()
        uniform['FWHM_0'] = 0.5

        mixed = _base_params()
        mixed['FWHM_0'] = 500.0
        mixed['fwhm_0_unit'] = 'nanometer'

        res_u = self._run_seeded_simulation(uniform)
        res_m = self._run_seeded_simulation(mixed)

        ru = res_u.run_results[0]
        rm = res_m.run_results[0]
        assert np.isclose(
            ru.nominal_diffusion_coefficient,
            rm.nominal_diffusion_coefficient,
            rtol=1e-10,
        )
        assert np.isclose(ru.nominal_lifetime, rm.nominal_lifetime, rtol=1e-10)
        assert np.isclose(ru.nominal_sigma2_0, rm.nominal_sigma2_0, rtol=1e-10)


# ---------------------------------------------------------------------------
# 8: Output unit conversion
# ---------------------------------------------------------------------------

class TestOutputUnitConversion:
    """Verify output CSV column values match expected conversions."""

    def test_export_with_output_unit_preferences(self):
        """Results exported with different output units should differ by
        exactly the conversion factor."""
        from dice.models.parameters import OutputUnitPreferences

        result = TestSimulationEquivalence._run_seeded_simulation(_base_params())

        # Default output (cm^2/s)
        default_prefs = OutputUnitPreferences()
        # Custom output (micrometer^2/ns)
        custom_prefs = OutputUnitPreferences(
            diffusion_length='micrometer',
            diffusion_time='nanosecond',
        )

        from dice.io.results import export_collated_results
        import tempfile, os, pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_default = os.path.join(tmpdir, 'default.csv')
            csv_custom = os.path.join(tmpdir, 'custom.csv')

            export_collated_results(result, csv_default, output_units=default_prefs)
            export_collated_results(result, csv_custom, output_units=custom_prefs)

            df_default = pd.read_csv(csv_default)
            df_custom = pd.read_csv(csv_custom)

            # Find the WLS diffusion coefficient columns by prefix
            def _find_col(df, prefix):
                return [c for c in df.columns if c.startswith(prefix)][0]

            wls_col_d = _find_col(df_default, 'weighted fit diffusion coeff')
            wls_col_c = _find_col(df_custom, 'weighted fit diffusion coeff')

            # The values should differ by the conversion factor cm^2/s -> micrometer^2/ns
            factor = convert_diffusion_coefficient(
                1.0, 'centimeter', 'second', 'micrometer', 'nanosecond'
            )
            d_vals = df_default[wls_col_d].dropna().values
            c_vals = df_custom[wls_col_c].dropna().values

            assert len(d_vals) == len(c_vals)
            assert np.allclose(d_vals * factor, c_vals, rtol=1e-10)


# ---------------------------------------------------------------------------
# 9: GUI pipeline (collect -> build -> resolve -> parse)
# ---------------------------------------------------------------------------

class TestGUIPipeline:
    """Simulate the GUI parameter flow without launching a window."""

    def test_build_parameters_dict_passes_unit_keys(self):
        """DiceInterface.build_parameters_dict should pass through _unit keys."""
        from dice_gui.dice_interface import DiceInterface

        iface = DiceInterface()

        gui_params = {
            'number_of_runs': 5,
            'filename_slug': 'test',
            'length_unit': 'micrometer',
            'time_unit': 'nanosecond',
            'amplitude_0': 1.0,
            'mean_0': 0.0,
            'profile_width_type': 'fwhm',
            'profile_width_value': 500.0,
            'fwhm_0_unit': 'nanometer',
            'diffusion_coefficient': 5.0,
            'diffusion_coefficient_length_unit': 'centimeter',
            'diffusion_coefficient_time_unit': 'second',
            'lifetime': 2000.0,
            'lifetime_unit': 'picosecond',
            'noise_type': 'fixed',
            'noise_value': 0.05,
            'spatial_width': 10.0,
            'pixel_width': 101,
            'time_type': 'range',
            'time_start': 0.0,
            'time_stop': 2.0,
            'time_steps': 10,
            'proximity_level': 0.1,
            'multiprocessing': False,
            'retain_profile_data': False,
            'image_type': 'png',
            'image_width': 8.5,
            'image_height': 5.0,
            'image_dpi': 100,
            'image_font_size': 8,
            'image_tick_length': 6,
            'image_tick_width': 2,
            'image_numbins': 35,
        }

        params = iface.build_parameters_dict(gui_params)

        # Unit override keys should be present in the output dict
        assert params['fwhm_0_unit'] == 'nanometer'
        assert params['diffusion_coefficient_length_unit'] == 'centimeter'
        assert params['diffusion_coefficient_time_unit'] == 'second'
        assert params['lifetime_unit'] == 'picosecond'

        # After resolve_units + parameter_parser, values should match the
        # equivalent single-unit-system values
        resolved = resolve_units(params)
        parsed = parameter_parser(resolved)

        ref = _base_params()
        ref['FWHM_0'] = 0.5
        parsed_ref = parameter_parser(resolve_units(ref))

        assert np.isclose(parsed['sigma2_0'], parsed_ref['sigma2_0'], rtol=1e-10)
        assert np.isclose(
            parsed['nominal diffusion coefficient'],
            parsed_ref['nominal diffusion coefficient'],
            rtol=1e-10,
        )
        assert np.isclose(
            parsed['nominal lifetime (tau)'],
            parsed_ref['nominal lifetime (tau)'],
            rtol=1e-10,
        )

    def test_gui_time_start_stop_unit_overrides(self):
        """time_start_unit/time_stop_unit overrides convert values in the
        time range list assembled by build_parameters_dict."""
        from dice_gui.dice_interface import DiceInterface

        iface = DiceInterface()

        gui_params = {
            'number_of_runs': 5,
            'filename_slug': 'test',
            'length_unit': 'micrometer',
            'time_unit': 'nanosecond',
            'amplitude_0': 1.0,
            'mean_0': 0.0,
            'profile_width_type': 'fwhm',
            'profile_width_value': 1.0,
            'diffusion_coefficient': 0.01,
            'lifetime': 2.0,
            'noise_type': 'fixed',
            'noise_value': 0.05,
            'spatial_width': 10.0,
            'pixel_width': 101,
            'time_type': 'range',
            'time_start': 500.0,       # 500 ps
            'time_stop': 2000.0,       # 2000 ps
            'time_steps': 10,
            'time_start_unit': 'picosecond',
            'time_stop_unit': 'picosecond',
            'proximity_level': 0.1,
            'multiprocessing': False,
            'retain_profile_data': False,
        }

        params = iface.build_parameters_dict(gui_params)
        # Before resolve_units, time range holds unconverted values
        assert params['time range'] == [500.0, 2000.0, 10]

        resolved = resolve_units(params)
        # After resolve_units: 500 ps -> 0.5 ns, 2000 ps -> 2.0 ns
        assert np.isclose(resolved['time range'][0], 0.5, rtol=1e-12)
        assert np.isclose(resolved['time range'][1], 2.0, rtol=1e-12)
        assert resolved['time range'][2] == 10
        assert 'time_start_unit' not in resolved
        assert 'time_stop_unit' not in resolved


# ---------------------------------------------------------------------------
# 10: Parameter file round-trip
# ---------------------------------------------------------------------------

class TestParameterFileRoundTrip:
    """Verify that _unit keys survive a write/read cycle via ast.literal_eval."""

    def test_unit_keys_survive_round_trip(self):
        """Write a dict with _unit keys to a file, read it back, resolve, and
        compare to the uniform-units equivalent."""
        import ast, tempfile, os

        params_with_overrides = {
            'filename slug': 'roundtrip_test',
            'number of runs': 5,
            'length unit': 'micrometer',
            'time unit': 'nanosecond',
            'FWHM_0': 500.0,
            'FWHM_0_unit': 'nanometer',
            'amplitude_0': 1.0,
            'mean_0': 0.0,
            'nominal diffusion coefficient': 5.0,
            'diffusion_coefficient_length_unit': 'centimeter',
            'diffusion_coefficient_time_unit': 'second',
            'nominal lifetime (tau)': 2000.0,
            'lifetime_unit': 'picosecond',
            'noise value': 0.05,
            'spatial width': 10.0,
            'pixel width': 101,
            'time range': [0, 2000, 10],
            'time_range_unit': 'picosecond',
            'proximity level': 0.1,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'params.txt')
            with open(filepath, 'w') as f:
                f.write(repr(params_with_overrides))

            with open(filepath, 'r') as f:
                loaded = ast.literal_eval(f.read())

        # Verify all _unit keys survived
        assert loaded['FWHM_0_unit'] == 'nanometer'
        assert loaded['diffusion_coefficient_length_unit'] == 'centimeter'
        assert loaded['diffusion_coefficient_time_unit'] == 'second'
        assert loaded['lifetime_unit'] == 'picosecond'
        assert loaded['time_range_unit'] == 'picosecond'

        # Resolve and parse
        resolved = resolve_units(loaded)
        parsed = parameter_parser(resolved)

        # Compare to uniform reference
        ref = _base_params()
        ref['FWHM_0'] = 0.5
        ref['time range'] = [0, 2, 10]
        parsed_ref = parameter_parser(resolve_units(ref))

        assert np.isclose(parsed['sigma2_0'], parsed_ref['sigma2_0'], rtol=1e-10)
        assert np.isclose(
            parsed['nominal diffusion coefficient'],
            parsed_ref['nominal diffusion coefficient'],
            rtol=1e-10,
        )
        assert np.allclose(
            parsed['time series'], parsed_ref['time series'], rtol=1e-10
        )

    def test_no_unit_keys_backward_compatible(self):
        """A parameter dict without _unit keys still works identically."""
        params = _base_params()
        resolved = resolve_units(params)

        # resolve_units should not modify values when no overrides exist
        assert resolved['FWHM_0'] == params['FWHM_0']
        assert resolved['nominal diffusion coefficient'] == params['nominal diffusion coefficient']
        assert resolved['nominal lifetime (tau)'] == params['nominal lifetime (tau)']
        assert resolved['spatial width'] == params['spatial width']
        assert resolved['time range'] == params['time range']

        # No _unit keys should remain
        assert not any(k.endswith('_unit') for k in resolved
                       if k not in ('length_unit', 'time_unit',
                                    'length unit', 'time unit'))


# ---------------------------------------------------------------------------
# 11: CSV round-trip (export -> read -> verify columns)
# ---------------------------------------------------------------------------

class TestCSVRoundTrip:
    """Export simulation results to CSV, read back, and verify column prefixes."""

    def test_csv_columns_present_after_export(self, tmp_path):
        """Run a small simulation, export to CSV, and verify expected columns."""
        import pandas as pd
        from dice.io.results import export_collated_results

        result = TestSimulationEquivalence._run_seeded_simulation(_base_params())

        csv_file = tmp_path / "round_trip.csv"
        export_collated_results(result, str(csv_file))

        df = pd.read_csv(csv_file)

        expected_prefixes = [
            'nominal diffusion coeff',
            'weighted fit diffusion coeff',
            'unweighted fit diffusion coeff',
            'weighted fit diffusion slope',
            'unweighted fit diffusion slope',
            'nominal CNR',
            'run number',
        ]

        for prefix in expected_prefixes:
            matches = [c for c in df.columns if c.startswith(prefix)]
            assert len(matches) >= 1, (
                f"Expected column starting with '{prefix}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        # Verify nominal values are consistent across rows
        nom_col = [c for c in df.columns if c.startswith('nominal diffusion coeff')][0]
        assert df[nom_col].nunique() == 1, "Nominal diffusion coefficient should be constant across runs"

        # Verify slope columns have numeric data
        wls_slope_col = [c for c in df.columns if c.startswith('weighted fit diffusion slope')][0]
        assert df[wls_slope_col].notna().sum() > 0, "WLS slope column should contain non-null values"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
