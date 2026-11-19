"""
Tests for dice.visualization.histograms module.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for testing
import matplotlib.pyplot as plt
import tempfile
import os
from pathlib import Path
from dice.visualization.histograms import (
    plot_accuracy_histogram,
    plot_diffusion_coefficient_histogram,
    plot_cnr_histogram,
    plot_precision_vs_parameter,
)


class TestPlotAccuracyHistogram:
    """Test accuracy histogram plotting."""
    
    def test_basic_histogram(self):
        """Test basic accuracy histogram creation."""
        # Create mock simulation result
        result = {
            'collated results': {
                'd_est_over_d_nom': [0.95, 1.02, 0.98, 1.05, 0.92, 1.08, 0.97, 1.01]
            }
        }
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_filename = temp_file.name
        
        try:
            fig = plot_accuracy_histogram(
                simulation_result=result,
                filename=temp_filename,
                width=8.0,
                height=6.0
            )
            
            assert isinstance(fig, plt.Figure)
            ax = fig.axes[0]
            
            # Check basic plot elements
            assert ax.get_xlabel() == '$D_{est}/D_{nom}$'
            assert ax.get_ylabel() == 'Probability'
            assert len(ax.patches) > 0  # Histogram bars
            assert len(ax.lines) > 0   # Normal distribution overlay
            
            # Check that file was created
            assert os.path.exists(temp_filename)
            
            plt.close(fig)
            
        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
    
    def test_missing_data_error(self):
        """Test error handling for missing data."""
        # Missing 'collated results'
        result1 = {}
        
        with pytest.raises(ValueError, match="'collated results' not found"):
            plot_accuracy_histogram(result1, 0.1, "test.png")
        
        # Missing 'd_est_over_d_nom'
        result2 = {'collated results': {}}
        
        with pytest.raises(ValueError, match="'d_est_over_d_nom' not found"):
            plot_accuracy_histogram(result2, 0.1, "test.png")
    
    def test_custom_parameters(self, tmp_path):
        """Test with custom plotting parameters."""
        result = {
            'collated results': {
                'd_est_over_d_nom': np.random.normal(1.0, 0.1, 100).tolist()
            }
        }

        test_file = tmp_path / "test_custom_parameters.png"

        fig = plot_accuracy_histogram(
            simulation_result=result,
            filename=str(test_file),
            image_type='svg',
            width=12.0,
            height=8.0,
            dpi=150,
            font_size=14,
            num_bins=25,
            x_lim=[0.7, 1.3]
        )

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert ax.get_xlim() == (0.7, 1.3)

        plt.close(fig)

    def test_perfect_data(self, tmp_path):
        """Test with perfect data (all values = 1.0)."""
        result = {
            'collated results': {
                'd_est_over_d_nom': [1.0] * 50
            }
        }

        test_file = tmp_path / "test_perfect_data.png"

        fig = plot_accuracy_histogram(result, filename=str(test_file))

        assert isinstance(fig, plt.Figure)
        assert test_file.exists()
        plt.close(fig)


class TestPlotDiffusionCoefficientHistogram:
    """Test diffusion coefficient histogram plotting."""
    
    def test_basic_diffusion_histogram(self):
        """Test basic diffusion coefficient histogram."""
        estimates = np.random.normal(1.0, 0.1, 100)
        nominal = 1.0
        
        fig = plot_diffusion_coefficient_histogram(estimates, nominal)
        
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        
        # Check basic elements
        assert len(ax.patches) > 0  # Histogram bars
        assert len(ax.lines) >= 2  # Normal fit + nominal line
        assert ax.get_xlabel() == 'Diffusion Coefficient'
        assert ax.get_ylabel() == 'Probability'
        
        plt.close(fig)
    
    def test_with_filename(self):
        """Test diffusion histogram with file saving."""
        estimates = np.array([0.9, 1.0, 1.1, 0.95, 1.05])
        nominal = 1.0
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_filename = temp_file.name
        
        try:
            fig = plot_diffusion_coefficient_histogram(
                estimates, nominal, 
                filename=temp_filename,
                title="Custom Diffusion Analysis"
            )
            
            assert isinstance(fig, plt.Figure)
            assert os.path.exists(temp_filename)
            
            plt.close(fig)
            
        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
    
    def test_proximity_calculation(self):
        """Test proximity calculation in diffusion histogram."""
        # Create data where exactly half are within ±10%
        estimates = np.array([0.85, 0.95, 1.05, 1.15])  # 2 within ±10%, 2 outside
        nominal = 1.0
        
        fig = plot_diffusion_coefficient_histogram(
            estimates, nominal, proximity=0.1
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_custom_parameters(self):
        """Test with custom parameters."""
        estimates = np.random.normal(0.5, 0.05, 50)
        nominal = 0.5
        
        fig = plot_diffusion_coefficient_histogram(
            estimates, nominal,
            width=15.0,
            height=10.0,
            num_bins=20,
            font_size=16
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotCNRHistogram:
    """Test CNR histogram plotting."""
    
    def test_basic_cnr_histogram(self):
        """Test basic CNR histogram."""
        cnr_estimates = np.random.exponential(20, 100)  # Exponential distribution
        
        fig = plot_cnr_histogram(cnr_estimates)
        
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        
        assert len(ax.patches) > 0  # Histogram bars
        assert len(ax.lines) >= 2   # Mean and median lines
        assert ax.get_xlabel() == 'Contrast-to-Noise Ratio'
        assert ax.get_ylabel() == 'Probability'
        
        plt.close(fig)
    
    def test_cnr_statistics(self):
        """Test CNR histogram with known statistics."""
        cnr_estimates = np.array([10, 20, 30, 20, 25])
        
        fig = plot_cnr_histogram(
            cnr_estimates,
            title="Test CNR Distribution"
        )
        
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert ax.get_title() == "Test CNR Distribution"
        
        plt.close(fig)
    
    def test_cnr_with_filename(self):
        """Test CNR histogram with file saving."""
        cnr_estimates = np.random.gamma(2, 10, 75)
        
        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as temp_file:
            temp_filename = temp_file.name
        
        try:
            fig = plot_cnr_histogram(
                cnr_estimates,
                filename=temp_filename,
                image_type='svg'
            )
            
            assert isinstance(fig, plt.Figure)
            assert os.path.exists(temp_filename)
            
            plt.close(fig)
            
        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)


class TestPlotPrecisionVsParameter:
    """Test precision vs parameter plotting."""
    
    def test_basic_precision_plot(self):
        """Test basic precision vs parameter plot."""
        param_values = np.array([1, 2, 5, 10, 20, 50])
        precision_values = np.array([10, 25, 50, 75, 90, 95])
        
        fig = plot_precision_vs_parameter(
            param_values, precision_values,
            parameter_name="CNR"
        )
        
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        
        assert len(ax.lines) == 1
        assert ax.get_xlabel() == "CNR"
        assert ax.get_ylabel() == "Precision (%)"
        assert ax.get_ylim() == (0, 100)
        
        plt.close(fig)
    
    def test_custom_title_and_labels(self):
        """Test with custom title and labels."""
        param_values = np.array([0.1, 0.2, 0.5])
        precision_values = np.array([20, 60, 90])
        
        fig = plot_precision_vs_parameter(
            param_values, precision_values,
            parameter_name="Noise Level",
            title="Custom Precision Analysis"
        )
        
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert ax.get_title() == "Custom Precision Analysis"
        assert ax.get_xlabel() == "Noise Level"
        
        plt.close(fig)
    
    def test_precision_with_filename(self):
        """Test precision plot with file saving."""
        param_values = np.linspace(1, 10, 5)
        precision_values = np.linspace(20, 80, 5)
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_filename = temp_file.name
        
        try:
            fig = plot_precision_vs_parameter(
                param_values, precision_values,
                parameter_name="Test Parameter",
                filename=temp_filename,
                image_type='pdf'
            )
            
            assert isinstance(fig, plt.Figure)
            assert os.path.exists(temp_filename)
            
            plt.close(fig)
            
        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_data(self):
        """Test with empty data arrays."""
        empty_array = np.array([])
        
        # These should not crash
        fig1 = plot_diffusion_coefficient_histogram(empty_array, 1.0)
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)
        
        fig2 = plot_cnr_histogram(empty_array)
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)
        
        fig3 = plot_precision_vs_parameter(empty_array, empty_array)
        assert isinstance(fig3, plt.Figure)
        plt.close(fig3)
    
    def test_single_value(self):
        """Test with single value."""
        single_value = np.array([1.0])

        fig1 = plot_diffusion_coefficient_histogram(single_value, 1.0)
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)

        fig2 = plot_cnr_histogram(single_value)
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)

    def test_accuracy_histogram_single_value(self, tmp_path):
        """Test accuracy histogram with a single D estimate."""
        result = {
            'collated results': {
                'd_est_over_d_nom': [1.05]
            }
        }
        fig = plot_accuracy_histogram(
            simulation_result=result,
            filename=str(tmp_path / "test.png")
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_accuracy_histogram_empty(self, tmp_path):
        """Test accuracy histogram with empty D estimates raises error."""
        result = {
            'collated results': {
                'd_est_over_d_nom': []
            }
        }
        with pytest.raises((ValueError, IndexError)):
            plot_accuracy_histogram(
                simulation_result=result,
                filename=str(tmp_path / "test.png")
            )

    def test_identical_values(self):
        """Test with identical values."""
        identical_values = np.ones(10)
        
        fig1 = plot_diffusion_coefficient_histogram(identical_values, 1.0)
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)
        
        fig2 = plot_cnr_histogram(identical_values)
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

class TestHistogramOptions:
    """Parameter-file image keys become plot_accuracy_histogram keyword arguments."""

    def test_every_image_key_maps_to_its_keyword(self):
        import inspect
        from dice.visualization.histograms import histogram_options

        parameters = {
            'image type': 'svg', 'image width': 12, 'image height': 8, 'image dpi': 150,
            'image font size': 7, 'image tick length': 3, 'image tick width': 1.5,
            'image numbins': 11, 'image x_lim': [0.5, 1.5],
            'number of runs': 4,  # unrelated keys are ignored
        }

        options = histogram_options(parameters)

        assert options == {
            'image_type': 'svg', 'width': 12, 'height': 8, 'dpi': 150, 'font_size': 7,
            'tick_length': 3, 'tick_width': 1.5, 'num_bins': 11, 'x_lim': [0.5, 1.5],
        }
        accepted = set(inspect.signature(plot_accuracy_histogram).parameters)
        assert set(options) <= accepted

    def test_absent_and_none_keys_are_left_to_the_plot_defaults(self):
        from dice.visualization.histograms import histogram_options

        assert histogram_options({}) == {}
        assert histogram_options({'image numbins': None, 'image dpi': 200}) == {'dpi': 200}

    @pytest.mark.parametrize('x_lim', [
        [1.5, 0.5], [1.0, 1.0], [0.5], [0.5, 1.0, 1.5], 'wide', [0.5, 'high'], [0.5, float('nan')],
    ])
    def test_malformed_x_lim_is_rejected_by_name(self, x_lim):
        from dice.visualization.histograms import histogram_options

        with pytest.raises(ValueError, match="'image x_lim' must be two increasing finite numbers"):
            histogram_options({'image x_lim': x_lim})

    def test_options_reach_the_figure(self, tmp_path):
        from dice.visualization.histograms import histogram_options

        rng = np.random.default_rng(0)
        result = {'collated results': {'d_est_over_d_nom': rng.normal(1.0, 0.1, 200).tolist()}}
        options = histogram_options({'image numbins': 11, 'image x_lim': (0.6, 1.4), 'image font size': 7})

        fig = plot_accuracy_histogram(
            simulation_result=result, filename=str(tmp_path / 'options.png'), **options)

        ax = fig.axes[0]
        assert ax.get_xlim() == (0.6, 1.4)
        assert len(ax.patches) == 11
        assert ax.xaxis.label.get_fontsize() == 7
        plt.close(fig)


class TestImageType:
    """'image type' is checked against the formats matplotlib can write, before any simulation."""

    def test_documented_types_are_supported(self):
        from dice.visualization.histograms import supported_image_types
        assert {'png', 'jpg', 'svg', 'tif', 'pdf'} <= set(supported_image_types())

    @pytest.mark.parametrize('image_type', ['png', 'PNG', 'jpeg', 'eps', 'svgz'])
    def test_supported_type_is_returned_unchanged(self, image_type):
        from dice.visualization.histograms import validate_image_type
        assert validate_image_type(image_type) == image_type

    @pytest.mark.parametrize('image_type', ['bmp', 'gif', '', 5, None])
    def test_unsupported_type_is_rejected_with_the_supported_list(self, image_type):
        from dice.visualization.histograms import supported_image_types, validate_image_type
        with pytest.raises(ValueError) as error:
            validate_image_type(image_type)
        assert str(error.value) == (
            f"Unknown image type '{image_type}'. Valid types: {', '.join(supported_image_types())}")

    @pytest.mark.parametrize('image_type', ['png', 'jpg', 'svg', 'tif', 'pdf', 'eps'])
    def test_supported_type_is_written(self, tmp_path, image_type):
        """Every listed type the docs name gives a file, so the list matches what savefig writes."""
        result = {'collated results': {'d_est_over_d_nom': [0.9, 1.0, 1.1, 1.0]}}
        target = tmp_path / f'histogram.{image_type}'
        fig = plot_accuracy_histogram(simulation_result=result, filename=str(target), image_type=image_type)
        plt.close(fig)
        assert target.stat().st_size > 0



class TestAccuracyVsCnr:
    """A simulation with a noise range plots each run's D_est/D_nom against its initial CNR."""

    @staticmethod
    def _data(count=60):
        rng = np.random.default_rng(4)
        cnr = rng.uniform(20, 100, count)
        return cnr, 1 + rng.normal(0, 1, count) / cnr

    def test_points_reference_line_and_labels(self, tmp_path):
        from dice.visualization.histograms import plot_accuracy_vs_cnr
        cnr, ratios = self._data()
        target = tmp_path / 'scatter.png'
        fig = plot_accuracy_vs_cnr(cnr, ratios, str(target))

        ax = fig.axes[0]
        np.testing.assert_array_equal(ax.collections[0].get_offsets(), np.column_stack([cnr, ratios]))
        [reference] = ax.lines
        assert list(reference.get_ydata()) == [1.0, 1.0]
        assert (ax.get_xlabel(), ax.get_ylabel()) == ('Initial CNR', '$D_{est}/D_{nom}$')
        assert ax.get_xscale() == 'linear'
        assert ax.get_ylim() == pytest.approx((ratios.mean() - 3 * ratios.std(), ratios.mean() + 3 * ratios.std()))
        assert target.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'
        plt.close(fig)

    def test_logarithmic_cnr_axis(self, tmp_path):
        from dice.visualization.histograms import plot_accuracy_vs_cnr
        cnr, ratios = self._data()
        fig = plot_accuracy_vs_cnr(cnr, ratios, str(tmp_path / 'scatter.png'), log_cnr=True)
        assert fig.axes[0].get_xscale() == 'log'
        plt.close(fig)

    def test_non_finite_ratios_do_not_set_the_limits(self, tmp_path):
        from dice.visualization.histograms import plot_accuracy_vs_cnr
        cnr, ratios = self._data()
        with_nan = np.append(ratios, np.nan)
        fig = plot_accuracy_vs_cnr(np.append(cnr, 50.0), with_nan, str(tmp_path / 'scatter.png'))
        assert np.isfinite(fig.axes[0].get_ylim()).all()
        plt.close(fig)

    def test_lengths_must_match(self, tmp_path):
        from dice.visualization.histograms import plot_accuracy_vs_cnr
        with pytest.raises(ValueError, match="differ in length: 3 and 2"):
            plot_accuracy_vs_cnr([20, 30, 40], [1.0, 1.1], str(tmp_path / 'scatter.png'))

    def test_image_keys_apply_with_x_lim_on_the_accuracy_axis(self, tmp_path):
        import inspect
        from dice.visualization.histograms import accuracy_scatter_options, plot_accuracy_vs_cnr
        parameters = {
            'image type': 'svg', 'image width': 12, 'image height': 8, 'image dpi': 150,
            'image font size': 7, 'image tick length': 3, 'image tick width': 1.5,
            'image numbins': 11, 'image x_lim': [0.5, 1.5],
        }

        options = accuracy_scatter_options(parameters)

        assert options == {
            'image_type': 'svg', 'width': 12, 'height': 8, 'dpi': 150, 'font_size': 7,
            'tick_length': 3, 'tick_width': 1.5, 'y_lim': [0.5, 1.5],
        }
        assert set(options) <= set(inspect.signature(plot_accuracy_vs_cnr).parameters)
        cnr, ratios = self._data()
        target = tmp_path / 'scatter.svg'
        fig = plot_accuracy_vs_cnr(cnr, ratios, str(target), **options)
        assert fig.axes[0].get_ylim() == (0.5, 1.5)
        assert fig.axes[0].xaxis.label.get_fontsize() == 7
        assert target.read_text(encoding='utf-8').lstrip().startswith('<?xml')
        plt.close(fig)

    def test_malformed_x_lim_is_rejected_by_name(self):
        from dice.visualization.histograms import accuracy_scatter_options
        with pytest.raises(ValueError, match="'image x_lim' must be two increasing finite numbers"):
            accuracy_scatter_options({'image x_lim': [1.5, 0.5]})
