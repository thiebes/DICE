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
                'd_wls_over_d_nom': [0.95, 1.02, 0.98, 1.05, 0.92, 1.08, 0.97, 1.01]
            }
        }
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_filename = temp_file.name
        
        try:
            fig = plot_accuracy_histogram(
                simulation_result=result,
                proximity=0.1,
                filename=temp_filename,
                width=8.0,
                height=6.0
            )
            
            assert isinstance(fig, plt.Figure)
            ax = fig.axes[0]
            
            # Check basic plot elements
            assert ax.get_xlabel() == '$D_{est}/D_{nom}$'
            assert ax.get_ylabel() == 'Probability density'
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
        
        # Missing 'd_wls_over_d_nom'
        result2 = {'collated results': {}}
        
        with pytest.raises(ValueError, match="'d_wls_over_d_nom' not found"):
            plot_accuracy_histogram(result2, 0.1, "test.png")
    
    def test_custom_parameters(self):
        """Test with custom plotting parameters."""
        result = {
            'collated results': {
                'd_wls_over_d_nom': np.random.normal(1.0, 0.1, 100).tolist()
            }
        }
        
        # Create output directory for tests
        output_dir = Path.cwd() / 'output' / 'tests'
        output_dir.mkdir(parents=True, exist_ok=True)
        test_file = output_dir / "test_custom_parameters.png"
        
        try:
            fig = plot_accuracy_histogram(
                simulation_result=result,
                proximity=0.05,
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
            
        finally:
            # Clean up test file
            if test_file.exists():
                test_file.unlink()
    
    def test_perfect_data(self):
        """Test with perfect data (all values = 1.0)."""
        result = {
            'collated results': {
                'd_wls_over_d_nom': [1.0] * 50
            }
        }
        
        # Create output directory for tests
        output_dir = Path.cwd() / 'output' / 'tests'
        output_dir.mkdir(parents=True, exist_ok=True)
        test_file = output_dir / "test_perfect_data.png"
        
        try:
            fig = plot_accuracy_histogram(result, 0.1, str(test_file))
            
            assert isinstance(fig, plt.Figure)
            assert test_file.exists()
            plt.close(fig)
            
        finally:
            # Clean up test file
            if test_file.exists():
                test_file.unlink()


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
        assert ax.get_ylabel() == 'Probability Density'
        
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
        assert ax.get_ylabel() == 'Probability Density'
        
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