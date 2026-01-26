"""
Tests for dice.visualization.plots module.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for testing
import matplotlib.pyplot as plt
from dice.visualization.plots import (
    get_color_definitions,
    plot_gaussian_profile,
    plot_profile_evolution,
    plot_diffusion_msd,
    plot_cnr_dependence,
    close_figure,
    close_all_figures,
    configure_matplotlib_style,
)


class TestGetColorDefinitions:
    """Test color definitions function."""
    
    def test_color_definitions(self):
        """Test that color definitions are returned correctly."""
        colors = get_color_definitions()
        
        assert isinstance(colors, dict)
        assert 'dice_blue' in colors
        assert 'dice_gold' in colors
        assert 'dice_green' in colors
        assert 'dice_gradient' in colors
        
        # Check color values
        assert colors['dice_blue'] == '#003f7f'
        assert colors['dice_gold'] == '#f7941e'
        assert colors['dice_green'] == '#0cce6b'


class TestPlotGaussianProfile:
    """Test Gaussian profile plotting."""
    
    def test_basic_plot(self):
        """Test basic Gaussian profile plotting."""
        x = np.linspace(-5, 5, 100)
        y = np.exp(-x**2 / 2)
        
        fig = plot_gaussian_profile(x, y, title="Test Profile")
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        
        ax = fig.axes[0]
        assert ax.get_title() == "Test Profile"
        assert ax.get_xlabel() == "Position"
        assert ax.get_ylabel() == "Intensity"
        
        close_figure(fig)
    
    def test_custom_labels(self):
        """Test with custom labels."""
        x = np.linspace(0, 10, 50)
        y = np.ones_like(x)
        
        fig = plot_gaussian_profile(
            x, y, 
            title="Custom Title",
            xlabel="Distance (μm)",
            ylabel="Signal (counts)"
        )
        
        ax = fig.axes[0]
        assert ax.get_title() == "Custom Title"
        assert ax.get_xlabel() == "Distance (μm)"
        assert ax.get_ylabel() == "Signal (counts)"
        
        close_figure(fig)
    
    def test_plot_kwargs(self):
        """Test with additional plotting parameters."""
        x = np.linspace(-2, 2, 20)
        y = np.exp(-x**2)
        
        fig = plot_gaussian_profile(
            x, y,
            width=8.0,
            height=5.0,
            font_size=14
        )
        
        assert isinstance(fig, plt.Figure)
        close_figure(fig)


class TestPlotProfileEvolution:
    """Test profile evolution plotting."""
    
    def test_evolution_plot(self):
        """Test profile evolution plotting."""
        x = np.linspace(-10, 10, 101)
        t = np.array([0, 1, 2, 3])
        profiles = [np.exp(-x**2 / (2*(1 + 0.5*ti))) for ti in t]
        
        fig = plot_profile_evolution(x, t, profiles)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        
        ax = fig.axes[0]
        assert len(ax.lines) == len(profiles)
        
        close_figure(fig)
    
    def test_many_profiles(self):
        """Test with many profiles (no legend)."""
        x = np.linspace(-5, 5, 50)
        t = np.linspace(0, 20, 15)  # 15 profiles
        profiles = [np.exp(-x**2 / (2*(1 + 0.1*ti))) for ti in t]
        
        fig = plot_profile_evolution(x, t, profiles)
        
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert len(ax.lines) == len(profiles)
        
        close_figure(fig)
    
    def test_custom_units(self):
        """Test with custom time units."""
        x = np.linspace(0, 5, 25)
        t = np.array([0, 0.5, 1.0])
        profiles = [np.ones_like(x) * (1 - 0.2*ti) for ti in t]
        
        fig = plot_profile_evolution(
            x, t, profiles,
            time_unit="ps",
            xlabel="Position (μm)",
            ylabel="Photoluminescence"
        )
        
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Position (μm)"
        assert ax.get_ylabel() == "Photoluminescence"
        
        close_figure(fig)


class TestPlotDiffusionMSD:
    """Test diffusion MSD plotting."""
    
    def test_basic_msd_plot(self):
        """Test basic MSD plotting."""
        t = np.array([0, 1, 2, 3, 4])
        sigma2 = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        
        fig = plot_diffusion_msd(t, sigma2)
        
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert len(ax.lines) >= 1  # At least data points
        
        close_figure(fig)
    
    def test_msd_with_errors(self):
        """Test MSD plotting with error bars."""
        t = np.array([0, 1, 2, 3])
        sigma2 = np.array([1.0, 1.2, 1.4, 1.6])
        errors = np.array([0.1, 0.1, 0.1, 0.1])
        
        fig = plot_diffusion_msd(t, sigma2, sigma2_errors=errors)
        
        assert isinstance(fig, plt.Figure)
        close_figure(fig)
    
    def test_msd_with_fits(self):
        """Test MSD plotting with OLS and WLS fits."""
        t = np.array([0, 1, 2, 3, 4])
        sigma2 = 1.0 + 0.5 * t + 0.1 * np.random.randn(5)
        
        ols_fit = {'slope': 0.48, 'intercept': 0.02}
        wls_fit = {'slope': 0.52, 'intercept': -0.01}
        
        fig = plot_diffusion_msd(
            t, sigma2,
            ols_fit=ols_fit,
            wls_fit=wls_fit
        )
        
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert len(ax.lines) >= 3  # Data + 2 fits
        
        close_figure(fig)
    
    def test_msd_custom_units(self):
        """Test MSD plotting with custom units."""
        t = np.array([0, 2, 4])
        sigma2 = np.array([0.5, 1.0, 1.5])
        
        fig = plot_diffusion_msd(
            t, sigma2,
            time_unit="ps",
            length_unit="nm"
        )
        
        ax = fig.axes[0]
        assert "ps" in ax.get_xlabel()
        assert "nm" in ax.get_ylabel()
        
        close_figure(fig)


class TestPlotCNRDependence:
    """Test CNR dependence plotting."""
    
    def test_cnr_plot(self):
        """Test CNR dependence plotting."""
        cnr = np.array([5, 10, 20, 50, 100])
        precision = np.array([20, 40, 70, 90, 95])
        
        fig = plot_cnr_dependence(cnr, precision)
        
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert len(ax.lines) == 1
        assert ax.get_ylim()[1] == 100  # Y-axis goes to 100%
        
        close_figure(fig)
    
    def test_cnr_custom_labels(self):
        """Test CNR plotting with custom labels."""
        cnr = np.array([1, 2, 5])
        precision = np.array([10, 30, 80])
        
        fig = plot_cnr_dependence(
            cnr, precision,
            title="Custom CNR Analysis",
            xlabel="Signal-to-Noise Ratio",
            ylabel="Accuracy (%)"
        )
        
        ax = fig.axes[0]
        assert ax.get_title() == "Custom CNR Analysis"
        assert ax.get_xlabel() == "Signal-to-Noise Ratio"
        assert ax.get_ylabel() == "Accuracy (%)"
        
        close_figure(fig)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_close_figure(self):
        """Test figure closing."""
        fig = plt.figure()
        close_figure(fig)
        # If this doesn't raise an error, the function works
    
    def test_close_all_figures(self):
        """Test closing all figures."""
        # Create multiple figures
        fig1 = plt.figure()
        fig2 = plt.figure()
        
        close_all_figures()
        # If this doesn't raise an error, the function works
    
    def test_configure_matplotlib_style(self):
        """Test matplotlib style configuration."""
        configure_matplotlib_style(font_size=14, font_family='sans-serif')
        
        # Check that rcParams were updated
        assert plt.rcParams['font.size'] == 14
        assert plt.rcParams['font.family'] == ['sans-serif']


class TestEmptyData:
    """Test plotting with edge cases."""
    
    def test_empty_arrays(self):
        """Test with empty arrays."""
        x = np.array([])
        y = np.array([])
        
        # This should not crash
        fig = plot_gaussian_profile(x, y)
        assert isinstance(fig, plt.Figure)
        close_figure(fig)
    
    def test_single_point(self):
        """Test with single data point."""
        x = np.array([0])
        y = np.array([1])
        
        fig = plot_gaussian_profile(x, y)
        assert isinstance(fig, plt.Figure)
        close_figure(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])