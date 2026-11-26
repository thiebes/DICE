"""
Parameter dataclasses for DICE simulations.

This module defines dataclasses that organize and validate simulation parameters.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
import numpy as np


@dataclass
class OutputUnitPreferences:
    """
    Units for output display.

    When a field is None, the output uses the global simulation unit or
    the default output convention (cm^2/s for diffusion coefficients).

    Attributes
    ----------
    length : str, optional
        Length unit for output display.
    time : str, optional
        Time unit for output display.
    diffusion_length : str, optional
        Length unit for diffusion coefficient output (defaults to 'centimeter').
    diffusion_time : str, optional
        Time unit for diffusion coefficient output (defaults to 'second').
    """
    length: Optional[str] = None
    time: Optional[str] = None
    diffusion_length: Optional[str] = None
    diffusion_time: Optional[str] = None


@dataclass
class GaussianParameters:
    """
    Parameters defining the initial Gaussian profile.

    Attributes
    ----------
    amplitude_0 : float
        Initial amplitude of the Gaussian (typically 1.0).
    sigma2_0 : float
        Initial variance (sigma^2) of the Gaussian at t=0.
    mu_0 : float
        Mean position of the Gaussian (typically 0.0).
    """
    amplitude_0: float = 1.0
    sigma2_0: float = 1.0
    mu_0: float = 0.0

    def __post_init__(self):
        if self.amplitude_0 < 0:
            raise ValueError("Amplitude must be non-negative")
        if self.sigma2_0 <= 0:
            raise ValueError("Variance (sigma2_0) must be positive")

    @classmethod
    def from_fwhm(cls, fwhm: float, amplitude_0: float = 1.0, mu_0: float = 0.0):
        """Create from FWHM instead of sigma^2."""
        from ..utils.converters import fwhm_to_sigma2
        return cls(amplitude_0=amplitude_0, sigma2_0=fwhm_to_sigma2(fwhm), mu_0=mu_0)

    @classmethod
    def from_sigma(cls, sigma: float, amplitude_0: float = 1.0, mu_0: float = 0.0):
        """Create from sigma instead of sigma^2."""
        return cls(amplitude_0=amplitude_0, sigma2_0=sigma**2, mu_0=mu_0)

    @property
    def fwhm(self) -> float:
        """Get FWHM from sigma^2."""
        from ..utils.converters import sigma2_to_fwhm
        return sigma2_to_fwhm(self.sigma2_0)

    @property
    def sigma(self) -> float:
        """Get sigma from sigma^2."""
        return np.sqrt(self.sigma2_0)


@dataclass
class NoiseParameters:
    """
    Parameters for noise in the simulation.
    
    Attributes
    ----------
    mode : str
        Type of noise specification ('single', 'range', 'estimate').
    value : float, optional
        Single noise standard deviation value.
    range : tuple, optional
        Range of noise values (min, max).
    data_file : str, optional
        Path to file for noise estimation.
    num_values : int
        Number of different noise values to simulate.
    logarithmic : bool
        Whether to use logarithmic spacing for noise range.
    """
    mode: Literal['single', 'range', 'estimate'] = 'single'
    value: Optional[float] = None
    range: Optional[tuple] = None
    data_file: Optional[str] = None
    num_values: int = 1
    logarithmic: bool = False
    
    def __post_init__(self):
        if self.mode == 'single' and self.value is None:
            raise ValueError("Value required for single noise mode")
        if self.mode == 'range' and self.range is None:
            raise ValueError("Range required for range noise mode")
        if self.mode == 'estimate' and self.data_file is None:
            raise ValueError("Data file required for estimate noise mode")
        
        if self.value is not None and self.value < 0:
            raise ValueError("Noise value must be non-negative")
        
        if self.range is not None:
            if len(self.range) != 2:
                raise ValueError("Range must be a tuple of (min, max)")
            if self.range[0] < 0 or self.range[1] < 0:
                raise ValueError("Noise range values must be non-negative")
            if self.range[0] >= self.range[1]:
                raise ValueError("Range min must be less than max")


@dataclass(frozen=True)
class NoiseRange:
    """
    Range of noise standard deviations from which each run draws its own value.

    The reciprocal 1/sigma, which is the initial CNR for an amplitude of 1, is
    uniformly distributed between 1/high and 1/low. With logarithmic=True,
    its base-10 logarithm is uniformly distributed instead.

    Attributes
    ----------
    low : float
        Smallest noise standard deviation (the highest CNR).
    high : float
        Largest noise standard deviation (the lowest CNR).
    logarithmic : bool
        Whether log10(1/sigma), rather than 1/sigma, is uniformly distributed.

    Raises
    ------
    ValueError
        If a bound is not a positive, finite number, or low is not less than high.
    """
    low: float
    high: float
    logarithmic: bool = False

    def __post_init__(self):
        bounds = [self.low, self.high]
        if not all(np.isfinite(bound) and bound > 0 for bound in bounds):
            raise ValueError(f"Invalid noise range {bounds}. Both values must be positive and finite.")
        if self.low >= self.high:
            raise ValueError(
                f"Invalid noise range {bounds}. The first value (low) must be less than the second (high)."
            )


@dataclass
class SpatialParameters:
    """
    Spatial axis parameters.
    
    Attributes
    ----------
    width : float
        Total width of the spatial axis in physical units.
    pixels : int
        Number of pixels (points) in the spatial axis.
    center : float
        Center position of the spatial axis.
    """
    width: float
    pixels: int
    center: float = 0.0
    
    def __post_init__(self):
        if self.width <= 0:
            raise ValueError("Spatial width must be positive")
        if self.pixels <= 0:
            raise ValueError("Number of pixels must be positive")
    
    @property
    def pixel_size(self) -> float:
        """The size of each pixel: the spacing of the spatial axis."""
        from ..utils.converters import calculate_pixel_size
        return calculate_pixel_size(self.width, self.pixels)
    
    def make_axis(self) -> np.ndarray:
        """Generate the spatial axis array."""
        from ..utils.axes import make_x_axis
        return make_x_axis(self.width, self.pixels, self.center)


@dataclass
class TemporalParameters:
    """
    Temporal axis parameters.
    
    Attributes
    ----------
    mode : str
        Type of time specification ('range' or 'series').
    start : float, optional
        Start time for range mode.
    end : float, optional
        End time for range mode.
    frames : int, optional
        Number of time frames for range mode.
    series : np.ndarray, optional
        Explicit time points for series mode.
    """
    mode: Literal['range', 'series'] = 'range'
    start: Optional[float] = None
    end: Optional[float] = None
    frames: Optional[int] = None
    series: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.mode == 'range':
            if any(x is None for x in [self.start, self.end, self.frames]):
                raise ValueError("Start, end, and frames required for range mode")
            if self.start >= self.end:
                raise ValueError("Start time must be less than end time")
            if self.frames <= 0:
                raise ValueError("Number of frames must be positive")
        elif self.mode == 'series':
            if self.series is None:
                raise ValueError("Time series required for series mode")
            self.series = np.asarray(self.series)
            if self.series.size == 0:
                raise ValueError("Time series cannot be empty")
    
    def make_axis(self) -> np.ndarray:
        """Generate the time axis array."""
        if self.mode == 'range':
            from ..utils.axes import make_time_axis
            return make_time_axis(self.start, self.end, self.frames)
        else:
            from ..utils.axes import make_time_series
            return make_time_series(self.series)


@dataclass
class OutputParameters:
    """
    Output and visualization parameters.
    
    Attributes
    ----------
    filename_slug : str
        Prefix for output files.
    image_type : str
        File format for plots (e.g., 'png', 'svg').
    image_width : float
        Width of output image in cm.
    image_height : float
        Height of output image in cm.
    image_dpi : int
        Resolution in dots per inch.
    font_size : int
        Font size for plot text.
    num_bins : int
        Number of bins for histograms.
    x_limits : Optional[tuple]
        X-axis limits for plots.
    retain_profiles : bool
        Whether to keep all profile data in memory.
    """
    filename_slug: str = "dice_output"
    image_type: str = "png"
    image_width: float = 15.0
    image_height: float = 10.0
    image_dpi: int = 300
    font_size: int = 12
    num_bins: int = 50
    x_limits: Optional[tuple] = None
    retain_profiles: bool = False
    
    def __post_init__(self):
        # Same rule as the run-time check. The common types skip the lookup,
        # which imports matplotlib, so constructing parameters stays light.
        if str(self.image_type).lower() not in ('png', 'svg', 'pdf', 'jpg', 'jpeg', 'tif', 'tiff'):
            from ..visualization.histograms import validate_image_type
            validate_image_type(self.image_type)
        
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("Image dimensions must be positive")
        
        if self.image_dpi <= 0:
            raise ValueError("DPI must be positive")
        
        if self.num_bins <= 0:
            raise ValueError("Number of bins must be positive")


@dataclass
class SimulationParameters:
    """
    Complete set of parameters for a DICE simulation.
    
    Attributes
    ----------
    num_runs : int
        Number of Monte Carlo simulation runs.
    diffusion_coefficient : float
        Nominal diffusion coefficient in length^2/time units.
    lifetime : float
        Nominal excited state lifetime (tau).
    gaussian : GaussianParameters
        Initial Gaussian profile parameters.
    noise : NoiseParameters
        Noise parameters.
    spatial : SpatialParameters
        Spatial axis parameters.
    temporal : TemporalParameters
        Temporal axis parameters.
    output : OutputParameters
        Output and visualization parameters.
    length_unit : str
        Unit of length (e.g., 'micrometer').
    time_unit : str
        Unit of time (e.g., 'nanosecond').
    multiprocessing : bool
        Whether to use parallel processing.
    """
    num_runs: int
    diffusion_coefficient: float
    lifetime: float
    gaussian: GaussianParameters
    noise: NoiseParameters
    spatial: SpatialParameters
    temporal: TemporalParameters
    output: OutputParameters = field(default_factory=OutputParameters)
    length_unit: str = "micrometer"
    time_unit: str = "nanosecond"
    multiprocessing: bool = True
    output_units: OutputUnitPreferences = field(default_factory=OutputUnitPreferences)

    def __post_init__(self):
        if self.num_runs <= 0:
            raise ValueError("Number of runs must be positive")
        if self.diffusion_coefficient < 0:
            raise ValueError("Diffusion coefficient must be non-negative")
        if self.lifetime < 0:
            raise ValueError("Lifetime must be non-negative")
    
    @property
    def diffusion_length(self) -> float:
        """Calculate the diffusion length √(D*τ); infinite for τ = 0 (no decay)."""
        from ..core.diffusion import calculate_diffusion_length
        return calculate_diffusion_length(self.diffusion_coefficient, self.lifetime)
    
    @classmethod
    def from_diffusion_length(cls, diffusion_length: float,
                             lifetime: float, **kwargs):
        """
        Create parameters from diffusion length instead of coefficient.

        Parameters
        ----------
        diffusion_length : float
            Diffusion length √(D*τ).
        lifetime : float
            Excited state lifetime.
        **kwargs
            Other parameters passed to constructor.
        """
        diff_coeff = (diffusion_length ** 2) / lifetime if lifetime > 0 else 0
        return cls(diffusion_coefficient=diff_coeff, lifetime=lifetime, **kwargs)
