"""
Parameter dataclasses for DICE simulations.

This module defines dataclasses that organize and validate simulation parameters.
"""

from dataclasses import dataclass, field
from typing import Optional, Union, List, Literal, Dict, Any
import warnings
import numpy as np


@dataclass
class GaussianParameters:
    """
    Parameters defining the initial Gaussian profile.
    
    Attributes
    ----------
    amplitude : float
        Initial amplitude of the Gaussian (typically 1.0).
    sigma2 : float
        Initial variance (sigma^2) of the Gaussian.
    mu : float
        Mean position of the Gaussian (typically 0.0).
    """
    amplitude: float = 1.0
    sigma2: float = 1.0
    mu: float = 0.0
    
    def __post_init__(self):
        if self.amplitude < 0:
            raise ValueError("Amplitude must be non-negative")
        if self.sigma2 <= 0:
            raise ValueError("Variance (sigma2) must be positive")
    
    @classmethod
    def from_fwhm(cls, fwhm: float, amplitude: float = 1.0, mu: float = 0.0):
        """Create from FWHM instead of sigma^2."""
        from ..utils.converters import fwhm_to_sigma2
        return cls(amplitude=amplitude, sigma2=fwhm_to_sigma2(fwhm), mu=mu)
    
    @classmethod
    def from_sigma(cls, sigma: float, amplitude: float = 1.0, mu: float = 0.0):
        """Create from sigma instead of sigma^2."""
        return cls(amplitude=amplitude, sigma2=sigma**2, mu=mu)
    
    @property
    def fwhm(self) -> float:
        """Get FWHM from sigma^2."""
        from ..utils.converters import sigma2_to_fwhm
        return sigma2_to_fwhm(self.sigma2)
    
    @property
    def sigma(self) -> float:
        """Get sigma from sigma^2."""
        return np.sqrt(self.sigma2)

    @classmethod
    def from_legacy(cls, params: Dict[str, Any]) -> 'GaussianParameters':
        """
        Create GaussianParameters from legacy parameter dictionary.

        Parameters
        ----------
        params : dict
            Legacy parameters dictionary. Can contain keys in either
            legacy format (e.g., 'sigma^2_0') or canonical format.

        Returns
        -------
        GaussianParameters
            New instance with values from the dictionary.
        """
        from .parameter_keys import normalize_parameters
        normalized = normalize_parameters(params)

        # Handle sigma2 from various sources
        if 'sigma2_0' in normalized:
            sigma2 = normalized['sigma2_0']
        elif 'sigma_0' in normalized:
            sigma2 = normalized['sigma_0'] ** 2
        elif 'fwhm_0' in normalized:
            from ..utils.converters import fwhm_to_sigma2
            sigma2 = fwhm_to_sigma2(normalized['fwhm_0'])
        else:
            sigma2 = 1.0

        return cls(
            amplitude=normalized.get('amplitude_0', 1.0),
            sigma2=sigma2,
            mu=normalized.get('mu_0', 0.0)
        )

    # Deprecated property aliases for legacy compatibility
    @property
    def sigma2_0(self) -> float:
        """Legacy alias for sigma2. Deprecated."""
        warnings.warn(
            "sigma2_0 is deprecated, use sigma2 instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.sigma2

    @property
    def amplitude_0(self) -> float:
        """Legacy alias for amplitude. Deprecated."""
        warnings.warn(
            "amplitude_0 is deprecated, use amplitude instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.amplitude

    @property
    def mu_0(self) -> float:
        """Legacy alias for mu. Deprecated."""
        warnings.warn(
            "mu_0 is deprecated, use mu instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.mu


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

    @classmethod
    def from_legacy(cls, params: Dict[str, Any]) -> 'NoiseParameters':
        """
        Create NoiseParameters from legacy parameter dictionary.

        Parameters
        ----------
        params : dict
            Legacy parameters dictionary with noise configuration.

        Returns
        -------
        NoiseParameters
            New instance with values from the dictionary.
        """
        from .parameter_keys import normalize_parameters
        normalized = normalize_parameters(params)

        if 'noise_value' in normalized:
            return cls(mode='single', value=normalized['noise_value'])
        elif 'estimate_noise_from_data' in normalized:
            return cls(mode='estimate', data_file=normalized['estimate_noise_from_data'])
        elif 'noise_range_reciprocal' in normalized:
            range_val = normalized['noise_range_reciprocal']
            return cls(mode='range', range=tuple(range_val[:2]),
                       num_values=int(range_val[2]) if len(range_val) > 2 else 1)
        elif 'noise_range_reciprocal_log' in normalized:
            range_val = normalized['noise_range_reciprocal_log']
            return cls(mode='range', range=tuple(range_val[:2]),
                       num_values=int(range_val[2]) if len(range_val) > 2 else 1,
                       logarithmic=True)
        else:
            # Default to single mode with default value
            return cls(mode='single', value=0.01)


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
        """Calculate the size of each pixel."""
        return self.width / self.pixels
    
    def make_axis(self) -> np.ndarray:
        """Generate the spatial axis array."""
        from ..utils.axes import make_x_axis
        return make_x_axis(self.width, self.pixels, self.center)

    @classmethod
    def from_legacy(cls, params: Dict[str, Any]) -> 'SpatialParameters':
        """
        Create SpatialParameters from legacy parameter dictionary.

        Parameters
        ----------
        params : dict
            Legacy parameters dictionary with spatial axis configuration.

        Returns
        -------
        SpatialParameters
            New instance with values from the dictionary.
        """
        from .parameter_keys import normalize_parameters
        normalized = normalize_parameters(params)

        return cls(
            width=normalized['spatial_width'],
            pixels=int(normalized['pixel_width']),
            center=normalized.get('mu_0', 0.0)
        )


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

    @classmethod
    def from_legacy(cls, params: Dict[str, Any]) -> 'TemporalParameters':
        """
        Create TemporalParameters from legacy parameter dictionary.

        Parameters
        ----------
        params : dict
            Legacy parameters dictionary with temporal axis configuration.

        Returns
        -------
        TemporalParameters
            New instance with values from the dictionary.
        """
        from .parameter_keys import normalize_parameters
        normalized = normalize_parameters(params)

        if 'time_series' in normalized:
            return cls(mode='series', series=np.asarray(normalized['time_series']))
        elif 'time_range' in normalized:
            time_range = normalized['time_range']
            return cls(
                mode='range',
                start=time_range[0],
                end=time_range[1],
                frames=int(time_range[2])
            )
        else:
            raise ValueError("Either 'time_series' or 'time_range' must be specified")


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
        valid_image_types = ['png', 'svg', 'jpg', 'tif', 'pdf']
        if self.image_type not in valid_image_types:
            raise ValueError(f"Image type must be one of {valid_image_types}")
        
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("Image dimensions must be positive")
        
        if self.image_dpi <= 0:
            raise ValueError("DPI must be positive")
        
        if self.num_bins <= 0:
            raise ValueError("Number of bins must be positive")

    @classmethod
    def from_legacy(cls, params: Dict[str, Any]) -> 'OutputParameters':
        """
        Create OutputParameters from legacy parameter dictionary.

        Parameters
        ----------
        params : dict
            Legacy parameters dictionary with output configuration.

        Returns
        -------
        OutputParameters
            New instance with values from the dictionary.
        """
        from .parameter_keys import normalize_parameters
        normalized = normalize_parameters(params)

        return cls(
            filename_slug=normalized.get('filename_slug', 'dice_output'),
            image_type=normalized.get('image_type', 'png'),
            image_width=normalized.get('image_width', 15.0),
            image_height=normalized.get('image_height', 10.0),
            image_dpi=normalized.get('image_dpi', 300),
            font_size=normalized.get('image_font_size', 12),
            num_bins=normalized.get('image_numbins', 50),
            x_limits=normalized.get('image_x_lim'),
            retain_profiles=normalized.get('retain_profile_data', False)
        )


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
    proximity_level : float
        Proximity level for accuracy assessment (e.g., 0.1 for ±10%).
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
    proximity_level: float = 0.1
    length_unit: str = "micrometer"
    time_unit: str = "nanosecond"
    multiprocessing: bool = True
    
    def __post_init__(self):
        if self.num_runs <= 0:
            raise ValueError("Number of runs must be positive")
        if self.diffusion_coefficient < 0:
            raise ValueError("Diffusion coefficient must be non-negative")
        if self.lifetime < 0:
            raise ValueError("Lifetime must be non-negative")
        if not 0 < self.proximity_level < 1:
            raise ValueError("Proximity level must be between 0 and 1")
    
    @property
    def diffusion_length(self) -> float:
        """Calculate the diffusion length √(D*τ)."""
        return np.sqrt(self.diffusion_coefficient * self.lifetime)
    
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

    @classmethod
    def from_legacy(cls, params: Dict[str, Any]) -> 'SimulationParameters':
        """
        Create SimulationParameters from legacy dictionary format.

        This method handles all legacy key naming conventions including
        space-separated keys, special characters, and alternative names.

        Parameters
        ----------
        params : dict
            Legacy parameters dictionary with any mix of legacy and
            canonical key formats.

        Returns
        -------
        SimulationParameters
            New instance with values from the dictionary.

        Examples
        --------
        >>> params = {
        ...     'number of runs': 100,
        ...     'nominal diffusion length': 1.0,
        ...     'sigma^2_0': 1.0,
        ...     'amplitude_0': 1.0,
        ...     'mean_0': 0.0,
        ...     'spatial width': 10.0,
        ...     'pixel width': 100,
        ...     'time range': (0, 5, 6),
        ...     'noise value': 0.01
        ... }
        >>> sim_params = SimulationParameters.from_legacy(params)
        """
        from .parameter_keys import normalize_parameters
        normalized = normalize_parameters(params)

        # Build nested parameter objects
        gaussian = GaussianParameters.from_legacy(normalized)
        noise = NoiseParameters.from_legacy(normalized)
        spatial = SpatialParameters.from_legacy(normalized)
        temporal = TemporalParameters.from_legacy(normalized)
        output = OutputParameters.from_legacy(normalized)

        # Extract common parameters
        num_runs = normalized['number_of_runs']
        proximity_level = normalized.get('proximity_level', 0.1)
        length_unit = normalized.get('length_unit', 'micrometer')
        time_unit = normalized.get('time_unit', 'nanosecond')
        multiprocessing = normalized.get('multiprocessing', True)

        # Extract diffusion parameters - either from diffusion_length or coefficient+lifetime
        if 'diffusion_length' in normalized:
            # Derive coefficient from diffusion length
            # Need lifetime - use default if not provided
            lifetime = normalized.get('lifetime', 1.0)
            return cls.from_diffusion_length(
                diffusion_length=normalized['diffusion_length'],
                lifetime=lifetime,
                num_runs=num_runs,
                gaussian=gaussian,
                noise=noise,
                spatial=spatial,
                temporal=temporal,
                output=output,
                proximity_level=proximity_level,
                length_unit=length_unit,
                time_unit=time_unit,
                multiprocessing=multiprocessing
            )
        else:
            return cls(
                num_runs=num_runs,
                diffusion_coefficient=normalized['diffusion_coefficient'],
                lifetime=normalized['lifetime'],
                gaussian=gaussian,
                noise=noise,
                spatial=spatial,
                temporal=temporal,
                output=output,
                proximity_level=proximity_level,
                length_unit=length_unit,
                time_unit=time_unit,
                multiprocessing=multiprocessing
            )