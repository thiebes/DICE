"""
Profile data structures for DICE simulations.

This module defines dataclasses for organizing spatial profile data
at different time points during diffusion simulations.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union
import numpy as np


@dataclass
class GaussianProfile:
    """
    A single Gaussian profile at a specific time point.
    
    Attributes
    ----------
    time : float
        Time point for this profile.
    x_axis : np.ndarray
        Spatial coordinate array.
    y_values : np.ndarray
        Profile intensity values.
    amplitude : float
        Gaussian amplitude.
    sigma2 : float
        Gaussian variance.
    mu : float
        Gaussian mean position.
    integrated_intensity : float
        Total intensity (area under curve).
    noise_free : bool
        Whether this is a nominal (noise-free) profile.
    """
    time: float
    x_axis: np.ndarray
    y_values: np.ndarray
    amplitude: float
    sigma2: float
    mu: float
    integrated_intensity: Optional[float] = None
    noise_free: bool = True
    
    def __post_init__(self):
        # Validate array shapes
        if self.x_axis.shape != self.y_values.shape:
            raise ValueError("x_axis and y_values must have same shape")
        
        # Calculate integrated intensity if not provided
        if self.integrated_intensity is None:
            self.integrated_intensity = self.amplitude * np.sqrt(2 * np.pi * self.sigma2)
    
    @property
    def fwhm(self) -> float:
        """Calculate FWHM from sigma^2."""
        from ..utils.converters import sigma2_to_fwhm
        return sigma2_to_fwhm(self.sigma2)
    
    @property
    def sigma(self) -> float:
        """Calculate sigma from sigma^2."""
        return np.sqrt(self.sigma2)
    
    @property
    def cnr(self) -> Optional[float]:
        """
        Calculate contrast-to-noise ratio if noisy.
        
        Returns None for noise-free profiles.
        """
        if self.noise_free:
            return None
        # Would need noise estimation logic here
        return None
    
    def add_noise(self, noise_sigma: float) -> 'GaussianProfile':
        """
        Create a noisy version of this profile.
        
        Parameters
        ----------
        noise_sigma : float
            Standard deviation of Gaussian noise to add.
        
        Returns
        -------
        GaussianProfile
            New profile with added noise.
        """
        rng = np.random.default_rng()
        noise = rng.normal(0, noise_sigma, size=self.y_values.shape)
        noisy_y = self.y_values + noise
        
        return GaussianProfile(
            time=self.time,
            x_axis=self.x_axis,
            y_values=noisy_y,
            amplitude=self.amplitude,  # Nominal values
            sigma2=self.sigma2,
            mu=self.mu,
            integrated_intensity=self.integrated_intensity,
            noise_free=False
        )


@dataclass
class ProfileData:
    """
    Container for profile data from a single simulation run.
    
    Attributes
    ----------
    nominal_profiles : List[GaussianProfile]
        Noise-free theoretical profiles at each time point.
    noisy_profiles : List[GaussianProfile]
        Profiles with added noise.
    fitted_profiles : List[GaussianProfile]
        Profiles fitted to noisy data.
    noise_sigma : float
        Noise standard deviation used.
    """
    nominal_profiles: List[GaussianProfile]
    noisy_profiles: List[GaussianProfile]
    fitted_profiles: List[GaussianProfile]
    noise_sigma: float
    
    def __post_init__(self):
        # Validate that all lists have same length
        n_nominal = len(self.nominal_profiles)
        n_noisy = len(self.noisy_profiles)
        n_fitted = len(self.fitted_profiles)
        
        if not (n_nominal == n_noisy == n_fitted):
            raise ValueError(
                f"Profile lists must have same length: "
                f"nominal={n_nominal}, noisy={n_noisy}, fitted={n_fitted}"
            )
    
    @property
    def num_timepoints(self) -> int:
        """Number of time points."""
        return len(self.nominal_profiles)
    
    @property
    def time_axis(self) -> np.ndarray:
        """Extract time axis from profiles."""
        return np.array([p.time for p in self.nominal_profiles])
    
    @property
    def nominal_sigma2_t(self) -> np.ndarray:
        """Extract nominal variance evolution."""
        return np.array([p.sigma2 for p in self.nominal_profiles])
    
    @property
    def fitted_sigma2_t(self) -> np.ndarray:
        """Extract fitted variance evolution."""
        return np.array([p.sigma2 for p in self.fitted_profiles])
    
    @property
    def msd_t(self) -> np.ndarray:
        """Calculate mean squared displacement from fitted profiles."""
        sigma2_0 = self.fitted_profiles[0].sigma2
        return self.fitted_sigma2_t - sigma2_0
    
    def get_profile_at_time(self, time: float, 
                           profile_type: str = 'nominal') -> Optional[GaussianProfile]:
        """
        Get profile at specific time point.
        
        Parameters
        ----------
        time : float
            Time point to retrieve.
        profile_type : str
            Type of profile ('nominal', 'noisy', or 'fitted').
        
        Returns
        -------
        GaussianProfile or None
            Profile at requested time, or None if not found.
        """
        if profile_type == 'nominal':
            profiles = self.nominal_profiles
        elif profile_type == 'noisy':
            profiles = self.noisy_profiles
        elif profile_type == 'fitted':
            profiles = self.fitted_profiles
        else:
            raise ValueError(f"Invalid profile_type: {profile_type}")
        
        for profile in profiles:
            if np.isclose(profile.time, time):
                return profile
        
        return None


@dataclass
class TimeSeriesProfile:
    """
    Time series representation of profile evolution.
    
    This is an alternative representation that stores the evolution
    as 2D arrays rather than lists of profiles.
    
    Attributes
    ----------
    x_axis : np.ndarray
        Spatial coordinate array (1D).
    time_axis : np.ndarray
        Time coordinate array (1D).
    y_values : np.ndarray
        Profile values (2D: time x space).
    parameters : Dict
        Dictionary of time-dependent parameters.
    """
    x_axis: np.ndarray
    time_axis: np.ndarray
    y_values: np.ndarray
    parameters: dict = field(default_factory=dict)
    
    def __post_init__(self):
        # Validate shapes
        n_time = len(self.time_axis)
        n_space = len(self.x_axis)
        
        if self.y_values.shape != (n_time, n_space):
            raise ValueError(
                f"y_values shape {self.y_values.shape} doesn't match "
                f"expected ({n_time}, {n_space})"
            )
    
    @property
    def num_timepoints(self) -> int:
        """Number of time points."""
        return len(self.time_axis)
    
    @property
    def num_spatial_points(self) -> int:
        """Number of spatial points."""
        return len(self.x_axis)
    
    def get_profile(self, time_index: int) -> np.ndarray:
        """
        Get profile at specific time index.
        
        Parameters
        ----------
        time_index : int
            Index of time point.
        
        Returns
        -------
        np.ndarray
            Profile values at that time.
        """
        return self.y_values[time_index, :]
    
    def to_profile_list(self) -> List[GaussianProfile]:
        """
        Convert to list of GaussianProfile objects.
        
        Returns
        -------
        List[GaussianProfile]
            List of profiles at each time point.
        """
        profiles = []
        
        for i, t in enumerate(self.time_axis):
            # Extract parameters if available
            amp = self.parameters.get('amplitude_t', [None] * len(self.time_axis))[i]
            sig2 = self.parameters.get('sigma2_t', [None] * len(self.time_axis))[i]
            mu = self.parameters.get('mu_t', [0.0] * len(self.time_axis))[i]
            
            profile = GaussianProfile(
                time=t,
                x_axis=self.x_axis,
                y_values=self.y_values[i, :],
                amplitude=amp if amp is not None else np.max(self.y_values[i, :]),
                sigma2=sig2 if sig2 is not None else 1.0,
                mu=mu,
            )
            profiles.append(profile)
        
        return profiles