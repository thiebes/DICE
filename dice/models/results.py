"""
Result dataclasses for DICE simulations.

This module defines dataclasses for organizing simulation results,
including individual run results and aggregated statistics.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd


@dataclass
class DiffusionFitResult:
    """
    Results from fitting MSD vs time to extract diffusion coefficient.
    
    Attributes
    ----------
    slope : float
        Slope of the linear fit (2*D in 1D).
    slope_error : float
        Standard error of the slope.
    intercept : float
        Y-intercept of the linear fit.
    intercept_error : float
        Standard error of the intercept.
    r_squared : float
        Coefficient of determination.
    diffusion_coefficient : float
        Calculated diffusion coefficient (slope/2).
    method : str
        Fitting method used ('OLS' or 'WLS').
    """
    slope: float
    slope_error: float
    intercept: float
    intercept_error: float
    r_squared: float
    diffusion_coefficient: float
    method: str = 'OLS'
    
    @property
    def relative_error(self) -> float:
        """Calculate relative error of the slope."""
        if self.slope != 0:
            return abs(self.slope_error / self.slope)
        return float('inf')


@dataclass
class RunResult:
    """
    Results from a single Monte Carlo simulation run.
    
    Attributes
    ----------
    run_number : int
        Identifier for this run.
    nominal_diffusion : float
        Input diffusion coefficient.
    nominal_lifetime : float
        Input lifetime.
    nominal_diffusion_length : float
        Input diffusion length.
    noise_sigma : float
        Noise standard deviation used.
    cnr_estimate : float
        Estimated contrast-to-noise ratio.
    sigma2_t : np.ndarray
        Fitted variance values at each time point.
    msd_t : np.ndarray
        Mean squared displacement at each time point.
    ols_fit : DiffusionFitResult
        Ordinary least squares fit result.
    wls_fit : Optional[DiffusionFitResult]
        Weighted least squares fit result.
    profiles : Optional[Dict]
        Raw profile data if retained.
    """
    run_number: int
    nominal_diffusion: float
    nominal_lifetime: float
    nominal_diffusion_length: float
    noise_sigma: float
    cnr_estimate: float
    sigma2_t: np.ndarray
    msd_t: np.ndarray
    ols_fit: DiffusionFitResult
    wls_fit: Optional[DiffusionFitResult] = None
    profiles: Optional[Dict] = None
    
    @property
    def ols_accuracy(self) -> float:
        """Calculate OLS estimate accuracy relative to nominal."""
        if self.nominal_diffusion > 0:
            return self.ols_fit.diffusion_coefficient / self.nominal_diffusion
        return float('nan')
    
    @property
    def wls_accuracy(self) -> float:
        """Calculate WLS estimate accuracy relative to nominal."""
        if self.wls_fit and self.nominal_diffusion > 0:
            return self.wls_fit.diffusion_coefficient / self.nominal_diffusion
        return float('nan')


@dataclass
class StatisticalAnalysis:
    """
    Statistical analysis of simulation results.
    
    Attributes
    ----------
    proximity_level : float
        The proximity level used for analysis (e.g., 0.1 for ±10%).
    fraction_within_proximity : float
        Fraction of estimates within proximity of nominal.
    mean_estimate : float
        Mean of all diffusion estimates.
    median_estimate : float
        Median of all diffusion estimates.
    std_estimate : float
        Standard deviation of all diffusion estimates.
    mean_relative_error : float
        Mean of D_est/D_nom ratios.
    percentiles : Dict[int, float]
        Key percentiles of the distribution.
    """
    proximity_level: float
    fraction_within_proximity: float
    mean_estimate: float
    median_estimate: float
    std_estimate: float
    mean_relative_error: float
    percentiles: Dict[int, float] = field(default_factory=dict)
    
    def __post_init__(self):
        # Add default percentiles if not provided
        if not self.percentiles:
            self.percentiles = {}
    
    def summary_string(self) -> str:
        """Generate a text summary of the analysis."""
        lines = [
            f"Statistical Analysis Summary",
            f"=" * 40,
            f"Proximity Level: ±{self.proximity_level*100:.1f}%",
            f"Fraction Within Proximity: {self.fraction_within_proximity:.3f}",
            f"",
            f"Diffusion Coefficient Estimates:",
            f"  Mean: {self.mean_estimate:.6e}",
            f"  Median: {self.median_estimate:.6e}",
            f"  Std Dev: {self.std_estimate:.6e}",
            f"  Mean Relative Error: {self.mean_relative_error:.3f}",
        ]
        
        if self.percentiles:
            lines.append("")
            lines.append("Percentiles:")
            for p, val in sorted(self.percentiles.items()):
                lines.append(f"  {p}th: {val:.6e}")
        
        return "\n".join(lines)


@dataclass
class SimulationResults:
    """
    Complete results from a DICE simulation.
    
    Attributes
    ----------
    parameters : Dict[str, Any]
        Input parameters used for the simulation.
    run_results : List[RunResult]
        Individual results from each Monte Carlo run.
    analysis : StatisticalAnalysis
        Statistical analysis of all runs.
    dataframe : pd.DataFrame
        Collated results in DataFrame format.
    filename_slug : str
        Base filename for output files.
    """
    parameters: Dict[str, Any]
    run_results: List[RunResult]
    analysis: StatisticalAnalysis
    dataframe: pd.DataFrame
    filename_slug: str
    
    def __post_init__(self):
        # Ensure consistency
        if len(self.run_results) == 0:
            raise ValueError("No run results provided")
        
        # Create dataframe if not provided
        if self.dataframe is None:
            self.dataframe = self._create_dataframe()
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame from run results."""
        data = []
        for run in self.run_results:
            row = {
                'run_number': run.run_number,
                'nominal_diffusion': run.nominal_diffusion,
                'nominal_lifetime': run.nominal_lifetime,
                'nominal_diffusion_length': run.nominal_diffusion_length,
                'noise_sigma': run.noise_sigma,
                'cnr_estimate': run.cnr_estimate,
                'ols_diffusion': run.ols_fit.diffusion_coefficient,
                'ols_slope': run.ols_fit.slope,
                'ols_slope_error': run.ols_fit.slope_error,
                'ols_intercept': run.ols_fit.intercept,
                'ols_r_squared': run.ols_fit.r_squared,
                'ols_accuracy': run.ols_accuracy,
            }
            
            if run.wls_fit:
                row.update({
                    'wls_diffusion': run.wls_fit.diffusion_coefficient,
                    'wls_slope': run.wls_fit.slope,
                    'wls_slope_error': run.wls_fit.slope_error,
                    'wls_intercept': run.wls_fit.intercept,
                    'wls_r_squared': run.wls_fit.r_squared,
                    'wls_accuracy': run.wls_accuracy,
                })
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def save_csv(self, filepath: Optional[str] = None):
        """Save results to CSV file."""
        if filepath is None:
            filepath = f"{self.filename_slug}_results.csv"
        self.dataframe.to_csv(filepath, index=False)
    
    def save_summary(self, filepath: Optional[str] = None):
        """Save summary to text file."""
        if filepath is None:
            filepath = f"{self.filename_slug}_summary.txt"
        
        with open(filepath, 'w') as f:
            f.write(self.analysis.summary_string())
            f.write("\n\n")
            f.write(f"Total runs: {len(self.run_results)}\n")
            f.write(f"Output files: {self.filename_slug}_*\n")
    
    def filter_by_cnr(self, min_cnr: float = None, 
                     max_cnr: float = None) -> 'SimulationResults':
        """
        Filter results by CNR range.
        
        Parameters
        ----------
        min_cnr : float, optional
            Minimum CNR value.
        max_cnr : float, optional
            Maximum CNR value.
        
        Returns
        -------
        SimulationResults
            New SimulationResults object with filtered data.
        """
        mask = np.ones(len(self.run_results), dtype=bool)
        
        if min_cnr is not None:
            cnrs = [run.cnr_estimate for run in self.run_results]
            mask &= np.array(cnrs) >= min_cnr
        
        if max_cnr is not None:
            cnrs = [run.cnr_estimate for run in self.run_results]
            mask &= np.array(cnrs) <= max_cnr
        
        filtered_runs = [run for i, run in enumerate(self.run_results) if mask[i]]
        
        # Recalculate analysis for filtered data
        # This would need the analysis calculation logic
        # For now, return with same analysis
        return SimulationResults(
            parameters=self.parameters,
            run_results=filtered_runs,
            analysis=self.analysis,  # Should recalculate
            dataframe=self.dataframe[mask],
            filename_slug=self.filename_slug + "_filtered"
        )


# Alias for compatibility
SimulationResult = SimulationResults