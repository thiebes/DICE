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
class RunResult:
    """
    Results from a single Monte Carlo simulation run.

    Attributes
    ----------
    run_id : int
        Identifier for this run.
    nominal_diffusion_coefficient : float
        Input diffusion coefficient.
    nominal_lifetime : float
        Input lifetime.
    nominal_diffusion_length : float
        Input diffusion length.
    nominal_sigma2_0 : float
        Input initial variance.
    noise_sigma : float
        Noise standard deviation used.
    cnr_0_estimate : float
        Estimated contrast-to-noise ratio at t=0.
    estimated_sigma2_0 : float, optional
        Estimated initial variance from fitting.
    ols_slope : float, optional
        OLS fit slope (2*D).
    ols_slope_stderr : float, optional
        Standard error of OLS slope.
    ols_intercept : float, optional
        OLS fit intercept.
    ols_intercept_stderr : float, optional
        Standard error of OLS intercept.
    wls_slope : float, optional
        WLS fit slope (2*D).
    wls_slope_stderr : float, optional
        Standard error of WLS slope.
    wls_intercept : float, optional
        WLS fit intercept.
    wls_intercept_stderr : float, optional
        Standard error of WLS intercept.
    nominal_profiles : np.ndarray, optional
        Nominal profile data if retained.
    noisy_profiles : np.ndarray, optional
        Noisy profile data if retained.
    fitted_sigma2_t : np.ndarray, optional
        Fitted variance values at each time point.
    fitted_sigma2_stderrs : np.ndarray, optional
        Standard errors of fitted variances.
    weights : np.ndarray, optional
        Weights used in WLS fitting.
    """
    # Core identifier
    run_id: int

    # Nominal physics parameters
    nominal_diffusion_coefficient: float
    nominal_lifetime: float
    nominal_diffusion_length: float
    nominal_sigma2_0: float

    # Noise and CNR
    noise_sigma: float
    cnr_0_estimate: float

    # Estimated values
    estimated_sigma2_0: Optional[float] = None

    # OLS fit results
    ols_slope: Optional[float] = None
    ols_slope_stderr: Optional[float] = None
    ols_intercept: Optional[float] = None
    ols_intercept_stderr: Optional[float] = None

    # WLS fit results
    wls_slope: Optional[float] = None
    wls_slope_stderr: Optional[float] = None
    wls_intercept: Optional[float] = None
    wls_intercept_stderr: Optional[float] = None

    # Optional profile data
    nominal_profiles: Optional[np.ndarray] = None
    noisy_profiles: Optional[np.ndarray] = None
    fitted_sigma2_t: Optional[np.ndarray] = None
    fitted_sigma2_stderrs: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None

    @property
    def ols_accuracy(self) -> float:
        """Calculate OLS estimate accuracy relative to nominal."""
        if self.ols_slope is not None and self.nominal_diffusion_coefficient > 0:
            return (self.ols_slope / 2) / self.nominal_diffusion_coefficient
        return float('nan')

    @property
    def wls_accuracy(self) -> float:
        """Calculate WLS estimate accuracy relative to nominal."""
        if self.wls_slope is not None and self.nominal_diffusion_coefficient > 0:
            return (self.wls_slope / 2) / self.nominal_diffusion_coefficient
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
class MonteCarloOutput:
    """
    Raw output from Monte Carlo simulation runs.

    Contains the direct results from run_monte_carlo_simulation() before
    any statistical analysis is performed.

    Attributes
    ----------
    parameters : Any
        Simulation parameters object (SimulationParameters or None for CSV-loaded data).
    run_results : List[RunResult]
        Individual results from each Monte Carlo run.
    num_runs : int
        Total number of simulation runs completed.
    noise_values : List[float]
        List of noise sigma values used in the simulation.
    """
    parameters: Any
    run_results: List[RunResult]
    num_runs: int
    noise_values: List[float] = field(default_factory=list)


@dataclass
class ProcessedSimulationResult:
    """
    Processed results from a DICE simulation with statistical analysis.

    Contains fully analyzed simulation data with DataFrame representation
    and statistical metrics, ready for export and reporting.

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
                'run_id': run.run_id,
                'nominal_diffusion_coefficient': run.nominal_diffusion_coefficient,
                'nominal_lifetime': run.nominal_lifetime,
                'nominal_diffusion_length': run.nominal_diffusion_length,
                'nominal_sigma2_0': run.nominal_sigma2_0,
                'noise_sigma': run.noise_sigma,
                'cnr_0_estimate': run.cnr_0_estimate,
                'estimated_sigma2_0': run.estimated_sigma2_0,
                'ols_slope': run.ols_slope,
                'ols_slope_stderr': run.ols_slope_stderr,
                'ols_intercept': run.ols_intercept,
                'ols_intercept_stderr': run.ols_intercept_stderr,
                'ols_accuracy': run.ols_accuracy,
            }

            if run.wls_slope is not None:
                row.update({
                    'wls_slope': run.wls_slope,
                    'wls_slope_stderr': run.wls_slope_stderr,
                    'wls_intercept': run.wls_intercept,
                    'wls_intercept_stderr': run.wls_intercept_stderr,
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
                     max_cnr: float = None) -> 'ProcessedSimulationResult':
        """
        Filter results by CNR range and recalculate analysis.

        Parameters
        ----------
        min_cnr : float, optional
            Minimum CNR value.
        max_cnr : float, optional
            Maximum CNR value.

        Returns
        -------
        ProcessedSimulationResult
            New ProcessedSimulationResult object with filtered data
            and recalculated statistical analysis.
        """
        from ..analysis.statistics import calculate_precision, calculate_accuracy_metrics

        mask = np.ones(len(self.run_results), dtype=bool)

        if min_cnr is not None or max_cnr is not None:
            cnrs = np.array([run.cnr_0_estimate for run in self.run_results])
            if min_cnr is not None:
                mask &= cnrs >= min_cnr
            if max_cnr is not None:
                mask &= cnrs <= max_cnr

        filtered_runs = [run for i, run in enumerate(self.run_results) if mask[i]]

        # Recalculate analysis for the filtered subset
        nominal_d = filtered_runs[0].nominal_diffusion_coefficient if filtered_runs else 0.0
        wls_d_estimates = np.array([
            r.wls_slope / 2 for r in filtered_runs if r.wls_slope is not None
        ])

        proximity_level = self.analysis.proximity_level
        if len(wls_d_estimates) > 0 and nominal_d > 0:
            precision = calculate_precision(wls_d_estimates, nominal_d, proximity_level)
            accuracy = calculate_accuracy_metrics(wls_d_estimates, nominal_d)
            new_analysis = StatisticalAnalysis(
                proximity_level=proximity_level,
                fraction_within_proximity=precision,
                mean_estimate=accuracy['mean'],
                median_estimate=accuracy['median'],
                std_estimate=accuracy['std'],
                mean_relative_error=accuracy['mean'] / nominal_d if nominal_d != 0 else np.nan,
            )
        else:
            new_analysis = StatisticalAnalysis(
                proximity_level=proximity_level,
                fraction_within_proximity=0.0,
                mean_estimate=np.nan,
                median_estimate=np.nan,
                std_estimate=np.nan,
                mean_relative_error=np.nan,
            )

        return ProcessedSimulationResult(
            parameters=self.parameters,
            run_results=filtered_runs,
            analysis=new_analysis,
            dataframe=self.dataframe[mask].reset_index(drop=True),
            filename_slug=self.filename_slug + "_filtered"
        )
