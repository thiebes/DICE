# DICE Glossary

Definitions of the technical terms, abbreviations, and concepts used in DICE.

## A

### Amplitude

The peak height of a Gaussian profile. In DICE simulations, the initial amplitude is usually 1.

### Anisotropic Diffusion

Diffusion that proceeds at different rates in different directions. DICE models only isotropic (direction-independent) diffusion.

## B

### Bias

The mean of the diffusion coefficient estimates minus the nominal diffusion coefficient. A bias of zero means that the estimates are correct on average.

## C

### CNR (Contrast-to-Noise Ratio)

The amplitude of the signal above the baseline divided by the standard deviation of the noise. A higher CNR means a clearer signal.

**Formula:** CNR = amplitude / σ_noise

CNR is distinct from the signal-to-noise ratio (SNR), which has several definitions, for example the ratio of signal power to noise power.

### CNR Bin

A group of runs with similar initial CNR. In a simulation with a noise range, each run has its own initial CNR. DICE sorts the runs by initial CNR, splits them into up to five bins with equal numbers of runs, and reports the statistics of each bin in the summary file.

## D

### Diffusion Coefficient (D)

A measure of how fast particles spread through space by random motion. In DICE, it describes how fast the excited state population spreads.

**Units:** (length)² / time, for example μm²/ns or cm²/s

**Relationship:** L_D = √(Dτ)

### Diffusion Length (L_D)

The characteristic distance that an excited state travels before it decays.

**Formula:** L_D = √(Dτ)

where D is the diffusion coefficient and τ is the lifetime.

**Physical meaning:** for a point excitation in one dimension, about 63% of the excited states decay within L_D of their origin.

## E

### Excited State

An electronic state of a molecule or material with a higher energy than the ground state. DICE models the transport and decay of excited states.

### Excited State Transport

The movement of electronic excitations through a material, by diffusion, hopping, or other mechanisms.

## F

### FFT (Fast Fourier Transform)

An efficient algorithm for the discrete Fourier transform. DICE estimates the CNR of a profile with the [fft-cnr](https://pypi.org/project/fft-cnr/) package, which separates signal and noise by their frequency content. See the [fft-cnr documentation](https://fft-cnr.readthedocs.io).

### Fickian Diffusion

Normal diffusion, which follows Fick's laws. The mean squared displacement grows linearly with time (MSD ∝ t). DICE models normal diffusion only.

**See also:** Non-Fickian Diffusion

### FWHM (Full Width at Half Maximum)

The width of a peak at half of its maximum height. For a Gaussian profile, FWHM and standard deviation are related by:

**Formula:** FWHM = 2√(2 ln 2) σ ≈ 2.355 σ

## G

### Gaussian Profile

A bell-shaped distribution described by a Gaussian (normal) function. DICE models the excited state population as a Gaussian profile.

**Parameters:** amplitude (A), mean (μ), standard deviation (σ)

## L

### Lifetime (τ)

The characteristic time for an excited state to decay to the ground state.

**Definition:** for a single-exponential decay, the time in which the population decreases to 1/e (about 37%) of its initial value.

**Units:** time, for example ns or ps

## M

### Monte Carlo Simulation

A statistical method that uses repeated random sampling to obtain numerical results. DICE simulates many noisy measurements with the same parameters and analyzes the distribution of the resulting estimates.

### MSD (Mean Squared Displacement)

The average squared distance that a particle has moved from its starting position.

**Formula in one dimension:** MSD(t) = ⟨(x(t) − x(0))²⟩ = 2Dt

For a Gaussian profile, the MSD equals the change in variance: MSD(t) = σ²(t) − σ²(0).

## N

### Noise (White)

Random variations in a signal with equal power at all frequencies. DICE adds white Gaussian noise to the simulated profiles.

**Characterization:** standard deviation (σ_noise)

### Nominal Value

The value of a parameter that DICE uses to generate the simulated data, for example the nominal diffusion coefficient D_nom. DICE compares the estimates (D_est) with the nominal value.

### Non-Fickian Diffusion

Diffusion that does not follow Fick's laws, also called anomalous diffusion. Examples are subdiffusion (the MSD grows more slowly than linearly with time) and superdiffusion (the MSD grows faster than linearly).

**Note:** DICE does not model non-Fickian diffusion.

## O

### OLS (Ordinary Least Squares)

A linear regression method that minimizes the sum of the squared residuals, with equal weight for every data point. DICE uses OLS and WLS to fit a line to the variance of the profiles against time, which gives the diffusion coefficient.

### Optoelectronic Materials

Materials that interact with light and electric current, used for example in solar cells, light-emitting diodes, and photodetectors. DICE was developed to study excited state transport in these materials.

## P

### Point Spread Function (PSF)

The response of an imaging system to a point source. In microscopy, the PSF is often approximated by a Gaussian function.

### Proximity Level

A fractional distance from the nominal value, used to count how many estimates are close to it. For example, a proximity level of 0.1 counts the estimates within ±10% of the nominal diffusion coefficient. The fraction of estimates within the proximity level describes the accuracy and precision of a measurement with the simulated parameters.

The proximity level is not a simulation parameter. The summary file reports the fraction at ±5%, ±10%, ±20%, and ±50%. In the GUI, you choose any level from 1% to 50% in the Analysis Settings tab after a run.

## R

### Relative Standard Error

The standard deviation of the diffusion coefficient estimates divided by the nominal diffusion coefficient.

### RMSE (Root-Mean-Square Error)

The square root of the mean squared difference between the estimates and the nominal value. The RMSE combines the bias and the spread of the estimates.

## S

### Sigma (σ)

The standard deviation of a Gaussian distribution, which describes its width.

**Relationship to variance:** variance = σ²

**Relationship to FWHM:** σ = FWHM / (2√(2 ln 2))

### Single-Exponential Decay

A decay in which the population decreases exponentially with time: I(t) = I₀ exp(−t/τ). DICE uses this lifetime model.

### Subdiffusion

Diffusion that is slower than normal diffusion: the MSD grows sublinearly with time (MSD ∝ t^α with α < 1).

**Note:** DICE does not model subdiffusion.

## T

### Time-Resolved Microscopy

Imaging techniques that record how a system changes over time, often on nanosecond to microsecond time scales. DICE was designed for time-resolved measurements of excited state transport.

## W

### WLS (Weighted Least Squares)

A linear regression method that gives different weights to different data points, often to account for heteroscedasticity (variance that is not constant). DICE uses WLS to fit the variance of the profiles against time. It weights each time point by the inverse square of the relative standard error of the fitted variance at that point.

**Advantage over OLS:** WLS accounts for the uncertainty of the fitted variance, which differs between time points.

### White Noise

See [Noise (White)](#noise-white).

## Mathematical notation

### Common symbols

- **D**: diffusion coefficient
- **τ** (tau): lifetime
- **L_D**: diffusion length
- **σ** (sigma): standard deviation
- **μ** (mu): mean position
- **t**: time
- **x**: spatial position
- **I**: intensity
- **A**: amplitude

### Statistical notation

- **⟨x⟩**: mean (average) of x
- **σ_x**: standard deviation of x
- **x²**: x squared
- **√x**: square root of x

## Abbreviations

- **CNR**: contrast-to-noise ratio
- **DICE**: Diffusion Insight Computation Engine
- **FFT**: fast Fourier transform
- **FWHM**: full width at half maximum
- **MSD**: mean squared displacement
- **OLS**: ordinary least squares
- **PSF**: point spread function
- **RMSE**: root-mean-square error
- **WLS**: weighted least squares

## See also

- [Parameter Reference](parameters.md)
- [CLI Guide](cli-guide.md)
- [Installation Guide](installation.md)

## References

For the scientific background, see the DICE paper:

Joseph J. Thiebes, Erik M. Grumstrup; Quantifying noise effects in optical measures of excited state transport. J. Chem. Phys. 28 March 2024; 160 (12): 124201. [https://doi.org/10.1063/5.0190347](https://doi.org/10.1063/5.0190347)
