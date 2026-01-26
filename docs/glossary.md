# DICE Glossary

Definitions of technical terms, acronyms, and concepts used in DICE.

## A

### Anisotropic Diffusion

Diffusion that occurs at different rates in different directions. DICE currently only models isotropic (direction-independent) diffusion.

### Amplitude

The peak intensity of a Gaussian profile. In DICE simulations, the initial amplitude is typically set to unity (1.0).

## C

### CNR (Contrast-to-Noise Ratio)

The ratio of the signal amplitude to the standard deviation of the noise. A higher CNR indicates clearer signal above noise.

**Formula:** CNR = Amplitude / σ_noise

**Note:** This is distinct from SNR (Signal-to-Noise Ratio), which compares signal power to noise power.

## D

### Diffusion Coefficient (D)

A measure of how quickly particles spread through space via random motion. In DICE, it describes the rate at which an excited state population spreads.

**Units:** (length)² / time (e.g., μm²/ns, cm²/s)

**Relationship:** Related to diffusion length by L_D = √(Dτ)

### Diffusion Length (L_D)

The characteristic distance an excited state travels before decay.

**Formula:** L_D = √(Dτ)

where D is the diffusion coefficient and τ is the lifetime.

**Physical Meaning:** Approximately 63% of particles decay within this distance from their origin.

## E

### Excited State

An electronic state of a molecule or material with higher energy than the ground state. In DICE, we model the transport and decay of these excited states.

### Excited State Transport

The movement of electronic excitations through a material via diffusion, hopping, or other mechanisms.

## F

### FFT (Fast Fourier Transform)

An algorithm for efficiently computing the Fourier transform. DICE uses FFT to estimate noise levels from experimental data by analyzing frequency components.

### Fickian Diffusion

Normal diffusion that follows Fick's laws, characterized by a mean squared displacement that grows linearly with time (MSD ∝ t). This is the type of diffusion DICE models.

**See also:** Non-Fickian diffusion (not currently supported)

### FWHM (Full Width at Half Maximum)

The width of a peak at half of its maximum height. For Gaussian profiles, FWHM relates to standard deviation by:

**Formula:** FWHM = σ × 2√(2 ln 2) ≈ σ × 2.355

## G

### Gaussian Profile

A bell-shaped distribution described by a Gaussian (normal) function. DICE models excited state populations as Gaussian profiles.

**Parameters:** amplitude (A), mean (μ), standard deviation (σ)

## M

### Monte Carlo Simulation

A statistical method that uses repeated random sampling to obtain numerical results. DICE uses Monte Carlo methods to simulate many noisy measurements and assess their statistical properties.

### MSD (Mean Squared Displacement)

A measure of the average squared distance a particle has moved from its starting position.

**Formula for 1D:** MSD(t) = ⟨(x(t) - x(0))²⟩ = 2Dt

In DICE, MSD is equivalent to the change in variance: MSD(t) = σ²(t) - σ²(0)

## N

### Non-Fickian Diffusion

Diffusion that does not follow Fick's laws. Examples include subdiffusion (MSD grows slower than linearly with time) and superdiffusion (MSD grows faster than linearly).

**Note:** DICE currently only models Fickian diffusion.

### Noise (White)

Random variations in signal that are frequency-independent (equal power at all frequencies). DICE adds white Gaussian noise to simulated profiles.

**Characterization:** Standard deviation (σ_noise)

## O

### OLS (Ordinary Least Squares)

A linear regression method that minimizes the sum of squared residuals. DICE uses OLS (and WLS) to fit a line to MSD vs. time data to extract the diffusion coefficient.

### Optoelectronic Materials

Materials that interact with light and electricity, often used in solar cells, LEDs, and photodetectors. DICE was developed to study excited state transport in these materials.

## P

### Proximity Level

A threshold used in DICE to assess accuracy of diffusion coefficient estimates.

**Example:** A proximity level of 0.1 means estimates within ±10% of the nominal value are considered accurate.

### Point Spread Function (PSF)

The response of an imaging system to a point source. In microscopy, the PSF is often approximated as a Gaussian.

## S

### Sigma (σ)

The standard deviation of a Gaussian distribution, describing its width.

**Relationship to variance:** σ² = variance

**Relationship to FWHM:** σ = FWHM / (2√(2 ln 2))

### Single-Exponential Decay

A decay process where intensity decreases exponentially with time: I(t) = I₀ exp(-t/τ). This is the lifetime model used in DICE.

### Subdiffusion

A diffusion process slower than normal Fickian diffusion, where MSD grows sublinearly with time (MSD ∝ t^α with α < 1).

**Note:** Not currently supported in DICE.

## T

### Time-Resolved Microscopy

Imaging techniques that capture how a system changes over time, often on nanosecond to microsecond timescales. DICE was designed to analyze data from time-resolved measurements of excited state transport.

### Lifetime (τ)

The characteristic time for an excited state to decay to the ground state.

**Definition:** Time for intensity to decrease to 1/e (≈37%) of its initial value in single-exponential decay.

**Units:** time (e.g., ns, ps)

## W

### WLS (Weighted Least Squares)

A linear regression method that assigns different weights to different data points, often to account for heteroscedasticity (non-constant variance). DICE uses WLS to fit MSD vs. time data.

**Advantage over OLS:** Better handles the fact that uncertainty in MSD typically increases with time.

### White Noise

See [Noise (White)](#noise-white)

## Mathematical Notation

### Common Symbols

- **D**: Diffusion coefficient
- **τ** (tau): Lifetime
- **L_D**: Diffusion length
- **σ** (sigma): Standard deviation
- **μ** (mu): Mean position
- **t**: Time
- **x**: Spatial position
- **I**: Intensity
- **A**: Amplitude

### Statistical Notation

- **⟨x⟩**: Mean (average) of x
- **σ_x**: Standard deviation of x
- **x²**: x squared
- **√x**: Square root of x

## Acronyms Quick Reference

- **CNR**: Contrast-to-Noise Ratio
- **DICE**: Diffusion Insight Computation Engine
- **FFT**: Fast Fourier Transform
- **FWHM**: Full Width at Half Maximum
- **MSD**: Mean Squared Displacement
- **OLS**: Ordinary Least Squares
- **PSF**: Point Spread Function
- **WLS**: Weighted Least Squares

## See Also

- [Parameter Reference](parameters.md)
- [CLI Guide](cli-guide.md)
- [Installation Guide](installation.md)

## References

For more detailed scientific background, see the DICE paper:

Joseph J. Thiebes, Erik M. Grumstrup; Quantifying noise effects in optical measures of excited state transport. J. Chem. Phys. 28 March 2024; 160 (12): 124201. [https://doi.org/10.1063/5.0190347](https://doi.org/10.1063/5.0190347)
