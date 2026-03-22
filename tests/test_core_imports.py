"""
Test script to verify core module imports and basic functionality.
"""

import sys
import numpy as np

def _run_core_import_checks():
    """Run core import checks. Returns 0 on success, 1 on failure."""

    print("Testing core module imports...")
    print("-" * 50)

    # Test imports
    try:
        from dice.core import (
            gaussian,
            integrated_intensity,
            make_diffusion_decay,
            add_noise,
            fft_cnr,
            gauss_fitting,
            diffusion_ols_fit,
            calculate_diffusion_coefficient,
        )
        print("[OK] All core functions imported successfully")
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return 1

    # Test basic functionality
    print("\nTesting basic core functionality...")
    print("-" * 50)

    try:
        # Test Gaussian generation
        x = np.linspace(-5, 5, 101)
        y = gaussian(x, mu=0, sig2=1, amp=1)
        print(f"[OK] Gaussian generated with max value: {np.max(y):.4f}")

        # Test integrated intensity
        intensity = integrated_intensity(sig2=1, amp=1)
        expected = np.sqrt(2 * np.pi)
        print(f"[OK] Integrated intensity: {intensity:.4f} (expected: {expected:.4f})")

        # Test noise addition
        profiles = np.array([y])
        noisy = add_noise(profiles, noise_sigma=0.1, seed=42)
        print(f"[OK] Noise added, shape: {noisy['y_values_t'].shape}")

        # Test CNR estimation
        cnr = fft_cnr(noisy['y_values_t'][0]).cnr
        print(f"[OK] CNR estimated: {cnr:.2f}")

        # Test diffusion coefficient calculation
        slope = 2.0  # MSD slope
        D = calculate_diffusion_coefficient(slope, dimensions=1)
        print(f"[OK] Diffusion coefficient from slope {slope}: D = {D}")

    except Exception as e:
        print(f"[FAIL] Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("-" * 50)
    print("\n[SUCCESS] All core module tests passed!")
    return 0


def test_core_imports():
    """Test that all core modules can be imported."""
    assert _run_core_import_checks() == 0


if __name__ == "__main__":
    sys.exit(_run_core_import_checks())