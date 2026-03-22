"""
Test script to verify all package imports work correctly.
"""

import sys
import traceback

def _run_import_checks():
    """Run import checks. Returns 0 on success, 1 on failure."""

    success = True
    failed_imports = []

    modules_to_test = [
        # Main package
        "dice",

        # Utils module
        "dice.utils",
        "dice.utils.converters",
        "dice.utils.validators",
        "dice.utils.axes",

        # Models module
        "dice.models",
        "dice.models.parameters",
        "dice.models.results",
        "dice.models.profiles",
    ]

    print("Testing package imports...")
    print("-" * 50)

    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"[OK] {module_name}")
        except ImportError as e:
            print(f"[FAIL] {module_name}: {e}")
            failed_imports.append((module_name, str(e)))
            success = False
        except Exception as e:
            print(f"[ERROR] {module_name}: Unexpected error - {e}")
            traceback.print_exc()
            failed_imports.append((module_name, str(e)))
            success = False

    print("-" * 50)

    if success:
        print("\n[SUCCESS] All imports successful!")
    else:
        print(f"\n[FAILED] {len(failed_imports)} import(s) failed:")
        for module, error in failed_imports:
            print(f"  - {module}: {error}")
        return 1

    # Test specific functionality
    print("\nTesting basic functionality...")
    print("-" * 50)

    try:
        from dice.utils.converters import sigma_to_fwhm
        result = sigma_to_fwhm(1.0)
        print(f"[OK] sigma_to_fwhm(1.0) = {result:.6f}")
    except Exception as e:
        print(f"[FAIL] Function test failed: {e}")
        success = False

    try:
        from dice.utils.axes import make_x_axis
        import numpy as np
        x = make_x_axis(10.0, 11, 0.0)
        print(f"[OK] make_x_axis created array with shape {x.shape}")
    except Exception as e:
        print(f"[FAIL] Axis generation failed: {e}")
        success = False

    try:
        from dice.models.parameters import GaussianParameters
        params = GaussianParameters(amplitude=1.0, sigma2=2.0, mu=0.0)
        print(f"[OK] GaussianParameters created with FWHM={params.fwhm:.6f}")
    except Exception as e:
        print(f"[FAIL] Dataclass creation failed: {e}")
        success = False

    print("-" * 50)

    if success:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print("\n[FAILED] Some tests failed")
        return 1


def test_imports():
    """Test that all modules can be imported."""
    assert _run_import_checks() == 0


if __name__ == "__main__":
    sys.exit(_run_import_checks())