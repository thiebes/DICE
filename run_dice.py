"""
DICE Runner - Entry point for DICE simulations.

This script provides backward compatibility with the original run_dice.py interface
while also supporting the new modular CLI functionality.
"""

import argparse
import sys
import os

def main():
    """Main entry point for DICE simulations."""
    
    try:
        # Use the modularized DICE CLI
        from dice.cli.main import main as cli_main
        return cli_main()
        
    except ImportError as e:
        print(f"Error: Could not import DICE CLI: {e}")
        print("Please ensure the dice package is properly installed.")
        return 1
    except Exception as e:
        print(f"Error running DICE simulation: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
