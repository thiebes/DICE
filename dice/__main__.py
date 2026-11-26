"""Allow running dice as a module: python -m dice"""
import sys

from dice.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
