import argparse
from dice.simulation import run_simulation_cli

def main():
    parser = argparse.ArgumentParser(description="Run DICE simulation from parameters file.")
    parser.add_argument("filename", help="Path to parameter file (e.g., data/parameters.txt)")
    args = parser.parse_args()
    run_simulation_cli(args.filename)
    print("Simulation complete. Check output files.")

if __name__ == "__main__":
    main()
