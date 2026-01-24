import argparse
from dice.simulation import run_simulation_cli

def main():
    parser = argparse.ArgumentParser(description="Run the DICE simulation from a parameters file.")
    parser.add_argument("filename", help="Path to the parameter file (e.g., data/parameters.txt)")
    args = parser.parse_args()

    run_simulation_cli(args.filename)

    print("Simulation complete. Check the output files.")

if __name__ == "__main__":
    main()
