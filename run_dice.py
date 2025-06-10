import argparse
from dicecore.simulation import dice_runner

def main():
    parser = argparse.ArgumentParser(description="Run the dice module.")
    parser.add_argument("filename", help="Filename of the parameters file")
    args = parser.parse_args()

    # Run the dice_runner function, results are saved in a file within the function
    dice_runner(args.filename)

    print("The script has been executed. Please check the output file for results.")

if __name__ == "__main__":
    main()
