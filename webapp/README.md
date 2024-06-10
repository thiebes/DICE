# Contrast-to-Noise Ratio (CNR) Estimation Web App

This web application estimates the contrast-to-noise ratio (CNR) from a list of comma-separated values representing a one-dimensional excited state population profile. The application utilizes a Fourier transform-based algorithm to calculate the CNR.

This web app is currently available for your use on the web at: https://dice-thiebes.pythonanywhere.com/

## Features

- Accepts comma-separated or space-separated lists of values.
- Provides an estimation of the contrast-to-noise ratio (CNR).
- Displays errors if the input is invalid.
- Provides citations and links to relevant publications and source code.

## Getting Started

### Prerequisites

- Python 3.x
- Flask
- NumPy

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/thiebes/DICE.git
    cd DICE
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Running the Application

1. Start the Flask development server:
    ```sh
    python dice_web_app.py
    ```

2. Open your web browser and navigate to `http://127.0.0.1:5000/`.

### Usage

1. Enter a list of comma-separated or space-separated values representing your noisy Gaussian profile.
2. Click the "Calculate CNR estimate" button.
3. View the estimated contrast-to-noise ratio (CNR) on the result page.

## Algorithm

The CNR estimation algorithm works as follows:
1. The data undergoes a unitary, single-sided, one-dimensional discrete Fourier transform (FT).
2. The modulus of each complex value in the transformed data is calculated.
3. The first local minimum subsequent to the signal peak is found.
4. The noise level is determined by calculating the root mean squared (RMS) amplitude for frequencies above the initial local minimum.
5. Since the normalized peak amplitude was unity, the CNR is simply the reciprocal of the noise level.
6. The estimate is rounded to the nearest integer value.

## References

- [DICE Source Code](https://github.com/thiebes/DICE)
- [One-dimensional Discrete Fourier Transform (numpy.fft.rfft)](https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html)

## Citation

- Code:
    ```
    Joseph J. Thiebes. (2023). thiebes/DICE. Zenodo. https://doi.org/10.5281/zenodo.10258191
    ```

- Paper:
    ```
    Joseph J. Thiebes, Erik M. Grumstrup; Quantifying noise effects in optical measures of excited state transport. J. Chem. Phys. 28 March 2024; 160 (12): 124201. https://doi.org/10.1063/5.0190347
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Joseph J. Thiebes for developing the DICE application.
- Erik M. Grumstrup for contributing to the research.

## Contact

- Joseph Thiebes: [Website](http://thiebes.org) | [LinkedIn](https://www.linkedin.com/in/thiebes/) | [Buy me a coffee](https://www.buymeacoffee.com/leufebkxkn)
