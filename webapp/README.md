# Contrast-to-Noise Ratio (CNR) Estimation Web App

This web application estimates the contrast-to-noise ratio (CNR) of a one-dimensional profile, for example an excited state population profile, entered as a list of values. The estimate comes from the [fft-cnr](https://pypi.org/project/fft-cnr/) package, which DICE also uses.

A hosted copy is available at <https://dice-thiebes.pythonanywhere.com/>. No installation is needed to use it.

## Features

- Accepts comma-separated or space-separated lists of values.
- Reports the CNR estimate.
- Displays an error message if the input is not a list of numbers.
- Provides citations and links to the source code and the publication.

## Run the app locally

### Prerequisites

- Python 3.11 or later
- A clone of the complete DICE repository. The app uses the DICE installation and its dependencies.

### Install

From the root of the repository, install DICE with the `webapp` extra, which adds Flask.

With uv:

```bash
git clone https://github.com/thiebes/DICE.git
cd DICE
uv sync --extra webapp
```

With pip, in an active virtual environment:

```bash
git clone https://github.com/thiebes/DICE.git
cd DICE
pip install ".[webapp]"
```

See the [Installation Guide](../docs/installation.md) for details.

### Start the development server

From the root of the repository:

```bash
uv run python webapp/dice_web_app.py      # with uv
python webapp/dice_web_app.py             # with pip, environment active
```

Then open <http://127.0.0.1:5000/> in a web browser. The Flask development server is meant for local use only.

### Use the app

1. Enter the values of your noisy profile, separated by commas or spaces. `webapp/example_profile.csv` contains an example profile.
2. Click "Calculate CNR estimate".
3. Read the CNR estimate on the result page.

## Method

The app passes the values to the `fft_cnr` function of the fft-cnr package and reports its CNR estimate. In outline, fft-cnr transforms the profile into frequency space, where the signal occupies the low frequencies and white noise spreads over all frequencies. It chooses the boundary frequency between signal and noise with the Akaike information criterion (AIC), estimates the root-mean-square (RMS) noise from the frequencies above that boundary, and divides the height of the signal peak above the baseline by the RMS noise. The [fft-cnr documentation](https://fft-cnr.readthedocs.io) describes the method and its limits.

## References

- [DICE source code](https://github.com/thiebes/DICE)
- [fft-cnr on PyPI](https://pypi.org/project/fft-cnr/) and [fft-cnr documentation](https://fft-cnr.readthedocs.io)

## Citation

Code:

```text
Joseph J. Thiebes. (2023). thiebes/DICE. Zenodo. https://doi.org/10.5281/zenodo.10258191
```

Paper:

```text
Joseph J. Thiebes, Erik M. Grumstrup; Quantifying noise effects in optical measures of excited state transport. J. Chem. Phys. 28 March 2024; 160 (12): 124201. https://doi.org/10.1063/5.0190347
```

## License

The web app is part of DICE and is licensed under the MIT License. See [LICENSE.md](../LICENSE.md).

## Acknowledgements

- Joseph J. Thiebes for developing the DICE application.
- Erik M. Grumstrup for contributing to the research.

## Contact

- Joseph Thiebes: [Website](http://thiebes.org) | [LinkedIn](https://www.linkedin.com/in/thiebes/) | [Buy me a coffee](https://www.buymeacoffee.com/leufebkxkn)
