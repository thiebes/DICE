"""Tests for the CNR estimator web app, using the repository's example profile."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("flask")

from fft_cnr import fft_cnr  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = REPO_ROOT / "webapp" / "dice_web_app.py"
EXAMPLE_PROFILE = REPO_ROOT / "data" / "examples" / "example_profile.csv"


@pytest.fixture(scope="module")
def web_app_module():
    spec = importlib.util.spec_from_file_location("dice_web_app_under_test", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    # Flask resolves the templates folder through sys.modules[module.__name__].
    sys.modules[spec.name] = module
    path_before = list(sys.path)
    try:
        spec.loader.exec_module(module)
        module.path_before_import = path_before
        module.path_after_import = list(sys.path)
        yield module
    finally:
        del sys.modules[spec.name]


@pytest.fixture
def client(web_app_module):
    web_app_module.app.config["TESTING"] = True
    return web_app_module.app.test_client()


def test_importing_the_app_leaves_sys_path_unchanged(web_app_module) -> None:
    assert web_app_module.path_after_import == web_app_module.path_before_import


def test_home_page_loads(client) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert b"fft-cnr" in response.data


@pytest.mark.parametrize("separator", [",", " "])
def test_example_profile_gives_the_fft_cnr_estimate(client, separator: str) -> None:
    profile = np.loadtxt(EXAMPLE_PROFILE, delimiter=",")
    expected = fft_cnr(profile.tolist()).cnr
    values = separator.join(str(value) for value in profile.tolist())

    response = client.post("/", data={"values": values})

    assert response.status_code == 200
    assert f"Contrast-to-noise estimate: {expected}" in response.get_data(as_text=True)


@pytest.mark.parametrize("scale, offset", [(1.0 / 1530.0, 0.0), (1.0, 500.0)])
def test_estimate_does_not_depend_on_normalization_or_background(
    client, scale: float, offset: float
) -> None:
    """The home page tells users that neither preprocessing step is needed."""
    profile = np.loadtxt(EXAMPLE_PROFILE, delimiter=",")
    expected = fft_cnr(profile.tolist()).cnr
    values = ",".join(repr(value * scale + offset) for value in profile.tolist())

    response = client.post("/", data={"values": values})

    text = response.get_data(as_text=True)
    reported = float(text.split("Contrast-to-noise estimate: ")[1].split("<")[0])
    assert reported == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize(
    "values, message",
    [
        ("", "No data provided"),
        ("1, 2, not-a-number", "Please enter a valid list"),
        ("1, 2, 3", "Error processing data"),
    ],
)
def test_malformed_input_shows_an_error_page(client, values: str, message: str) -> None:
    response = client.post("/", data={"values": values})

    assert response.status_code == 200
    assert message in response.get_data(as_text=True)
