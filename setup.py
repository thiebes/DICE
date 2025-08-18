"""
Setup script for DICE package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dice-diffusion",
    version="1.2.0",
    author="Joseph J. Thiebes",
    author_email="joseph@thiebes.org",
    description="Diffusion Insight Computation Engine - A tool for quantifying noise effects in optical measures of excited state transport",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/thiebes/DICE",
    packages=find_packages(exclude=["tests", "tests.*", "webapp", "webapp.*"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "statsmodels>=0.12.0",
        "joblib>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "webapp": [
            "flask>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dice=dice.cli.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)