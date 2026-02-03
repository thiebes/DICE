"""
Example Parameter Configurations

Pre-defined parameter sets for common simulation scenarios.
"""

EXAMPLE_PARAMETERS = {
    "quick_test": {
        'filename slug': 'quick_test',
        'number of runs': 100,
        'nominal diffusion length': 0.1,
        'FWHM_0': 1,
        'amplitude_0': 1,
        'mean_0': 0,
        'noise value': 0.02,
        'spatial width': 5,
        'pixel width': 50,
        'time range': [0, 1, 5],
        'proximity level': 0.1,
    },
    "high_precision": {
        'filename slug': 'high_precision',
        'number of runs': 10000,
        'nominal diffusion length': 0.1,
        'FWHM_0': 1,
        'amplitude_0': 1,
        'mean_0': 0,
        'noise value': 0.02,
        'spatial width': 10,
        'pixel width': 200,
        'time range': [0, 2, 20],
        'proximity level': 0.1,
    },
    "publication": {
        'filename slug': 'publication_example',
        'number of runs': 1000,
        'nominal diffusion length': 0.1,
        'FWHM_0': 1,
        'amplitude_0': 1,
        'mean_0': 0,
        'noise value': 0.02,
        'spatial width': 5,
        'pixel width': 100,
        'time range': [0, 1, 10],
        'proximity level': 0.5,
    }
}
