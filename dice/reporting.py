from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from utils import colordefs

def plot_accuracy_histogram(
        simulation_result: dict,                        # Results from the simulation
        proximity: float,                               # Accuracy & precision: nominal proximity threshold 
        filename: str,                                  # Base name for the output file
        image_type: str,                                # Image format (e.g., 'png', 'jpg')
        width: float,                                   # Width of the image in cm
        height: float,                                  # Height of the image in cm
        dpi:float,                                      # Resolution of the image in dots per inch
        font_size: float,                               # Font size in points, for labels
        tick_length: float,                             # tick length in points
        tick_width: float,                              # tick width in points
        num_bins: int,                                  # Nuber of bins in the histogram
        x_lim: Optional[Tuple[float, float]] = None     # Optional x-axis limits
    ):
    
    """
    Exports a histogram of diffusion accuracy values (D_est/D_nom) from simulation results. 
    """

    # convert width and height from cm to inches
    inch = 1/2.54
    width, height = width * inch, height * inch

    # colors
    color_definitions = colordefs()
    dice_blue = color_definitions['dice_blue']
    dice_gold = color_definitions['dice_gold']
    
    # Verify required data is present
    if 'collated results' not in simulation_result or 'd_wls_over_d_nom' not in simulation_result['collated results']:
        raise ValueError("Required data not found in simulation_result.")

    # Convert list of values to NumPy array
    dest_over_d0 = np.array(simulation_result['collated results']['d_wls_over_d_nom'])
    
    # Initialize array to flag accuracy ratio values in proximity
    dest_d0_proximity_flag = np.abs(dest_over_d0 - 1) <= proximity

    # Calculate and report percentage of values that are within proximity threshold
    proxpct = 100 * np.sum(dest_d0_proximity_flag) / len(dest_d0_proximity_flag)
    print(f'Percent of D estimates within {proximity * 100:.2f}% of nominal: {proxpct:.1f}')

    fig, ax = plt.subplots(layout='constrained', figsize = (width,height))

    n_dd0, bins_dd0, patches_dd0 = ax.hist(
        dest_over_d0,
        bins=num_bins, density = True,
        color=dice_gold, edgecolor='w',
        )

    binspace_dd0 = np.linspace(bins_dd0[0], bins_dd0[-1], 100)
    mu_dd0 = np.mean(dest_over_d0)
    sigma_dd0 = np.std(dest_over_d0)
    y_dd0 = norm.pdf(binspace_dd0, mu_dd0, sigma_dd0)

    ax.set_xlabel('$D_{est}/D_{nom}$', fontsize = font_size)
    ax.set_ylabel('Probability density', fontsize = font_size)

    # default limits of x: 99.97% confidence interval
    if x_lim is None:
        x_lim = (
            float(mu_dd0 - 3 * sigma_dd0),
            float(mu_dd0 + 3 * sigma_dd0),
        )
    ax.set_xlim(x_lim)
    
    ax.tick_params(axis='both', which='both', 
                   labelsize=font_size,
                   direction='in', 
                   length=tick_length, 
                   width=tick_width,
                   # left=False, labelleft=False,
                )

    ax.plot(binspace_dd0, y_dd0, 
            color = dice_blue, linewidth=2,
            label = 'mean ' + str(np.round(mu_dd0,3)) + 
                    '\nmedian ' + str(np.round(np.median(dest_over_d0),3)) + 
                    '\nstdev ' + str(np.round(sigma_dd0,3))
            )

    ax.legend(fontsize=font_size, handlelength=0, 
            labelspacing = 2, frameon=False)

    # make the image background white and opaque
    fig.patch.set_facecolor('w')
    fig.patch.set_alpha(1)

    try:
        # Export the image
        export_file = f"{filename}"
        plt.savefig(export_file, dpi=dpi, format=image_type)
    except Exception as e:
        print(f"Error saving file: {e}")

    plt.close(fig)  # Close the figure to free up memory

def summarize_results():
    # placeholder for refactoring
    return

def export_results():
    # placeholder for refactoring
    return

'''
Potential future additions to this module:
- heatmaps
- performance vs pixel/time resolution
- param-sweep visualizations
'''