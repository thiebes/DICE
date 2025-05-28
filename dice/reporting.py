from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from dice.utils import colordefs

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
        x_lim: Optional[Tuple[float, float]] = None,    # Optional x-axis limits
        return_figure: bool = False                     # If True, returns the figure object
    ):
    
    """
    Generates or exports a histogram of diffusion accuracy values (D_est/D_nom) from simulation results.
    If return_figure is True, returns the matplotlib figure object. Otherwise, saves to file and closes.
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
        # Export the image if not returning the figure (or always save, then decide to close)
        if not return_figure: # Only save to file if not returning the figure for embedding
            export_file = f"{filename}"
            plt.savefig(export_file, dpi=dpi, format=image_type)
    except Exception as e:
        print(f"Error saving file: {e}")

    if return_figure:
        return fig
    else:
        plt.close(fig)  # Close the figure to free up memory if not returning it

def summarize_results(result_dict):
    """
    Return summary lines from the simulation for CLI or GUI display.
    """
    p = result_dict['parameters']
    i = result_dict['indices']

    lines = []
    lines.append(f"Running {i['total runs']} simulations with the following parameters (rounded):\n")
    lines.append(f"Spatial width: {p['scan width']} {p['length units']}")
    lines.append(f"Pixel width: {p['scan pixels']} pixels")
    lines.append(f"Number of time frames: {len(i['time axis'])} frames")
    lines.append(f"Noise stdev: {round(i['noise sigmas'][0], 3)}")
    lines.append(f"Initial CNR: {round(1 / i['noise sigmas'][0], 3)}")
    lines.append(f"Initial profile sigma^2: {round(p['sigma^2_0'], 3)} {p['length units']}²")
    lines.append(f"Nominal diffusion length: {round(p['nominal diffusion length'], 3)} {p['length units']}")
    lines.append(f"Nominal diffusion coeff: {round(p['nominal diffusion coeff'], 5)} {p['length units']}² per {p['time units']}")
    lines.append(f"Nominal lifetime: {p['nominal lifetime']} {p['time units']}\n")

    if 'analysis' in result_dict:
        proximity = result_dict['parameters']['proximity level']
        ols_pct = result_dict['analysis']['% fits within proximity']['unweighted fit']
        wls_pct = result_dict['analysis']['% fits within proximity']['weighted fit']
        lines.append(f"Portion of fits where D_est / D_nom = 1 ± {proximity}:")
        lines.append(f"-- Unweighted fit: {round(ols_pct, 2)}%")
        lines.append(f"-- Weighted fit: {round(wls_pct, 2)}%\n")
    else:
        lines.append("Only one time frame; diffusion fits not computed.\n")

    lines.append("Exporting result data and histogram.")
    lines.append(f"-- Summary file: {p['summary filename']}")
    lines.append(f"-- Collated CSV file: {p['result filename']}")
    lines.append(f"-- Histogram image file: {p['image filename']}")
    lines.append("Done!\n")

    return lines

def export_results(result_dict):
    """
    Save the collated CSV and accuracy histogram image to disk.
    """
    df = result_dict['collated results']
    p = result_dict['parameters']

    # Write CSV
    df.to_csv(p['result filename'], index=False)

    # Write image
    plot_accuracy_histogram(
        simulation_result=result_dict,
        proximity=result_dict['parameters']['proximity level'],
        filename=p['image filename'],
        image_type=p['image type'],
        width=result_dict['parameters']['image width'],
        height=result_dict['parameters']['image height'],
        dpi=result_dict['parameters']['image dpi'],
        font_size=result_dict['parameters']['image font size'],
        tick_length=result_dict['parameters']['image tick length'],
        tick_width=result_dict['parameters']['image tick width'],
        num_bins=result_dict['parameters']['image numbins'],
        x_lim=result_dict['parameters']['image x_lim']
    )

'''
Potential future additions to this module:
- heatmaps
- performance vs pixel/time resolution
- param-sweep visualizations
'''