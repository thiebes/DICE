import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Optional, Tuple # For type hinting x_lim

def colordefs() -> dict:
    """
    Defines a set of colors for plotting purposes. The color choices are made with 
    considerations for color blindness accessibility and clarity in black & white printing.

    Returns:
    - A dictionary of color names mapped to their hexadecimal color codes.
    """
    return {
        'dice_blue': '#003f7f',    # Montana State blue, good contrast and colorblind safe
        'dice_gold': '#f7941e',    # Montana State gold, vibrant and distinguishable in grayscale
        'dice_green': '#0cce6b',   # Bright green, good visibility and colorblind safe
        'dice_gradient': sns.cubehelix_palette(start=1, rot=0.9, gamma=1.0, hue=1, light=0.75, dark=0.20, reverse=True, as_cmap=True)
    }

def plot_accuracy_histogram(
        simulation_result: dict, 
        proximity: float,        
        filename: str,           
        image_type: str,         
        width: float,            
        height: float,           
        dpi:float,               
        font_size: float,        
        tick_length: float,      
        tick_width: float,       
        num_bins: int,           
        x_lim: Optional[Tuple[float, float]] = None # Made x_lim optional with type hint
    ):
    """
    Exports a histogram of diffusion accuracy values (D_est/D_nom) from simulation results. 
    """
    # convert width and height from cm to inches
    inch = 1/2.54
    fig_width_in, fig_height_in = width * inch, height * inch # Renamed to avoid conflict

    colors = colordefs() # Use 'colors' to avoid conflict if colordefs is also a variable name somewhere
    dice_blue = colors['dice_blue']
    dice_gold = colors['dice_gold']
    
    # Verify required data is present and valid
    if not isinstance(simulation_result, dict):
        print("Warning: Invalid simulation_result (not a dict). Histogram cannot be generated.")
        return

    if 'collated results' not in simulation_result:
        print("Warning: 'collated results' not found in simulation_result. Histogram cannot be generated.")
        return

    collated_df = simulation_result['collated results']

    if not isinstance(collated_df, pd.DataFrame): # Make sure pd is imported
        print("Warning: 'collated results' is not a DataFrame. Histogram cannot be generated.")
        return

    if 'd_wls_over_d_nom' not in collated_df.columns:
        print("Warning: Column 'd_wls_over_d_nom' not found in 'collated results' DataFrame. Histogram cannot be generated.")
        return

    if collated_df['d_wls_over_d_nom'].isnull().all():
        print("Warning: Column 'd_wls_over_d_nom' contains all NaN values. Histogram cannot be generated.")
        return

    # If all checks pass, proceed with the rest of the function
    dest_over_d0 = np.array(collated_df['d_wls_over_d_nom'].dropna()) # dropna just in case some are NaN
    if dest_over_d0.size == 0:
        print("Warning: Column 'd_wls_over_d_nom' has no valid data after dropping NaNs. Histogram cannot be generated.")
        return

    fig, ax = plt.subplots(layout='constrained', figsize=(fig_width_in, fig_height_in))

    # Filter out NaN/inf values before histogramming
    valid_data = dest_over_d0[np.isfinite(dest_over_d0)]
    if len(valid_data) == 0:
        print("Warning: No finite data available for histogram after filtering. Histogram cannot be generated.")
        plt.close(fig)
        return

    n_dd0, bins_dd0, patches_dd0 = ax.hist(
        valid_data,
        bins=num_bins, density=True,
        color=dice_gold, edgecolor='w',
    )

    mu_dd0 = np.mean(valid_data)
    sigma_dd0 = np.std(valid_data)
    
    if sigma_dd0 > 0: # Ensure std is positive before creating norm distribution
        binspace_dd0 = np.linspace(bins_dd0[0], bins_dd0[-1], 100)
        y_dd0 = norm.pdf(binspace_dd0, mu_dd0, sigma_dd0)
        ax.plot(binspace_dd0, y_dd0, 
                color=dice_blue, linewidth=2,
                label=f'mean {mu_dd0:.3f}\nmedian {np.median(valid_data):.3f}\nstdev {sigma_dd0:.3f}'
        )
    else: # Handle case with zero std (all data points are the same)
         ax.axvline(mu_dd0, color=dice_blue, linewidth=2, 
                    label=f'mean {mu_dd0:.3f}\nmedian {np.median(valid_data):.3f}\nstdev {sigma_dd0:.3f} (all data identical)')


    ax.set_xlabel('$D_{est}/D_{nom}$', fontsize=font_size)
    ax.set_ylabel('Probability density', fontsize=font_size)

    if x_lim is None and sigma_dd0 > 0 : # Default limits only if sigma is positive
        x_lim_calc = [mu_dd0 - 3 * sigma_dd0, mu_dd0 + 3 * sigma_dd0]
        # Ensure default limits are reasonable (e.g. not identical if sigma is tiny)
        if x_lim_calc[0] >= x_lim_calc[1]: x_lim_calc = [mu_dd0 -1, mu_dd0+1] # Fallback for tiny sigma
        ax.set_xlim(x_lim_calc)
    elif x_lim: # If x_lim is provided
        ax.set_xlim(x_lim)
    # If x_lim is None and sigma_dd0 is 0, autoscale will be used.
    
    ax.tick_params(axis='both', which='both', 
                   labelsize=font_size,
                   direction='in', 
                   length=tick_length, 
                   width=tick_width,
                )

    ax.legend(fontsize=font_size, handlelength=0, 
              labelspacing=2, frameon=False)

    fig.patch.set_facecolor('w')
    fig.patch.set_alpha(1)

    try:
        plt.savefig(filename, dpi=dpi, format=image_type) # Use filename directly
    except Exception as e:
        print(f"Error saving file: {e}")
    finally:
        plt.close(fig)
