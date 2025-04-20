import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from cycler import cycler
import itertools
import random
import math

random.seed(42)

def line_color_style_cycler():
    linestyles = ['-', '--', '-.', ':']

    # Get default matplotlib color cycle
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Generate all color + linestyle combinations
    style_combinations = list(itertools.product(default_colors, linestyles))

    # Remove the ('r', '-') combo reserved for the first line
    style_combinations = [combo for combo in style_combinations if combo != ('r', '-')]

    # Shuffle to randomize the order of the combinations
    random.shuffle(style_combinations)

    # Convert to a cycler
    custom_cycler = cycler(color=[c for c, l in style_combinations],
                        linestyle=[l for c, l in style_combinations])
    
    return custom_cycler

if __name__ == '__main__':

    # Paths, keys, titles
    # Build a list of dictionaries similar to the dataToCompare array in MATLAB.
    save_path='C:/Users/medemirhan/Desktop/mdpi/figures/results'
    data_to_compare = [
        {"path": r"D:\results\comparison\normal\buildingblock.mat", "key": "data", "label": "Ground Truth"},
        {"path": r"D:\results\comparison\ours\buildingblock.mat",   "key": "ref",  "label": "SS-HSLIE (Ours)"},
        {"path": r"D:\results\comparison\low\buildingblock.mat",    "key": "data", "label": "Low-light"},
        {"path": r"D:\results\comparison\bm4d\buildingblock.mat",    "key": "data", "label": "BM4D"},
        {"path": r"D:\results\comparison\clahe\buildingblock.mat",   "key": "data", "label": "CLAHE"},
        # The following line was commented out in MATLAB:
        # {"path": r"D:\results\comparison\deep_hs_prior\buildingblock.mat", "key": "pred", "label": "Deep HS Prior"},
        {"path": r"D:\results\comparison\fast_hy_mix\buildingblock.mat", "key": "data", "label": "FastHyMix"},
        {"path": r"D:\results\comparison\hcanet\buildingblock.mat",       "key": "data", "label": "HCANet"},
        {"path": r"D:\results\comparison\he\buildingblock.mat",           "key": "data", "label": "HE"},
        {"path": r"D:\results\comparison\lrtdtv\buildingblock.mat",        "key": "data", "label": "LRTDTV"},
        {"path": r"D:\results\comparison\mr\buildingblock.mat",            "key": "data", "label": "MR"},
        {"path": r"D:\results\comparison\msr\buildingblock.mat",           "key": "data", "label": "MSR"},
        {"path": r"D:\results\comparison\retinexnet\buildingblock.mat",     "key": "data", "label": "RetinexNet"}
    ]

    # Parameters
    # These locations and parameters come directly from the MATLAB code.
    x_locs = [250, 110, 200, 370]
    y_locs = [310, 190, 240, 200]
    window_size = 5  # must be an odd number
    sample_band = 20  # MATLAB index (1-indexed); will subtract one for Python (0-indexed)

    # Load data
    # Load the first hyperspectral image to get its dimensions.
    first_entry = data_to_compare[0]
    mat_contents = loadmat(first_entry["path"])
    # Extract the field specified by the key.
    data_field = mat_contents[first_entry["key"]]
    # Assume the data is in shape (h, w, c)
    h, w, c = data_field.shape
    # Create a list to accumulate the hyperspectral images.
    hs_images = [data_field]  # first image

    # Load remaining images, reshape each to (h, w, c) and append to the list.
    for entry in data_to_compare[1:]:
        mat_contents = loadmat(entry["path"])
        cur_data = mat_contents[entry["key"]]
        # Optionally, check that cur_data.shape matches (h,w,c)
        hs_images.append(cur_data)

    # Stack all the images into a single NumPy array of shape (num_images, h, w, c)
    data_array = np.stack(hs_images, axis=0)
    num_hsis = data_array.shape[0]

    # Plot hsis (Each plot is saved separately)
    # For each hyperspectral image, display the sample band.
    # In MATLAB, imshow(curData(:,:,sampleBand)) is used.
    # In Python, remember that indexing is 0-based.
    for i in range(num_hsis):
        if i != 0:
            continue

        # Extract the i-th image of shape (h, w, c)
        cur_data = data_array[i, :, :, :]
        
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 16,
            'axes.linewidth': 1.2,
            'xtick.direction': 'in',
            'ytick.direction': 'in'
            })

        plt.figure(figsize=(8, 6))
        # Display the selected spectral band (subtract one for 0-index)
        plt.imshow(cur_data[:, :, sample_band - 1], cmap='gray')
        plt.axis('off')
        
        # For the first image, overlay red circles and text at given locations.
        if i == 0:
            # Adjust coordinates: MATLAB indices are 1-based, so subtract 1 for Python array coordinates.
            x_coords = np.array(x_locs) - 1
            y_coords = np.array(y_locs) - 1
            
            # Plot red circles.
            plt.plot(x_coords, y_coords, 'ro', markersize=8, linewidth=2)
            
            # Add text labels near the points.
            for (x, y) in zip(x_coords, y_coords):
                # Offset the text (adjust as necessary)
                plt.text(
                    x + 5,
                    y - 5,
                    f'[{x+1}, {y+1}]',
                    color='blue',
                    fontsize=10,
                    weight='bold',
                    bbox=dict(
                        facecolor='white',      # box background color
                        edgecolor='black',      # box border color
                        boxstyle='round,pad=0.5',
                        alpha=0.3               # transparency (0 = fully transparent, 1 = opaque)
                        )
                    )
        
        # Set the title (using the label from data_to_compare)
        cur_label = data_to_compare[i]["label"]
        #plt.title(cur_label)
        
        # Save the figure; here we save each figure separately.
        # For example, name files "hs_image_0.png", "hs_image_1.png", etc.
        filename = f"hs_image_{i}.eps"
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
        plt.close()

    # Plot spectra (Each plot is saved separately)
    # For each (x,y) location, compute the local spectrum across all algorithms.
    # Create one figure per (x,y) location.
    # Note: Convert MATLAB 1-indexing to Python 0-indexing.
    for idx in range(len(x_locs)):
        # Convert coordinates to zero-indexed values.
        x_loc = x_locs[idx] - 1
        y_loc = y_locs[idx] - 1

        # Compute window indices (ensuring we stay within image boundaries).
        # For x direction:
        x_start = max(0, x_loc - window_size // 2)
        x_end = min(w, x_start + window_size)
        # For y direction:
        y_start = max(0, y_loc - window_size // 2)
        y_end = min(h, y_start + window_size)

        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 12,
            'axes.linewidth': 1.2,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'axes.prop_cycle': line_color_style_cycler()
        })
        
        # Create a new figure for this spectrum.
        plt.figure(figsize=(12, 8))
        
        # Prepare a list to collect labels for the legend.
        legend_labels = []
        
        # Plot the spectrum for each hyperspectral image.
        for j in range(num_hsis):
            cur_data = data_array[j, :, :, :]  # shape: (h, w, c)
            # Extract the window from the image.
            # In Python slicing, the window has shape (window_size, window_size, c)
            window = cur_data[y_start:y_end, x_start:x_end, :]
            # Compute the spectrum as the average over the window pixels.
            # Sum over the first two axes then divide by (window_size**2)
            spectrum = np.sum(window, axis=(0, 1)) / (window_size ** 2)
            if data_to_compare[j]["label"]=="SS-HSLIE (Ours)":
                plt.plot(spectrum, linewidth=2, color='r', linestyle='-')
            elif data_to_compare[j]["label"]=="Ground Truth":
                plt.plot(spectrum, linewidth=2, color='b', linestyle='-')
            elif data_to_compare[j]["label"]=="Low-light":
                plt.plot(spectrum, linewidth=2, color='g', linestyle='--')
            else:
                plt.plot(spectrum, linewidth=1.1)
            legend_labels.append(data_to_compare[j]["label"])
        
        plt.xlabel('Band Number')
        plt.ylabel('Reflectance')
        plt.legend(legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=math.ceil(len(data_to_compare)/2))
        plt.tight_layout()
        # Save the spectrum plot as a separate file.
        spectrum_filename = f"spectrum_at_{x_locs[idx]}-{y_locs[idx]}.eps"
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, spectrum_filename), bbox_inches='tight')
        #plt.savefig(spectrum_filename, dpi=300)
        plt.close()
