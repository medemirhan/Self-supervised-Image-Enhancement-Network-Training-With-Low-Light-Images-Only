import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from cycler import cycler
import itertools
import random
import math
import string

random.seed(42)

def visualize_hsi_false_color(HSI, wave_start_nm, wave_end_nm, normalize=False):
    """
    Create a false-color RGB image from a hyperspectral cube.
    
    Parameters
    ----------
    HSI : numpy.ndarray
        Hyperspectral image cube of shape (height, width, bands).
    wave_start_nm : float
        Wavelength (nm) corresponding to band index 0.
    wave_end_nm : float
        Wavelength (nm) corresponding to band index -1.
    normalize : bool, optional
        If True, each channel is scaled to [0,1]. Default is False.
        
    Returns
    -------
    rgb : numpy.ndarray
        False-color RGB image of shape (height, width, 3).
    """
    # number of bands and their wavelengths
    bands = HSI.shape[2]
    wavelengths = np.linspace(wave_start_nm, wave_end_nm, bands)
    
    # find band indices closest to target wavelengths
    idx_nir   = np.argmin(np.abs(wavelengths - 800))
    idx_red   = np.argmin(np.abs(wavelengths - 670))
    idx_green = np.argmin(np.abs(wavelengths - 550))
    
    # extract each channel
    R = HSI[:, :, idx_nir]
    G = HSI[:, :, idx_red]
    B = HSI[:, :, idx_green]
    
    if normalize:
        # scale each channel to [0,1]
        def norm(channel):
            cmin, cmax = channel.min(), channel.max()
            return (channel - cmin) / (cmax - cmin) if cmax > cmin else channel
        R, G, B = norm(R), norm(G), norm(B)
    
    # stack into an RGB image
    rgb = np.stack([R, G, B], axis=-1)
    
    return rgb

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
    save_path='C:/Users/medemirhan/Desktop/tez/thesis/figures/results'

    dataset = 2 # 0: jyu_outdoor, 2: jyu_indoor, 1: other indoor

    if dataset == 1:
        mat_name = 'vegatables5'
        data_to_compare = [
            {"path": rf"D:\results\comparison\normal\indoor\{mat_name}.mat", "key": "data", "label": "Ground Truth"},
            {"path": rf"D:\results\comparison\ours\indoor\{mat_name}.mat",   "key": "ref",  "label": "SS-HSLIE (Ours)"},
            {"path": rf"D:\results\comparison\low\indoor\{mat_name}.mat",    "key": "data", "label": "Low-light"},
            {"path": rf"D:\results\comparison\bm4d\indoor\{mat_name}.mat",    "key": "data", "label": "BM4D"},
            {"path": rf"D:\results\comparison\clahe\indoor\{mat_name}.mat",   "key": "data", "label": "CLAHE"},
            {"path": rf"D:\results\comparison\deep_hs_prior\indoor\{mat_name}.mat", "key": "pred", "label": "DeepHS Pr."},
            {"path": rf"D:\results\comparison\fast_hy_mix\indoor\{mat_name}.mat", "key": "data", "label": "FastHyMix"},
            {"path": rf"D:\results\comparison\hcanet\indoor\{mat_name}.mat",       "key": "data", "label": "HCANet"},
            {"path": rf"D:\results\comparison\enlightengan\indoor\{mat_name}.mat",           "key": "data", "label": "EnlightenGAN"},
            {"path": rf"D:\results\comparison\lrtdtv\indoor\{mat_name}.mat",        "key": "data", "label": "LRTDTV"},
            {"path": rf"D:\results\comparison\mr\indoor\{mat_name}.mat",            "key": "data", "label": "MR"},
            {"path": rf"D:\results\comparison\rcild\indoor\{mat_name}.mat",           "key": "data", "label": "RCILD"},
            {"path": rf"D:\results\comparison\retinexnet\indoor\{mat_name}.mat",     "key": "data", "label": "RetinexNet"},
            {"path": rf"D:\results\comparison\exposure_diffusion\indoor\{mat_name}.mat",     "key": "data", "label": "ExposureDiff"},
            {"path": rf"D:\results\comparison\cdan\indoor\{mat_name}.mat",     "key": "data", "label": "CDAN"},
            {"path": rf"D:\results\comparison\mafnet\indoor\{mat_name}.mat",     "key": "data", "label": "MAFNet"},
            {"path": rf"D:\results\comparison\retinexformer\indoor\{mat_name}.mat",     "key": "data", "label": "Retinexformer"},
            {"path": rf"D:\results\comparison\retinexmamba\indoor\{mat_name}.mat",     "key": "data", "label": "Retinexmamba"}
        ]
        #bblock:
        '''x_locs = [131, 134, 253, 327]
        y_locs = [47, 214, 146, 158]'''
        
        #corn
        '''x_locs = [95, 252, 233, 330]
        y_locs = [190, 101, 274, 190]'''

        #huanggua
        '''x_locs = [160, 115, 306, 337]
        y_locs = [106, 267, 183, 239]'''

        #orange_apple
        '''x_locs = [219, 220, 100]
        y_locs = [109, 336, 200]'''

        #vegatables5
        x_locs = [129, 127, 254, 344]
        y_locs = [59, 219, 144, 147]

        waveStart_nm = 453.8117
        waveEnd_nm = 962.3318

    elif dataset == 0:
        mat_name = '486'
        data_to_compare = [
            {"path": rf"D:\results\comparison\normal\jyu_outdoor\{mat_name}.mat", "key": "data", "label": "Ground Truth"},
            {"path": rf"D:\results\comparison\ours\jyu_outdoor\{mat_name}.mat",   "key": "ref",  "label": "SS-HSLIE (Ours)"},
            {"path": rf"D:\results\comparison\low\jyu_outdoor\{mat_name}.mat",    "key": "data", "label": "Low-light"},
            {"path": rf"D:\results\comparison\bm4d\jyu_outdoor\{mat_name}.mat",    "key": "data", "label": "BM4D"},
            {"path": rf"D:\results\comparison\clahe\jyu_outdoor\{mat_name}.mat",   "key": "data", "label": "CLAHE"},
            {"path": rf"D:\results\comparison\deep_hs_prior\jyu_outdoor\{mat_name}.mat", "key": "pred", "label": "DeepHS Pr."},
            {"path": rf"D:\results\comparison\fast_hy_mix\jyu_outdoor\{mat_name}.mat", "key": "data", "label": "FastHyMix"},
            {"path": rf"D:\results\comparison\hcanet\jyu_outdoor\{mat_name}.mat",       "key": "data", "label": "HCANet"},
            {"path": rf"D:\results\comparison\enlightengan\jyu_outdoor\{mat_name}.mat",           "key": "data", "label": "EnlightenGAN"},
            {"path": rf"D:\results\comparison\lrtdtv\jyu_outdoor\{mat_name}.mat",        "key": "data", "label": "LRTDTV"},
            {"path": rf"D:\results\comparison\mr\jyu_outdoor\{mat_name}.mat",            "key": "data", "label": "MR"},
            {"path": rf"D:\results\comparison\rcild\jyu_outdoor\{mat_name}.mat",           "key": "data", "label": "RCILD"},
            {"path": rf"D:\results\comparison\retinexnet\jyu_outdoor\{mat_name}.mat",     "key": "data", "label": "RetinexNet"},
            {"path": rf"D:\results\comparison\exposure_diffusion\jyu_outdoor\{mat_name}.mat",     "key": "data", "label": "ExposureDiff"},
            {"path": rf"D:\results\comparison\cdan\jyu_outdoor\{mat_name}.mat",     "key": "data", "label": "CDAN"},
            {"path": rf"D:\results\comparison\mafnet\jyu_outdoor\{mat_name}.mat",     "key": "data", "label": "MAFNet"},
            {"path": rf"D:\results\comparison\retinexformer\jyu_outdoor\{mat_name}.mat",     "key": "data", "label": "Retinexformer"},
            {"path": rf"D:\results\comparison\retinexmamba\jyu_outdoor\{mat_name}.mat",     "key": "data", "label": "Retinexmamba"}
        ]
        x_locs = [272, 110, 374, 56]
        y_locs = [149, 71, 385, 459]
        waveStart_nm = 414.63
        waveEnd_nm = 985.05

    elif dataset == 2:
        mat_name = '678'
        data_to_compare = [
            {"path": rf"D:\results\comparison\normal\jyu_indoor\{mat_name}.mat", "key": "data", "label": "Ground Truth"},
            {"path": rf"D:\results\comparison\ours\jyu_indoor\{mat_name}.mat",   "key": "ref",  "label": "SS-HSLIE (Ours)"},
            {"path": rf"D:\results\comparison\low\jyu_indoor\{mat_name}.mat",    "key": "data", "label": "Low-light"},
            {"path": rf"D:\results\comparison\bm4d\jyu_indoor\{mat_name}.mat",    "key": "data", "label": "BM4D"},
            {"path": rf"D:\results\comparison\clahe\jyu_indoor\{mat_name}.mat",   "key": "data", "label": "CLAHE"},
            {"path": rf"D:\results\comparison\deep_hs_prior\jyu_indoor\{mat_name}.mat", "key": "pred", "label": "DeepHS Pr."},
            {"path": rf"D:\results\comparison\fast_hy_mix\jyu_indoor\{mat_name}.mat", "key": "data", "label": "FastHyMix"},
            {"path": rf"D:\results\comparison\hcanet\jyu_indoor\{mat_name}.mat",       "key": "data", "label": "HCANet"},
            {"path": rf"D:\results\comparison\enlightengan\jyu_indoor\{mat_name}.mat",           "key": "data", "label": "EnlightenGAN"},
            {"path": rf"D:\results\comparison\lrtdtv\jyu_indoor\{mat_name}.mat",        "key": "data", "label": "LRTDTV"},
            {"path": rf"D:\results\comparison\mr\jyu_indoor\{mat_name}.mat",            "key": "data", "label": "MR"},
            {"path": rf"D:\results\comparison\rcild\jyu_indoor\{mat_name}.mat",           "key": "data", "label": "RCILD"},
            {"path": rf"D:\results\comparison\retinexnet\jyu_indoor\{mat_name}.mat",     "key": "data", "label": "RetinexNet"},
            {"path": rf"D:\results\comparison\exposure_diffusion\jyu_indoor\{mat_name}.mat",     "key": "data", "label": "ExposureDiff"},
            {"path": rf"D:\results\comparison\cdan\jyu_indoor\{mat_name}.mat",     "key": "data", "label": "CDAN"},
            {"path": rf"D:\results\comparison\mafnet\jyu_indoor\{mat_name}.mat",     "key": "data", "label": "MAFNet"},
            {"path": rf"D:\results\comparison\retinexformer\jyu_indoor\{mat_name}.mat",     "key": "data", "label": "Retinexformer"},
            {"path": rf"D:\results\comparison\retinexmamba\jyu_indoor\{mat_name}.mat",     "key": "data", "label": "Retinexmamba"}
        ]
        x_locs = [194, 300, 110, 245]
        y_locs = [103, 180, 325, 440]
        waveStart_nm = 414.63
        waveEnd_nm = 985.05

    # Parameters
    # These locations and parameters come directly from the MATLAB code.
    window_size = 5  # must be an odd number

    # Load data
    # Load the first hyperspectral image to get its dimensions.
    first_entry = data_to_compare[0]
    mat_contents = loadmat(first_entry["path"])
    # Extract the field specified by the key.
    data_field = mat_contents[first_entry["key"]]
    #data_field = np.rot90(data_field, k=-1, axes=(0, 1))
    # Assume the data is in shape (h, w, c)
    h, w, c = data_field.shape
    # Create a list to accumulate the hyperspectral images.
    hs_images = [data_field]  # first image

    # Load remaining images, reshape each to (h, w, c) and append to the list.
    for entry in data_to_compare[1:]:
        mat_contents = loadmat(entry["path"])
        cur_data = mat_contents[entry["key"]]
        #cur_data = np.rot90(data_field, k=-1, axes=(0, 1))
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

        rgb_data = visualize_hsi_false_color(cur_data, waveStart_nm, waveEnd_nm, normalize=False)
        
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 16,
            'axes.linewidth': 1.2,
            'xtick.direction': 'in',
            'ytick.direction': 'in'
            })

        plt.figure(figsize=(8, 6))
        # Display the selected spectral band (subtract one for 0-index)
        #plt.imshow(cur_data[:, :, sample_band - 1], cmap='gray')
        plt.imshow(rgb_data)
        plt.axis('off')
        
        # For the first image, overlay red circles and text at given locations.
        if i == 0:
            # Adjust coordinates: MATLAB indices are 1-based, so subtract 1 for Python array coordinates.
            x_coords = np.array(x_locs) - 1
            y_coords = np.array(y_locs) - 1
            letters = string.ascii_lowercase  # 'abcdefghijklmnopqrstuvwxyz'
            
            # Plot red circles.
            plt.plot(x_coords, y_coords, 'ro', markersize=8, linewidth=2)
            
            # Add text labels near the points.
            for j, (x, y) in enumerate(zip(x_coords, y_coords)):
                letter = letters[j]
                # Offset the text (adjust as necessary)
                plt.text(
                    x + 15,
                    y - 15,
                    f'({letter})',
                    color='blue',
                    fontsize=20,
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
        filename = "hs_image_" + mat_name + "_" + str(dataset) + ".eps"
        plt.tight_layout()
        #plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
        plt.close()

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 19,
        'axes.linewidth': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'axes.prop_cycle': line_color_style_cycler()
    })
    
    # Plot spectra (Each plot is saved separately)
    # For each (x,y) location, compute the local spectrum across all algorithms.
    # Create one figure per (x,y) location.
    # Note: Convert MATLAB 1-indexing to Python 0-indexing.
    for idx in range(len(x_locs)):
        letter = letters[idx]
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
        
        # Create a new figure for this spectrum.
        plt.figure(figsize=(16, 9))
        
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
                plt.plot(spectrum, linewidth=3.5, color='r', linestyle='-')
            elif data_to_compare[j]["label"]=="Ground Truth":
                plt.plot(spectrum, linewidth=3.5, color='b', linestyle='-')
            elif data_to_compare[j]["label"]=="Low-light":
                plt.plot(spectrum, linewidth=3.5, color='g', linestyle='--')
            else:
                plt.plot(spectrum, linewidth=2.5)
            legend_labels.append(data_to_compare[j]["label"])
        
        plt.xlabel('Band Number')
        plt.ylabel('Intensity')
        #plt.legend(legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=math.ceil(len(data_to_compare)/2))
        plt.legend(legend_labels, loc='upper left', bbox_to_anchor=(1.01, 1.0), ncol=1)
        plt.tight_layout()
        # Save the spectrum plot as a separate file.
        spectrum_filename = f"spectrum_at_{letter}_{str(dataset)}_{mat_name}.eps"
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, spectrum_filename), bbox_inches='tight')
        plt.close()
