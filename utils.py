import numpy as np
import scipy.io as sio
from pylab import histogram, interp

class Struct:
    pass

def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)

def load_hsi(file, matContentHeader='ref', normalization=None, max_val=None, min_val=None):
    mat = sio.loadmat(file)
    mat = mat[matContentHeader]
    mat = mat.astype('float32')

    x = np.array(mat, dtype='float32')

    if normalization == 'self':
        x = self_normalization(x)
    elif normalization == 'global_normalization':
        x = global_normalization(x, max_val, min_val)
    elif normalization == 'per_channel_normalization':
        x = per_channel_normalization(x)
    elif normalization == 'per_channel_standardization':
        x = per_channel_standardization(x)
    elif normalization == None:
        pass
    else:
        raise NotImplementedError(normalization + ' is not implemented')

    return x

def per_channel_normalization(x):
    """
    Returns normalized datacube with values scaled between 0 and 1 for each channel.
    """
    # Get the min and max values per channel
    min_vals = np.min(x, axis=(0, 1), keepdims=True)  # Shape: (1, 1, channels)
    max_vals = np.max(x, axis=(0, 1), keepdims=True)  # Shape: (1, 1, channels)

    # Avoid division by zero: if max == min, set the range to 1
    range_vals = np.where(max_vals > min_vals, max_vals - min_vals, 1)
    
    # Normalize each channel independently
    result = (x - min_vals) / range_vals

    return result

def global_normalization(x, max_val=None, min_val=None):
    """
    Normalizes the entire image (all channels together) using global min/max values
    """
    if max_val == None:
        raise("Error: max value is not provided for normalization.")
    else:
        if min_val == None:
            min_val = 0.
        if min_val > max_val:
            raise("Error: min value cannot be larger than the max value for normalization.")
    
    return (x - min_val) / (max_val - min_val)

def self_normalization(x):
    """
    Max of the datacube is mapped to 1, other values are mapped accordingly.
    """
    return x / np.max(x)

def per_channel_standardization(x):
    """
    Returns standardized datacube where each channel has a mean of 0 and a standard deviation of 1.
    """
    # Compute mean and standard deviation per channel
    mean_vals = np.mean(x, axis=(0, 1), keepdims=True)  # Shape: (1, 1, channels)
    std_vals = np.std(x, axis=(0, 1), keepdims=True)    # Shape: (1, 1, channels)

    # Avoid division by zero: if std == 0, set std to 1
    std_vals = np.where(std_vals > 0, std_vals, 1)

    # Standardize each channel independently
    result = (x - mean_vals) / std_vals

    return result

def inverse_per_channel_normalization(predictions, min_vals, max_vals):
    """
    Inverse scaling for Per-Channel Normalization.

    Parameters:
        predictions (numpy.ndarray): Normalized predictions, shape (H, W, C).
        min_vals (numpy.ndarray): Minimum values per channel, shape (1, 1, C).
        max_vals (numpy.ndarray): Maximum values per channel, shape (1, 1, C).

    Returns:
        numpy.ndarray: Predictions scaled back to the original range.
    """
    # Ensure the input is a NumPy array
    predictions = np.asarray(predictions)
    
    # Undo normalization
    original_predictions = predictions * (max_vals - min_vals) + min_vals

    return original_predictions

def inverse_global_normalization(predictions, global_min, global_max):
    """
    Inverse scaling for Global Normalization.

    Parameters:
        predictions (numpy.ndarray): Normalized predictions, shape (H, W, C).
        global_min (float): Global minimum value from the training set.
        global_max (float): Global maximum value from the training set.

    Returns:
        numpy.ndarray: Predictions scaled back to the original range.
    """
    # Ensure the input is a NumPy array
    predictions = np.asarray(predictions)
    
    # Undo normalization
    original_predictions = predictions * (global_max - global_min) + global_min

    return original_predictions

def inverse_per_channel_standardization(predictions, mean_vals, std_vals):
    """
    Inverse scaling for Per-Channel Standardization.

    Parameters:
        predictions (numpy.ndarray): Standardized predictions, shape (H, W, C).
        mean_vals (numpy.ndarray): Mean values per channel, shape (1, 1, C).
        std_vals (numpy.ndarray): Standard deviation per channel, shape (1, 1, C).

    Returns:
        numpy.ndarray: Predictions scaled back to the original range.
    """
    # Ensure the input is a NumPy array
    predictions = np.asarray(predictions)
    
    # Undo standardization
    original_predictions = predictions * std_vals + mean_vals

    return original_predictions

def save_hsi(filepath, data, postfix=None, key='ref'):
    """Save hyperspectral image as .mat file"""

    data = np.squeeze(data)

    savepath = filepath[:-4]
    if postfix != None:
        savepath += postfix
    
    sio.savemat(savepath + '.mat', {key: data})

def histeq(im, nbr_bins=256):
    """Perform histogram equalization on each channel independently"""
    output = np.zeros_like(im)
    
    for i in range(im.shape[-1]):
        imhist, bins = histogram(im[:,:,i].flatten(), nbr_bins, normed=True)
        cdf = imhist.cumsum()
        cdf = 1.0 * cdf / cdf[-1]
        im2 = interp(im[:,:,i].flatten(), bins[:-1], cdf)
        output[:,:,i] = im2.reshape(im[:,:,i].shape)
    
    return output

def adaptive_lr(num_epochs, divide_period, divide_by, initial_value):
    # Create an array to store the result
    arr = np.zeros(num_epochs)
    
    current_value = initial_value
    for i in range(0, num_epochs, divide_period):
        # Set the next divide_period elements to the current value
        arr[i:i+divide_period] = current_value
        # Divide the value by divide_by for the next block
        current_value /= divide_by
    
    return arr