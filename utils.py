#coding=utf-8
import numpy as np
from PIL import Image
from pylab import *
import tensorflow as tf
import cv2
from PIL import ImageFilter
import scipy.io as sio

tf.compat.v1.disable_eager_execution()
tf = tf.compat.v1  # Alias tf.compat.v1 as tf

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

def load_hsi(file, matContentHeader='ref', normalize='self', globalMax=None):
    mat = sio.loadmat(file)
    mat = mat[matContentHeader]
    mat = mat.astype('float32')

    x = np.array(mat, dtype='float32')

    if normalize is not None:
        if normalize == 'self':
            x = x / np.amax(mat)
        elif normalize == 'global':
            if globalMax == None:
                raise("Error: max value is not provided for normalization.")
            else:
                x = x / globalMax

    return x

def load_images(file, matContentHeader='ref', normalize=None, globalMax=None):
    """Load hyperspectral images. Assumes data is stored in a format that can be read as a numpy array."""

    # Modified to handle hyperspectral data
    try:
        # Try loading as hyperspectral data
        data = load_hsi(file, matContentHeader, normalize, globalMax)
        # Normalize to [0, 1]
        return (data.astype("float32") / np.max(data))
    except:
        # Fallback to regular image loading
        im = Image.open(file)
        h = im.size[0]-im.size[0]%4
        w = im.size[1]-im.size[1]%4
        x = (np.array(im,dtype="float32") / 255.0)
        return x

def channel_wise_normalization(image):
    """Normalize each channel independently"""
    mean_channels = np.mean(np.mean(image, axis=0), axis=0)
    ratio = np.clip(mean_channels/mean_channels.min(), 1.0, 1.1)
    normalized_image = image.copy()
    
    for i in range(image.shape[-1]):
        normalized_image[:,:,i] = normalized_image[:,:,i] / ratio[i]
    
    return normalized_image

def save_images(filepath, result_1, result_2 = None):
    """Save hyperspectral images as .mat files with key 'ref'"""

    result_1 = np.squeeze(result_1)
    
    if result_2 is not None:
        result_2 = np.squeeze(result_2)
        # Save both results as .mat files
        sio.savemat(filepath[:-4] + '_part1.mat', {'ref': result_1})
        sio.savemat(filepath[:-4] + '_part2.mat', {'ref': result_2})
        
        # Also save a RGB visualization for quick viewing
        # Use first three channels or average channels into three bands
        if result_1.shape[-1] >= 3:
            rgb_1 = result_1[:,:,:3]
            rgb_2 = result_2[:,:,:3]
        else:
            # Average channels into three bands for visualization
            splits = np.array_split(range(result_1.shape[-1]), 3)
            rgb_1 = np.stack([np.mean(result_1[:,:,split], axis=-1) for split in splits], axis=-1)
            rgb_2 = np.stack([np.mean(result_2[:,:,split], axis=-1) for split in splits], axis=-1)
            
        cat_image = np.concatenate([rgb_1, rgb_2], axis=1)
        im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
        im.save(filepath, 'png')
    else:
        sio.savemat(filepath[:-4] + '.mat', {'ref': result_1})
        # Save RGB visualization
        if result_1.shape[-1] >= 3:
            im = Image.fromarray(np.clip(result_1[:,:,:3] * 255.0, 0, 255.0).astype('uint8'))
            im.save(filepath[:-4] + '.png', 'png')

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

def adapthisteq(im, NumTiles=8, ClipLimit=0.01, NBins=256):
    """Apply adaptive histogram equalization to each channel"""
    mri_img = im * 255.0
    mri_img = mri_img.astype('uint8')
    
    r, c, h = mri_img.shape
    for k in range(h):
        temp = mri_img[:,:,k]
        clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8,8))
        mri_img[:,:,k] = clahe.apply(temp)
    
    return (np.array(mri_img, dtype="float32") / 255.0)

class Struct:
    pass