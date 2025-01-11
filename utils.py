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

def load_hsi(file, matContentHeader='ref', normalize=None, maxVal=None, minVal=None):
    mat = sio.loadmat(file)
    mat = mat[matContentHeader]
    mat = mat.astype('float32')

    x = np.array(mat, dtype='float32')

    if normalize == 'self':
        x = x / np.amax(mat)
    elif normalize == 'global':
        if maxVal == None:
            raise("Error: max value is not provided for normalization.")
        else:
            if minVal == None:
                minVal = 0.
            if minVal > maxVal:
                raise("Error: min value cannot be larger than the max value for normalization.")
            x = (x - minVal) / (maxVal - minVal)

    #return (data.astype("float32") / np.max(data))

    return x

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