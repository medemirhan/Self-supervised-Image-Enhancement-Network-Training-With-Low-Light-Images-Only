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

def load_images(file, img_type='rgb', matContentHeader='data', normalize='self', globalMax=None):
    if img_type == 'rgb':
        return load_rgb(file)
    elif img_type == 'hsi':
        return load_hsi(file, matContentHeader, normalize, globalMax)

def load_rgb(file):
    im = Image.open(file)
    h= im.size[0]-im.size[0]%4
    w= im.size[1]-im.size[1]%4
    x=(np.array(im,dtype="float32") / 255.0)
    return x

def load_hsi(file, matContentHeader='data', normalize='self', globalMax=None):
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

def save_images(filepath, result_1, result_2 = None):
    result_1 = np.squeeze(result_1)
    result_2 = np.squeeze(result_2)

    if not result_2.any():
        cat_image = result_1
    else:
        cat_image = np.concatenate([result_1, result_2], axis = 1)

    im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
    im.save(filepath, 'png')

def histeq(im,nbr_bins = 256):
    """对一幅灰度图像进行直方图均衡化"""
    #计算图像的直方图
    #在numpy中，也提供了一个计算直方图的函数histogram(),第一个返回的是直方图的统计量，第二个为每个bins的中间值
    imhist,bins = histogram(im.flatten(),nbr_bins,normed= True)
    cdf = imhist.cumsum()   #
    cdf = 1.0*cdf / cdf[-1]
    #使用累积分布函数的线性插值，计算新的像素值
    im2 = interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape)

def adapthisteq(im,NumTiles=8,ClipLimit=0.01,NBins=256):
# other methods can be tried too！not only histeq ，like LAHE or others in the max channel
    mri_img = im * 255.0;
    mri_img = mri_img.astype('uint8')

    r, c, h = mri_img.shape
    if h==1:
        temp = mri_img
        clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8,8))
        mri_img = clahe.apply(temp)
    elif h==3: 
        for k in range(h):
            temp = mri_img[:,:,k]
            clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8,8))
            mri_img[:,:,k] = clahe.apply(temp)
    return  (np.array(mri_img, dtype="float32") / 255.0).reshape(im.shape)
