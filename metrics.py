import os
import glob
import tensorflow as tf
from utils import load_hsi

def ssim(input, target, data_min=0.0, data_max=1.0, multiscale=False):
    if data_max <= data_min:
        raise("Error: Max value of the data must be greater than the min value.")

    data_range = data_max - data_min

    # Expand dimensions to add batch and channel dimensions (needed for SSIM)
    im1 = tf.expand_dims(input, axis=0)  # Shape: [1, H, W, C]
    im2 = tf.expand_dims(target, axis=0)  # Shape: [1, H, W, C]

    if multiscale:
        return tf.image.ssim_multiscale(im1, im2, data_range)
    else:
        return tf.image.ssim(im1, im2, data_range)

def psnr(input, target, data_min=0.0, data_max=1.0):
    if data_max <= data_min:
        raise("Error: Max value of the data must be greater than the min value.")

    data_range = data_max - data_min

    return tf.image.psnr(input, target, data_range)

def sam(input, target, reduction='elementwise_mean'):
    #TODO
    return tf.constant(0.0)

def metrics(im_dir, label_dir, data_min=0.0, data_max=1.0, matKeyPrediction='data', matKeyGt='data'):    
    avg_psnr = 0
    avg_ssim = 0
    avg_ssim_ms = 0
    avg_sam = 0
    n = 0
    for item in sorted(glob.glob(im_dir)):
        n += 1
        im1 = load_hsi(item, matContentHeader=matKeyPrediction)
        name = item.split('\\')[-1]
        im2 = load_hsi(os.path.join(label_dir, name), matContentHeader=matKeyGt)

        if im1.shape[0] < im2.shape[0]:
            im2 = im2[:im1.shape[0], :, :]

        score_psnr = psnr(im1, im2, data_min=data_min, data_max=data_max) # data range onemli. incele!
        score_ssim = ssim(im1, im2, data_min=data_min, data_max=data_max) # data range onemli. incele!
        score_ssim_ms = ssim(im1, im2, data_min=data_min, data_max=data_max, multiscale=True) # data range onemli. incele!
        score_sam = sam(im1, im2, reduction='elementwise_mean') # reduction onemli. incele!
    
        avg_psnr += score_psnr
        avg_ssim += score_ssim
        avg_ssim_ms += score_ssim_ms
        avg_sam += score_sam

    avg_psnr = avg_psnr / n
    avg_ssim = avg_ssim / n
    avg_ssim_ms = avg_ssim_ms / n
    avg_sam = avg_sam / n

    return avg_psnr.numpy().item(), avg_ssim.numpy().item(), avg_ssim_ms.numpy().item(), avg_sam.numpy().item()

if __name__ == '__main__':

    globalMin = 0.0708354
    globalMax = 1.7410845
    lowLightMin = 0.0708354
    lowLightMax = 0.2173913
    normalLightMin = 0.0708354
    normalLightMax = 1.7410845

    globalMin = 0.
    globalMax = 0.005019044472441
    
    '''im_dir = 'D:/sslie/test_results/non_scaled/renamed/*.mat'
    label_dir = '../PairLIEdata/label_ll'''

    im_dir = 'D:/sslie/test_results_2nd/non_scaled/renamed/*.mat'
    label_dir = '../PairLIE/data/CZ_hsdb/lowered_1.9/gt'

    avg_psnr, avg_ssim, avg_ssim_ms, avg_sam = metrics(
        im_dir=os.path.normpath(im_dir),
        label_dir=os.path.normpath(label_dir),
        data_min=globalMin,
        data_max=globalMax,
        matKeyPrediction='ref',
        matKeyGt='ref'
        )

    print("\n===> Avg.PSNR            : {:.4f} dB".format(avg_psnr))
    print("===> Avg.SSIM            : {:.4f}".format(avg_ssim))
    print("===> Avg.SSIM Multiscale : {:.4f}".format(avg_ssim_ms))
    print("===> Avg.SAM             : {:.4f}".format(avg_sam))
