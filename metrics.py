import os
import glob
import torch
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure, spectral_angle_mapper
from utils import load_hsi
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
from skimage.metrics import structural_similarity as ssim_skimage

def psnr(input, target, data_range=None):
    return peak_signal_noise_ratio(input, target, data_range=data_range)

def ssim(input, target, data_range=None):
    im1 = input.unsqueeze(0)
    im2 = target.unsqueeze(0)
    return structural_similarity_index_measure(im1, im2, data_range=data_range)

def sam(input, target, reduction='elementwise_mean'):
    im1 = input.permute(2, 0, 1).unsqueeze(0)
    im2 = target.permute(2, 0, 1).unsqueeze(0)
    return spectral_angle_mapper(im1, im2, reduction=reduction)

def psnr_sk(input, target, data_range=None):
    if data_range == None:
        return psnr_skimage(input, target)
    else:
        return psnr_skimage(input, target, data_range=data_range[1] - data_range[0])

def ssim_sk(input, target, data_range=None):
    if data_range == None:
        return ssim_skimage(input, target)
    else:
        return ssim_skimage(input, target, data_range=data_range[1] - data_range[0])

def calc_metrics(im_dir, label_dir, data_min=None, data_max=None, matKeyPrediction='data', matKeyGt='data'):    
    avg_psnr = 0
    avg_ssim = 0
    avg_sam = 0
    n = 0
    for item in sorted(glob.glob(im_dir)):
        n += 1
        im1 = load_hsi(item, matContentHeader=matKeyPrediction)
        im1 = torch.from_numpy(im1).to(dtype=torch.float32)
        name = item.split('\\')[-1]
        im2 = load_hsi(os.path.join(label_dir, name), matContentHeader=matKeyGt)
        im2 = torch.from_numpy(im2).to(dtype=torch.float32)

        if im1.shape[0] < im2.shape[0]:
            im2 = im2[:im1.shape[0], :, :]

        data_range = None
        if data_min != None and data_max != None:
            data_range = (data_min, data_max)
            print("\n====> WARNING: Data will be clamped between data range values <====".format(avg_psnr))
        elif data_max != None:
            data_range = data_max

        score_psnr = psnr(im1, im2, data_range=data_range) # data range onemli. incele!
        score_ssim = ssim(im1, im2, data_range=data_range) # data range onemli. incele!
        score_sam = sam(im1, im2, reduction='elementwise_mean') # reduction onemli. incele!
    
        avg_psnr += score_psnr
        avg_ssim += score_ssim
        avg_sam += score_sam

    avg_psnr = avg_psnr / n
    avg_ssim = avg_ssim / n
    avg_sam = avg_sam / n

    return avg_psnr, avg_ssim, avg_sam

if __name__ == '__main__':

    globalMin = 0.0708354
    globalMax = 1.7410845
    lowLightMin = 0.0708354
    lowLightMax = 0.2173913
    normalLightMin = 0.0708354
    normalLightMax = 1.7410845

    '''globalMin = 0.
    globalMax = 0.005019044472441'''

    globalMin = 0.0708354
    globalMax = 0.2173913

    im_dir = 'D:/sslie/test_results_3rd/non_scaled/renamed/*.mat'
    label_dir = '../PairLIE/data/label_ll'

    '''im_dir = 'D:/sslie/test_results_2nd/non_scaled/renamed/*.mat'
    label_dir = '../PairLIE/data/CZ_hsdb/lowered_1.9/gt'''

    '''im_dir = 'D:/sslie/test_results_20250112_165938/*.mat'
    label_dir = '../PairLIE/data/label_ll'''

    avg_psnr, avg_ssim, avg_sam = calc_metrics(
        im_dir=os.path.normpath(im_dir),
        label_dir=os.path.normpath(label_dir),
        data_min=None,
        data_max=globalMax,
        matKeyPrediction='ref',
        matKeyGt='data'
        )

    print("\n===> Avg.PSNR : {:.4f} dB ".format(avg_psnr))
    print("===> Avg.SSIM : {:.4f} ".format(avg_ssim))
    print("===> Avg.SAM  : {:.4f} ".format(avg_sam))
