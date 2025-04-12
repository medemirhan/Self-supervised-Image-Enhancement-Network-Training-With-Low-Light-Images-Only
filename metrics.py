import os
import glob
import torch
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure, spectral_angle_mapper
from utils import load_hsi
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
from skimage.metrics import structural_similarity as ssim_skimage
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')

def psnr(input, target, data_range=None):
    return peak_signal_noise_ratio(input, target, data_range=data_range)

def ssim(input, target, data_range=None):
    im1 = input.unsqueeze(0)
    im2 = target.unsqueeze(0)
    return structural_similarity_index_measure(im1, im2, data_range=data_range)

def sam_bandwise(input, target, reduction='elementwise_mean'):
    im1 = input.unsqueeze(0).unsqueeze(0)
    im2 = target.unsqueeze(0).unsqueeze(0)
    return spectral_angle_mapper(im1, im2, reduction=reduction)

def ssim_bandwise(input, target, data_range=None):
    im1 = input.unsqueeze(0).unsqueeze(0)
    im2 = target.unsqueeze(0).unsqueeze(0)
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

def single_img_bandwise_metrics(pred_path, label_path, data_min=None, data_max=None, matKeyPrediction='data', matKeyGt='data'):
    im1 = load_hsi(pred_path, matContentHeader=matKeyPrediction)
    im2 = load_hsi(label_path, matContentHeader=matKeyGt)

    im1 = torch.from_numpy(im1).to(dtype=torch.float32)
    im2 = torch.from_numpy(im2).to(dtype=torch.float32)

    data_range = None
    if data_min != None and data_max != None:
        data_range = (data_min, data_max)
        print("====> WARNING: Data will be clamped between data range values <====")
    elif data_max != None:
        data_range = data_max

    h, w, c = im1.shape
    psnr_vec = []
    ssim_vec = []
    for i in range(c):
        score_psnr = psnr(im1[:,:,i], im2[:,:,i], data_range=data_range) # data range onemli. incele!
        score_ssim = ssim_bandwise(im1[:,:,i], im2[:,:,i], data_range=data_range) # data range onemli. incele!

        psnr_vec.append(score_psnr)
        ssim_vec.append(score_ssim)

    return np.array(psnr_vec), np.array(ssim_vec)

def multi_img_bandwise_metrics(preds_path, labels_path, data_min=None, data_max=None, matKeyPrediction='data', matKeyGt='data'):
    preds = glob.glob(os.path.join(preds_path, '*.mat'))

    psnr_sum = None
    ssim_sum = None
    count = 0
    for pred_img in preds:
        filename = os.path.basename(pred_img)
        label_img = os.path.join(labels_path, filename)

        psnr_cur, ssim_cur = single_img_bandwise_metrics(
            pred_img,
            label_img,
            data_min=data_min,
            data_max=data_max,
            matKeyPrediction=matKeyPrediction,
            matKeyGt=matKeyGt
            )
        
        if psnr_sum is None:
            psnr_sum = psnr_cur.copy()
        else:
            psnr_sum += psnr_cur

        if ssim_sum is None:
            ssim_sum = ssim_cur.copy()
        else:
            ssim_sum += ssim_cur

        count += 1
    
    psnr_avg_vec = np.array(psnr_sum / count)
    ssim_avg_vec = np.array(ssim_sum / count)

    return psnr_avg_vec, ssim_avg_vec

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
            print("====> WARNING: Data will be clamped between data range values <====".format(avg_psnr))
        elif data_max != None:
            data_range = data_max

        score_psnr = psnr(im1, im2, data_range=data_range) # data range onemli. incele!
        score_ssim = ssim(im1, im2, data_range=data_range) # data range onemli. incele!
        score_sam = sam(im1, im2, reduction='elementwise_mean') # reduction onemli. incele!
    
        print(f'\n===> {name} | PSNR : {score_psnr:.4f}')
        print(f'===> {name} | SSIM : {score_ssim:.4f}')
        print(f'===> {name} | SAM  : {score_sam:.4f}')

        avg_psnr += score_psnr
        avg_ssim += score_ssim
        avg_sam += score_sam

    avg_psnr = avg_psnr / n
    avg_ssim = avg_ssim / n
    avg_sam = avg_sam / n

    return avg_psnr, avg_ssim, avg_sam

if __name__ == '__main__':

    globalMin = 0.0708354
    globalMax = 1.6697606
    lowLightMin = 0.0708354
    lowLightMax = 0.2173913
    normalLightMin = 0.0708354
    normalLightMax = 1.7410845

    im_dir = '../denoising/HCANet/results/dim32/*.mat'
    label_dir = '../PairLIE/data/label_ll'

    avg_psnr, avg_ssim, avg_sam = calc_metrics(
        im_dir=os.path.normpath(im_dir),
        label_dir=os.path.normpath(label_dir),
        data_min=None,
        data_max=globalMax,
        matKeyPrediction='data',
        matKeyGt='data'
        )

    print(f'\n===> Avg.PSNR : {avg_psnr:.4f}')
    print(f'===> Avg.SSIM : {avg_ssim:.4f}')
    print(f'===> Avg.SAM  : {avg_sam:.4f}')
    
    '''psnr_vec, ssim_vec = single_img_bandwise_metrics(
        pred_path='C:/Users/medemirhan/Desktop/comparison/results/msr/5/buildingblock.mat',
        label_path='C:/Users/medemirhan/Desktop/n2n/PairLIE/data/label_ll/buildingblock.mat',
        data_min=None,
        data_max=globalMax,
        matKeyPrediction='data',
        matKeyGt='data'
        )
    
    sio.savemat('./psnr_vec6.mat', {"data": np.array(psnr_vec)})
    sio.savemat('./ssim_vec6.mat', {"data": np.array(ssim_vec)})'''

    multi_img_bandwise_metrics(
        preds_path='C:/Users/medemirhan/Desktop/comparison/results/msr/5',
        labels_path='C:/Users/medemirhan/Desktop/n2n/PairLIE/data/label_ll',
        data_min=None,
        data_max=globalMax,
        matKeyPrediction='data',
        matKeyGt='data'
        )