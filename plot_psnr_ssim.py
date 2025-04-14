import numpy as np
import matplotlib.pyplot as plt
from metrics import multi_img_bandwise_metrics
from utils import select_hsi_wavelengths

def get_metrics(algorithms, data_min=None, data_max=None):
    # Pre-compute metrics for all algorithms and store them in a results dictionary.
    metrics = {}
    count = 0
    print('\n')
    for algo_name, values in algorithms.items():
        psnr, ssim = multi_img_bandwise_metrics(
            preds_path=values['preds_path'],
            labels_path=values['labels_path'],
            data_min=data_min,
            data_max=data_max,
            matKeyPrediction=values['matKeyPred'],
            matKeyGt=values['matKeyGt']
            )
        metrics[algo_name] = {'psnr': psnr, 'ssim': ssim}
        count += 1
        print(f'Calculated metrics for image {count}/{len(algorithms)}')
    
    return metrics

def plot_vectors(wavelengths, metrics, figsize=(8, 6), legend='upper right', font_family='serif', font_size=12, linewidth=2, axes_linewidth=1.2, save_path=''):
    plt.rcParams.update({
        'font.family': font_family,
        'font.size': font_size,
        'axes.linewidth': axes_linewidth,
        'xtick.direction': 'in',
        'ytick.direction': 'in'
        })

    # ------------------------------
    # Figure 1: PSNR Plot
    # ------------------------------
    plt.figure(figsize=figsize)
    for algo_name, metric in metrics.items():
        if algo_name=='SS-HSLIE (Ours)':
            plt.plot(wavelengths, metric['psnr'], label=algo_name, linewidth=linewidth, color='red')
        else:
            plt.plot(wavelengths, metric['psnr'], label=algo_name, linewidth=linewidth)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("PSNR (dB)")
    plt.grid(True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(metrics)//2+1)
    plt.tight_layout()
    plt.savefig(save_path + "/psnr_vector.eps")

    # ------------------------------
    # Figure 2: SSIM Plot
    # ------------------------------
    plt.figure(figsize=figsize)
    for algo_name, metric in metrics.items():
        if algo_name=='SS-HSLIE (Ours)':
            plt.plot(wavelengths, metric['ssim'], label=algo_name, linewidth=linewidth, color='red')
        else:
            plt.plot(wavelengths, metric['ssim'], label=algo_name, linewidth=linewidth)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("SSIM")
    plt.grid(True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(metrics)//2+1)
    plt.tight_layout()
    plt.savefig(save_path + "/ssim_vector.eps")

if __name__ == '__main__':
    globalMax=1.6697606

    wavelengths, num_bands = select_hsi_wavelengths(
        range_start=400,
        range_end=1000,
        total_channels=224,
        d_head=20,
        d_tail=12,
        s=3
        )

    # Dictionary of algorithms with corresponding file paths for PSNR and SSIM calculations.
    label_path = '../PairLIE/data/label_ll'

    '''algorithms = {
        'Ours': {
            'preds_path': 'D:/results/comparison/ours',
            'labels_path': label_path,
            'matKeyPred': 'ref',
            'matKeyGt': 'data'
        },
        'BM4D': {
            'preds_path': 'D:/results/comparison/bm4d',
            'labels_path': label_path,
            'matKeyPred': 'data',
            'matKeyGt': 'data'
        },
    }'''

    # deep hs prior eklenecek. nedense 384 h
    algorithms = {
        'SS-HSLIE (Ours)': {
            'preds_path': 'D:/results/comparison/ours',
            'labels_path': label_path,
            'matKeyPred': 'ref',
            'matKeyGt': 'data'
        },
        'BM4D': {
            'preds_path': 'D:/results/comparison/bm4d',
            'labels_path': label_path,
            'matKeyPred': 'data',
            'matKeyGt': 'data'
        },
        'FastHyMix': {
            'preds_path': 'D:/results/comparison/fast_hy_mix',
            'labels_path': label_path,
            'matKeyPred': 'data',
            'matKeyGt': 'data'
        },
        'HCANet': {
            'preds_path': 'D:/results/comparison/hcanet',
            'labels_path': label_path,
            'matKeyPred': 'data',
            'matKeyGt': 'data'
        },
        'HE': {
            'preds_path': 'D:/results/comparison/he',
            'labels_path': label_path,
            'matKeyPred': 'data',
            'matKeyGt': 'data'
        },
        'LRTDTV': {
            'preds_path': 'D:/results/comparison/lrtdtv',
            'labels_path': label_path,
            'matKeyPred': 'data',
            'matKeyGt': 'data'
        },
        'MR': {
            'preds_path': 'D:/results/comparison/mr',
            'labels_path': label_path,
            'matKeyPred': 'data',
            'matKeyGt': 'data'
        },
        'MSR': {
            'preds_path': 'D:/results/comparison/msr',
            'labels_path': label_path,
            'matKeyPred': 'data',
            'matKeyGt': 'data'
        },
        'RetinexNet': {
            'preds_path': 'D:/results/comparison/retinexnet',
            'labels_path': label_path,
            'matKeyPred': 'data',
            'matKeyGt': 'data'
        },
    }
    
    metrics = get_metrics(algorithms=algorithms, data_min=None, data_max=globalMax)
    
    plot_vectors(
        wavelengths=wavelengths,
        metrics=metrics,
        figsize=(12, 8),
        legend='upper right',
        font_family='serif',
        font_size=12,
        linewidth=2,
        axes_linewidth=1.2,
        save_path='C:/Users/medemirhan/Desktop/mdpi/figures/results'
        )