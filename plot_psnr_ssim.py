import numpy as np
import matplotlib.pyplot as plt
from metrics import multi_img_bandwise_metrics
from utils import select_hsi_wavelengths
from cycler import cycler
import itertools
import random
import math

random.seed(42)

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

def plot_vectors(wavelengths, metrics, env, figsize=None, font_family='serif', font_size=12, linewidth=2, axes_linewidth=1.2, save_path=''):   
    # Set rcParams with full update
    plt.rcParams.update({
        'font.family': font_family,
        'font.size': font_size,
        'axes.linewidth': axes_linewidth,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'axes.prop_cycle': line_color_style_cycler()
    })

    # ------------------------------
    # Figure 1: PSNR Plot
    # ------------------------------
    plt.figure(figsize=figsize)
    for algo_name, metric in metrics.items():
        if algo_name=='SS-HSLIE (Ours)':
            plt.plot(wavelengths, metric['psnr'], label=algo_name, linestyle='-', linewidth=linewidth, color='r')
        else:
            plt.plot(wavelengths, metric['psnr'], label=algo_name, linewidth=linewidth)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("MPSNR (dB)")
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=math.ceil(len(metrics)/2))
    plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), ncol=1)
    plt.tight_layout()
    plt.savefig(save_path + "/psnr_vector_" + env + ".eps", bbox_inches='tight')

    # ------------------------------
    # Figure 2: SSIM Plot
    # ------------------------------
    plt.figure(figsize=figsize)
    for algo_name, metric in metrics.items():
        if algo_name=='SS-HSLIE (Ours)':
            plt.plot(wavelengths, metric['ssim'], label=algo_name, linestyle='-', linewidth=linewidth, color='r')
        else:
            plt.plot(wavelengths, metric['ssim'], label=algo_name, linewidth=linewidth)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("MSSIM")
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=math.ceil(len(metrics)/2))
    plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), ncol=1)
    plt.tight_layout()
    plt.savefig(save_path + "/ssim_vector_" + env + ".eps", bbox_inches='tight')

if __name__ == '__main__':
    
    env = 'jyu_outdoor'
    
    if env == 'indoor':
        globalMax=1.6697606
        total_channels=224
        d_head=20
        d_tail=12

    else:
        globalMax=4095
        total_channels=204
        d_head=6
        d_tail=6

    label_path = 'D:/jyu/selected_outdoor_64_registration_nonSaturated_splitted_v3/high/test'
    
    algorithms = {
        'SS-HSLIE (Ours)': {
            'preds_path': 'D:/results/comparison/ours/' + env,
            'labels_path': label_path,
            'matKeyPred': 'ref',
            'matKeyGt': 'data'
        },
        'BM4D': {
            'preds_path': 'D:/results/comparison/bm4d/' + env,
            'labels_path': label_path,
            'matKeyPred': 'data',
            'matKeyGt': 'data'
        },
        'FastHyMix': {
            'preds_path': 'D:/results/comparison/fast_hy_mix/' + env,
            'labels_path': label_path,
            'matKeyPred': 'data',
            'matKeyGt': 'data'
        },
        'HCANet': {
            'preds_path': 'D:/results/comparison/hcanet/' + env,
            'labels_path': label_path,
            'matKeyPred': 'data',
            'matKeyGt': 'data'
        },
        'EnlightenGAN': {
            'preds_path': 'D:/results/comparison/enlightengan/' + env,
            'labels_path': label_path,
            'matKeyPred': 'data',
            'matKeyGt': 'data'
        },
        'LRTDTV': {
            'preds_path': 'D:/results/comparison/lrtdtv/' + env,
            'labels_path': label_path,
            'matKeyPred': 'data',
            'matKeyGt': 'data'
        },
        'MR': {
            'preds_path': 'D:/results/comparison/mr/' + env,
            'labels_path': label_path,
            'matKeyPred': 'data',
            'matKeyGt': 'data'
        },
        'CLAHE': {
            'preds_path': 'D:/results/comparison/clahe/' + env,
            'labels_path': label_path,
            'matKeyPred': 'data',
            'matKeyGt': 'data'
        },
        'RetinexNet': {
            'preds_path': 'D:/results/comparison/retinexnet/' + env,
            'labels_path': label_path,
            'matKeyPred': 'data',
            'matKeyGt': 'data'
        },
        'RCILD': {
            'preds_path': 'D:/results/comparison/rcild/' + env,
            'labels_path': label_path,
            'matKeyPred': 'data',
            'matKeyGt': 'data'
        },
        'CDAN': {
            'preds_path': 'D:/results/comparison/cdan/' + env,
            'labels_path': label_path,
            'matKeyPred': 'data',
            'matKeyGt': 'data'
        },
        'ExposureDiff': {
            'preds_path': 'D:/results/comparison/exposure_diffusion/' + env,
            'labels_path': label_path,
            'matKeyPred': 'data',
            'matKeyGt': 'data'
        },
        'Retinexformer': {
            'preds_path': 'D:/results/comparison/retinexformer/' + env,
            'labels_path': label_path,
            'matKeyPred': 'data',
            'matKeyGt': 'data'
        },
        'Retinexmamba': {
            'preds_path': 'D:/results/comparison/retinexmamba/' + env,
            'labels_path': label_path,
            'matKeyPred': 'data',
            'matKeyGt': 'data'
        },
        'MAFNet': {
            'preds_path': 'D:/results/comparison/mafnet/' + env,
            'labels_path': label_path,
            'matKeyPred': 'data',
            'matKeyGt': 'data'
        }
    }

    if env != 'outdoor':
        algorithms['DHS Pr'] = {
            'preds_path': 'D:/results/comparison/deep_hs_prior/' + env,
            'labels_path': label_path,
            'matKeyPred': 'pred',
            'matKeyGt': 'data'
        }

    wavelengths, num_bands = select_hsi_wavelengths(
        range_start=400,
        range_end=1000,
        total_channels=total_channels,
        d_head=d_head,
        d_tail=d_tail,
        s=3
        )

    metrics = get_metrics(algorithms=algorithms, data_min=None, data_max=globalMax)
    
    plot_vectors(
        wavelengths=wavelengths,
        metrics=metrics,
        env=env,
        figsize=(16, 9),
        font_family='serif',
        font_size=19,
        linewidth=3.5,
        axes_linewidth=1.2,
        save_path='C:/Users/medemirhan/Desktop/jstsp_versions/revision_2/figures/results'
        )