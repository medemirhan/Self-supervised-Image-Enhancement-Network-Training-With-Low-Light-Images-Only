import os
from glob import glob
import numpy as np
import random
from datetime import datetime
import torch
from model import LowLightEnhance
from utils import load_hsi, Struct
from metrics import calc_metrics

def train(model, args):
    model.train_model(
        train_data_path=args.train_data,
        eval_data_path=args.eval_data,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        num_epochs=args.epoch,
        start_lr=args.start_lr,
        ckpt_dir=args.model_ckpt_dir,
        eval_result_dir=args.eval_result_dir,
        eval_every_epoch=args.eval_every_epoch,
        plot_every_epoch=args.plot_every_epoch
        )

def test(model, args):
    if not os.path.exists(args.test_result_dir):
        os.makedirs(args.test_result_dir)

    test_low_data_name = sorted(glob(os.path.join(args.test_data) + '/*.*'))  # Sorted for reproducibility

    test_low_data = []
    
    print("Found test files:", test_low_data_name)
    for idx in range(len(test_low_data_name)):
        test_low_im = load_hsi(test_low_data_name[idx], matContentHeader=args.mat_key, normalization=args.normalization, max_val=args.global_max, min_val=args.global_min)
        test_low_data.append(test_low_im)

    model.test_model(
        model_dir=args.test_model_dir,
        test_low_data=test_low_data, 
        test_low_data_names=test_low_data_name, 
        save_dir=args.test_result_dir, 
        )

def eval_metrics(args):
    im_dir = args.test_result_dir + '/*.mat'

    data_min = None
    avg_psnr, avg_ssim, avg_sam = calc_metrics(
        im_dir=os.path.normpath(im_dir),
        label_dir=os.path.normpath(args.label_dir),
        data_min=data_min,
        data_max=args.global_max,
        matKeyPrediction='ref',
        matKeyGt='data'
        )

    if data_min == None:
        strMin = str(data_min)
    else:
        strMin = f"{data_min:.3f}"

    # Format the log entry
    log_entry = f"min:{strMin}, max:{args.global_max:.3f}, mpsnr:{avg_psnr:.3f}, mssim:{avg_ssim:.3f}, msam:{avg_sam:.3f}\n"

    with open(args.log_file_path, "a") as log_file:
        # Write the log entry to the file
        log_file.write(log_entry)

def main(args):
    # Set random seeds for reproducibility
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Create model
    model = LowLightEnhance(
        input_channels=args.channels,
        time_stamp=args.timestamp,
        coeff_recon_loss_low=args.coeff_recon_loss_low, 
        coeff_Ismooth_loss_low=args.coeff_Ismooth_loss_low,
        coeff_recon_loss_low_eq=args.coeff_recon_loss_low_eq,
        coeff_R_low_loss_smooth=args.coeff_R_low_loss_smooth,
        coeff_relight_loss=args.coeff_relight_loss,
        coeff_Ismooth_loss_delta=args.coeff_Ismooth_loss_delta,
        coeff_fourier_loss=args.coeff_fourier_loss,
        coeff_spectral_loss=args.coeff_spectral_loss,
        device=device
    )
    
    model.to(device)
    
    # If channels is not given, try to infer from the first training image.
    train_files = sorted(glob(os.path.join(args.train_data, "*.*")))
    if len(train_files) == 0:
        print("No training files found.")
        return
    first_image = load_hsi(train_files[0], matContentHeader=args.mat_key,
                           normalization=args.normalization,
                           max_val=args.global_max, min_val=args.global_min)
    if args.channels is None:
        args.channels = first_image.shape[-1]
    
    if args.phase == 'train':
        train(model, args)
    elif args.phase == 'test':
        test(model, args)
        eval_metrics(args)
    elif args.phase == 'train_and_test':
        train(model, args)
        test(model, args)
        eval_metrics(args)
    else:
        print('[!] Unknown phase')
        exit(0)

if __name__ == '__main__':
    args = Struct()

    # Common args
    args.use_gpu = 1
    args.gpu_idx = '0'
    args.gpu_mem = float(0.8)
    args.decom = 0
    args.timestamp = f'{datetime.now():{""}%Y%m%d_%H%M%S}'

    # Data related args
    args.mat_key = 'data'
    args.channels = 64
    #args.global_min = 0.
    #args.global_max = 0.005019044472441
    args.global_min = 0.0708354
    args.global_max = 1.7410845
    args.normalization = 'global_normalization'

    args.coeff_recon_loss_low = 10
    args.coeff_Ismooth_loss_low = 1
    args.coeff_recon_loss_low_eq = 1
    args.coeff_R_low_loss_smooth = 1
    args.coeff_relight_loss = 0.2
    args.coeff_Ismooth_loss_delta = 20
    args.coeff_fourier_loss = 0.2
    args.coeff_spectral_loss = 1

    args.batch_size = 1
    args.patch_size = 128
    args.start_lr = 1e-3

    # Change if necessary
    args.train_data = '../PairLIE/data/hsi_dataset_indoor_only/train'
    args.eval_data = '../PairLIE/data/hsi_dataset_indoor_only/eval'
    args.test_data = '../PairLIE/data/hsi_dataset_indoor_only/test'
    args.label_dir = '../PairLIE/data/label_ll'
    args.model_name = 'torch1'
    args.phase = 'train_and_test'
    args.epoch = 400
    args.eval_every_epoch = 100
    args.plot_every_epoch = 100

    if args.phase == 'test':
        args.timestamp = '' # enter timestamp manually

    # Don't change
    args.full_model_name = args.model_name + '_' + args.timestamp
    args.model_ckpt_dir = './checkpoint/' + args.model_name
    args.eval_result_dir = 'D:/sslie/eval_results_' + args.full_model_name
    args.test_result_dir = 'D:/sslie/test_results_' + args.full_model_name
    args.test_model_dir = './checkpoint/' + args.model_name + '/Decom_' + args.timestamp
    args.log_file_path = './logs/' + args.full_model_name + '.log'

    main(args)
