import os
import argparse
import yaml
from glob import glob
import numpy as np
import random
from datetime import datetime
import torch
import mlflow
from model import LowLightEnhance
from utils import load_hsi, Struct
from metrics import calc_metrics

def parse_args():
    # Use default=None so we can detect if the user provided a value.
    parser = argparse.ArgumentParser(description="Parse config from YAML and command-line.")
    parser.add_argument('--config', type=str, default='./config/config_indoor.yml')
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--seed_value', type=int, default=41)
    parser.add_argument('--gpu_idx', type=str, default='0')
    parser.add_argument('--gpu_mem', type=float, default=0.8)
    parser.add_argument('--decom', type=int, default=0)
    parser.add_argument('--mat_key', type=str, default='data')
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--global_min', type=float, default=0.)
    parser.add_argument('--global_max', type=float, default=1.)
    parser.add_argument('--normalization', type=str, default='global_normalization')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--start_lr', type=float, default=0.001)
    parser.add_argument('--lr_update_factor', type=int, default=1)
    parser.add_argument('--lr_update_period', type=int, default=400)
    parser.add_argument('--train_data', type=str, default='../PairLIE/data/hsi_dataset_indoor_only/train')
    parser.add_argument('--eval_data', type=str, default='../PairLIE/data/hsi_dataset_indoor_only/eval')
    parser.add_argument('--test_data', type=str, default='../PairLIE/data/hsi_dataset_indoor_only/test')
    parser.add_argument('--label_dir', type=str, default='../PairLIE/data/label_ll')
    parser.add_argument('--phase', type=str, default='train_and_test')
    parser.add_argument('--epoch', type=int, default=400)
    parser.add_argument('--eval_every_epoch', type=int, default=200)
    parser.add_argument('--plot_every_epoch', type=int, default=200)
    parser.add_argument('--c_loss_reconstruction', type=float, default=10.)
    parser.add_argument('--c_loss_r_fidelity', type=float, default=1.)
    parser.add_argument('--c_loss_i_smooth_low', type=float, default=1.)
    parser.add_argument('--c_loss_i_smooth_delta', type=float, default=20.)
    parser.add_argument('--c_loss_fourier', type=float, default=0.2)
    parser.add_argument('--c_loss_spectral_cons', type=float, default=1.)
    parser.add_argument('--save_reflectance', type=bool, default=False)
    parser.add_argument('--save_illumination', type=bool, default=False)
    parser.add_argument('--save_i_delta', type=bool, default=False)
    parser.add_argument('--model_name', type=str, default=None)

    args = parser.parse_args()
    
    # Load the YAML configuration.
    with open(args.config, 'r') as file:
        config_data = yaml.safe_load(file)
    
    # For each key in the YAML file, set the attribute on args if not already provided.
    for key, value in config_data.items():
        # If the argument was not provided on the command line, assign the YAML value.
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    
    args.timestamp = f'{datetime.now():{""}%Y%m%d_%H%M%S}'
    if args.phase == 'test':
        args.timestamp = '' # enter timestamp manually

    # Don't change
    args.full_model_name = args.model_name + '_' + args.timestamp
    args.model_ckpt_dir = './checkpoint/' + args.model_name
    args.eval_result_dir = 'D:/sslie/eval_results_' + args.full_model_name
    args.test_result_dir = 'D:/sslie/test_results_' + args.full_model_name
    args.test_model_dir = './checkpoint/' + args.model_name + '/Decom_' + args.timestamp
    args.log_file_path = './logs/' + args.full_model_name + '.log'
    
    return args

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
        max_val=args.global_max,
        min_val=args.global_min, 
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
        save_reflectance=args.save_reflectance,
        save_illumination=args.save_illumination,
        save_i_delta=args.save_i_delta
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

    mlflow.log_metric("PSNR_dB", avg_psnr.item())
    mlflow.log_metric("SSIM", avg_ssim.item())
    mlflow.log_metric("SAM", avg_sam.item())

def main(args):
    print("------ PARAMETERS ------")
    for arg, value in vars(args).items():
        print(f"{arg} : {value}")
    print("------------------------")

    # Set random seeds for reproducibility
    random.seed(args.seed_value)
    np.random.seed(args.seed_value)
    torch.manual_seed(args.seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Create model
    model = LowLightEnhance(
        input_channels=args.channels,
        lr=args.start_lr,
        lr_update_factor=args.lr_update_factor,
        lr_update_period=args.lr_update_period,
        time_stamp=args.timestamp,
        c_loss_reconstruction=args.c_loss_reconstruction,
        c_loss_r_fidelity=args.c_loss_r_fidelity,
        c_loss_i_smooth_low=args.c_loss_i_smooth_low,
        c_loss_i_smooth_delta=args.c_loss_i_smooth_delta,
        c_loss_fourier=args.c_loss_fourier,
        c_loss_spectral_cons=args.c_loss_spectral_cons,
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
    
    mlflow.set_experiment(args.full_model_name)
    with mlflow.start_run():
        mlflow.log_param('phase', args.phase)
        mlflow.log_param('data_min', args.global_min)
        mlflow.log_param('data_max', args.global_max)
        mlflow.log_param('seed', args.seed_value)
        mlflow.log_param('patch_size', args.patch_size)

        mlflow.log_artifact('main.py')
        mlflow.log_artifact('model.py')
        mlflow.log_artifact('utils.py')
        mlflow.log_artifact('metrics.py')
        mlflow.log_artifact('main.py')
        mlflow.log_artifact(args.config)
        
        if args.phase == 'train':
            mlflow.log_param('data_train', args.train_data)
            train(model, args)
        elif args.phase == 'test':
            mlflow.log_param('data_test', args.test_data)
            test(model, args)
            eval_metrics(args)
        elif args.phase == 'train_and_test':
            mlflow.log_param('data_train', args.train_data)
            mlflow.log_param('data_test', args.test_data)
            train(model, args)
            test(model, args)
            eval_metrics(args)
        else:
            print('[!] Unknown phase')
            exit(0)

if __name__ == '__main__':
    args = parse_args()
    main(args)
    print("Job finished...")
    '''
    Sample Usage:
        python main_args.py --model_name "baseline" --coeff_recon_loss_low 10 --coeff_Ismooth_loss_low 1 --coeff_recon_loss_low_eq 1 --coeff_R_low_loss_smooth 1 --coeff_relight_loss 0.2 --coeff_Ismooth_loss_delta 20 --coeff_fourier_loss 0.2 --coeff_spectral_loss 1
    '''