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
    parser.add_argument('--config', type=str, default='./config/config.yml')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--train_data', type=str, default='../PairLIE/data/hsi_dataset_indoor_only/train')
    parser.add_argument('--eval_data', type=str, default='../PairLIE/data/hsi_dataset_indoor_only/eval')
    parser.add_argument('--test_data', type=str, default='../PairLIE/data/hsi_dataset_indoor_only/test')
    parser.add_argument('--label_dir', type=str, default='../PairLIE/data/label_ll')
    parser.add_argument('--coeff_recon_loss_low', type=float, default=10)
    parser.add_argument('--coeff_Ismooth_loss_low', type=float, default=1)
    parser.add_argument('--coeff_recon_loss_low_eq', type=float, default=1)
    parser.add_argument('--coeff_R_low_loss_smooth', type=float, default=1)
    parser.add_argument('--coeff_relight_loss', type=float, default=0.2)
    parser.add_argument('--coeff_Ismooth_loss_delta', type=float, default=20)
    parser.add_argument('--coeff_fourier_loss', type=float, default=0.2)
    parser.add_argument('--coeff_spectral_loss', type=float, default=1)

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
    
    mlflow.set_experiment(args.full_model_name)
    with mlflow.start_run():
        mlflow.log_param('phase', args.phase)
        mlflow.log_param('data_min', args.global_min)
        mlflow.log_param('data_max', args.global_max)
        mlflow.log_param('seed', args.seed_value)

        mlflow.log_artifact('main.py')
        mlflow.log_artifact('model.py')
        mlflow.log_artifact('utils.py')
        mlflow.log_artifact('metrics.py')
        mlflow.log_artifact('main_args.py')
        mlflow.log_artifact('./config/config.yml')
        
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