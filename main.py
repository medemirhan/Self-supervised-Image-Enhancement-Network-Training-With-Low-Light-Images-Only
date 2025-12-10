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
from logger import Logger
import sys

def parse_args():
    default_values = {
        'use_gpu': 1,
        'seed_value': 41,
        'gpu_idx': '0',
        'gpu_mem': 0.8,
        'decom': 0,
        'mat_key': 'data',
        'channels': 64,
        'global_min': 0.,
        'global_max': 1.,
        'normalization': 'global_normalization',
        'batch_size': 1,
        'patch_size': 128,
        'start_lr': 0.001,
        'lr_update_factor': 1,
        'lr_update_period': 400,
        'train_data': '../PairLIE/data/hsi_dataset_indoor_only/train',
        'eval_data': '../PairLIE/data/hsi_dataset_indoor_only/eval',
        'test_data': '../PairLIE/data/hsi_dataset_indoor_only/test',
        'label_dir': '../PairLIE/data/label_ll',
        'phase': 'train_and_test',
        'epoch': 400,
        'eval_every_epoch': 200,
        'plot_every_epoch': 200,
        'c_loss_reconstruction': 10.,
        'c_loss_r_fidelity': 1.,
        'c_loss_i_smooth_low': 1.,
        'c_loss_i_smooth_delta': 20.,
        'c_loss_fourier': 0.2,
        'c_loss_spectral_cons': 1.,
        'alpha_i_smooth_low': 1.,
        'alpha_i_smooth_delta': 10.,
        'save_reflectance': False,
        'save_illumination': False,
        'save_i_delta': False,
        'model_name': 'no_name_model',
        'pretrained_model': '',
        'freeze_decom_epochs': 0
    }

    parser = argparse.ArgumentParser(description="Parse config from YAML and command-line.")
    parser.add_argument('--config', type=str, default='./config/config_indoor.yml')

    # Add rest with default=None
    for key, val in default_values.items():
        parser.add_argument(f'--{key}', type=type(val), default=None)

    args = parser.parse_args()
    
    # Load config from YAML
    with open(args.config, 'r') as file:
        config_data = yaml.safe_load(file)

    # Apply priorities: CLI > YAML > Default
    for key, default_val in default_values.items():
        current_val = getattr(args, key)
        if current_val is None:
            setattr(args, key, config_data.get(key, default_val))
    
    postfix = ''
    args.timestamp = f'{datetime.now():{""}%Y%m%d_%H%M%S}'
    if args.phase == 'test':
        postfix = '_test_' + args.timestamp
        args.timestamp = '20250926_140412' # enter timestamp manually

    # Don't change
    args.full_model_name = args.model_name + '_' + args.timestamp + postfix
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
        label_dir=args.label_dir,
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
    log_filepath = os.path.join('logs', 'console_output_' + args.full_model_name + '.log')
    original_stdout = sys.stdout
    logger = Logger(log_filepath)
    sys.stdout = logger
    
    try:
        print(f"Console output is being logged to: {log_filepath}")
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
            alpha_i_smooth_low=args.alpha_i_smooth_low,
            alpha_i_smooth_delta=args.alpha_i_smooth_delta,
            device=device,
            global_min=args.global_min,
            global_max=args.global_max,
            save_reflectance=args.save_reflectance,
            save_illumination=args.save_illumination,
            save_i_delta=args.save_i_delta
        )
        
        model.to(device)
        
        # Load pretrained model if specified
        if hasattr(args, 'pretrained_model') and args.pretrained_model and os.path.exists(args.pretrained_model):
            print(f"Loading pretrained model from: {args.pretrained_model}")
            checkpoint = torch.load(args.pretrained_model, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                model.load_state_dict(checkpoint)
            
            print("Pretrained model loaded successfully!")
            
            # Optionally freeze DecomNet for initial epochs
            if hasattr(args, 'freeze_decom_epochs') and args.freeze_decom_epochs > 0:
                print(f"DecomNet will be frozen for the first {args.freeze_decom_epochs} epochs")
                model.freeze_decom_epochs = args.freeze_decom_epochs
        
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
            mlflow.log_param('model_name', args.full_model_name)

            mlflow.log_param('c_loss_reconstruction', args.c_loss_reconstruction)
            mlflow.log_param('c_loss_r_fidelity', args.c_loss_r_fidelity)
            mlflow.log_param('c_loss_i_smooth_low', args.c_loss_i_smooth_low)
            mlflow.log_param('c_loss_i_smooth_delta', args.c_loss_i_smooth_delta)
            mlflow.log_param('c_loss_fourier', args.c_loss_fourier)
            mlflow.log_param('c_loss_spectral_cons', args.c_loss_spectral_cons)
            mlflow.log_param('alpha_i_smooth_low', args.alpha_i_smooth_low)
            mlflow.log_param('alpha_i_smooth_delta', args.alpha_i_smooth_delta)

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
            
            mlflow.log_artifact(log_filepath, artifact_path="run_logs")
    
    except Exception as e:
        print(f"\n--- An error occurred: {e} ---")
        import traceback
        traceback.print_exc()

    finally:
        # --- Restore stdout and close the log file ---
        if 'logger' in locals() and sys.stdout == logger:
            sys.stdout = original_stdout
            logger.close()
        print(f"Final console output log is available at: {log_filepath}")

if __name__ == '__main__':
    args = parse_args()
    main(args)
    print("Job finished...")
    '''
    Sample Usage:
        python main_args.py --model_name "baseline" --coeff_recon_loss_low 10 --coeff_Ismooth_loss_low 1 --coeff_recon_loss_low_eq 1 --coeff_R_low_loss_smooth 1 --coeff_relight_loss 0.2 --coeff_Ismooth_loss_delta 20 --coeff_fourier_loss 0.2 --coeff_spectral_loss 1
    '''