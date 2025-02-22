import os
import argparse
from glob import glob
import tensorflow as tf
from model import lowlight_enhance
from utils import *
from datetime import datetime
import metrics
import random

tf.compat.v1.disable_eager_execution()
tf = tf.compat.v1  # Alias tf.compat.v1 as tf

seed_value = 42
tf.set_random_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)

# Configure GPU options
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = False  # Set allow_growth to False
config.intra_op_parallelism_threads = 1
config.inter_op_parallelism_threads = 1

os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.config.set_visible_devices([], 'GPU')

# Set the session with the config
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

def lowlight_train(lowlight_enhance, args):
    if not os.path.exists(args.model_ckpt_dir):
        os.makedirs(args.model_ckpt_dir)
    if not os.path.exists(args.eval_result_dir):
        os.makedirs(args.eval_result_dir)

    lr = args.start_lr * np.ones([args.epoch])
    lr[40:] = lr[0] / 10.0
    #lr = adaptive_lr(args.epoch, args.lr_div_period, args.lr_div_factor, args.start_lr)

    train_low_data = []
    train_high_data = []
    train_low_data_eq = []

    # Load training data
    train_low_data_names = glob(args.train_data + '/*.*')  # Modified to accept any extension
    train_low_data_names.sort()
    train_high_data_names = glob(args.train_data + '/*.*')  # Modified to accept any extension
    train_high_data_names.sort()
    assert len(train_low_data_names) == len(train_high_data_names)
    print('[*] Number of training data: %d' % len(train_low_data_names))

    for idx in range(len(train_low_data_names)):
        low_im = load_hsi(train_low_data_names[idx], matContentHeader=args.mat_key, normalization=args.normalization, max_val=args.global_max, min_val=args.global_min)
        train_low_data.append(low_im)
        high_im = load_hsi(train_high_data_names[idx], matContentHeader=args.mat_key, normalization=args.normalization, max_val=args.global_max, min_val=args.global_min)
        train_high_data.append(high_im)
        
        # Calculate max channel for equalization (across all spectral bands)
        '''train_low_data_max_chan = np.max(high_im, axis=2, keepdims=True)
        train_low_data_max_channel = histeq(train_low_data_max_chan)'''
        train_low_data_max_channel = pca_projection(high_im)
        train_low_data_eq.append(train_low_data_max_channel)

    eval_low_data = []
    eval_low_data_name = glob(args.eval_data + '/*.*')  # Modified to accept any extension

    for idx in range(len(eval_low_data_name)):
        eval_low_im = load_hsi(eval_low_data_name[idx], matContentHeader=args.mat_key, normalization=args.normalization, max_val=args.global_max, min_val=args.global_min)
        eval_low_data.append(eval_low_im)

    if args.decom == 1:
        tr_phase = "Decom"
    else:
        tr_phase = "Relight"

    lowlight_enhance.train(
        train_low_data,
        train_low_data_eq, 
        eval_low_data,
        train_high_data, 
        batch_size=args.batch_size, 
        patch_size=args.patch_size, 
        epoch=args.epoch, 
        lr=lr, 
        eval_dir=args.eval_result_dir, 
        ckpt_dir=args.model_ckpt_dir, 
        eval_every_epoch=args.eval_every_epoch, 
        train_phase=tr_phase,
        plot_every_epoch=args.plot_every_epoch
    )

def lowlight_test(lowlight_enhance, args):
    if args.test_data == None:
        print("[!] please provide --test_dir")
        exit(0)
    
    if args.test_model_dir == None:
        print("[!] please provide --test_model_dir")
        exit(0)

    if not os.path.exists(args.test_result_dir):
        os.makedirs(args.test_result_dir)

    test_low_data_name = glob(os.path.join(args.test_data) + '/*.*')  # Modified to accept any extension
    test_low_data = []
    test_high_data = []
    
    print("Found test files:", test_low_data_name)
    for idx in range(len(test_low_data_name)):
        test_low_im = load_hsi(test_low_data_name[idx], matContentHeader=args.mat_key, normalization=args.normalization, max_val=args.global_max, min_val=args.global_min)
        test_low_data.append(test_low_im)

    data_min = None
    data_max = None
    if args.post_scale:
        data_min = args.global_min
        data_max = args.global_max

    lowlight_enhance.test(
        model_dir=args.test_model_dir,
        test_low_data=test_low_data, 
        test_high_data=test_high_data, 
        test_low_data_names=test_low_data_name, 
        save_dir=args.test_result_dir, 
        decom_flag=args.decom,
        lum_factor=args.lum_factor,
        data_min=data_min,
        data_max=data_max
    )

    im_dir = args.test_result_dir + '/*.mat'

    data_min = None
    avg_psnr, avg_ssim, avg_sam = metrics.calc_metrics(
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
    if args.use_gpu:
        print("[*] GPU\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            # Get number of channels from first image if not specified
            if args.channels is None:
                first_image = load_hsi(glob(args.train_data + '/*.*')[0], matContentHeader=args.mat_key, normalization=args.normalization, max_val=args.global_max, min_val=args.global_min)
                args.channels = first_image.shape[-1]
            model = lowlight_enhance(sess, input_channels=args.channels, time_stamp=args.timestamp)
            if args.phase == 'train':
                lowlight_train(model, args)
            elif args.phase == 'test':
                lowlight_test(model, args)
            elif args.phase == 'train_and_test':
                lowlight_train(model, args)
                tf.get_variable_scope().reuse_variables()
                lowlight_test(model, args)
            else:
                print('[!] Unknown phase')
                exit(0)
    else:
        print("[*] CPU\n")
        with tf.Session() as sess:
            model = lowlight_enhance(sess, time_stamp=args.timestamp)
            if args.phase == 'train':
                lowlight_train(model, args)
            elif args.phase == 'test':
                lowlight_test(model, args)
            elif args.phase == 'train_and_test':
                lowlight_train(model, args)
                tf.get_variable_scope().reuse_variables()
                lowlight_test(model, args)
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
    args.lum_factor = 0.2
    args.post_scale = False
    args.lr_div_period = 100
    args.lr_div_factor = 3
    args.eval_every_epoch = 100
    args.plot_every_epoch = 10

    # Data related args
    args.mat_key = 'data'
    args.channels = 64
    '''args.global_min = 0.
    args.global_max = 0.005019044472441'''
    args.global_min = 0.0708354
    args.global_max = 1.7410845
    args.normalization = 'global_normalization'

    # Change if necessary
    args.train_data = '../PairLIE/data/hsi_dataset_indoor_only/train'
    args.eval_data = '../PairLIE/data/hsi_dataset_indoor_only/eval'
    args.test_data = '../PairLIE/data/hsi_dataset_indoor_only/test'
    args.label_dir = '../PairLIE/data/label_ll'
    args.model_name = 'pca_0_1'
    args.phase = 'train_and_test'

    # Train and Eval related args
    args.epoch = 400
    args.batch_size = 1
    args.patch_size = 128
    args.start_lr = 1e-3

    if args.phase == 'test':
        args.timestamp = '' # enter timestamp manually

    args.full_model_name = args.model_name + '_' + args.timestamp

    # Don't change
    args.model_ckpt_dir = './checkpoint/' + args.model_name
    args.eval_result_dir = 'D:/sslie/eval_results_' + args.full_model_name
    args.test_result_dir = 'D:/sslie/test_results_' + args.full_model_name
    args.test_model_dir = './checkpoint/' + args.model_name + '/Decom_' + args.timestamp
    args.log_file_path = './logs/' + args.full_model_name + '.log'

    main(args)