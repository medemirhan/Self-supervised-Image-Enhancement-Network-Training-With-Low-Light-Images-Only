import os
import argparse
from glob import glob
import tensorflow as tf
from model import lowlight_enhance
from utils import *

tf.compat.v1.disable_eager_execution()
tf = tf.compat.v1  # Alias tf.compat.v1 as tf

seed_value = 42
tf.set_random_seed(seed_value)
np.random.seed(seed_value)

def parse_args():
    parser = argparse.ArgumentParser(description='Hyperspectral Image Enhancement')

    parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
    parser.add_argument('--gpu_idx', dest='gpu_idx', default="0", help='GPU idx')
    parser.add_argument('--gpu_mem', dest='gpu_mem', type=float, default=0.8, help="0 to 1, gpu memory usage")
    parser.add_argument('--phase', dest='phase', default='train', help='train or test')

    parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='number of total epoches')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of samples in one batch')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=48, help='patch size')
    parser.add_argument('--start_lr', dest='start_lr', type=float, default=0.001, help='initial learning rate for adam')
    parser.add_argument('--eval_every_epoch', dest='eval_every_epoch', default=100, help='evaluating and saving checkpoints every # epoch')
    parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='directory for checkpoints')
    parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='directory for evaluating outputs')

    parser.add_argument('--save_dir', dest='save_dir', default='./test_results', help='directory for testing outputs')
    parser.add_argument('--test_dir', dest='test_dir', default='./data/test', help='directory for testing inputs')
    parser.add_argument('--decom', dest='decom', default=1, help='decom flag, 0 for enhanced results only and 1 for decomposition results')
    
    # Add new argument for number of spectral channels
    parser.add_argument('--channels', dest='channels', type=int, default=None, help='number of spectral channels')

    return parser.parse_args()

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
        train_low_data_max_chan = np.max(high_im, axis=2, keepdims=True)
        train_low_data_max_channel = histeq(train_low_data_max_chan)
        train_low_data_eq.append(train_low_data_max_channel)

    eval_low_data = []
    eval_low_data_name = glob(args.eval_data + '/*.*')  # Modified to accept any extension

    for idx in range(len(eval_low_data_name)):
        eval_low_im = load_hsi(eval_low_data_name[idx], matContentHeader=args.mat_key, normalization=args.normalization, max_val=args.global_max, min_val=args.global_min)
        eval_low_data.append(eval_low_im)

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
        train_phase="Decom",
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
            model = lowlight_enhance(sess, input_channels=args.channels)
            if args.phase == 'train':
                lowlight_train(model, args)
            elif args.phase == 'test':
                lowlight_test(model, args)
            else:
                print('[!] Unknown phase')
                exit(0)
    else:
        print("[*] CPU\n")
        with tf.Session() as sess:
            model = lowlight_enhance(sess)
            if args.phase == 'train':
                lowlight_train(model, args)
            elif args.phase == 'test':
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
    args.decom = 1

    # Data related args
    args.mat_key = 'data'
    args.channels = 64
    '''args.global_min = 0.
    args.global_max = 0.005019044472441'''
    args.global_min = 0.0708354
    args.global_max = 1.7410845
    args.normalization = 'global_normalization'

    # Directories
    args.model_ckpt_dir = './checkpoint/global_norm_max_1_74_divide_128p_indoor_recon'
    args.train_data = '../PairLIE/data/hsi_dataset_indoor_only/train'
    args.eval_data = '../PairLIE/data/hsi_dataset_indoor_only/eval'
    args.test_data = '../PairLIE/data/hsi_dataset/test'
    
    args.eval_result_dir = 'D:/sslie/eval_results_global_norm_max_1_74_divide_128p_indoor_recon'
    args.test_result_dir = 'D:/sslie/test_results_20250120_124743/temp1'
    args.test_model_dir = './checkpoint/Decom_20250120_124743'

    # Train and Eval related args
    args.phase = 'train'
    args.epoch = 1000
    args.batch_size = 1
    args.patch_size = 128
    args.start_lr = 1e-3
    args.lr_div_period = 100
    args.lr_div_factor = 3
    args.eval_every_epoch = 100
    args.plot_every_epoch = 5
    args.lum_factor = 0.2
    args.post_scale = False

    main(args)
