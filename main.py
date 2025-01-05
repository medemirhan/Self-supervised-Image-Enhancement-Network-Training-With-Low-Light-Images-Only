#coding=utf-8
from __future__ import print_function
import os
import argparse
from glob import glob

import tensorflow as tf
from model import lowlight_enhance
from utils import *

tf.compat.v1.disable_eager_execution()
tf = tf.compat.v1  # Alias tf.compat.v1 as tf

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
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    lr = args.start_lr * np.ones([args.epoch])
    lr[40:] = lr[0] / 10.0

    train_low_data = []
    train_high_data = []
    train_low_data_eq = []

    # Load training data
    train_low_data_names = glob(args.train_low_dir + '/*.*')  # Modified to accept any extension
    train_low_data_names.sort()
    train_high_data_names = glob(args.train_low_dir + '/*.*')  # Modified to accept any extension
    train_high_data_names.sort()
    assert len(train_low_data_names) == len(train_high_data_names)
    print('[*] Number of training data: %d' % len(train_low_data_names))

    for idx in range(len(train_low_data_names)):
        low_im = load_images(train_low_data_names[idx])
        train_low_data.append(low_im)
        high_im = load_images(train_high_data_names[idx])
        train_high_data.append(high_im)
        
        # Calculate max channel for equalization (across all spectral bands)
        train_low_data_max_chan = np.max(high_im, axis=2, keepdims=True)
        train_low_data_max_channel = histeq(train_low_data_max_chan)
        train_low_data_eq.append(train_low_data_max_channel)

    eval_low_data = []
    eval_low_data_name = glob(args.eval_low_dir + '/*.*')  # Modified to accept any extension

    for idx in range(len(eval_low_data_name)):
        eval_low_im = load_images(eval_low_data_name[idx])
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
        sample_dir=args.sample_dir, 
        ckpt_dir=os.path.join(args.ckpt_dir, 'Decom'), 
        eval_every_epoch=args.eval_every_epoch, 
        train_phase="Decom",
        plot_every_epoch=args.plot_every_epoch
    )

def lowlight_test(lowlight_enhance, args):
    if args.test_dir == None:
        print("[!] please provide --test_dir")
        exit(0)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    test_low_data_name = glob(os.path.join(args.test_dir) + '/*.*')  # Modified to accept any extension
    test_low_data = []
    test_high_data = []
    
    print("Found test files:", test_low_data_name)
    for idx in range(len(test_low_data_name)):
        test_low_im = load_images(test_low_data_name[idx])
        test_low_data.append(test_low_im)

    lowlight_enhance.test(
        test_low_data, 
        test_high_data, 
        test_low_data_name, 
        save_dir=args.save_dir, 
        decom_flag=args.decom
    )

def main(args):
    if args.use_gpu:
        print("[*] GPU\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            # Get number of channels from first image if not specified
            if args.channels is None:
                first_image = load_images(glob(args.train_low_dir + '/*.*')[0])
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

    args.use_gpu = 1
    args.gpu_idx = '0'
    args.gpu_mem = float(0.8)
    args.decom = 1
    args.sample_dir = './data/eval_results'

    args.phase = 'train'
    args.epoch = 1000
    args.batch_size = 1
    args.patch_size = 48
    args.start_lr = 1e-3
    args.eval_every_epoch = 100
    args.plot_every_epoch = 5

    args.ckpt_dir = './checkpoint'
    args.save_dir = './data/test_results'
    args.test_dir = '../PairLIE/data/CZ_hsdb/lowered_1.9/test'
    args.channels = 31

    args.train_low_dir = '../PairLIE/data/CZ_hsdb/lowered_1.9/train'
    args.eval_low_dir = '../PairLIE/data/CZ_hsdb/lowered_1.9/eval'

    main(args)
