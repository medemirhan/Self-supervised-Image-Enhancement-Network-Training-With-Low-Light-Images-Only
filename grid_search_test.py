import os
import argparse
from glob import glob
import tensorflow as tf
from model import lowlight_enhance
from utils import *
import metrics

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
        low_im = load_hsi(train_low_data_names[idx], matContentHeader=args.mat_key, normalization='global', max_val=args.global_max, min_val=args.global_min)
        train_low_data.append(low_im)
        high_im = load_hsi(train_high_data_names[idx], matContentHeader=args.mat_key, normalization='global', max_val=args.global_max, min_val=args.global_min)
        train_high_data.append(high_im)
        
        # Calculate max channel for equalization (across all spectral bands)
        train_low_data_max_chan = np.max(high_im, axis=2, keepdims=True)
        train_low_data_max_channel = histeq(train_low_data_max_chan)
        train_low_data_eq.append(train_low_data_max_channel)

    eval_low_data = []
    eval_low_data_name = glob(args.eval_data + '/*.*')  # Modified to accept any extension

    for idx in range(len(eval_low_data_name)):
        eval_low_im = load_hsi(eval_low_data_name[idx], matContentHeader=args.mat_key, normalization='global', max_val=args.global_max, min_val=args.global_min)
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
        test_low_im = load_hsi(test_low_data_name[idx], matContentHeader=args.mat_key, normalization='global', max_val=args.global_max, min_val=args.global_min)
        test_low_data.append(test_low_im)

    lowlight_enhance.test(
        model_dir=args.test_model_dir,
        test_low_data=test_low_data, 
        test_high_data=test_high_data, 
        test_low_data_names=test_low_data_name, 
        save_dir=args.test_result_dir, 
        decom_flag=args.decom,
        lum_factor=args.lum_factor
    )

def main(args):
    if args.use_gpu:
        print("[*] GPU\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            # Get number of channels from first image if not specified
            if args.channels is None:
                first_image = load_hsi(glob(args.train_data + '/*.*')[0], matContentHeader=args.mat_key, normalization='global', max_val=args.global_max, min_val=args.global_min)
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
    args.global_max = 0.2173913

    # Directories
    args.model_ckpt_dir = './checkpoint'
    args.train_data = '../PairLIE/data/hsi_dataset/train'
    args.eval_data = '../PairLIE/data/hsi_dataset/eval'
    args.test_data = '../PairLIE/data/hsi_dataset/test'
    
    args.eval_result_dir = 'D:/sslie/eval_results'
    args.test_result_dir = 'D:/sslie/test_results'
    args.test_model_dir = './checkpoint/Decom_20250112_024528'

    # Train and Eval related args
    args.phase = 'test'
    args.epoch = 600
    args.batch_size = 1
    args.patch_size = 96
    args.start_lr = 1e-3
    args.lr_div_period = 100
    args.lr_div_factor = 3
    args.eval_every_epoch = 100
    args.plot_every_epoch = 5
    
    lums = np.arange(20., 30., 0.5)
    mins = (0.0708354, None)

    log_file_path = "lf_logs.log"
    label_dir = '../PairLIE/data/label_ll'
    test_result_dir = args.test_result_dir
    with open(log_file_path, "a") as log_file:
        for lf in lums:
            for min in mins:
                args.test_result_dir = os.path.join(test_result_dir, 'lf_' + f"{lf:.1f}")
                args.lum_factor = lf
                args.global_min = min
                main(args)
                
                tf.get_variable_scope().reuse_variables()

                im_dir = args.test_result_dir + '/*.mat'

                avg_psnr, avg_ssim, avg_sam = metrics.calc_metrics(
                    im_dir=os.path.normpath(im_dir),
                    label_dir=os.path.normpath(label_dir),
                    data_min=args.global_min,
                    data_max=args.global_max,
                    matKeyPrediction='ref',
                    matKeyGt='data'
                    )
                
                if args.global_min == None:
                    strMin = str(args.global_min)
                else:
                    strMin = f"{args.global_min:.3f}"

                # Format the log entry
                log_entry = f"lf:{lf:.1f}, min:{strMin}, max:{args.global_max:.3f}, mpsnr:{avg_psnr:.3f}, mssim:{avg_ssim:.3f}, msam:{avg_sam:.3f}\n"

                # Write the log entry to the file
                log_file.write(log_entry)