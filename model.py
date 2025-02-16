import os
import time
import tensorflow as tf
import numpy as np
from utils import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()
tf = tf.compat.v1  # Alias tf.compat.v1 as tf

seed_value = 42
tf.set_random_seed(seed_value)
np.random.seed(seed_value)

def concat(layers):
    return tf.concat(layers, axis=3)

def DecomNet(input_im, layer_num, channel=64, kernel_size=3, is_training=True):
    # Get static number of input channels
    input_shape = input_im.get_shape().as_list()
    input_channels = input_shape[-1]
    
    # Calculate channel-wise maximum
    input_max = tf.reduce_max(input_im, axis=3, keepdims=True)
    input_concat = concat([input_max, input_im])
    
    # Get the concatenated channel dimension (static)
    concat_channels = input_channels + 1  # original channels + max channel
    
    with tf.variable_scope('DecomNet', reuse=tf.AUTO_REUSE):
        # First layer with fixed number of output channels
        conv_0 = tf.layers.conv2d(input_concat, channel//2, kernel_size, padding='same', activation=tf.nn.relu, name="first_layer")
        conv = tf.layers.conv2d(input_concat, channel, kernel_size * 3, padding='same', activation=None, name="shallow_feature_extraction")
        
        conv1 = tf.layers.conv2d(conv, channel, kernel_size, padding='same', activation=tf.nn.relu, name='activated_layer_1')
        conv2 = tf.layers.conv2d(conv1, channel*2, kernel_size, strides=2, padding='same', activation=tf.nn.relu, name='activated_layer_2')
        conv3 = tf.layers.conv2d(conv2, channel*2, kernel_size, padding='same', activation=tf.nn.relu, name='activated_layer_3')
        conv4 = tf.layers.conv2d_transpose(conv3, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu, name='activated_layer_4')
        
        conv4_ba2 = concat([conv4, conv1])
        conv5 = tf.layers.conv2d(conv4_ba2, channel, kernel_size, padding='same', activation=tf.nn.relu, name='activated_layer_5')
        conv6 = concat([conv5, conv_0])
        conv7 = tf.layers.conv2d(conv6, channel, kernel_size, padding='same', activation=None, name='activated_layer_7')
        
        # Get static number of output channels
        output_channels = input_im.get_shape().as_list()[-1]
        conv8 = tf.layers.conv2d(conv7, output_channels + 1, kernel_size, padding='same', activation=None, name='recon_layer')

    # R maintains input channel dimension, L is single channel
    R = tf.sigmoid(conv8[:,:,:,:output_channels])
    L = tf.sigmoid(conv8[:,:,:,output_channels:])  # Illumination map is still single channel

    return R, L

def RelightNet(input_L, input_R, channel=64, kernel_size=3):
    # Get number of spectral channels from reflectance input
    num_spectral_channels = input_R.get_shape().as_list()[3]  # input_R is H×W×C (64 in your case)
    # input_L is H×W×1
    
    # Concatenate reflectance (C channels) and illumination (1 channel)
    input_im = concat([input_R, input_L])  # Results in H×W×(C+1) tensor
    
    with tf.variable_scope('RelightNet'):
        # Encoder path
        conv0 = tf.layers.conv2d(input_im, channel, kernel_size, padding='same', activation=None)
        conv1 = tf.layers.conv2d(conv0, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(conv2, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
        
        # Decoder path with skip connections
        up1 = tf.image.resize_nearest_neighbor(conv3, (tf.shape(conv2)[1], tf.shape(conv2)[2]))
        deconv1 = tf.layers.conv2d(up1, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv2
        up2 = tf.image.resize_nearest_neighbor(deconv1, (tf.shape(conv1)[1], tf.shape(conv1)[2]))
        deconv2 = tf.layers.conv2d(up2, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv1
        up3 = tf.image.resize_nearest_neighbor(deconv2, (tf.shape(conv0)[1], tf.shape(conv0)[2]))
        deconv3 = tf.layers.conv2d(up3, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv0
        
        # Multi-scale feature fusion
        deconv1_resize = tf.image.resize_nearest_neighbor(deconv1, (tf.shape(deconv3)[1], tf.shape(deconv3)[2]))
        deconv2_resize = tf.image.resize_nearest_neighbor(deconv2, (tf.shape(deconv3)[1], tf.shape(deconv3)[2]))
        feature_gather = concat([deconv1_resize, deconv2_resize, deconv3])
        
        # Feature fusion with preserved spectral dimensionality
        feature_fusion = tf.layers.conv2d(feature_gather, channel, 1, padding='same', activation=None)
        
        # Final layer outputs a single channel (illumination delta)
        output = tf.layers.conv2d(feature_fusion, 1, 3, padding='same', activation=None)
    
    return output

class lowlight_enhance(object):
    def __init__(self, sess, input_channels=3, cutoff=0.1, time_stamp=None):
        self.sess = sess
        self.DecomNet_layer_num = 5
        self.input_channels = input_channels  # Store channel count
        self.model_timestamp = time_stamp
        self.cutoff = cutoff

        # Generate band-specific weights
        #self.band_weights = self.generate_band_weights()

        # Store average losses per epoch
        self.epoch_losses = {
            'total_loss': [],
            'recon_loss': [],
            'recon_loss_eq': [],
            'R_smooth_loss': [],
            'I_smooth_loss': [],
            'I_smooth_loss_delta': [],
            'relight_loss': [],
            'fourier_loss': [],
            'decom_zhangyu': [],
            'relightNet_loss': [],
            'spectral_loss': []
        }
        # For accumulating losses within each epoch
        self.current_epoch_losses = {
            'total_loss': 0,
            'recon_loss': 0,
            'recon_loss_eq': 0,
            'R_smooth_loss': 0,
            'I_smooth_loss': 0,
            'I_smooth_loss_delta': 0,
            'relight_loss': 0,
            'fourier_loss': 0,
            'decom_zhangyu': 0,
            'relightNet_loss': 0,
            'spectral_loss': 0,
            'steps': 0
        }

        # Modified placeholders with specific channel dimension
        self.input_low = tf.placeholder(tf.float32, [None, None, None, self.input_channels], name='input_low')
        self.input_high = tf.placeholder(tf.float32, [None, None, None, self.input_channels], name='input_high')
        self.input_low_eq = tf.placeholder(tf.float32, [None, None, None, 1], name='input_low_eq')

        [R_low, I_low] = DecomNet(self.input_low, layer_num=self.DecomNet_layer_num)
        I_delta = RelightNet(I_low, R_low)  # Will be H×W×1

        # Expand I_low and I_delta to match the number of channels
        I_low_expanded = tf.tile(I_low, [1, 1, 1, self.input_channels])  # Expands to H×W×64
        I_delta_expanded = tf.tile(I_delta, [1, 1, 1, self.input_channels])  # Expands to H×W×64

        self.output_R_low = R_low  # H×W×64
        self.output_I_low = I_low_expanded  # H×W×64
        self.output_S_low_zy = R_low * I_low_expanded  # H×W×64
        self.output_I_delta = I_delta_expanded  # H×W×64
        self.output_S = R_low * I_delta_expanded  # H×W×64 * H×W×64 -> H×W×64
        self.output_S = self.output_S + R_low * I_low_expanded

        # Loss calculations
        self.recon_loss_low = tf.reduce_mean(tf.abs(R_low * I_low_expanded - self.input_high))
        '''self.recon_loss_low = self.weighted_reconstruction_loss(
            R_low * I_low_expanded, 
            self.input_high
        )'''
        #self.recon_loss_low = tf.reduce_mean(tf.square(R_low * I_low_expanded - self.input_high))
        #self.recon_loss_low = tf.reduce_mean(tf.abs(tf.math.pow(I_low_expanded, 0.2) * R_low - self.input_high))
        
        # Modified to handle multiple channels
        R_low_max = tf.reduce_max(R_low, axis=3, keepdims=True)
        self.recon_loss_low_eq = tf.reduce_mean(tf.abs(R_low_max - self.input_low_eq))
        #self.recon_loss_low_eq = tf.reduce_mean(tf.square(R_low_max - self.input_low_eq))
        
        # Calculate smoothness loss using average across channels
        R_low_gray = tf.reduce_mean(R_low, axis=3, keepdims=True)
        self.R_low_loss_smooth = tf.reduce_mean(tf.abs(self.gradient(R_low_gray, "x")) + 
                                              tf.abs(self.gradient(R_low_gray, "y")))
        '''self.R_low_loss_smooth = tf.reduce_mean(tf.square(self.gradient(R_low_gray, "x")) + 
                                              tf.square(self.gradient(R_low_gray, "y")))'''
        
        self.Ismooth_loss_low = self.smooth(I_low, R_low_gray)

        self.loss_Decom_zhangyu = (self.recon_loss_low + 
                                  1 * self.Ismooth_loss_low + 
                                  0 * self.recon_loss_low_eq + 
                                  1 * self.R_low_loss_smooth)

        self.Ismooth_loss_delta = self.smooth(I_delta, R_low)
        self.relight_loss = tf.reduce_mean(tf.abs(R_low * I_delta_expanded - self.input_high))
        self.loss_Relight = (0.1 * self.relight_loss + 
                             10 * self.Ismooth_loss_delta)
        
        self.fourier_loss = self.fourier_spectrum_loss(self.input_low, self.output_S, cutoff=self.cutoff, loss_type="l1")

        self.spectral_loss = self.spectral_smoothness_loss(self.output_S)

        #self.loss_combined = self.loss_Decom_zhangyu + self.loss_Relight
        self.loss_combined = (1 * self.loss_Decom_zhangyu + 
                              2 * self.loss_Relight + 
                              0.2 * self.fourier_loss +
                              2 * self.spectral_loss)

        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')

        self.var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
        self.train_op_Decom = optimizer.minimize(self.loss_Decom_zhangyu, var_list=self.var_Decom)

        self.var_Relight = [var for var in tf.trainable_variables() if 'RelightNet' in var.name]
        self.train_op_Relight = optimizer.minimize(self.loss_Relight, var_list = self.var_Relight)

        self.var_Combined = [var for var in tf.trainable_variables() if 'DecomNet' or 'RelightNet' in var.name]
        self.train_op_Combined = optimizer.minimize(self.loss_combined, var_list=self.var_Combined)

        self.sess.run(tf.global_variables_initializer())
        self.saver_Decom = tf.train.Saver(var_list=self.var_Decom)
        self.saver_Relight = tf.train.Saver(var_list=self.var_Relight)
        self.saver_Combined = tf.train.Saver(var_list=self.var_Combined)
        print("[*] Initialize model successfully...")

        #self.print_band_weights()

    def generate_band_weights(self):
        """Generate weights for each band using polynomial decay"""
        weights = np.zeros(self.input_channels)
        for band in range(self.input_channels):
            # Convert band index to step for polynomial decay
            # band 0 should have highest penalty
            step = band
            
            # Parameters for polynomial decay
            initial_value = 5.0
            decay_rate = 0.3  # Adjust this to control decay speed
            power = 2.0       # Adjust this to control decay shape
            
            weights[band] = polynomial_decay(initial_value, decay_rate, power, step)
            
        # Normalize weights
        #weights = weights / np.max(weights)
        return tf.constant(weights, dtype=tf.float32)

    def weighted_reconstruction_loss(self, pred, target):
        """Calculate band-weighted reconstruction loss"""
        # Expand weights to match batch dimensions
        weights_expanded = tf.reshape(self.band_weights, [1, 1, 1, -1])
        
        # Calculate absolute difference for each band
        pixel_wise_loss = tf.abs(pred - target)
        
        # Apply band-specific weights
        weighted_loss = pixel_wise_loss * weights_expanded
        
        # Average across all dimensions
        return tf.reduce_mean(weighted_loss)

    def gradient(self, input_tensor, direction):
        # Create kernels for each channel
        channels = input_tensor.get_shape().as_list()[3]
        
        # Reshape kernels to work with multiple channels
        if direction == "x":
            kernel = tf.reshape(tf.constant([[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1])
            kernel = tf.tile(kernel, [1, 1, channels, 1])  # Expand for each input channel
        elif direction == "y":
            kernel = tf.reshape(tf.constant([[0, -1], [0, 1]], tf.float32), [2, 2, 1, 1])
            kernel = tf.tile(kernel, [1, 1, channels, 1])  # Expand for each input channel
        
        return tf.abs(tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))

    def smooth(self, input_I, input_R):
        return tf.reduce_mean(self.gradient(input_I, "x") * tf.exp(-10 * self.gradient(input_R, "x")) + 
                            self.gradient(input_I, "y") * tf.exp(-10 * self.gradient(input_R, "y")))

    def evaluate(self, epoch_num, eval_low_data, sample_dir, train_phase):
        print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))

        for idx in range(len(eval_low_data)):
            input_low_eval = np.expand_dims(eval_low_data[idx], axis=0)

            if train_phase == "Decom":
                result_1, result_2 = self.sess.run([self.output_R_low, self.output_I_low], 
                                                 feed_dict={self.input_low: input_low_eval})

            if train_phase == "Relight":
                result_1, result_2 = self.sess.run([self.output_S, self.output_I_delta],
                                                   feed_dict={self.input_low: input_low_eval})

            '''result_1 = result_1 * (0.2173913 - 0.0708354) + 0.0708354
            result_2 = result_2 * (0.2173913 - 0.0708354) + 0.0708354'''
            
            mat_path = os.path.join(sample_dir, 'eval_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num))
            #save_hsi(mat_path, result_1, postfix='_R')
            #save_hsi(mat_path, result_2, postfix='_I')

    def train(self, train_low_data, train_low_data_eq, eval_low_data, train_high_data, batch_size, patch_size, epoch, lr, eval_dir, ckpt_dir, eval_every_epoch, train_phase, plot_every_epoch=10):
        """
        Added plot_every_epoch parameter to control how often we plot losses
        """
        eval_dir += '_' + self.model_timestamp
        ckpt_dir = os.path.join(ckpt_dir, 'Decom_' + self.model_timestamp)
        
        # Get channel dimension from input data
        h, w, channels = train_low_data[0].shape  # 3D array: height, width, channels
        numBatch = len(train_low_data) // int(batch_size)

        # load pretrained model
        '''if train_phase == "Decom":
            train_op = self.train_op_Decom
            train_loss = self.loss_Decom_zhangyu
            saver = self.saver_Decom
        elif train_phase == "Relight":
            train_op = self.train_op_Relight
            train_loss = self.loss_Relight
            saver = self.saver_Relight'''
        
        train_op = self.train_op_Combined
        train_loss = self.loss_combined
        saver = self.saver_Combined

        load_model_status, global_step = self.load(saver, ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")

        print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))

        start_time = time.time()
        image_id = 0
        boolflag = True
        
        for epoch in range(start_epoch, epoch):
            boolflag = True
            for batch_id in range(start_step, numBatch):
                # generate data for a batch - now using correct channel dimension
                batch_input_low = np.zeros((batch_size, patch_size, patch_size, channels), dtype="float32")
                batch_input_high = np.zeros((batch_size, patch_size, patch_size, channels), dtype="float32")
                batch_input_low_eq = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")

                for patch_id in range(batch_size):
                    h, w, _ = train_low_data[image_id].shape
                    x = np.random.randint(0, h - patch_size)
                    y = np.random.randint(0, w - patch_size)

                    rand_mode = np.random.randint(0, 7)
                    
                    batch_input_low[patch_id, :, :, :] = data_augmentation(
                        train_low_data[image_id][x:x+patch_size, y:y+patch_size, :], 
                        rand_mode
                    )
                    batch_input_high[patch_id, :, :, :] = data_augmentation(
                        train_high_data[image_id][x:x+patch_size, y:y+patch_size, :], 
                        rand_mode
                    )
                    batch_input_low_eq[patch_id, :, :, :] = data_augmentation(
                        train_low_data_eq[image_id][x:x+patch_size, y:y+patch_size, :], 
                        rand_mode
                    )

                    image_id = (image_id + 1) % len(train_low_data)
                    if image_id == 0:
                        tmp = list(zip(train_low_data, train_low_data))
                        np.random.shuffle(list(tmp))
                        train_low_data, _ = zip(*tmp)

                # train
                if not boolflag:
                    _, batch_loss, recon_loss, recon_loss_eq, r_smooth_loss, i_smooth_loss, i_smooth_loss_delta, relight_loss, fourier_loss, decom_loss, relightNet_loss, spectral_loss = self.sess.run(
                        [train_op, train_loss, self.recon_loss_low, self.recon_loss_low_eq, 
                        self.R_low_loss_smooth, self.Ismooth_loss_low, self.Ismooth_loss_delta, self.relight_loss,
                        self.fourier_loss, self.loss_Decom_zhangyu, self.loss_Relight, self.spectral_loss], 
                        feed_dict={
                            self.input_low: batch_input_low,
                            self.input_high: batch_input_high,
                            self.input_low_eq: batch_input_low_eq,
                            self.lr: lr[epoch]
                        }
                    )
                else:
                    boolflag = False
                    _, batch_loss, recon_loss, recon_loss_eq, r_smooth_loss, i_smooth_loss, i_smooth_loss_delta, relight_loss, fourier_loss, decom_loss, relightNet_loss, spectral_loss = self.sess.run(
                        [train_op, train_loss, self.recon_loss_low, self.recon_loss_low_eq, 
                        self.R_low_loss_smooth, self.Ismooth_loss_low, self.Ismooth_loss_delta, self.relight_loss,
                        self.fourier_loss, self.loss_Decom_zhangyu, self.loss_Relight, self.spectral_loss], 
                        feed_dict={
                            self.input_low: batch_input_low,
                            self.input_high: batch_input_high,
                            self.input_low_eq: batch_input_low_eq,
                            self.lr: lr[epoch]
                        }
                    )

                print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, total_loss: %.6f" % 
                      (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, batch_loss))

                # Accumulate losses for averaging
                '''if train_phase == "Decom":
                    self.current_epoch_losses['total_loss'] += batch_loss
                    self.current_epoch_losses['recon_loss'] += recon_loss
                    self.current_epoch_losses['recon_loss_eq'] += recon_loss_eq
                    self.current_epoch_losses['R_smooth_loss'] += r_smooth_loss
                    self.current_epoch_losses['I_smooth_loss'] += i_smooth_loss
                    self.current_epoch_losses['steps'] += 1
                    print("Losses - Recon: %.6f, ReconEq: %.6f, RSmooth: %.6f, ISmooth: %.6f" %
                        (recon_loss, recon_loss_eq, r_smooth_loss, i_smooth_loss))
                elif train_phase == "Relight":
                    self.current_epoch_losses['total_loss'] += batch_loss
                    self.current_epoch_losses['I_smooth_loss_delta'] += i_smooth_loss_delta
                    self.current_epoch_losses['relight_loss'] += relight_loss
                    self.current_epoch_losses['steps'] += 1
                    print("Losses - ISmoothDelta: %.6f, Relight: %.6f" %
                        (i_smooth_loss_delta, relight_loss))'''

                self.current_epoch_losses['total_loss'] += batch_loss
                self.current_epoch_losses['recon_loss'] += recon_loss
                self.current_epoch_losses['recon_loss_eq'] += recon_loss_eq
                self.current_epoch_losses['R_smooth_loss'] += r_smooth_loss
                self.current_epoch_losses['I_smooth_loss'] += i_smooth_loss
                self.current_epoch_losses['I_smooth_loss_delta'] += i_smooth_loss_delta
                self.current_epoch_losses['relight_loss'] += relight_loss
                self.current_epoch_losses['fourier_loss'] += fourier_loss
                self.current_epoch_losses['decom_zhangyu'] += decom_loss
                self.current_epoch_losses['relightNet_loss'] += relightNet_loss
                self.current_epoch_losses['spectral_loss'] += spectral_loss
                self.current_epoch_losses['steps'] += 1
                print("Losses - Recon: %.6f, ReconEq: %.6f, RSmooth: %.6f, ISmooth: %.6f, ISmoothDelta: %.6f, Relight: %.6f, Fourier: %.6f, Decom: %.6f, RelNet: %.6f, Spectral: %.6f" %
                    (recon_loss, recon_loss_eq, r_smooth_loss, i_smooth_loss, i_smooth_loss_delta, relight_loss, fourier_loss, decom_loss, relightNet_loss, spectral_loss))
            
                iter_num += 1

            # At the end of each epoch, compute average losses
            steps = self.current_epoch_losses['steps']
            if steps > 0:  # Ensure we don't divide by zero
                '''if train_phase == "Decom":
                    self.epoch_losses['total_loss'].append(self.current_epoch_losses['total_loss'] / steps)
                    self.epoch_losses['recon_loss'].append(self.current_epoch_losses['recon_loss'] / steps)
                    self.epoch_losses['recon_loss_eq'].append(self.current_epoch_losses['recon_loss_eq'] / steps)
                    self.epoch_losses['R_smooth_loss'].append(self.current_epoch_losses['R_smooth_loss'] / steps)
                    self.epoch_losses['I_smooth_loss'].append(self.current_epoch_losses['I_smooth_loss'] / steps)
                elif train_phase == "Relight":
                    self.epoch_losses['total_loss'].append(self.current_epoch_losses['total_loss'] / steps)
                    self.epoch_losses['I_smooth_loss_delta'].append(self.current_epoch_losses['I_smooth_loss_delta'] / steps)
                    self.epoch_losses['relight_loss'].append(self.current_epoch_losses['relight_loss'] / steps)'''
                self.epoch_losses['total_loss'].append(self.current_epoch_losses['total_loss'] / steps)
                self.epoch_losses['recon_loss'].append(self.current_epoch_losses['recon_loss'] / steps)
                self.epoch_losses['recon_loss_eq'].append(self.current_epoch_losses['recon_loss_eq'] / steps)
                self.epoch_losses['R_smooth_loss'].append(self.current_epoch_losses['R_smooth_loss'] / steps)
                self.epoch_losses['I_smooth_loss'].append(self.current_epoch_losses['I_smooth_loss'] / steps)
                self.epoch_losses['I_smooth_loss_delta'].append(self.current_epoch_losses['I_smooth_loss_delta'] / steps)
                self.epoch_losses['relight_loss'].append(self.current_epoch_losses['relight_loss'] / steps)
                self.epoch_losses['fourier_loss'].append(self.current_epoch_losses['fourier_loss'] / steps)
                self.epoch_losses['decom_zhangyu'].append(self.current_epoch_losses['decom_zhangyu'] / steps)
                self.epoch_losses['relightNet_loss'].append(self.current_epoch_losses['relightNet_loss'] / steps)
                self.epoch_losses['spectral_loss'].append(self.current_epoch_losses['spectral_loss'] / steps)

            # Reset current epoch losses
            for key in self.current_epoch_losses:
                self.current_epoch_losses[key] = 0

            # Plot losses every plot_every_epoch epochs
            if plot_every_epoch > 0 and (epoch + 1) % plot_every_epoch == 0:
                print(f"\nPlotting loss curves at epoch {epoch + 1}")
                '''if train_phase == "Decom":
                    loss_plot_path = os.path.join(eval_dir, 'loss_curves_decom.png')
                    self.plot_loss_curve_decom(loss_plot_path)
                elif train_phase == "Relight":
                    loss_plot_path = os.path.join(eval_dir, 'loss_curves_relight.png')
                    self.plot_loss_curve_relight(loss_plot_path)'''
                loss_plot_path = os.path.join(eval_dir, 'loss_curves_combined.png')
                self.plot_loss_curve_combined(loss_plot_path)

            # evaluate the model and save a checkpoint file for it
            if (epoch + 1) % eval_every_epoch == 0:
                self.evaluate(epoch + 1, eval_low_data, sample_dir=eval_dir, train_phase='Combined')
                self.save(saver, iter_num, ckpt_dir, "RetinexNet-%s" % 'Combined')

        print("[*] Finish training for phase %s." % train_phase)

    def save(self, saver, iter_num, ckpt_dir, model_name):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        print("[*] Saving model %s" % model_name)
        saver.save(self.sess,
                  os.path.join(ckpt_dir, model_name),
                  global_step=iter_num)

    def load(self, saver, ckpt_dir):
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(ckpt_dir)
            try:
                global_step = int(full_path.split('/')[-1].split('-')[-1])
            except ValueError:
                global_step = None
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            print("[*] Failed to load model from %s" % ckpt_dir)
            return False, 0

    def test(self, model_dir, test_low_data, test_high_data, test_low_data_names, save_dir, decom_flag, lum_factor, data_min=None, data_max=None):
        tf.global_variables_initializer().run()

        model_dir_decom = os.path.join(model_dir)

        print("[*] Reading checkpoint...")

        load_model_status_Decom = False
        load_model_status_Relight = False
        load_model_status_Combined = False
        
        '''if decom_flag == 1:
            load_model_status_Decom, _ = self.load(self.saver_Decom, model_dir_decom)
        else:
            load_model_status_Relight, _ = self.load(self.saver_Relight, model_dir_decom)'''
        
        load_model_status_Combined, _ = self.load(self.saver_Combined, model_dir_decom)
        
        '''if load_model_status_Decom or load_model_status_Relight:
            print("[*] Load weights successfully...")'''
        
        if load_model_status_Combined:
            print("[*] Load weights successfully...")
        
        print("[*] Testing...")
        total_run_time = 0.0
        for idx in range(len(test_low_data)):
            print(test_low_data_names[idx])
            [_, name] = os.path.split(test_low_data_names[idx])
            suffix = name[name.find('.') + 1:]
            name = name[:name.find('.')]

            input_low_test = np.expand_dims(test_low_data[idx], axis=0)

            start_time = time.time()

            '''if decom_flag == 1:
                R_low, I_low, output_S_low_zy = self.sess.run(
                    [self.output_R_low, self.output_I_low, self.output_S_low_zy], 
                    feed_dict={self.input_low: input_low_test}
                )
            else:
                [I_delta, S] = self.sess.run(
                    [self.output_I_delta, self.output_S],
                    feed_dict = {self.input_low: input_low_test})'''
            
            R_low, I_low, output_S_low_zy, I_delta, S = self.sess.run(
                [self.output_R_low, self.output_I_low, self.output_S_low_zy, self.output_I_delta, self.output_S], 
                feed_dict={self.input_low: input_low_test}
            )

            '''if data_min != None and data_max != None:
                I_low = I_low * (data_max - data_min) + data_min
                R_low = R_low * (data_max - data_min) + data_min'''
            #enhanced_im = np.power(I_low, lum_factor) * R_low

            if(idx != 0):
                total_run_time += time.time() - start_time

            if decom_flag == 1:
                #save_hsi(os.path.join(save_dir, name + "_R." + suffix), R_low)
                #save_hsi(os.path.join(save_dir, name + "_I." + suffix), I_low)
                #save_hsi(os.path.join(save_dir, name + "_enhanced." + suffix), enhanced_im)
                #save_hsi(os.path.join(save_dir, name + "." + suffix), enhanced_im)
                pass
            else:
                save_hsi(os.path.join(save_dir, name + "." + suffix), S)

        ave_run_time = total_run_time / (float(len(test_low_data) + 1)-1)
        print("[*] Average run time: %.4f" % ave_run_time)

    def plot_loss_curve_decom(self, save_path='loss_curve.png'):
        """Plot and save all training loss curves with epoch numbers"""
        
        epochs = range(1, len(self.epoch_losses['total_loss']) + 1)
        
        plt.figure(figsize=(15, 10))
        
        # Plot each loss in a separate subplot
        plt.subplot(2, 3, 1)
        plt.plot(epochs, self.epoch_losses['total_loss'], 'k-', label='Total Loss')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 3, 2)
        plt.plot(epochs, self.epoch_losses['recon_loss'], 'r-', label='Reconstruction Loss')
        plt.title('Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 3, 3)
        plt.plot(epochs, self.epoch_losses['recon_loss_eq'], 'b-', label='Eq Reconstruction Loss')
        plt.title('Equalization Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 3, 4)
        plt.plot(epochs, self.epoch_losses['R_smooth_loss'], 'g-', label='R Smoothness Loss')
        plt.title('R Smoothness Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 3, 5)
        plt.plot(epochs, self.epoch_losses['I_smooth_loss'], 'm-', label='I Smoothness Loss')
        plt.title('I Smoothness Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path)
        plt.close()
        print(f"Loss curves saved to {save_path}")

    def plot_loss_curve_relight(self, save_path='loss_curve.png'):
        """Plot and save all training loss curves with epoch numbers"""
        
        epochs = range(1, len(self.epoch_losses['total_loss']) + 1)
        
        plt.figure(figsize=(15, 10))
        
        # Plot each loss in a separate subplot
        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.epoch_losses['total_loss'], 'k-', label='Total Loss')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.epoch_losses['I_smooth_loss_delta'], 'r-', label='I Smoothness Delta Loss')
        plt.title('I Smoothness Delta Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.epoch_losses['relight_loss'], 'b-', label='Relightness Loss')
        plt.title('Relightness Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path)
        plt.close()
        print(f"Loss curves saved to {save_path}")

    def plot_loss_curve_combined(self, save_path='loss_curve.png'):
        """Plot and save all training loss curves with epoch numbers"""
        
        epochs = range(1, len(self.epoch_losses['total_loss']) + 1)
        
        plt.figure(figsize=(20, 10))
        
        # Plot each loss in a separate subplot
        plt.subplot(3, 4, 1)
        plt.plot(epochs, self.epoch_losses['total_loss'], 'k-', label='Total Loss')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(3, 4, 2)
        plt.plot(epochs, self.epoch_losses['recon_loss'], 'r-', label='Reconstruction Loss')
        plt.title('Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(3, 4, 3)
        plt.plot(epochs, self.epoch_losses['recon_loss_eq'], 'b-', label='Eq Reconstruction Loss')
        plt.title('Equalization Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(3, 4, 4)
        plt.plot(epochs, self.epoch_losses['R_smooth_loss'], 'g-', label='R Smoothness Loss')
        plt.title('R Smoothness Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(3, 4, 5)
        plt.plot(epochs, self.epoch_losses['I_smooth_loss'], 'm-', label='I Smoothness Loss')
        plt.title('I Smoothness Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 4, 6)
        plt.plot(epochs, self.epoch_losses['I_smooth_loss_delta'], 'r-', label='I Smoothness Delta Loss')
        plt.title('I Smoothness Delta Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(3, 4, 7)
        plt.plot(epochs, self.epoch_losses['relight_loss'], 'c-', label='Relightness Loss')
        plt.title('Relightness Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 4, 8)
        plt.plot(epochs, self.epoch_losses['decom_zhangyu'], 'y-', label='Decom Loss')
        plt.title('Decom Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 4, 9)
        plt.plot(epochs, self.epoch_losses['relightNet_loss'], 'g-', label='RelightNet Loss')
        plt.title('RelightNet Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 4, 10)
        plt.plot(epochs, self.epoch_losses['fourier_loss'], 'm-', label='Fourier Loss')
        plt.title('Fourier Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 4, 11)
        plt.plot(epochs, self.epoch_losses['spectral_loss'], 'r-', label='Spectral Smoothness Loss')
        plt.title('Spectral Smoothness Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path)
        plt.close()
        print(f"Loss curves saved to {save_path}")

    def print_band_weights(self):
        """Debug function to print band weights"""
        weights = self.sess.run(self.band_weights)
        plt.figure()
        plt.plot(weights)
        plt.show()
        plt.title('band weights for reconstruction loss')
        plt.xlabel('band #')
        plt.ylabel('weight')
        plt.grid(True)

    def dynamic_meshgrid(self, y, x):
        """
        Create a meshgrid from dynamic 1D tensors y and x.
        
        Args:
            y: Tensor of shape [H].
            x: Tensor of shape [W].
            
        Returns:
            Y: Tensor of shape [H, W] with rows equal to y.
            X: Tensor of shape [H, W] with columns equal to x.
        """
        H = tf.shape(y)[0]
        W = tf.shape(x)[0]
        Y = tf.tile(tf.expand_dims(y, axis=1), [1, W])
        X = tf.tile(tf.expand_dims(x, axis=0), [H, 1])
        return Y, X

    def high_pass_filter(self, fourier_image, cutoff=0.1):
        """
        Apply a high-pass filter in the frequency domain by zeroing out
        the low-frequency components.
        
        Args:
            fourier_image: A complex Tensor of shape (B, H, W, C).
            cutoff: A float specifying the cutoff radius in normalized coordinates.
                    Frequencies with a normalized radius < cutoff will be suppressed.
                    
        Returns:
            A Tensor of the same shape as fourier_image with low frequencies zeroed out.
        """
        # Check if H and W are known statically.
        static_shape = fourier_image.get_shape().as_list()
        if static_shape[1] is not None and static_shape[2] is not None:
            H, W = static_shape[1], static_shape[2]
            # When dimensions are known, use tf.linspace and tf.meshgrid.
            y = tf.linspace(-1.0, 1.0, H)  # shape: [H]
            x = tf.linspace(-1.0, 1.0, W)  # shape: [W]
            # Note: In TF 1.x, specifying the 'indexing' argument may or may not be supported.
            # If you run into issues here, you can replace this branch with the dynamic version.
            Y, X = tf.meshgrid(y, x, indexing='ij')
        else:
            # Fully dynamic branch.
            shape = tf.shape(fourier_image)
            H = shape[1]
            W = shape[2]
            # Create coordinate vectors using tf.range.
            y = tf.cast(tf.range(H), tf.float32)  # shape: [H]
            x = tf.cast(tf.range(W), tf.float32)  # shape: [W]
            # Scale these to the interval [-1, 1].
            y = 2.0 * (y / tf.cast(H - 1, tf.float32)) - 1.0
            x = 2.0 * (x / tf.cast(W - 1, tf.float32)) - 1.0
            # Build the meshgrid using our custom dynamic_meshgrid.
            Y, X = self.dynamic_meshgrid(y, x)
        
        # Compute the Euclidean distance from the center (0,0).
        radius = tf.sqrt(tf.square(X) + tf.square(Y))
        
        # Create a mask: 1 where radius >= cutoff, 0 elsewhere.
        mask = tf.cast(radius >= cutoff, tf.complex64)
        # Reshape mask to broadcast to shape (B, H, W, C).
        mask = tf.reshape(mask, [1, tf.shape(fourier_image)[1], tf.shape(fourier_image)[2], 1])
        
        return fourier_image * mask

    def fourier_spectrum_loss(self, input_hsi, target_hsi, cutoff=0.1, loss_type="l1"):
        """
        Compute the Fourier spectrum loss between two images.
        
        Args:
            input_hsi: Low-light hyperspectral image, Tensor of shape (B, H, W, C), dtype tf.float32.
            target_hsi: Enhanced hyperspectral image, Tensor of shape (B, H, W, C), dtype tf.float32.
            cutoff: High-pass filter cutoff in normalized coordinates.
            loss_type: "l1" for L1 loss or "l2" for L2 loss.
        
        Returns:
            A scalar Tensor representing the Fourier spectrum loss.
        """
        # Convert images to complex and compute the 2D FFT.
        fft_input = tf.spectral.fft2d(tf.complex(input_hsi, 0.0))
        fft_target = tf.spectral.fft2d(tf.complex(target_hsi, 0.0))
        
        # Apply the high-pass filter to each FFT.
        high_freq_input = self.high_pass_filter(fft_input, cutoff)
        high_freq_target = self.high_pass_filter(fft_target, cutoff)
        
        # Compute the magnitude of the high-frequency components.
        abs_high_freq_input = tf.abs(high_freq_input)
        abs_high_freq_target = tf.abs(high_freq_target)
        
        # Calculate the loss.
        if loss_type == "l1":
            loss = tf.reduce_mean(tf.abs(abs_high_freq_input - abs_high_freq_target))
        else:  # "l2"
            loss = tf.reduce_mean(tf.square(abs_high_freq_input - abs_high_freq_target))
        
        return loss
    
    def spectral_smoothness_loss(self, hsi, loss_type="l1"):
        """
        Computes a spectral smoothness loss that penalizes large differences 
        between adjacent spectral bands.
        
        Args:
            hsi: Tensor of shape [B, H, W, C] (e.g., your enhanced hyperspectral image).
            loss_type: "l1" for L1 loss or "l2" for L2 loss.
        
        Returns:
            A scalar Tensor representing the spectral smoothness loss.
        """
        # Compute differences along the spectral (last) dimension.
        # This results in a tensor of shape [B, H, W, C-1].
        spectral_diff = hsi[:,:,:,1:] - hsi[:,:,:,:-1]
        
        if loss_type == "l1":
            loss = tf.reduce_mean(tf.abs(spectral_diff))
        else:  # L2 loss
            loss = tf.reduce_mean(tf.square(spectral_diff))
        
        return loss