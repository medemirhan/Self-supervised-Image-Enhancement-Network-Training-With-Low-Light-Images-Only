#coding=utf-8
from __future__ import print_function

import os
import time
import tensorflow as tf
import numpy as np
from utils import *
from scipy.ndimage import maximum_filter
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()
tf = tf.compat.v1  # Alias tf.compat.v1 as tf

def concat(layers):
    return tf.concat(layers, axis=4)  # Changed to axis 4 for 3D convolutions

def DecomNet3D(input_im, layer_num, channel=64, kernel_size=3, spectral_kernel_size=3, is_training=True):
    """
    Modified DecomNet to use 3D convolutions for spectral-spatial processing
    Args:
        input_im: Input tensor of shape [batch, height, width, channels]
        spectral_kernel_size: Size of kernel in spectral dimension
    """
    # Get static number of input channels
    input_shape = input_im.get_shape().as_list()
    input_channels = input_shape[-1]
    
    # Calculate channel-wise maximum and reshape it to match input channels
    input_max = tf.reduce_max(input_im, axis=3, keepdims=True)
    input_max_expanded = tf.tile(input_max, [1, 1, 1, input_channels])  # Expand to match channels
    
    # Reshape both tensors to 5D for 3D convolutions [batch, height, width, spectral_depth, 1]
    input_5d = tf.expand_dims(input_im, axis=4)
    input_max_5d = tf.expand_dims(input_max_expanded, axis=4)
    input_concat = concat([input_max_5d, input_5d])
    
    with tf.variable_scope('DecomNet', reuse=tf.AUTO_REUSE):
        # First 3D convolution layer
        conv_0 = tf.layers.conv3d(
            input_concat, 
            channel//2,
            kernel_size=(kernel_size, kernel_size, spectral_kernel_size),
            padding='same',
            activation=tf.nn.relu,
            name="first_layer"
        )

        # Shallow feature extraction with 3D conv
        conv = tf.layers.conv3d(
            input_concat,
            channel,
            kernel_size=(kernel_size * 3, kernel_size * 3, spectral_kernel_size),
            padding='same',
            activation=None,
            name="shallow_feature_extraction"
        )

        # Deeper layers with 3D convolutions
        conv1 = tf.layers.conv3d(
            conv,
            channel,
            kernel_size=(kernel_size, kernel_size, spectral_kernel_size),
            padding='same',
            activation=tf.nn.relu,
            name='activated_layer_1'
        )

        conv2 = tf.layers.conv3d(
            conv1,
            channel*2,
            kernel_size=(kernel_size, kernel_size, spectral_kernel_size),
            strides=(2, 2, 1),  # Don't downsample in spectral dimension
            padding='same',
            activation=tf.nn.relu,
            name='activated_layer_2'
        )

        conv3 = tf.layers.conv3d(
            conv2,
            channel*2,
            kernel_size=(kernel_size, kernel_size, spectral_kernel_size),
            padding='same',
            activation=tf.nn.relu,
            name='activated_layer_3'
        )

        # Transpose conv for upsampling
        conv4 = tf.layers.conv3d_transpose(
            conv3,
            channel,
            kernel_size=(kernel_size, kernel_size, spectral_kernel_size),
            strides=(2, 2, 1),  # Don't upsample in spectral dimension
            padding='same',
            activation=tf.nn.relu,
            name='activated_layer_4'
        )

        # Skip connections
        conv4_ba2 = concat([conv4, conv1])
        conv5 = tf.layers.conv3d(
            conv4_ba2,
            channel,
            kernel_size=(kernel_size, kernel_size, spectral_kernel_size),
            padding='same',
            activation=tf.nn.relu,
            name='activated_layer_5'
        )

        conv6 = concat([conv5, conv_0])
        conv7 = tf.layers.conv3d(
            conv6,
            channel,
            kernel_size=(kernel_size, kernel_size, spectral_kernel_size),
            padding='same',
            activation=None,
            name='activated_layer_7'
        )

        # Final layer to output R and L
        '''conv8 = tf.layers.conv3d(
            conv7,
            1,  # We want only one feature map
            kernel_size=(kernel_size, kernel_size, spectral_kernel_size),
            padding='same',
            activation=None,
            name='recon_layer'
        )'''
        conv8 = tf.layers.conv3d(
            conv7,
            input_channels + 1,  # Output input_channels + 1 feature maps
            kernel_size=(kernel_size, kernel_size, spectral_kernel_size),
            padding='same',
            activation=None,
            name='recon_layer'
        )

        # Get dynamic shape using tf.shape instead of get_shape()
        shape = tf.shape(conv8)
        #conv8_reshaped = tf.reshape(conv8, [shape[0], shape[1], shape[2], input_channels + 1])
        conv8_reshaped = tf.reshape(conv8, [-1, shape[1], shape[2], input_channels + 1])

        # Split channels for R and L
        '''R = tf.sigmoid(conv8_reshaped[:,:,:,:input_channels])
        L = tf.sigmoid(conv8_reshaped[:,:,:,input_channels:])  # Illumination map'''
        R = tf.sigmoid(conv8_reshaped[:, :, :, :input_channels])
        L = tf.sigmoid(conv8_reshaped[:, :, :, input_channels:])

        return R, L

class lowlight_enhance(object):
    def __init__(self, sess, input_channels=3, spectral_kernel_size=3):
        self.sess = sess
        self.DecomNet_layer_num = 5
        self.input_channels = input_channels
        self.spectral_kernel_size = spectral_kernel_size
        
        # Store average losses per epoch
        self.epoch_losses = {
            'total_loss': [],
            'recon_loss': [],
            'recon_loss_eq': [],
            'R_smooth_loss': [],
            'I_smooth_loss': []
        }
        
        # For accumulating losses within each epoch
        self.current_epoch_losses = {
            'total_loss': 0,
            'recon_loss': 0,
            'recon_loss_eq': 0,
            'R_smooth_loss': 0,
            'I_smooth_loss': 0,
            'steps': 0
        }

        # Placeholders
        self.input_low = tf.placeholder(tf.float32, [None, None, None, self.input_channels], name='input_low')
        self.input_high = tf.placeholder(tf.float32, [None, None, None, self.input_channels], name='input_high')
        self.input_low_eq = tf.placeholder(tf.float32, [None, None, None, 1], name='input_low_eq')

        # Use 3D DecomNet
        [R_low, I_low] = DecomNet3D(
            self.input_low, 
            layer_num=self.DecomNet_layer_num,
            spectral_kernel_size=self.spectral_kernel_size
        )

        # Repeat illumination map for all channels
        I_low_expanded = tf.tile(I_low, [1, 1, 1, self.input_channels])

        self.output_R_low = R_low
        self.output_I_low = I_low_expanded
        self.output_S_low_zy = (R_low * I_low_expanded)

        # Loss calculations with spectral considerations
        self.recon_loss_low = tf.reduce_mean(tf.abs(R_low * I_low_expanded - self.input_high))
        
        # Modified to handle spectral dimension
        R_low_max = tf.reduce_max(R_low, axis=3, keepdims=True)
        self.recon_loss_low_eq = tf.reduce_mean(tf.abs(R_low_max - self.input_low_eq))
        
        # Smoothness loss across spatial and spectral dimensions
        R_low_gray = tf.reduce_mean(R_low, axis=3, keepdims=True)
        self.R_low_loss_smooth = tf.reduce_mean(tf.abs(self.gradient(R_low_gray, "x")) + 
                                              tf.abs(self.gradient(R_low_gray, "y")))
        
        self.Ismooth_loss_low = self.smooth(I_low, R_low_gray)

        # Combined loss
        self.loss_Decom_zhangyu = (self.recon_loss_low + 
                                  0.1 * self.Ismooth_loss_low + 
                                  0.1 * self.recon_loss_low_eq + 
                                  0.01 * self.R_low_loss_smooth)

        # Optimizer setup
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')

        self.var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
        self.train_op_Decom = optimizer.minimize(self.loss_Decom_zhangyu, var_list=self.var_Decom)

        self.sess.run(tf.global_variables_initializer())
        self.saver_Decom = tf.train.Saver(var_list=self.var_Decom)
        print("[*] Initialize model successfully...")

    def gradient(self, input_tensor, direction):
        """Modified gradient calculation for spectral-spatial processing"""
        input_shape = tf.shape(input_tensor)
        
        # Expand the kernels for 3D convolution
        self.smooth_kernel_x = tf.reshape(tf.constant([[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1, 1])
        self.smooth_kernel_y = tf.transpose(self.smooth_kernel_x, [1, 0, 2, 3, 4])

        # Add an extra dimension to input tensor if needed
        if len(input_tensor.get_shape()) == 4:
            input_tensor = tf.expand_dims(input_tensor, axis=-1)

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        
        return tf.abs(tf.nn.conv3d(input_tensor, kernel, strides=[1, 1, 1, 1, 1], padding='SAME'))

    def ave_gradient(self, input_tensor, direction):
        """Modified average gradient for 3D data"""
        grad = self.gradient(input_tensor, direction)
        return tf.layers.average_pooling3d(grad, pool_size=[3, 3, 1], strides=[1, 1, 1], padding='SAME')

    def smooth(self, input_I, input_R):
        """Modified smoothness calculation for 3D data"""
        # Ensure 5D tensors
        if len(input_I.get_shape()) == 4:
            input_I = tf.expand_dims(input_I, axis=-1)
        if len(input_R.get_shape()) == 4:
            input_R = tf.expand_dims(input_R, axis=-1)

        return tf.reduce_mean(
            self.gradient(input_I, "x") * tf.exp(-10 * self.gradient(input_R, "x")) +
            self.gradient(input_I, "y") * tf.exp(-10 * self.gradient(input_R, "y"))
        )

    def evaluate(self, epoch_num, eval_low_data, sample_dir, train_phase):
        print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))

        for idx in range(len(eval_low_data)):
            input_low_eval = np.expand_dims(eval_low_data[idx], axis=0)

            if train_phase == "Decom":
                result_1, result_2 = self.sess.run([self.output_R_low, self.output_I_low], 
                                                 feed_dict={self.input_low: input_low_eval})

            save_images(os.path.join(sample_dir, 'eval_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), 
                       result_1, result_2)

    def plot_loss_curve(self, save_path='loss_curve.png'):
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

    def train(self, train_low_data, train_low_data_eq, eval_low_data, train_high_data, batch_size, patch_size, epoch, lr, sample_dir, ckpt_dir, eval_every_epoch, train_phase, plot_every_epoch=10):
        """Train the model with 3D convolution support"""
        # Get channel dimension from input data
        h, w, channels = train_low_data[0].shape
        numBatch = len(train_low_data) // int(batch_size)

        # load pretrained model
        if train_phase == "Decom":
            train_op = self.train_op_Decom
            train_loss = self.loss_Decom_zhangyu
            saver = self.saver_Decom

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
                    _, batch_loss, recon_loss, recon_loss_eq, r_smooth_loss, i_smooth_loss = self.sess.run(
                        [train_op, train_loss, self.recon_loss_low, self.recon_loss_low_eq, 
                         self.R_low_loss_smooth, self.Ismooth_loss_low], 
                        feed_dict={
                            self.input_low: batch_input_low,
                            self.input_high: batch_input_high,
                            self.input_low_eq: batch_input_low_eq,
                            self.lr: lr[epoch]
                        }
                    )
                else:
                    boolflag = False
                    _, batch_loss, recon_loss, recon_loss_eq, r_smooth_loss, i_smooth_loss = self.sess.run(
                        [train_op, train_loss, self.recon_loss_low, self.recon_loss_low_eq, 
                         self.R_low_loss_smooth, self.Ismooth_loss_low],
                        feed_dict={
                            self.input_low: batch_input_low,
                            self.input_high: batch_input_high,
                            self.input_low_eq: batch_input_low_eq,
                            self.lr: lr[epoch]
                        }
                    )

                # Accumulate losses for averaging
                self.current_epoch_losses['total_loss'] += batch_loss
                self.current_epoch_losses['recon_loss'] += recon_loss
                self.current_epoch_losses['recon_loss_eq'] += recon_loss_eq
                self.current_epoch_losses['R_smooth_loss'] += r_smooth_loss
                self.current_epoch_losses['I_smooth_loss'] += i_smooth_loss
                self.current_epoch_losses['steps'] += 1

                print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, total_loss: %.6f" % 
                      (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, batch_loss))
                print("Losses - Recon: %.6f, ReconEq: %.6f, RSmooth: %.6f, ISmooth: %.6f" %
                      (recon_loss, recon_loss_eq, r_smooth_loss, i_smooth_loss))
                iter_num += 1

            # At the end of each epoch, compute average losses
            steps = self.current_epoch_losses['steps']
            if steps > 0:  # Ensure we don't divide by zero
                self.epoch_losses['total_loss'].append(self.current_epoch_losses['total_loss'] / steps)
                self.epoch_losses['recon_loss'].append(self.current_epoch_losses['recon_loss'] / steps)
                self.epoch_losses['recon_loss_eq'].append(self.current_epoch_losses['recon_loss_eq'] / steps)
                self.epoch_losses['R_smooth_loss'].append(self.current_epoch_losses['R_smooth_loss'] / steps)
                self.epoch_losses['I_smooth_loss'].append(self.current_epoch_losses['I_smooth_loss'] / steps)

            # Reset current epoch losses
            for key in self.current_epoch_losses:
                self.current_epoch_losses[key] = 0

            # Plot losses every plot_every_epoch epochs
            if plot_every_epoch > 0 and (epoch + 1) % plot_every_epoch == 0:
                print(f"\nPlotting loss curves at epoch {epoch + 1}")
                loss_plot_path = os.path.join(sample_dir, 'loss_curves.png')
                self.plot_loss_curve(loss_plot_path)

            # evaluate the model and save a checkpoint file for it
            if (epoch + 1) % eval_every_epoch == 0:
                self.evaluate(epoch + 1, eval_low_data, sample_dir=sample_dir, train_phase=train_phase)
                self.save(saver, iter_num, ckpt_dir, "RetinexNet-%s" % train_phase)

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

    def test(self, test_low_data, test_high_data, test_low_data_names, save_dir, decom_flag, batch_size=1):
        '''tf.global_variables_initializer().run()

        print("[*] Reading checkpoint...")
        load_model_status_Decom, _ = self.load(self.saver_Decom, './checkpoint/Decom')
        if load_model_status_Decom:
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
            R_low, I_low, output_S_low_zy = self.sess.run(
                [self.output_R_low, self.output_I_low, self.output_S_low_zy], 
                feed_dict={self.input_low: input_low_test}
            )

            if(idx != 0):
                total_run_time += time.time() - start_time
                
            if decom_flag == decom_flag:
                save_images(os.path.join(save_dir, name + "_R_low." + suffix), R_low)
                save_images(os.path.join(save_dir, name + "_I_low." + suffix), I_low)

        ave_run_time = total_run_time / (float(len(test_low_data))-1)
        print("[*] Average run time: %.4f" % ave_run_time)'''

        tf.global_variables_initializer().run()

        print("[*] Reading checkpoint...")
        load_model_status_Decom, _ = self.load(self.saver_Decom, './checkpoint/Decom')
        if load_model_status_Decom:
            print("[*] Load weights successfully...")
        
        print("[*] Testing...")
        total_run_time = 0.0
        
        for idx in range(0, len(test_low_data), batch_size):
            batch_low_data = test_low_data[idx:idx + batch_size]
            batch_low_data = np.stack(batch_low_data, axis=0)

            start_time = time.time()
            R_low, I_low, output_S_low_zy = self.sess.run(
                [self.output_R_low, self.output_I_low, self.output_S_low_zy], 
                feed_dict={self.input_low: batch_low_data}
            )
            
            if idx != 0:
                total_run_time += time.time() - start_time
            
            for i in range(len(batch_low_data)):
                idx_img = idx + i
                print(test_low_data_names[idx_img])
                [_, name] = os.path.split(test_low_data_names[idx_img])
                suffix = name[name.find('.') + 1:]
                name = name[:name.find('.')]

                if decom_flag == decom_flag:
                    save_images(os.path.join(save_dir, name + "_R_low." + suffix), R_low[i])
                    save_images(os.path.join(save_dir, name + "_I_low." + suffix), I_low[i])

        ave_run_time = total_run_time / (float(len(test_low_data)) - 1)
        print("[*] Average run time: %.4f" % ave_run_time)