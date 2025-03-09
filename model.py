import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from glob import glob
import random
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from sklearn.decomposition import PCA

from utils import pca_projection, save_hsi, data_augmentation, load_hsi

def conv(in_channels, out_channels, kernel_size, stride=1, padding=None, activation=True):
    if padding is None:
        padding = (kernel_size - 1) // 2
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)]
    if activation:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class DecomNet(nn.Module):
    def __init__(self, in_channels, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        self.in_channels = in_channels
        self.channel = channel
        self.kernel_size = kernel_size
        
        # First layer (conv_0)
        self.conv0 = conv(in_channels, channel // 2, kernel_size, activation=True)
        # Shallow feature extraction with a larger kernel (kernel_size*3)
        self.shallow_conv = conv(in_channels, channel, kernel_size * 3, activation=False)
        self.conv1 = conv(channel, channel, kernel_size, activation=True)
        self.conv2 = conv(channel, channel * 2, kernel_size, stride=2, activation=True)
        self.conv3 = conv(channel * 2, channel * 2, kernel_size, activation=True)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(channel * 2, channel, kernel_size, stride=2,
                               padding=(kernel_size - 1) // 2, output_padding=1),
            nn.ReLU(inplace=True)
        )
        # After concatenation, the number of channels doubles
        self.conv5 = conv(channel + channel, channel, kernel_size, activation=True)
        self.conv7 = conv(channel + channel // 2, channel, kernel_size, activation=False)
        self.recon = nn.Conv2d(channel, in_channels + 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        
    def forward(self, x):
        # x: (N, in_channels, H, W)
        conv0 = self.conv0(x)
        shallow = self.shallow_conv(x)
        conv1 = self.conv1(shallow)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        deconv = self.deconv(conv3)
        concat1 = torch.cat([deconv, conv1], dim=1)
        conv5 = self.conv5(concat1)
        concat2 = torch.cat([conv5, conv0], dim=1)
        conv7 = self.conv7(concat2)
        conv8 = self.recon(conv7)
        R = torch.sigmoid(conv8[:, :self.in_channels, :, :])
        L = torch.sigmoid(conv8[:, self.in_channels:, :, :])
        return R, L

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    
    def forward(self, x):
        # x: (N, C, H, W)
        N, C, H, W = x.size()
        squeeze = x.view(N, C, -1).mean(dim=2)
        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation))
        excitation = excitation.view(N, C, 1, 1)
        return x * excitation

class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads=4, head_dim=16, ff_dim=64):
        super(TransformerBlock, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total_dim = num_heads * head_dim
        self.q_linear = nn.Linear(channels, self.total_dim)
        self.k_linear = nn.Linear(channels, self.total_dim)
        self.v_linear = nn.Linear(channels, self.total_dim)
        self.ff_linear1 = nn.Linear(self.total_dim, ff_dim)
        self.ff_linear2 = nn.Linear(ff_dim, channels)
    
    def forward(self, x):
        # x: (N, C, H, W) – flatten spatially
        N, C, H, W = x.size()
        seq_len = H * W
        x_flat = x.view(N, C, seq_len).permute(0, 2, 1)
        Q = self.q_linear(x_flat)
        K = self.k_linear(x_flat)
        V = self.v_linear(x_flat)
        Q = Q.view(N, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(N, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(N, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        scale = self.head_dim ** 0.5
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(N, seq_len, self.total_dim)
        ff_output = F.relu(self.ff_linear1(attn_output))
        ff_output = self.ff_linear2(ff_output)
        output_seq = x_flat + ff_output
        output = output_seq.permute(0, 2, 1).view(N, C, H, W)
        return output

class RelightNet(nn.Module):
    def __init__(self, in_channels, channel=64, kernel_size=3, use_attention=False, use_transformer=True):
        super(RelightNet, self).__init__()
        # After concatenation, input channels become in_channels + 1.
        self.conv0 = conv(in_channels + 1, channel, kernel_size, activation=False)
        self.conv1 = conv(channel, channel, kernel_size, stride=2, activation=True)
        self.conv2 = conv(channel, channel, kernel_size, stride=2, activation=True)
        self.conv3 = conv(channel, channel, kernel_size, stride=2, activation=True)
        
        if use_attention:
            self.attn = SEBlock(channel)
        elif use_transformer:
            self.attn = TransformerBlock(channel)
        else:
            self.attn = nn.Identity()
        
        self.deconv1 = conv(channel, channel, kernel_size, activation=True)
        self.deconv2 = conv(channel, channel, kernel_size, activation=True)
        self.deconv3 = conv(channel, channel, kernel_size, activation=True)
        self.feature_fusion = conv(channel * 3, channel, 1, activation=False)
        self.final_conv = nn.Conv2d(channel, 1, 3, stride=1, padding=1)
        
    def forward(self, I, R):
        # I: illumination (N, 1, H, W); R: reflectance (N, in_channels, H, W)
        x = torch.cat([R, I], dim=1)
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv3 = self.attn(conv3)
        up1 = F.interpolate(conv3, size=conv2.shape[2:], mode='nearest')
        deconv1 = self.deconv1(up1) + conv2
        up2 = F.interpolate(deconv1, size=conv1.shape[2:], mode='nearest')
        deconv2 = self.deconv2(up2) + conv1
        up3 = F.interpolate(deconv2, size=conv0.shape[2:], mode='nearest')
        deconv3 = self.deconv3(up3) + conv0
        deconv1_resize = F.interpolate(deconv1, size=deconv3.shape[2:], mode='nearest')
        deconv2_resize = F.interpolate(deconv2, size=deconv3.shape[2:], mode='nearest')
        feature_gather = torch.cat([deconv1_resize, deconv2_resize, deconv3], dim=1)
        feature_fusion = self.feature_fusion(feature_gather)
        output = self.final_conv(feature_fusion)
        return output

class LowLightEnhance(nn.Module):
    def __init__(self, input_channels=64, time_stamp=None, 
                 coeff_recon_loss_low=10, coeff_Ismooth_loss_low=1, coeff_recon_loss_low_eq=1,
                 coeff_R_low_loss_smooth=1, coeff_relight_loss=0.2, coeff_Ismooth_loss_delta=20,
                 coeff_fourier_loss=0.2, coeff_spectral_loss=1, device=torch.device("cpu")):
        super(LowLightEnhance, self).__init__()
        self.input_channels = input_channels
        self.device = device
        self.time_stamp = time_stamp
        self.coeff_recon_loss_low = coeff_recon_loss_low
        self.coeff_Ismooth_loss_low = coeff_Ismooth_loss_low
        self.coeff_recon_loss_low_eq = coeff_recon_loss_low_eq
        self.coeff_R_low_loss_smooth = coeff_R_low_loss_smooth
        self.coeff_relight_loss = coeff_relight_loss
        self.coeff_Ismooth_loss_delta = coeff_Ismooth_loss_delta
        self.coeff_fourier_loss = coeff_fourier_loss
        self.coeff_spectral_loss = coeff_spectral_loss
        
        self.decom_net = DecomNet(in_channels=input_channels)
        self.relight_net = RelightNet(in_channels=input_channels)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
        self.all_epoch_losses = {'total_loss': [], 'recon_loss': [], 'recon_loss_eq': [], 'R_smooth_loss': [],
                             'I_smooth_loss': [], 'I_smooth_loss_delta': [], 'relight_loss': [],
                             'fourier_loss': [], 'decom_loss': [], 'relightNet_loss': [], 'spectral_loss': []}
    
    def forward(self, input_low):
        # input_low: (N, input_channels, H, W)
        R_low, I_low = self.decom_net(input_low)
        I_delta = self.relight_net(I_low, R_low)
        S = R_low * I_delta + R_low * I_low
        return R_low, I_low, I_delta, S
    
    def compute_gradients(self, img):
        grad_x = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
        grad_y = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
        return grad_x, grad_y
    
    def smooth_loss(self, I, R):
        grad_Ix, grad_Iy = self.compute_gradients(I)
        grad_Rx, grad_Ry = self.compute_gradients(R)
        loss = torch.mean(grad_Ix * torch.exp(-10 * grad_Rx)) + torch.mean(grad_Iy * torch.exp(-10 * grad_Ry))
        return loss
    
    def fourier_spectrum_loss(self, input_hsi, target_hsi, cutoff=0.1, loss_type="l1"):
        fft_input = torch.fft.fft2(input_hsi)
        fft_target = torch.fft.fft2(target_hsi)
        N, C, H, W = input_hsi.size()
        y = torch.linspace(-1, 1, H, device=input_hsi.device)
        x = torch.linspace(-1, 1, W, device=input_hsi.device)
        Y, X = torch.meshgrid(y, x, indexing='ij')
        radius = torch.sqrt(X**2 + Y**2)
        mask = (radius >= cutoff).float().unsqueeze(0).unsqueeze(0)
        high_freq_input = fft_input * mask
        high_freq_target = fft_target * mask
        abs_input = torch.abs(high_freq_input)
        abs_target = torch.abs(high_freq_target)
        if loss_type == "l1":
            loss = torch.mean(torch.abs(abs_input - abs_target))
        else:
            loss = torch.mean((abs_input - abs_target) ** 2)
        return loss
    
    def spectral_smoothness_loss(self, hsi, loss_type="l1"):
        spectral_diff = hsi[:, 1:, :, :] - hsi[:, :-1, :, :]
        if loss_type == "l1":
            loss = torch.mean(torch.abs(spectral_diff))
        else:
            loss = torch.mean(spectral_diff ** 2)
        return loss
    
    def pca_projection_loss(self, R):
        # Assume batch size 1; convert tensor to numpy (H, W, C)
        R_np = R.detach().cpu().numpy()[0].transpose(1, 2, 0)
        pc1 = pca_projection(R_np)
        pc1 = torch.from_numpy(pc1).unsqueeze(0).permute(0, 3, 1, 2).to(R.device).float()
        return pc1
    
    def compute_loss(self, input_low, input_high, input_low_eq):
        R_low, I_low, I_delta, S = self.forward(input_low)
        recon_loss = torch.mean(torch.abs(R_low * I_low - input_high))

        R_low_pca = self.pca_projection_loss(R_low)
        recon_loss_eq = torch.mean(torch.abs(R_low_pca - input_low_eq))

        R_gray = torch.mean(R_low, dim=1, keepdim=True)
        R_smooth_loss = torch.mean(torch.abs(self.compute_gradients(R_gray)[0])) + torch.mean(torch.abs(self.compute_gradients(R_gray)[1]))
        
        I_smooth_loss = self.smooth_loss(I_low, R_gray)
        
        I_smooth_loss_delta = self.smooth_loss(I_delta, R_low)
        
        relight_loss = torch.mean(torch.abs(R_low * I_delta - input_high))
        
        fourier_loss = self.fourier_spectrum_loss(input_low, S, cutoff=0.1, loss_type="l1")
        
        spectral_loss = self.spectral_smoothness_loss(S, loss_type="l1")
        
        loss_Decom = (self.coeff_recon_loss_low * recon_loss + 
                      self.coeff_Ismooth_loss_low * I_smooth_loss + 
                      self.coeff_recon_loss_low_eq * recon_loss_eq + 
                      self.coeff_R_low_loss_smooth * R_smooth_loss)
        loss_Relight = (self.coeff_relight_loss * relight_loss + 
                        self.coeff_Ismooth_loss_delta * I_smooth_loss_delta)
        loss_combined = (self.coeff_recon_loss_low * recon_loss + 
                         self.coeff_Ismooth_loss_low * I_smooth_loss + 
                         self.coeff_recon_loss_low_eq * recon_loss_eq + 
                         self.coeff_R_low_loss_smooth * R_smooth_loss +
                         self.coeff_relight_loss * relight_loss + 
                         self.coeff_Ismooth_loss_delta * I_smooth_loss_delta + 
                         self.coeff_fourier_loss * fourier_loss + 
                         self.coeff_spectral_loss * spectral_loss)
        losses = {
            'total_loss': loss_combined.item(),
            'recon_loss': recon_loss.item(),
            'recon_loss_eq': recon_loss_eq.item(),
            'R_smooth_loss': R_smooth_loss.item(),
            'I_smooth_loss': I_smooth_loss.item(),
            'I_smooth_loss_delta': I_smooth_loss_delta.item(),
            'relight_loss': relight_loss.item(),
            'fourier_loss': fourier_loss.item(),
            'decom_loss': loss_Decom.item(),
            'relightNet_loss': loss_Relight.item(),
            'spectral_loss': spectral_loss.item()
        }
        return loss_combined, losses
    
    def train_model(self, train_data_path, eval_data_path, batch_size, patch_size, num_epochs, start_lr, ckpt_dir, eval_result_dir, eval_every_epoch, plot_every_epoch=10):
        ckpt_dir = os.path.join(ckpt_dir, 'Decom_' + self.time_stamp)
        
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(eval_result_dir, exist_ok=True)
        train_files = sorted(glob(os.path.join(train_data_path, "*.*")))
        train_low_data = []
        train_high_data = []
        train_low_data_eq = []
        for file in train_files:
            low_im = load_hsi(file, matContentHeader='data', normalization='global_normalization', max_val=1.7410845, min_val=0.0708354)
            train_low_data.append(low_im)
            high_im = load_hsi(file, matContentHeader='data', normalization='global_normalization', max_val=1.7410845, min_val=0.0708354)
            train_high_data.append(high_im)
            low_eq = pca_projection(low_im)
            if low_eq.ndim == 2:
                low_eq = low_eq[:, :, None]
            train_low_data_eq.append(low_eq)
        eval_low_data = []
        eval_files = sorted(glob(os.path.join(eval_data_path, "*.*")))
        for file in eval_files:
            eval_im = load_hsi(file, matContentHeader='data', normalization='global_normalization', max_val=1.7410845, min_val=0.0708354)
            eval_low_data.append(eval_im)
        
        num_batches = len(train_low_data) // batch_size
        lr_schedule = [start_lr] * num_epochs
        iter_num = 0
        for epoch in range(num_epochs):
            cur_epoch_losses = {
                'total_loss': 0,
                'recon_loss': 0,
                'recon_loss_eq': 0,
                'R_smooth_loss': 0,
                'I_smooth_loss': 0,
                'I_smooth_loss_delta': 0,
                'relight_loss': 0,
                'fourier_loss': 0,
                'decom_loss': 0,
                'relightNet_loss': 0,
                'spectral_loss': 0
                }
            count = 0
            for batch_id in range(num_batches):
                batch_input_low = np.zeros((batch_size, patch_size, patch_size, self.input_channels), dtype=np.float32)
                batch_input_high = np.zeros((batch_size, patch_size, patch_size, self.input_channels), dtype=np.float32)
                batch_input_low_eq = np.zeros((batch_size, patch_size, patch_size, 1), dtype=np.float32)
                for i in range(batch_size):
                    idx = (batch_id * batch_size + i) % len(train_low_data)
                    h, w, _ = train_low_data[idx].shape
                    x = np.random.randint(0, h - patch_size)
                    y = np.random.randint(0, w - patch_size)
                    rand_mode = np.random.randint(0, 8)
                    low_patch = data_augmentation(train_low_data[idx][x:x+patch_size, y:y+patch_size, :], rand_mode)
                    high_patch = data_augmentation(train_high_data[idx][x:x+patch_size, y:y+patch_size, :], rand_mode)
                    low_eq_patch = data_augmentation(train_low_data_eq[idx][x:x+patch_size, y:y+patch_size, :], rand_mode)
                    batch_input_low[i] = low_patch
                    batch_input_high[i] = high_patch
                    batch_input_low_eq[i] = low_eq_patch
                batch_input_low = torch.from_numpy(batch_input_low).permute(0, 3, 1, 2).to(self.device)
                batch_input_high = torch.from_numpy(batch_input_high).permute(0, 3, 1, 2).to(self.device)
                batch_input_low_eq = torch.from_numpy(batch_input_low_eq).permute(0, 3, 1, 2).to(self.device)
                self.optimizer.zero_grad()
                loss, batch_losses = self.compute_loss(batch_input_low, batch_input_high, batch_input_low_eq)
                loss.backward()
                self.optimizer.step()
                self.accumulate_loss_dict(cur_epoch_losses, batch_losses)
                count += 1
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_id+1}/{num_batches}] Loss: {loss.item():.6f}")
                iter_num += 1
            self.append_to_loss_dict(cur_epoch_losses, count)
            
            avg_epoch_loss = cur_epoch_losses['total_loss'] / count if count > 0 else 0
            if (epoch + 1) % plot_every_epoch == 0:
                self.plot_loss_curve(os.path.join(eval_result_dir, 'loss_curves_combined.png'))
            if (epoch + 1) % eval_every_epoch == 0:
                self.save_checkpoint(os.path.join(ckpt_dir, f"model_epoch_{epoch+1}.pth"), epoch+1)
                self.save_checkpoint(os.path.join(ckpt_dir, "model_epoch_latest.pth"), epoch+1)
            print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_epoch_loss:.6f}")
    
    def append_to_loss_dict(self, cur_epoch_losses, count):
        self.all_epoch_losses['total_loss'].append(cur_epoch_losses['total_loss'] / count if count > 0 else 0)
        self.all_epoch_losses['recon_loss'].append(cur_epoch_losses['recon_loss'] / count if count > 0 else 0)
        self.all_epoch_losses['recon_loss_eq'].append(cur_epoch_losses['recon_loss_eq'] / count if count > 0 else 0)
        self.all_epoch_losses['R_smooth_loss'].append(cur_epoch_losses['R_smooth_loss'] / count if count > 0 else 0)
        self.all_epoch_losses['I_smooth_loss'].append(cur_epoch_losses['I_smooth_loss'] / count if count > 0 else 0)
        self.all_epoch_losses['I_smooth_loss_delta'].append(cur_epoch_losses['I_smooth_loss_delta'] / count if count > 0 else 0)
        self.all_epoch_losses['relight_loss'].append(cur_epoch_losses['relight_loss'] / count if count > 0 else 0)
        self.all_epoch_losses['fourier_loss'].append(cur_epoch_losses['fourier_loss'] / count if count > 0 else 0)
        self.all_epoch_losses['decom_loss'].append(cur_epoch_losses['decom_loss'] / count if count > 0 else 0)
        self.all_epoch_losses['relightNet_loss'].append(cur_epoch_losses['relightNet_loss'] / count if count > 0 else 0)
        self.all_epoch_losses['spectral_loss'].append(cur_epoch_losses['spectral_loss'] / count if count > 0 else 0)

    def accumulate_loss_dict(self, cur_epoch_losses, batch_losses):
        cur_epoch_losses['total_loss'] += batch_losses['total_loss']
        cur_epoch_losses['recon_loss'] += batch_losses['recon_loss']
        cur_epoch_losses['recon_loss_eq'] += batch_losses['recon_loss_eq']
        cur_epoch_losses['R_smooth_loss'] += batch_losses['R_smooth_loss']
        cur_epoch_losses['I_smooth_loss'] += batch_losses['I_smooth_loss']
        cur_epoch_losses['I_smooth_loss_delta'] += batch_losses['I_smooth_loss_delta']
        cur_epoch_losses['relight_loss'] += batch_losses['relight_loss']
        cur_epoch_losses['fourier_loss'] += batch_losses['fourier_loss']
        cur_epoch_losses['decom_loss'] += batch_losses['decom_loss']
        cur_epoch_losses['relightNet_loss'] += batch_losses['relightNet_loss']
        cur_epoch_losses['spectral_loss'] += batch_losses['spectral_loss']

    def save_checkpoint(self, path, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Checkpoint saved at {path}")
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {path}")
    
    def test_model(self, model_dir, test_low_data, test_low_data_names, save_dir):
        self.load_checkpoint(os.path.join(model_dir, 'model_epoch_latest.pth'))
        self.eval()
        total_run_time = 0.0
        with torch.no_grad():
            for idx in range(len(test_low_data)):
                filename = os.path.basename(test_low_data_names[idx])
                print(f'Processing {filename}')
                
                low_im = test_low_data[idx]
                input_tensor = torch.from_numpy(low_im).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
                start_time = time.time()
                R_low, I_low, I_delta, S = self.forward(input_tensor)
                run_time = time.time() - start_time
                total_run_time += run_time
                S_np = S.squeeze(0).permute(1, 2, 0).cpu().numpy()

                save_hsi(os.path.join(save_dir, filename), S_np)
                print(f"Processed {filename} in {run_time:.4f} seconds.")
            avg_run_time = total_run_time / len(test_low_data) if len(test_low_data) > 0 else 0
            print(f"Average run time: {avg_run_time:.4f} seconds.")
    
    def plot_loss_curve(self, save_path):
        """Plot and save all training loss curves with epoch numbers"""
        
        epochs = range(1, len(self.all_epoch_losses['total_loss']) + 1)
        
        plt.figure(figsize=(20, 10))
        
        # Plot each loss in a separate subplot
        plt.subplot(3, 4, 1)
        plt.plot(epochs, self.all_epoch_losses['total_loss'], 'k-', label='Total Loss')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(3, 4, 2)
        plt.plot(epochs, self.all_epoch_losses['recon_loss'], 'r-', label='Reconstruction Loss')
        plt.title('Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(3, 4, 3)
        plt.plot(epochs, self.all_epoch_losses['recon_loss_eq'], 'b-', label='Eq Reconstruction Loss')
        plt.title('Equalization Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(3, 4, 4)
        plt.plot(epochs, self.all_epoch_losses['R_smooth_loss'], 'g-', label='R Smoothness Loss')
        plt.title('R Smoothness Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(3, 4, 5)
        plt.plot(epochs, self.all_epoch_losses['I_smooth_loss'], 'm-', label='I Smoothness Loss')
        plt.title('I Smoothness Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 4, 6)
        plt.plot(epochs, self.all_epoch_losses['I_smooth_loss_delta'], 'r-', label='I Smoothness Delta Loss')
        plt.title('I Smoothness Delta Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(3, 4, 7)
        plt.plot(epochs, self.all_epoch_losses['relight_loss'], 'c-', label='Relightness Loss')
        plt.title('Relightness Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 4, 8)
        plt.plot(epochs, self.all_epoch_losses['decom_loss'], 'y-', label='Decom Loss')
        plt.title('Decom Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 4, 9)
        plt.plot(epochs, self.all_epoch_losses['relightNet_loss'], 'g-', label='RelightNet Loss')
        plt.title('RelightNet Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 4, 10)
        plt.plot(epochs, self.all_epoch_losses['fourier_loss'], 'm-', label='Fourier Loss')
        plt.title('Fourier Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 4, 11)
        plt.plot(epochs, self.all_epoch_losses['spectral_loss'], 'r-', label='Spectral Smoothness Loss')
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
