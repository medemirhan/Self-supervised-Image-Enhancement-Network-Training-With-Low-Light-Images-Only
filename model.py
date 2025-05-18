import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import numpy as np
from glob import glob
import mlflow
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
        # x: (B, in_channels, H, W)
        conv0 = self.conv0(x) # conv0: (B, channel//2, H, W) 
        shallow = self.shallow_conv(x) # shallow: (B, channel, H, W)
        conv1 = self.conv1(shallow) # conv1: (B, channel, H, W)
        conv2 = self.conv2(conv1) # conv2: (B, 2*channel, H/2, W/2)  [Downsampled by stride 2]
        conv3 = self.conv3(conv2) # conv3: (B, 2*channel, H/2, W/2)
        deconv = self.deconv(conv3) # deconv: (B, channel, H, W)  [Upsampled to original H, W]
        
        # First skip connection: fuse deconv and conv1
        concat1 = torch.cat([deconv, conv1], dim=1) # concat1: (B, channel + channel, H, W) = (B, 2*channel, H, W)
        conv5 = self.conv5(concat1) # conv5: (B, channel, H, W)
        
        # Second skip connection: fuse conv5 and conv0
        concat2 = torch.cat([conv5, conv0], dim=1) # concat2: (B, channel + (channel//2), H, W)
        conv7 = self.conv7(concat2) # conv7: (B, channel, H, W)
        conv8 = self.recon(conv7) # conv8: (B, in_channels + 1, H, W)
        
        # Split the output into reflectance and illumination components
        R = torch.sigmoid(conv8[:, :self.in_channels, :, :]) # R: (B, in_channels, H, W)
        L = torch.sigmoid(conv8[:, self.in_channels:, :, :]) # L: (B, 1, H, W)
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
        # I: illumination (B, 1, H, W)
        # R: reflectance (B, in_channels, H, W)
        x = torch.cat([R, I], dim=1) # x: (B, in_channels + 1, H, W)
        conv0 = self.conv0(x) # conv0: (B, channel, H, W)
        conv1 = self.conv1(conv0) # conv1: (B, channel, H/2, W/2)  [Downsampled]
        conv2 = self.conv2(conv1) # conv2: (B, channel, H/4, W/4)  [Downsampled]
        conv3 = self.conv3(conv2) # conv3: (B, channel, H/8, W/8)  [Downsampled]
        
        # Attention block
        conv3 = self.attn(conv3) # conv3 (after attention): (B, channel, H/8, W/8)
        
        # Upsample and add skip connection from conv2
        up1 = F.interpolate(conv3, size=conv2.shape[2:], mode='nearest') # up1: (B, channel, H/4, W/4)
        deconv1 = self.deconv1(up1) + conv2 # deconv1: (B, channel, H/4, W/4)
        
        # Upsample and add skip connection from conv1
        up2 = F.interpolate(deconv1, size=conv1.shape[2:], mode='nearest') # up2: (B, channel, H/2, W/2)
        deconv2 = self.deconv2(up2) + conv1 # deconv2: (B, channel, H/2, W/2)
        
        # Upsample and add skip connection from conv0
        up3 = F.interpolate(deconv2, size=conv0.shape[2:], mode='nearest') # up3: (B, channel, H, W)
        deconv3 = self.deconv3(up3) + conv0 # deconv3: (B, channel, H, W)
        
        # Resize deconv1 and deconv2 to match deconv3's spatial size for gathering
        deconv1_resize = F.interpolate(deconv1, size=deconv3.shape[2:], mode='nearest') # deconv1_resize: (B, channel, H, W)
        deconv2_resize = F.interpolate(deconv2, size=deconv3.shape[2:], mode='nearest') # deconv2_resize: (B, channel, H, W)
        
        # Concatenate features along channel dimension
        feature_gather = torch.cat([deconv1_resize, deconv2_resize, deconv3], dim=1) # feature_gather: (B, 3 * channel, H, W)
        feature_fusion = self.feature_fusion(feature_gather) # feature_fusion: (B, channel, H, W)
        output = self.final_conv(feature_fusion) # output: (B, 1, H, W)
        return output

class LowLightEnhance(nn.Module):
    def __init__(self, input_channels=64, lr=1e-3, lr_update_factor=1, lr_update_period=None, time_stamp=None, 
                 c_loss_reconstruction=10, c_loss_r_fidelity=1, c_loss_i_smooth_low=1, c_loss_i_smooth_delta=20,
                 c_loss_fourier=0.2, c_loss_spectral_cons=1, alpha_i_smooth_low=1, alpha_i_smooth_delta=10,
                 device=torch.device("cpu")):
        super(LowLightEnhance, self).__init__()
        self.input_channels = input_channels
        self.device = device
        self.time_stamp = time_stamp
        self.c_loss_reconstruction = c_loss_reconstruction
        self.c_loss_r_fidelity = c_loss_r_fidelity
        self.c_loss_i_smooth_low = c_loss_i_smooth_low
        self.c_loss_i_smooth_delta = c_loss_i_smooth_delta
        self.c_loss_fourier = c_loss_fourier
        self.c_loss_spectral_cons = c_loss_spectral_cons
        self.alpha_i_smooth_low = alpha_i_smooth_low
        self.alpha_i_smooth_delta = alpha_i_smooth_delta
        self.lr = lr
        self.lr_update_factor = lr_update_factor
        self.lr_update_period = lr_update_period
        self.adaptive_lr = False

        if abs(self.lr_update_factor - 1) > 1e-6:
            self.adaptive_lr = True

        self.decom_net = DecomNet(in_channels=input_channels)
        self.relight_net = RelightNet(in_channels=input_channels)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        if self.adaptive_lr:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_update_period, gamma=self.lr_update_factor)
        
        self.all_epoch_losses = {
            'total_loss': [],
            'L_reconstruction': [],
            'L_R_fidelity': [],
            'L_I_smooth_low': [],
            'L_I_smooth_delta': [],
            'L_fourier': [],
            'L_spectral_cons': []
            }

    def forward(self, input_low):
        # input_low: (N, input_channels, H, W)
        R_low, I_low = self.decom_net(input_low)
        I_delta = self.relight_net(I_low, R_low)
        S = R_low * I_delta + R_low * I_low
        return R_low, I_low, I_delta, S
    
    def train_model(self, train_data_path, eval_data_path, batch_size, patch_size, num_epochs, start_lr, ckpt_dir, eval_result_dir, eval_every_epoch, max_val=None, min_val=None, plot_every_epoch=10):
        ckpt_dir = os.path.join(ckpt_dir, 'Decom_' + self.time_stamp)
        
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(eval_result_dir, exist_ok=True)
        train_files = sorted(glob(os.path.join(train_data_path, "*.*")))
        train_low_data = []
        train_high_data = []
        for file in train_files:
            low_im = load_hsi(file, matContentHeader='data', normalization='global_normalization', max_val=max_val, min_val=min_val)
            train_low_data.append(low_im)
            high_im = load_hsi(file, matContentHeader='data', normalization='global_normalization', max_val=max_val, min_val=min_val)
            train_high_data.append(high_im)
            low_eq = pca_projection(low_im)
            if low_eq.ndim == 2:
                low_eq = low_eq[:, :, None]
        eval_low_data = []
        eval_files = sorted(glob(os.path.join(eval_data_path, "*.*")))
        for file in eval_files:
            eval_im = load_hsi(file, matContentHeader='data', normalization='global_normalization', max_val=1.7410845, min_val=0.0708354)
            eval_low_data.append(eval_im)
        
        num_batches = len(train_low_data) // batch_size
        iter_num = 0

        log_params = {
            "epochs": num_epochs,
            "start_lr": start_lr,
            "adaptive_lr": self.adaptive_lr,
            "batch_size": batch_size,
            "optimizer": "Adam",
        }
        # Log training parameters.
        mlflow.log_params(log_params)

        # Log model summary.
        summary_path = os.path.join(ckpt_dir, "model_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(str(summary(self)))
        mlflow.log_artifact(summary_path)

        for epoch in range(num_epochs):
            cur_epoch_losses = {
                'total_loss': 0,
                'L_reconstruction': 0,
                'L_R_fidelity': 0,
                'L_I_smooth_low': 0,
                'L_I_smooth_delta': 0,
                'L_fourier': 0,
                'L_spectral_cons': 0
                }
            count = 0
            for batch_id in range(num_batches):
                batch_input_low = np.zeros((batch_size, patch_size, patch_size, self.input_channels), dtype=np.float32)
                batch_input_high = np.zeros((batch_size, patch_size, patch_size, self.input_channels), dtype=np.float32)
                
                for i in range(batch_size):
                    idx = (batch_id * batch_size + i) % len(train_low_data)
                    h, w, _ = train_low_data[idx].shape
                    x = np.random.randint(0, h - patch_size)
                    y = np.random.randint(0, w - patch_size)
                    rand_mode = np.random.randint(0, 8)
                    low_patch = data_augmentation(train_low_data[idx][x:x+patch_size, y:y+patch_size, :], rand_mode)
                    high_patch = data_augmentation(train_high_data[idx][x:x+patch_size, y:y+patch_size, :], rand_mode)
                    batch_input_low[i] = low_patch
                    batch_input_high[i] = high_patch
                
                batch_input_low = torch.from_numpy(batch_input_low).permute(0, 3, 1, 2).to(self.device)
                batch_input_high = torch.from_numpy(batch_input_high).permute(0, 3, 1, 2).to(self.device)
                self.optimizer.zero_grad()
                loss, batch_losses = self.compute_loss(batch_input_low, batch_input_high)
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

            mlflow.log_metric("learning_rate", self.optimizer.param_groups[0]['lr'], step=epoch)
            if self.adaptive_lr:
                self.scheduler.step()

            print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_epoch_loss:.6f}")

            mlflow.log_metrics(cur_epoch_losses, step=epoch)
        
        mlflow.log_param('model_path', os.path.normpath(os.path.join(ckpt_dir, "model_epoch_latest.pth")))
    
    def test_model(self, model_dir, test_low_data, test_low_data_names, save_dir, save_reflectance=False, save_illumination=False, save_i_delta=False):
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

                if save_reflectance:
                    R_np = R_low.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    save_hsi(os.path.join(save_dir, filename.split('.')[0] + '_R_low.mat'), R_np)
                if save_illumination:
                    I_np = I_low.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    save_hsi(os.path.join(save_dir, filename.split('.')[0] + '_I_low.mat'), I_np)
                if save_i_delta:
                    I_delta_np = I_delta.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    save_hsi(os.path.join(save_dir, filename.split('.')[0] + '_I_delta.mat'), I_delta_np)

                print(f"Processed {filename} in {run_time:.4f} seconds.")
            avg_run_time = total_run_time / len(test_low_data) if len(test_low_data) > 0 else 0
            print(f"Average run time: {avg_run_time:.4f} seconds.")

    def compute_gradients(self, img):
        grad_x = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
        grad_y = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
        return grad_x, grad_y
    
    def smooth_loss(self, I, R, alpha=10):
        grad_Ix, grad_Iy = self.compute_gradients(I)
        grad_Rx, grad_Ry = self.compute_gradients(R)
        loss = torch.mean(grad_Ix * torch.exp(-alpha * grad_Rx)) + torch.mean(grad_Iy * torch.exp(-alpha * grad_Ry))
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

    def gradient_x(self, x):
        # Computes forward differences along width dimension
        return x[..., :, 1:] - x[..., :, :-1]

    def gradient_y(self, x):
        # Computes forward differences along height dimension
        return x[..., 1:, :] - x[..., :-1, :]

    def structure_aware_loss(self, R, I, R_enh, alpha=1.0, beta=1.0, lambda_I=1.0, lambda_R=1.0):
        """
        Structure-aware loss that combines edge-aware illumination smoothness and reflectance fidelity.
        
        Parameters:
            R (torch.Tensor): Predicted reflectance, shape (B, C, H, W).
            I (torch.Tensor): Predicted illumination, shape (B, 1, H, W).
            R_enh (torch.Tensor): Reflectance of the enhanced image, shape (B, C, H, W).
            alpha (float): Weighting parameter for edge-aware weighting.
            beta (float): Weight for the gradient term in the reflectance loss.
            lambda_I (float): Weight for the illumination loss.
            lambda_R (float): Weight for the reflectance fidelity loss.
        
        Returns:
            torch.Tensor: The total loss.
        """
        
        # --------------------------
        # 1. Edge-Aware Illumination Loss
        # --------------------------
        # Compute gradients of reflectance R (for weight computation)
        grad_R_x = self.gradient_x(R)  # shape: (B, C, H, W-1)
        grad_R_y = self.gradient_y(R)  # shape: (B, C, H-1, W)
        
        # Average absolute gradients over channels to get a per-pixel weight
        # Resulting shapes: (B, 1, H, W-1) and (B, 1, H-1, W)
        weight_x = torch.exp(-alpha * grad_R_x.abs().mean(dim=1, keepdim=True))
        weight_y = torch.exp(-alpha * grad_R_y.abs().mean(dim=1, keepdim=True))
        
        # Compute gradients of illumination I
        grad_I_x = self.gradient_x(I)  # shape: (B, 1, H, W-1)
        grad_I_y = self.gradient_y(I)  # shape: (B, 1, H-1, W)
        
        # Compute weighted L1 loss on illumination gradients
        loss_I_x = torch.mean(weight_x * grad_I_x.abs())
        loss_I_y = torch.mean(weight_y * grad_I_y.abs())
        loss_I = loss_I_x + loss_I_y

        # --------------------------
        # 2. Reflectance Fidelity Loss
        # --------------------------
        # Pixel-wise L1 loss between two reflectances
        loss_R1 = torch.mean(torch.abs(R - R_enh))
        
        # Compute gradients for reflectances R and R_enh
        grad_R_x = self.gradient_x(R)
        grad_R_y = self.gradient_y(R)
        grad_R_enh_x = self.gradient_x(R_enh)
        grad_R_enh_y = self.gradient_y(R_enh)
        
        # L1 loss on the difference of gradients
        loss_R2_x = torch.mean(torch.abs(grad_R_x - grad_R_enh_x))
        loss_R2_y = torch.mean(torch.abs(grad_R_y - grad_R_enh_y))
        loss_R2 = loss_R2_x + loss_R2_y
        
        loss_R = loss_R1 + beta * loss_R2

        # --------------------------
        # 3. Total Loss
        # --------------------------
        i_loss = lambda_I * loss_I
        r_loss = lambda_R * loss_R

        return i_loss, r_loss

    def compute_loss(self, input_low, input_high):
        R_low, I_low, I_delta, S = self.forward(input_low)
        R_enh, I_enh = self.decom_net(S)

        alpha1 = self.alpha_i_smooth_low
        alpha2 = self.alpha_i_smooth_delta
        
        L_reconstruction = torch.mean(torch.abs(R_low * I_low - input_high))
        L_I_smooth_low, L_R_fidelity = self.structure_aware_loss(R_low, I_low, R_enh, alpha=alpha1, beta=0.5, lambda_I=1.0, lambda_R=1.0)
        L_I_smooth_delta = self.smooth_loss(I_delta, R_low, alpha=alpha2)
        L_fourier = self.fourier_spectrum_loss(input_low, S, cutoff=0.1, loss_type="l1")
        L_spectral_cons = self.spectral_smoothness_loss(S, loss_type="l1")
        
        total_loss = (
            self.c_loss_reconstruction * L_reconstruction + 
            self.c_loss_r_fidelity * L_R_fidelity +
            self.c_loss_i_smooth_low * L_I_smooth_low +
            self.c_loss_i_smooth_delta * L_I_smooth_delta + 
            self.c_loss_fourier * L_fourier + 
            self.c_loss_spectral_cons * L_spectral_cons
            )

        losses = {
            'total_loss': total_loss.item(),
            'L_reconstruction': L_reconstruction.item(),
            'L_R_fidelity': L_R_fidelity.item(),
            'L_I_smooth_low': L_I_smooth_low.item(),
            'L_I_smooth_delta': L_I_smooth_delta.item(),
            'L_fourier': L_fourier.item(),
            'L_spectral_cons': L_spectral_cons.item()
        }
        return total_loss, losses

    def append_to_loss_dict(self, cur_epoch_losses, count):
        self.all_epoch_losses['total_loss'].append(cur_epoch_losses['total_loss'] / count if count > 0 else 0)
        self.all_epoch_losses['L_reconstruction'].append(cur_epoch_losses['L_reconstruction'] / count if count > 0 else 0)
        self.all_epoch_losses['L_R_fidelity'].append(cur_epoch_losses['L_R_fidelity'] / count if count > 0 else 0)
        self.all_epoch_losses['L_I_smooth_low'].append(cur_epoch_losses['L_I_smooth_low'] / count if count > 0 else 0)
        self.all_epoch_losses['L_I_smooth_delta'].append(cur_epoch_losses['L_I_smooth_delta'] / count if count > 0 else 0)
        self.all_epoch_losses['L_fourier'].append(cur_epoch_losses['L_fourier'] / count if count > 0 else 0)
        self.all_epoch_losses['L_spectral_cons'].append(cur_epoch_losses['L_spectral_cons'] / count if count > 0 else 0)

    def accumulate_loss_dict(self, cur_epoch_losses, batch_losses):
        cur_epoch_losses['total_loss'] += batch_losses['total_loss']
        cur_epoch_losses['L_reconstruction'] += batch_losses['L_reconstruction']
        cur_epoch_losses['L_R_fidelity'] += batch_losses['L_R_fidelity']
        cur_epoch_losses['L_I_smooth_low'] += batch_losses['L_I_smooth_low']
        cur_epoch_losses['L_I_smooth_delta'] += batch_losses['L_I_smooth_delta']
        cur_epoch_losses['L_fourier'] += batch_losses['L_fourier']
        cur_epoch_losses['L_spectral_cons'] += batch_losses['L_spectral_cons']

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

    def plot_loss_curve(self, save_path):
        """Plot and save all training loss curves with epoch numbers"""
        
        epochs = range(1, len(self.all_epoch_losses['total_loss']) + 1)
        
        plt.figure(figsize=(20, 10))

        # Plot each loss in a separate subplot
        plt.subplot(3, 3, 1)
        plt.plot(epochs, self.all_epoch_losses['total_loss'], 'k-', label='total_loss')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 3, 2)
        plt.plot(epochs, self.all_epoch_losses['L_reconstruction'], 'r-', label='L_reconstruction')
        plt.title('Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 3, 3)
        plt.plot(epochs, self.all_epoch_losses['L_R_fidelity'], 'b-', label='L_R_fidelity')
        plt.title('Reflectance Fidelity Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 3, 4)
        plt.plot(epochs, self.all_epoch_losses['L_I_smooth_low'], 'y-', label='L_I_smooth_low')
        plt.title('Structure-aware Illumination Smoothness Loss (I_low)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 3, 5)
        plt.plot(epochs, self.all_epoch_losses['L_I_smooth_delta'], 'g-', label='L_I_smooth_delta')
        plt.title('Structure-aware Illumination Smoothness Loss (I_delta)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 3, 6)
        plt.plot(epochs, self.all_epoch_losses['L_fourier'], 'm-', label='L_fourier')
        plt.title('Fourier Spectrum Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 3, 7)
        plt.plot(epochs, self.all_epoch_losses['L_spectral_cons'], 'c-', label='L_spectral_cons')
        plt.title('Spectral Consistency Loss')
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
