import os
import argparse
import torch
from main import parse_args, main
from model import LowLightEnhance

def pretrain_with_indoor_weights():
    """
    Initialize outdoor model with indoor weights, then fine-tune
    """
    # Parse args for outdoor config
    import sys
    sys.argv = ['pretrain_outdoor.py', '--config', 'config/config_outdoor_balanced.yml', 
                '--model_name', 'outdoor_pretrained']
    args = parse_args()
    
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    
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
        device=device
    ).to(device)
    
    # Try to load indoor weights
    indoor_checkpoint = "./checkpoint/no_name_model/Decom_/model_epoch_latest.pth"
    if os.path.exists(indoor_checkpoint):
        print(f"Loading indoor weights from {indoor_checkpoint}")
        checkpoint = torch.load(indoor_checkpoint, map_location=device)
        
        # Load only the model weights, not optimizer state
        model_state = checkpoint['model_state_dict']
        model.load_state_dict(model_state, strict=False)
        print("Indoor weights loaded successfully!")
        
        # Reset optimizer with new learning rate
        model.optimizer = torch.optim.Adam(model.parameters(), lr=args.start_lr * 0.1)  # Lower LR for fine-tuning
    else:
        print("No indoor checkpoint found, training from scratch")
    
    # Now run normal training
    main(args)

if __name__ == "__main__":
    pretrain_with_indoor_weights()