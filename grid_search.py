import itertools
import subprocess
import yaml
import os

# Define parameter grid
param_grid = {
    'c_loss_reconstruction': [1, 0.1],
    'c_loss_r_fidelity': [10000, 500],
    'c_loss_i_smooth_low': [1000000, 1000],
    'c_loss_i_smooth_delta': [100, 10000],
    'c_loss_fourier': [0.05, 1],
    'c_loss_spectral_cons': [100, 10],
}

# Path to the base config file
base_config_path = './config/config_outdoor_grid.yml'
with open(base_config_path, 'r') as f:
    base_config = yaml.safe_load(f)

# Create output directory for generated configs
os.makedirs('configs', exist_ok=True)

# Generate all combinations
keys, values = zip(*param_grid.items())
for idx, combo in enumerate(itertools.product(*values)):
    config = base_config.copy()
    combo_dict = dict(zip(keys, combo))
    config.update(combo_dict)
    
    # Write new config file
    config_path = f'configs/config_{idx}.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"\nRunning config {idx}: {combo_dict}")
    try:
        subprocess.run([
            'python', 'main.py',
            '--config', config_path,
            '--model_name', 'outdoor_grid_search'
        ], check=True)
    finally:
        os.remove(config_path)
