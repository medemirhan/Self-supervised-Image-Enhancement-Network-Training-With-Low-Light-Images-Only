import itertools
import subprocess
import yaml
import os

#------------------------ USER INPUTS START ------------------------#

# Define parameter grid
param_grid = {
    'alpha_i_smooth_low': [0.1, 10, 50],
    'alpha_i_smooth_delta': [1, 100, 1000],
}

# Path to the base config file
base_config_path = './config/config_jyu_outdoor_64_aligned_simple.yml'

user_model_name = 'jyu_outdoor_64_aligned_simple_mini_grid_search'

#------------------------- USER INPUTS END -------------------------#

with open(base_config_path, 'r') as f:
    base_config = yaml.safe_load(f)

# Create output directory for generated configs
output_config_dir = 'configs'
os.makedirs(output_config_dir, exist_ok=True)

# Generate all combinations
keys, values = zip(*param_grid.items())
combinations = list(itertools.product(*values))
total_combinations = len(combinations)

print(f"Starting grid search with {total_combinations} combinations...")

# --- Loop through all combinations ---
for idx, combo in enumerate(combinations):
    config = base_config.copy()
    combo_dict = dict(zip(keys, combo))
    config.update(combo_dict)
    
    # Write new config file
    config_path = os.path.join(output_config_dir, f'config_{idx}.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    print("-" * 50)
    print(f"Running combination {idx + 1}/{total_combinations}: {combo_dict}")

    try:
        # The command to run the training script
        command = [
            'python', 'main.py',
            '--config', config_path,
            '--model_name', user_model_name
        ]
        # Run the subprocess. If it fails, CalledProcessError will be raised.
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Combination {idx + 1} completed successfully.")

    except subprocess.CalledProcessError as e:
        # This block catches the error, allowing the loop to continue.
        print(f"!!! Combination {idx + 1} FAILED !!!")
        print(f"Return Code: {e.returncode}")
        print("\n--- STDOUT ---")
        print(e.stdout)
        print("\n--- STDERR ---")
        print(e.stderr)
        print("-" * 50)
        # The script will now continue to the next iteration.

    finally:
        # This block always runs, ensuring the temporary config file is removed.
        if os.path.exists(config_path):
            os.remove(config_path)
