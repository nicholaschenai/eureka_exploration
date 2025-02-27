"""
this script loads pytorch model weights, finds the appropriate model definition,
then saves the full model to a file such that the entire NN can be loaded via just one file
"""
import torch
from pathlib import Path
import os
import sys
import yaml
import re

# Configuration variables - change these as needed
env_name = "franka_cabinet"  # Change this to load different environments

# You can manually specify checkpoint info (no auto-discovery fallback)
CHECKPOINT_DIR = "outputs/ckpt_pth/2025-02-13_09-26-08"  # Required - must be valid
CHECKPOINT_FILE = "FrankaCabinetGPT_epoch__iter0.pth"    # Required - must be valid

# Get project root directory
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(__file__)))

# Add necessary paths to find modules
ISAAC_ROOT_DIR = PROJECT_ROOT.parent / "isaacgymenvs"
sys.path.append(str(ISAAC_ROOT_DIR))
sys.path.append(str(ISAAC_ROOT_DIR.parent))  # For rl_games

def load_yaml_config(file_path):
    """Load YAML configuration file"""
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading YAML file {file_path}: {e}")
        return None

def get_task_name_from_env_yaml(env_name):
    """Get task name from environment YAML file"""
    env_yaml_path = PROJECT_ROOT / "eureka" / "cfg" / "env" / f"{env_name}.yaml"
    if not os.path.exists(env_yaml_path):
        env_yaml_path = ISAAC_ROOT_DIR / "isaacgymenvs" / "cfg" / "task" / f"{env_name}.yaml"
    
    if not os.path.exists(env_yaml_path):
        print(f"Warning: Environment YAML file not found at {env_yaml_path}")
        # Try to infer task name from env_name (capitalize each word)
        return ''.join(word.capitalize() for word in env_name.split('_'))
    
    config = load_yaml_config(env_yaml_path)
    if config and 'task' in config:
        return config['task']
    else:
        # Fallback to inferring task name
        return ''.join(word.capitalize() for word in env_name.split('_'))

def find_network_config(task_name):
    """Find the network configuration YAML file based on task name"""
    # Try different possible config file names
    config_paths = [
        ISAAC_ROOT_DIR / "isaacgymenvs" / "cfg" / "train" / f"{task_name}GPTPPO.yaml",
        ISAAC_ROOT_DIR / "isaacgymenvs" / "cfg" / "train" / f"{task_name}GPT.yaml",
        ISAAC_ROOT_DIR / "isaacgymenvs" / "cfg" / "train" / f"{task_name}PPO.yaml",
        ISAAC_ROOT_DIR / "isaacgymenvs" / "cfg" / "train" / f"{task_name}.yaml"
    ]
    
    for path in config_paths:
        if os.path.exists(path):
            return path
    
    print(f"Warning: Could not find network config for {task_name}. Using default network configuration.")
    return None

def analyze_py_file(env_name):
    """Analyze Python file to extract observation and action space dimensions"""
    py_file_path = ISAAC_ROOT_DIR / "isaacgymenvs" / "tasks" / f"{env_name}.py"
    
    if not os.path.exists(py_file_path):
        raise FileNotFoundError(f"Python file not found at {py_file_path}")
    
    # Read file content and find observation and action space dimensions using pattern matching
    with open(py_file_path, 'r') as file:
        content = file.read()
        
    # Look for self.cfg["env"] assignments
    env_obs_pattern = r'self\.cfg\["env"\]\["numObservations"\]\s*=\s*([^;,\n]+)'
    env_act_pattern = r'self\.cfg\["env"\]\["numActions"\]\s*=\s*([^;,\n]+)'
    
    env_obs_match = re.search(env_obs_pattern, content)
    env_act_match = re.search(env_act_pattern, content)
    
    # Helper function to extract dimension value from a match
    def extract_dimension(match, dimension_type):
        if not match:
            return None
            
        value = match.group(1).strip()
        # Check if it's a direct number
        if value.isdigit():
            return int(value)
        else:
            # It's likely a variable, try to find its assignment
            var_pattern = r'{}\s*=\s*(\d+)'.format(re.escape(value))
            var_match = re.search(var_pattern, content)
            if var_match:
                dim_value = int(var_match.group(1))
                print(f"Found {dimension_type} dimension {dim_value} from variable {value}")
                return dim_value
            else:
                print(f"Could not resolve variable {value} for {dimension_type} dimension")
                return None
    
    # Extract observation and action dimensions
    obs_dim = extract_dimension(env_obs_match, "observation")
    act_dim = extract_dimension(env_act_match, "action")
            
    return obs_dim, act_dim


def main():
    # Get task name from env_name
    task_name = get_task_name_from_env_yaml(env_name)
    print(f"Detected task name: {task_name}")
    
    # Find network configuration
    network_config_path = find_network_config(task_name)
    network_config = load_yaml_config(network_config_path) if network_config_path else {}
    
    # Analyze Python file for observation and action dimensions
    obs_dim, act_dim = analyze_py_file(env_name)
    print(f"Detected from Python file - Observation dimension: {obs_dim}, Action dimension: {act_dim}")
    
    # Get checkpoint path
    # Simply construct the full path and verify it exists once
    checkpoint_path = PROJECT_ROOT / CHECKPOINT_DIR / CHECKPOINT_FILE
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Import modules
    try:
        from rl_games.algos_torch import model_builder, models
        from rl_games.algos_torch import torch_ext
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print(f"PYTHONPATH: {sys.path}")
        return None
    
    # Load checkpoint
    try:
        checkpoint = torch_ext.load_checkpoint(str(checkpoint_path))
        print(f"Checkpoint loaded successfully")
        print(f"Checkpoint keys: {checkpoint.keys()}")
    except Exception as e:
        print(f"Error with torch_ext.load_checkpoint: {e}")
        try:
            checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
            print(f"Loaded with torch.load instead. Keys: {checkpoint.keys()}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None
    
    # Use observation and action dimensions directly from Python file analysis
    # We won't try to extract from model weights as it's unreliable
    if obs_dim is None:
        raise ValueError("Could not determine observation dimension from Python file. Please check the environment implementation.")
    
    if act_dim is None:
        raise ValueError("Could not determine action dimension from Python file. Please check the environment implementation.")
        
    obs_shape = obs_dim
    action_space_shape = act_dim
    print(f"Using observation shape from Python file: {obs_shape}")
    print(f"Using action space shape from Python file: {action_space_shape}")
    
    # For observation shape, we need a tuple for the network builder
    if isinstance(obs_shape, int):
        obs_shape_tuple = (obs_shape,)
    else:
        obs_shape_tuple = obs_shape
    
    # Create model builder
    mb = model_builder.ModelBuilder()
    
    def resolve_default_value(value):
        """Helper function to resolve Hydra-style ${resolve_default:...} values"""
        if isinstance(value, str) and value.startswith("${resolve_default:"):
            # Extract the default value (first value after resolve_default:)
            try:
                default_value = value.split("${resolve_default:")[1].split(",")[0]
                # Try to convert to appropriate type (int, float, bool)
                if default_value.isdigit():
                    return int(default_value)
                elif default_value.replace(".", "").isdigit():
                    return float(default_value)
                elif default_value.lower() in ["true", "false"]:
                    return default_value.lower() == "true"
                return default_value
            except:
                return value
        return value

    def resolve_config_defaults(config):
        """Recursively resolve defaults in a config dictionary"""
        if isinstance(config, dict):
            return {k: resolve_config_defaults(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [resolve_config_defaults(v) for v in config]
        else:
            return resolve_default_value(config)
    
    # Use the load method to create both network and model in one step
    print(f"Using model from YAML: {network_config['params']['model']['name']}")
    model = mb.load(network_config['params'])
    
    input_config = resolve_config_defaults(network_config['params']['config'].copy())
    # Remove the 'name' parameter to avoid conflict
    input_config.pop('name', None)  # Using pop with None default in case 'name' doesn't exist
    
    # Add required parameters that must come from the environment
    input_config['actions_num'] = action_space_shape
    input_config['input_shape'] = obs_shape_tuple
    
    print(f"Final input config: {input_config}")
    
    try:
        nn_model = model.build(input_config)
        # class NetworkBuilder.BaseNetwork.__init__.<locals>.<lambda>
        print("Model network built successfully!")
        
        # Load weights from checkpoint
        if 'model' in checkpoint:
            try:
                nn_model.load_state_dict(checkpoint['model'])
                print("Model weights loaded successfully!")
            except Exception as e:
                print(f"Error loading model weights: {e}")
                print("\nThis usually indicates a mismatch between the checkpoint and the model architecture.")
                print("The model structure is still valid, but the weights couldn't be loaded correctly.")
        else:
            print("Warning: 'model' key not found in checkpoint")
        
        # Print model information
        print("\nModel Structure:")
        print(nn_model)
        
        total_params = sum(p.numel() for p in nn_model.parameters())
        print(f"\nTotal parameters: {total_params:,}")
        
        # Set to evaluation mode
        # nn_model.eval()
        
        print("\nModel is ready for use!")
        return nn_model
        
    except Exception as e:
        print(f"Error building model: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    rl_games_model = main()


    # Save the complete model (architecture + weights) to a single file
    if rl_games_model is not None:
        # Create save path next to original checkpoint file
        checkpoint_name = os.path.splitext(CHECKPOINT_FILE)[0]  # Remove extension
        save_filename = f"{checkpoint_name}_pytorch.pt"
        save_path = os.path.join(PROJECT_ROOT, CHECKPOINT_DIR, save_filename)
        
        try:
            
            # TODO: generalize this
            torch.save(rl_games_model.a2c_network.actor_mlp, save_path)
            print(f"\nComplete model (architecture + weights) saved successfully to: {save_path}")
            
            # Verify by loading back
            print("\nVerifying saved model...")
            loaded_model = torch.load(save_path)
            print(f"Model loaded successfully! Type: {type(loaded_model).__name__}")
            
        except Exception as e:
            print(f"Error saving model: {e}")
            import traceback
            traceback.print_exc()
    
        # Example showing how to use the model
        print("\nExample: Access the neural network components")
        if hasattr(rl_games_model, 'a2c_network'):
            print(f"Network type: {type(rl_games_model.a2c_network).__name__}")
        
        try:
            # Example showing how to examine the model's state dictionary
            print("\nExamining model parameters:")
            if 'running_mean_std.running_mean' in rl_games_model.state_dict():
                mean = rl_games_model.state_dict()['running_mean_std.running_mean']
                print(f"Observation normalization mean shape: {mean.shape}")
            
        except Exception as e:
            print(f"Error examining model: {e}")
