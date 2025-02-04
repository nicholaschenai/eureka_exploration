import os
import numpy as np
import re
from pathlib import Path

artifacts_dir = Path("eureka_artifacts")

def extract_max_training_success(log_file):
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            # Look for the pattern "Max Training Success X" in the log
            match = re.search(r'Max Training Success ([\d.-]+)', content)
            if match:
                return float(match.group(1))
    except Exception as e:
        print(f"Warning: Could not read log file {log_file}: {e}")
    return None

def analyze_folder(folder_path):
    results = {}
    
    # Try to load final_eval.npz
    try:
        eval_data = np.load(os.path.join(folder_path, 'final_eval.npz'))
        final_successes = eval_data.get('reward_code_final_successes')
        final_correlations = eval_data.get('reward_code_correlations_final')
        
        if final_successes is not None:
            results['final_success_mean'] = np.mean(final_successes)
            results['final_success_std'] = np.std(final_successes)
        
        if final_correlations is not None:
            results['final_correlation_mean'] = np.mean(final_correlations)
            results['final_correlation_std'] = np.std(final_correlations)
    except Exception as e:
        print(f"Warning: Could not load final_eval.npz in {folder_path}: {e}")
    
    # Try to get max training success from log
    log_file = os.path.join(folder_path, 'eureka.log')
    max_training_success = extract_max_training_success(log_file)
    if max_training_success is not None:
        results['max_training_success'] = max_training_success
    
    return results

def main():
    if not artifacts_dir.exists():
        print(f"Error: {artifacts_dir} does not exist")
        return

    # Process both eureka and human_baseline folders
    for exp_type in ['eureka', 'human_baseline']:
        exp_dir = artifacts_dir / exp_type
        if not exp_dir.exists():
            continue

        print(f"\n=== {exp_type.upper()} RESULTS ===")
        
        # Process each timestamp folder
        for timestamp_dir in sorted(exp_dir.iterdir()):
            if not timestamp_dir.is_dir():
                continue
                
            results = analyze_folder(timestamp_dir)
            
            if not results:  # Skip if no results found
                continue
                
            print(f"\nFolder: {exp_type}/{timestamp_dir.name}")
            
            if 'max_training_success' in results:
                print(f"Max Training Success: {results['max_training_success']:.2f}")
            
            if 'final_success_mean' in results:
                print(f"Final Success: {results['final_success_mean']:.2f} ± {results['final_success_std']:.2f}")
            
            if 'final_correlation_mean' in results:
                print(f"Final Correlation: {results['final_correlation_mean']:.2f} ± {results['final_correlation_std']:.2f}")

if __name__ == "__main__":
    main() 