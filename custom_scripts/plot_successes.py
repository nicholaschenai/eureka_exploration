import os
import glob
import matplotlib.pyplot as plt

from eureka_task_processor import EurekaTaskProcessor

# Define results directory structure
RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


def ensure_results_dirs():
    """Create results directory structure if it doesn't exist"""
    for d in [RESULTS_DIR, PLOTS_DIR]:
        os.makedirs(d, exist_ok=True)


def generate_plots(task_folder: str):
    """Generate reward plots for each task"""
    try:
        processor = EurekaTaskProcessor(task_folder)
        print(f"Generating plots for task folder: {task_folder}")
        
        # Plot training progress (max consecutive successes per iteration)
        max_successes = []
        for iter_num in range(processor.iteration):
            iter_policies = processor.get_iteration_policies(iter_num)
            if iter_policies:
                max_success = max(policy[0] for policy in iter_policies)
                max_successes.append(max_success)
            else:
                # If all runs in iteration failed, use 0 or previous value
                prev_value = 0
                max_successes.append(prev_value)
                print(f"Warning: No valid policies for iteration {iter_num}, using value: {prev_value}")
        
        if max_successes:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            x = range(len(max_successes))
            plt.plot(x, max_successes, label='Max Consecutive Successes')
            plt.xlabel('Iteration')
            plt.ylabel('Max Consecutive Successes')
            plt.title(f'Training Progress - {processor.task_name}')
            plt.legend()
        
        # Plot best evaluation run timeseries
        eval_policies = processor.get_iteration_policies(None)  # Get eval runs
        if eval_policies:
            _, _, best_eval_logs = max(eval_policies, key=lambda x: x[0])
            
            plt.subplot(1, 2, 2)
            consecutive_successes = best_eval_logs['consecutive_successes']
            x = consecutive_successes['steps']  # Use actual step numbers
            y = consecutive_successes['values']  # Use values
            plt.plot(x, y, label='Consecutive Successes')
            plt.xlabel('Step')
            plt.ylabel('Consecutive Successes')
            plt.title('Best Evaluation Run')
            plt.legend()
        else:
            print("Warning: No valid evaluation runs found")
        
        if max_successes or eval_policies:  # Only save if we have something to plot
            plt.tight_layout()
            plot_path = os.path.join(PLOTS_DIR, f"{processor.task_name}{processor.suffix}_performance.png")
            plt.savefig(plot_path)
            print(f"Saved plot to {plot_path}")
        plt.close()
            
    except Exception as e:
        print(f"Error generating plots for {task_folder}: {e}")


def main():
    # Ensure results directories exist
    ensure_results_dirs()    
    task_folders = glob.glob("custom_checkpoints/eureka/*/")
    
    for task_folder in task_folders:
        generate_plots(task_folder)
    

if __name__ == "__main__":
    main()
