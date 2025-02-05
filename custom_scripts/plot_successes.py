import os
import glob
import matplotlib.pyplot as plt

from eureka_task_processor import EurekaTaskProcessor

# Define results directory structure
# CHECKPOINTS_DIR = "custom_checkpoints"
CHECKPOINTS_DIR = "eureka_artifacts"
# RESULTS_DIR = "results"
RESULTS_DIR = CHECKPOINTS_DIR
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


def ensure_results_dirs():
    """Create results directory structure if it doesn't exist"""
    for d in [RESULTS_DIR, PLOTS_DIR]:
        os.makedirs(d, exist_ok=True)


def generate_train_eval_plots(task_folder: str):
    """
    Generate plots for each task.
    Shows max consecutive success over iterations in left panel.
    Shows consecutive successes vs step for best evaluation run in right panel.
    """
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


def generate_comparison_plots():
    """
    Generate comparison plots between Eureka runs (default and 3000 epochs) and human baselines.
    For each task, creates two side-by-side plots:
    1. Bar plot showing performance ratio compared to human baseline
    2. Training progress plot with iteration data
    """
    # Get all task folders
    eureka_folders = glob.glob(os.path.join(CHECKPOINTS_DIR, "eureka/*/"))
    human_folders = glob.glob(os.path.join(CHECKPOINTS_DIR, "human_baseline/*/"))
    
    # Group folders by task name
    task_runs = {}
    for folder in eureka_folders + human_folders:
        try:
            processor = EurekaTaskProcessor(folder)
            task_name = processor.task_name
            if task_name not in task_runs:
                task_runs[task_name] = {
                    "default": None,  # Default epochs
                    "3000": None,    # 3000 epochs
                    "human": []      # Human baseline(s)
                }
            
            if "human_baseline" in folder:
                task_runs[task_name]["human"].append(processor)
            else:  # Eureka folder
                if processor.max_iterations == 3000:
                    task_runs[task_name]["3000"] = processor
                else:
                    task_runs[task_name]["default"] = processor
                    
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")
            continue
    
    # Generate plots for each task
    for task_name, runs in task_runs.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Get human baseline mean first since we need it for both plots
        human_mean = None
        if runs["human"]:
            human_means = []
            for processor in runs["human"]:
                eval_policies = processor.get_iteration_policies(None)
                if eval_policies:
                    mean = sum(policy[0] for policy in eval_policies) / len(eval_policies)
                    human_means.append(mean)
            if human_means:
                human_mean = sum(human_means) / len(human_means)
        
        if human_mean is None:
            print(f"Warning: No human baseline data for {task_name}")
            continue

        # Plot 1: Performance Ratio Bar Plot
        ratios = []
        labels = []
        colors = ['darkblue', 'darkgreen']
        
        for run_type, color in [("default", colors[0]), ("3000", colors[1])]:
            processor = runs[run_type]
            if processor:
                eval_policies = processor.get_iteration_policies(None)
                if eval_policies:
                    final_mean = sum(policy[0] for policy in eval_policies) / len(eval_policies)
                    ratio = final_mean / human_mean
                    ratios.append(ratio)
                    labels.append(f"{'3000' if run_type == '3000' else 'Default'} training epochs")

        ax1.bar(range(len(ratios)), ratios, color=colors[:len(ratios)])
        ax1.set_xticks(range(len(ratios)))
        ax1.set_xticklabels(labels)
        ax1.axhline(y=1.0, color='red', linestyle='--', label='Human')
        ax1.set_ylabel('Ratio: Eval Success vs Human')
        ax1.set_title('Performance Ratio')
        ax1.legend()

        # Plot 2: Training Progress
        for run_type, (color, label) in [
            ("default", ('darkblue', 'Eureka (default training epochs)')),
            ("3000", ('darkgreen', 'Eureka (3000 training epochs)'))
        ]:
            processor = runs[run_type]
            if processor:
                max_successes = []
                for iter_num in range(processor.iteration):
                    iter_policies = processor.get_iteration_policies(iter_num)
                    if iter_policies:
                        max_success = max(policy[0] for policy in iter_policies)
                        max_successes.append(max_success)
                    else:
                        prev_value = max_successes[-1] if max_successes else 0
                        max_successes.append(prev_value)
                
                x = range(len(max_successes))
                ax2.plot(x, max_successes, label=label, color=color, marker='o', markersize=4)

        # Add human baseline as horizontal line
        ax2.axhline(y=human_mean, color='red', linestyle='--', label='Human')
        
        ax2.set_xlabel('Eureka Iteration')
        ax2.set_ylabel('Max Consecutive Successes')
        ax2.set_title('Training Progress')
        ax2.legend()

        plt.suptitle(f'Performance Comparison - {task_name}')
        plt.tight_layout()
        plot_path = os.path.join(PLOTS_DIR, f"{task_name}_comparison.png")
        plt.savefig(plot_path, bbox_inches='tight')
        print(f"Saved comparison plot to {plot_path}")
        plt.close()


def main():
    """Main function to generate all plots"""
    ensure_results_dirs()


    
    # Generate individual training plots
    # task_folders = glob.glob(os.path.join(CHECKPOINTS_DIR, "eureka/*/"))
    # for task_folder in task_folders:
    #     generate_train_eval_plots(task_folder)
    
    # Generate comparison plots
    generate_comparison_plots()


if __name__ == "__main__":
    main()
