import os
import yaml
import glob
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def get_task_name_from_path(task_folder: str) -> str:
    """Extract a clean task name from the folder path for naming results"""
    return os.path.basename(os.path.normpath(task_folder))


def load_tensorboard_logs_with_steps(path):
    """Load tensorboard logs with both values and step numbers"""
    data = defaultdict(list)
    steps = defaultdict(list)
    event_acc = EventAccumulator(path)
    event_acc.Reload()  # Load all data written so far

    for tag in event_acc.Tags()["scalars"]:
        events = event_acc.Scalars(tag)
        for event in events:
            data[tag].append(event.value)
            steps[tag].append(event.step)
    
    return data, steps


class EurekaTaskProcessor:
    def __init__(self, task_folder: str):
        self.task_folder = task_folder
        try:
            self.config = self._load_config()
            self.iteration = self.config['iteration']
            self.sample = self.config['sample']
            self.num_eval = self.config['num_eval']
            self.task_name = self.config['env']['task']
            self.suffix = self.config.get('suffix', '')
            self.results_name = get_task_name_from_path(task_folder)
        except FileNotFoundError:
            print(f"Error: Config file not found in {task_folder}")
            raise
        except KeyError as e:
            print(f"Error: Missing required config key: {e}")
            raise

    def _load_config(self) -> Dict:
        """Load task configuration from hydra config.yaml"""
        config_path = os.path.join(self.task_folder, ".hydra/config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def get_iteration_policies(self, iter_num: Optional[int] = None) -> List[Tuple[float, str, Dict]]:
        """
        Get all policies for a given iteration, or evaluation policies if iter_num is None
        
        Args:
            iter_num: If provided, get policies for this iteration number.
                     If None, get evaluation policies.
        
        For example, with 5 iterations, 2 samples, 2 evals:
        - Iteration 0: folders 0-1
        - Iteration 1: folders 2-3
        ...
        - Iteration 4: folders 8-9
        - Evaluation: folders 10-11
        """

        try:
            # Get all policy folders and sort them chronologically
            policy_folders = sorted(glob.glob(os.path.join(self.task_folder, f"policy-*/")))
            
            if not policy_folders:
                print(f"Warning: No policy folders found in {self.task_folder}")
                return []
                
            # Calculate folder indices for this iteration
            total_training_folders = self.iteration * self.sample
            
            if iter_num is not None:
                # Get training folders for specific iteration
                start_idx = iter_num * self.sample
                end_idx = start_idx + self.sample
                relevant_folders = policy_folders[start_idx:end_idx]
                if not relevant_folders:
                    print(f"Warning: No policy folders found for iteration {iter_num}")
            else:
                # Get evaluation folders (after all training folders)
                relevant_folders = policy_folders[total_training_folders:total_training_folders + self.num_eval]
                if not relevant_folders:
                    print("Warning: No evaluation policy folders found")
                
            iter_policies = []
            for policy_folder in relevant_folders:
                try:
                    summary_path = os.path.join(policy_folder, "runs")

                    try:
                        # Get first subfolder in runs (the identifier)
                        identifier = next(os.walk(summary_path))[1][0]
                    except (StopIteration, IndexError):
                        print(f"Warning: No run identifier found in {summary_path}")
                        continue
                        
                    summary_path = os.path.join(summary_path, identifier, "summaries")
                    
                    # Load tensorboard logs
                    try:
                        logs, steps = load_tensorboard_logs_with_steps(summary_path)
                        if "consecutive_successes" not in logs:
                            print(f"Warning: No consecutive_successes found in {summary_path}")
                            continue
                            
                        max_consecutive_successes = max(logs["consecutive_successes"])
                        # Include both values and steps in the logs dictionary
                        combined_logs = {
                            key: {"values": values, "steps": steps[key]} 
                            for key, values in logs.items()
                        }
                        iter_policies.append((max_consecutive_successes, policy_folder, combined_logs))
                    except Exception as e:
                        print(f"Warning: Error loading tensorboard logs from {summary_path}: {e}")
                        continue
                        
                except Exception as e:
                    print(f"Warning: Error processing policy folder {policy_folder}: {e}")
                    continue
            
            if not iter_policies:
                phase = "evaluation" if iter_num is None else f"iteration {iter_num}"
                print(f"Warning: No valid policies found for {phase}")
            
            return iter_policies
            
        except Exception as e:
            print(f"Error in get_iteration_policies: {e}")
            return []

    def get_best_policy(self, iter_policies: List[Tuple[float, str, Dict]]) -> Tuple[str, Dict]:
        """Get policy with highest max consecutive successes"""
        if not iter_policies:
            return None, None
        best_policy = max(iter_policies, key=lambda x: x[0])
        return best_policy[1], best_policy[2]  # return folder and logs

    def get_checkpoint_path(self, policy_folder: str) -> Optional[str]:
        """Get path to policy checkpoint file"""
        try:
            nn_path = os.path.join(policy_folder, "runs", next(os.walk(os.path.join(policy_folder, "runs")))[1][0], "nn")
            checkpoint = os.path.join(nn_path, f"{self.task_name}{self.suffix}.pth")
            if not os.path.exists(checkpoint):
                print(f"Warning: Checkpoint not found: {checkpoint}")
                return None
            return checkpoint
        except Exception as e:
            print(f"Warning: Error getting checkpoint path: {e}")
            return None
