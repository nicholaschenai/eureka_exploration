import hydra
import numpy as np
import logging 
import os
import subprocess
from pathlib import Path
import time 
# import shutil

from utils.misc import * 
from utils.file_utils import load_tensorboard_logs
# from utils.create_task import create_task
from utils.extract_task_code import *

from custom_utils import wait_for_free_vram

EUREKA_ROOT_DIR = os.getcwd()
ISAAC_ROOT_DIR = f"{EUREKA_ROOT_DIR}/../isaacgymenvs/isaacgymenvs"

@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {EUREKA_ROOT_DIR}")

    task = cfg.env.task
    task_description = cfg.env.description
    suffix = ''
    
    logging.info("Task: " + task)
    logging.info("Task description: " + task_description)

    logging.info(f"Evaluating best reward code {cfg.num_eval} times")
    
    eval_runs = []
    for i in range(cfg.num_eval):
        set_freest_gpu()

        wait_for_free_vram(cfg.min_vram)
        
        # Execute the python file with flags
        rl_filepath = f"reward_code_eval{i}.txt"
        with open(rl_filepath, 'w') as f:
            cmd = ['python', '-u', f'{ISAAC_ROOT_DIR}/train.py',  
                    'hydra/output=subprocess',
                    f'task={task}{suffix}', f'wandb_activate={cfg.use_wandb}',
                    f'wandb_entity={cfg.wandb_username}', f'wandb_project={cfg.wandb_project}',
                    f'headless={not cfg.capture_video}', f'capture_video={cfg.capture_video}', 'force_render=False', f'seed={i}']
            # if cfg.num_envs:  # Only add if not empty string
            #     cmd.append(f'num_envs={cfg.num_envs}')
            process = subprocess.Popen(cmd, stdout=f, stderr=f)
        time.sleep(1)
        block_until_training(rl_filepath)
        eval_runs.append(process)

    reward_code_final_successes = []
    reward_code_correlations_final = []
    for i, rl_run in enumerate(eval_runs):
        rl_run.communicate()
        rl_filepath = f"reward_code_eval{i}.txt"
        with open(rl_filepath, 'r') as f:
            stdout_str = f.read() 
        lines = stdout_str.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('Tensorboard Directory:'):
                break 
        tensorboard_logdir = line.split(':')[-1].strip() 
        tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
        max_success = max(tensorboard_logs['consecutive_successes'])
        reward_code_final_successes.append(max_success)

        if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
            gt_reward = np.array(tensorboard_logs["gt_reward"])
            gpt_reward = np.array(tensorboard_logs["gpt_reward"])
            reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
            reward_code_correlations_final.append(reward_correlation)

    logging.info(f"Final Success Mean: {np.mean(reward_code_final_successes)}, Std: {np.std(reward_code_final_successes)}, Raw: {reward_code_final_successes}")
    # logging.info(f"Final Correlation Mean: {np.mean(reward_code_correlations_final)}, Std: {np.std(reward_code_correlations_final)}, Raw: {reward_code_correlations_final}")
    np.savez('final_eval.npz', reward_code_final_successes=reward_code_final_successes, reward_code_correlations_final=reward_code_correlations_final)


if __name__ == "__main__":
    main()