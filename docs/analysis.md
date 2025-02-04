# Analysis

This document explains how to analyze the training outputs after using `custom_scripts/copy_sanitize_checkpoints.py` to copy checkpoints from `eureka/outputs/` to `custom_checkpoints/`.

## Directory Structure

```
eureka_artifacts/
    eureka/
        DATE_TIME_FOLDER/           # Task run folder
            .hydra/                 # Configuration files
                config.yaml         # Task parameters (iteration, sample, num_eval)
            policy-DATE-TIME/       # Policy folders, multiple per task run
                runs/              
                    IDENTIFIER/    
                        nn/        
                            *.pth  # Model weights
                        summaries/ 
                            events.* # Tensorboard logs
```

### Policy Folder Organization
Each task run contains multiple policy folders based on:
- Training phase: `iteration × sample` folders
- Evaluation phase: `num_eval` folders

For example, with settings:
- iterations = 5
- samples = 2
- num_eval = 2

You'll see:
- 10 training folders (5 iterations × 2 samples)
- 2 evaluation folders
- Total: 12 policy folders

## Key Metrics

### Training Metrics
- `consecutive_successes`: Primary performance metric tracked over time
- For each iteration:
  1. Track max consecutive successes for each run
  2. Select best policy based on highest max consecutive successes

### Evaluation Metrics
- `gt_reward`: Ground truth reward using human reward function
  - Note: This is different from the human baseline mentioned in the paper

## Visualization Tools
These are saved in the `results` folder
### Success Plot
```bash
python custom_scripts/plot_successes.py
```
Generates two plots:
1. Max consecutive successes of best policy per iteration
2. Consecutive successes during evaluation vs training steps

### Policy Animation
```bash
python custom_scripts/animate.py
```
Creates visualizations of trained policies in action.

