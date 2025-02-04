# Experiment Runs

## Eureka Experiments
When `max_iterations` is not specified, it is set to the environment's default.
This is a training-only hyperparameter;
During evaluation (Final Success), the environment's default is used.
As such, the max training success can only be compared with the final success if `max_iterations` is the same as the environment's default.

The number after the $\pm$ is the standard deviation


| timestamp | `env_name` | `max_iterations` | Max Training Success | Final Success Mean | Final Correlation Mean |
|-----------|-----------|----------------|-------------------|----------------|--------------|
| 2025-02-02_14-49-34 | franka_cabinet | - | 0.37 | 0.05 ± 0.06 | 0.60 ± 0.19 |
| 2025-02-01_18-54-34 | humanoid | - | 6.45 | 5.30 ± 0.65 | 1.00 ± 0.00 |
| 2025-01-31_13-55-23 | ant | - | 8.61 | **8.21** ± 0.47 | 0.92 ± 0.01 |
| 2025-01-30_14-32-06 | ant | 3000 | 11.08 | 6.68 ± 0.26 | 0.97 ± 0.00 |
| 2025-01-27_20-24-22 | franka_cabinet | 3000 | 0.99 | **0.09** ± 0.14 | 0.77 ± 0.28 |
| 2025-01-25_04-08-48 | humanoid | 3000 | 8.52 | 5.60 ± 0.71 | 0.98 ± 0.01 |


## Human Baseline Experiments
Human baselines use the environment's default `max_iterations`.

| timestamp | `env_name` | Final Success Mean |
|-----------|-----------|----------------|
| 2025-01-31_23-25-03 | allegro_hand | 13.11 ± 1.39 |
| 2025-01-30_10-48-46 | ant | 6.93 ± 0.50 |
| 2025-01-27_01-22-01 | franka_cabinet | 0.05 ± 0.07 |
| 2025-01-27_00-07-02 | humanoid | **6.31** ± 0.55 |


---

# Old Experiment Runs
Might have appeared in previous versions of the repo but deprecate as they deviate from the research paper's settings.

(Originally due to GPU limitations)

## Old Eureka Experiments

| timestamp | `sample` | `num_eval` | `env_name` | `max_iterations` |
|-----------|---------|-----------|-----------|----------------|
| 2025-01-19_21-49-31 | 2 | 2 | allegro_hand | 3000 |
| 2025-01-18_19-07-32 | 2 | 2 | humanoid | 3000 |
| 2025-01-17_21-43-31 | 2 | 2 | shadow_hand | 3000 |

## Old Human Baseline Experiments

| timestamp | `num_eval` | `env_name` |
|-----------|-----------|-----------|
| 2025-01-23_18-21-42 | 2 | shadow_hand |
| 2025-01-24_11-23-55 | 2 | humanoid |