# Quick exploration of Eureka

## Introduction
This is a quick exploration of [Eureka](https://github.com/eureka-research/Eureka) with some modifications:
- Uses Azure OpenAI
- Added [instructions for WSL](#instructions-for-wsl)
- Added human baseline evaluation script `human_baseline.py` in `eureka` folder (modified from `eureka.py` to use the regular env from isaacgymenvs)
- Ran 3 examples and saved the outputs for reproducibility (see below)
  - Difference with original model is `gpt-4o-2024-08-06` instead of `gpt-4-0314`
  - use `custom_scripts/copy_sanitize_checkpoints.py` to copy outputs to the `eureka_artifacts` folder
- Various human baseline runs in `eureka_artifacts/human_baseline`
- Added VRAM management: `custom_utils.py` includes a `wait_for_free_vram` utility that blocks process execution until sufficient GPU memory (default 8GB) is available (Original Eureka spawns a lot of processes and requires ~128GB VRAM at a time)
- Bugfix: rename bidex folder to dexterity as script expects that

The outputs (logs, checkpoints, etc) are saved as a submodule to keep this repo light.
To only clone this repo, run
```bash
git clone https://github.com/nicholaschenai/eureka_exploration.git
```

To clone and include the submodule, run this instead
```bash
git clone --recurse-submodules https://github.com/nicholaschenai/eureka_exploration.git
```

## Documentation
- [Animation Guide](docs/animation.md) - Instructions for creating and capturing videos of trained policies
- [Analysis Guide](docs/analysis.md) - Understanding metrics, directory structure, and how to plot results
- [Inferring details guide](docs/infer_details.md) - Attempts at inferring details which are not clear from the paper
- [Experiment Runs](https://github.com/nicholaschenai/eureka_artifacts/blob/main/experiment_runs.md) - Details on the experiments runs
- [Results](https://github.com/nicholaschenai/eureka_artifacts) - Detailed analysis of training results, performance plots, and videos


## Instructions for WSL

> ⚠️ Important: Consider using native Linux if possible (and follow the original instructions in the Eureka repo) as WSL installation is complex.

### Installation

1. Create conda environment:
```bash
conda create -n eureka python=3.8
conda activate eureka
```

2. Install IsaacGym (Preview Release 4/4):
   - Download from [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym)
   - Extract and install:

```bash
tar -xvf IsaacGym_Preview_4_Package.tar.gz
cd isaacgym/python

# Fix numpy float deprecation
# Edit isaacgym/python/isaacgym/torch_utils.py - replace np.float with np.float64 at line 135

pip install -e .
```  

3. Fix common WSL-specific issues:

   a. Missing Python library:
    ```bash
    export LD_LIBRARY_PATH=/PATH/TO/ANACONDA/envs/eureka/lib:$LD_LIBRARY_PATH
    ```

    without this, you might get the error

    ```console
    # ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory
    ```


    b. CUDA library symlinks:
    ```bash
    cd /usr/lib/wsl/lib
    sudo rm libcuda.so libcuda.so.1
    sudo ln -s libcuda.so.1.1 libcuda.so.1
    sudo ln -s libcuda.so.1 libcuda.so
    sudo ldconfig
    ```

    without this, you might get the error

    ```console
    /buildAgent/work/.../source/physx/src/gpu/PxPhysXGpuModuleLoader.cpp (148) : internal error : libcuda.so!
    ```
    due to `libcuda.so` for windows host stubbed inside WSL2

   c. GLIBCXX version fix:
   - Copy `/usr/lib/x86_64-linux-gnu/libstdc++.so.6` to your conda environment's lib folder
   This is because the LD library path's `libstdc++.so.6` GLIBCXX does not contain ver GLIBCXX_3.4.32. you can check this via 
    ```bash
    strings PATH/TO/LIBSTDC | grep GLIBCXX
    ```

  d. (Optional, for visualization) Install Vulkan via mesa:
  To properly run any visualization, you need to get Vulkan in WSL which is tricky as it is not officially supported. There's a workaround by downloading an experimental version of mesa from ppa:kisak/kisak-mesa

  ```bash
  sudo add-apt-repository ppa:kisak/kisak-mesa
  sudo apt update
  sudo apt upgrade
  ```

Examples in IsaacGym:

```bash
# Test installation (with modifications):
cd examples  # Must be in examples directory due to relative imports
python joint_monkey.py
```

If didn't install Vulkan via mesa, the code works fine if you comment out anything viewer related, else you'll get

**Error: segmentation fault**

>  WSL's graphics support, especially for complex OpenGL applications like IsaacGym, can be unstable. The segmentation fault occurs when both the graphics pipeline and viewer are trying to access the GPU through WSL's graphics translation layer.

4. Install Eureka:

```bash
git clone https://github.com/eureka-research/Eureka.git
cd Eureka

# Comment out the numpy requirements in eureka cos need numpy 1.21 n above (package used 1.20) for the NDARRAY type which is used during Eureka training

pip install -e .

cd isaacgymenvs
pip install -e .

cd ../rl_games
pip install -e .

# Install additional dependency for Eureka training
pip install gpustat
```

## Getting started

Configure Azure OpenAI API:
```bash
export AZURE_OPENAI_API_KEY="YOUR_API_KEY"
export AZURE_OPENAI_ENDPOINT="YOUR_AZURE_ENDPOINT"
```

Then see the original README for the rest of the instructions

### Caveat on Eureka Parameters
The default settings in `env/config` differ from those mentioned in the original README (the latter reflects the settings used in the research paper). This repo has adjusted `env/config` to match the paper's settings. Also, see [Inferring details guide](docs/infer_details.md) for more details.

#### Parallel Processing and VRAM Requirements
Eureka runs multiple processes in parallel during:
- Training: `sample` (K) parallel environments
- Evaluation: `num_eval` parallel environments

Each process requires approximately 8GB of VRAM. To manage VRAM usage, you have several options:

1. **VRAM Management (Recommended)**
   - We've added automatic VRAM management that blocks execution until sufficient memory is available
   - Default threshold is 8GB per process
   - Adjust `min_vram` in the config to change the required free VRAM threshold

2. **Environment Count**
   - You can adjust `num_envs` in the config, which controls the number of environments in isaacgymenvs
   - Note: Reducing this may impact success rates

3. **Sample Size (Not Recommended)**
   - While reducing `sample` (K) would decrease VRAM usage, it significantly impacts performance
   - We notice that the first iteration's reward functions are only executable ~25-50% of the time, so reducing K severely limits the variety of reward functions
   - Our early experiments with K=2 (reduced from K=16) showed:
     - Less smooth training curves / higher variance
     - Sometimes unrecoverable performance issues where success rates remain near zero even after many iterations (e.g. AllegroHand, Humanoid)
     - Potential negative feedback loop where poor reward functions in the prompt confuse the language model (though it can improve reward functions that are already decent)

## Eureka Pen Spinning Demo in WSL

- Remove the headless arg (i.e. `headless=True`) if Vulkan via mesa is not installed
- Somehow for `headless=False`, we need `force_render=True`


## TODO
- see https://github.com/isaac-sim/IsaacLabEureka for newer implementation but task config doesnt seem done
- Sparse baseline: note that in normal eureka, 
  - the env from eureka is used, 
  - modified to set rewards from the reward fn
  - then the compute_bonus function adds a bonus to the reward if the goal is reached (which implements the sparse reward), and this env file is output in isaacgymenvs
  - so can do something to set reward fn to zero so we only get the compute bonus?
- Observations: since only most successful code is fed back to LM (which by definition is executable), it doesnt correct its mistakes from the runs that failed especially due to the code being non-executable

## Other resources
- [Eureka Research Paper](https://arxiv.org/abs/2310.12931)
- My [summary](https://github.com/nicholaschenai/agi-potential-notes/blob/main/papers/eureka.md) (WIP)