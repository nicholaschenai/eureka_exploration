# Quick exploration of Eureka

## Introduction
This is a quick exploration of [Eureka](https://github.com/eureka-research/Eureka) with some modifications:
- Uses Azure OpenAI
- Added instructions for WSL
- Ran 3 examples and saved the outputs including logs and checkpoints in `custom_checkpoints` folder
  - Differences with original: 
    - Model is `gpt-4o-2024-08-06` instead of `gpt-4-0314`
    - `sample=2` instead of 16 due to GPU limitations (16GB VRAM)
    - `num_eval=2` instead of 5 due to GPU limitations (16GB VRAM)
  - use `eureka/utils/copy_sanitize_checkpoints.py`


## Installation Instructions for WSL

> ⚠️ Important: Consider using native Linux if possible (and follow the original instructions in the Eureka repo) as WSL installation is complex.

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

5. Configure Azure OpenAI API:
```bash
export AZURE_OPENAI_API_KEY="YOUR_API_KEY"
export AZURE_OPENAI_ENDPOINT="YOUR_AZURE_ENDPOINT"
```

### Caveat on Eureka params
- Note the default settings are in env/config which is different from the original README (latter is the right settings for the research paper)
- Adjust `num_eval` (evaluating the final reward fn) based on available VRAM (this is run in parallel, default 5 evals need ~40GB)

### Visualization via isaacgymenvs
- Pen spinning demo: Remove the headless arg (i.e. `headless=True`) if Vulkan via mesa is not installed
- Somehow for `headless=False`, we need `force_render=True`

### Animation
First, install ffmpeg:
```bash
conda install ffmpeg -c conda-forge
# if this doesnt work, can use sudo apt-get install ffmpeg but the LD_LIBRARY_PATH messes this up, need to fix
```

Add this to your environment:
```bash
export DISPLAY=:0
```

Add this to `isaacgymenvs/isaacgymenvs/train.py` to handle virtual display initialization:
```python
os.environ['PYVIRTUALDISPLAY_DISPLAYFD'] = '0'
```

Set the right flags. Using the pen spinning demo as an example:
```bash
python train.py test=True headless=False task=ShadowHandSpin checkpoint=checkpoints/EurekaPenSpinning.pth capture_video=True
```
- strangely it requires `headless=False` and doesnt show the visualization
- `capture_video=True`
- additional settings can be found in the isaacgymenvs README
- this creates a train output in `isaacgymenvs/isaacgymenvs/outputs/DATETIME` where the video is in the `videos` folder as an mp4 file

## TODO
- Save plots from tensorboard to show that despite reduced sampling, Eureka generally produces better reward functions over time which beats human designed reward functions
- Animate policies

## Other resources
- [Eureka Research Paper](https://arxiv.org/abs/2310.12931)
- My [summary](https://github.com/nicholaschenai/agi-potential-notes/blob/main/papers/eureka.md) (WIP)