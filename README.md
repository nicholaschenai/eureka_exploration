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

> ⚠️ Important: Consider using native Linux if possible as WSL installation is complex.

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

Examples in IsaacGym:

**Error: segmentation fault**

>  WSL's graphics support, especially for complex OpenGL applications like IsaacGym, can be unstable. The segmentation fault occurs when both the graphics pipeline and viewer are trying to access the GPU through WSL's graphics translation layer.

main prob is the viewer -- comment out everything to do with it

```bash
# Test installation (with modifications):
cd examples  # Must be in examples directory due to relative imports
python joint_monkey.py
```


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

### Important Notes
- Eureka params: Note the default settings are in env/config which is different from README!
  - README is closer to the settings from the research paper
- When running tasks:
  - Use `headless=True` for rendering issues especially in the pen spinning demo
  - Adjust `num_eval` (evaluating the final reward fn) based on available VRAM (this is run in parallel, default 5 evals need ~40GB)
  - Avoid `capture_video` flag in WSL is it appears to always result in error? maybe due to the same display error in WSL as above?
```