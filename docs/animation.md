# Animation for WSL

First, install ffmpeg:
```bash
conda install ffmpeg -c conda-forge
# if this doesnt work, can use sudo apt-get install ffmpeg but the LD_LIBRARY_PATH messes this up, need to fix
```

Add this to your environment:
```bash
export DISPLAY=:0
```

Add this to `isaacgymenvs/isaacgymenvs/train.py` to handle virtual display initialization (already done for this repo):
```python
os.environ['PYVIRTUALDISPLAY_DISPLAYFD'] = '0'
```

Set the right flags. Using the pen spinning demo as an example (assuming you already cd into `isaacgymenvs/isaacgymenvs`):
```bash
python train.py test=True headless=False task=ShadowHandSpin checkpoint=checkpoints/EurekaPenSpinning.pth capture_video=True
```
- strangely it requires `headless=False` and doesnt show the visualization
- `capture_video=True`
- additional settings can be found in the isaacgymenvs README
- this creates a train output in `outputs/train/DATETIME` where the video is in the `videos` folder as an mp4 file
  - this directory is relative to where you ran the command
