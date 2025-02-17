"""
animates the trained policies in eureka_artifacts for both eureka and human baseline experiments
"""

import os
import glob
import shutil
import subprocess
import time
from typing import Optional, Tuple

from .eureka_task_processor import EurekaTaskProcessor

# Define results directory structure
# RESULTS_DIR = "results"
# CHECKPOINTS_DIR = "custom_checkpoints"
CHECKPOINTS_DIR = "eureka_artifacts"
RESULTS_DIR = CHECKPOINTS_DIR
VIDEOS_DIR = os.path.join(RESULTS_DIR, "videos")


class VideoGenerator:
    def __init__(self, output_dir: Optional[str] = None, num_envs: Optional[int] = None,
                 virtual_screen_capture: bool = True, force_render: bool = False,
                 capture_video_freq: Optional[int] = None, capture_video_len: Optional[int] = None,
                 save_task_folder: bool = False):
        self.output_dir = output_dir or VIDEOS_DIR
        self.num_envs = num_envs
        self.virtual_screen_capture = virtual_screen_capture
        self.force_render = force_render
        self.capture_video_freq = capture_video_freq
        self.capture_video_len = capture_video_len
        self.save_task_folder = save_task_folder

        os.makedirs(self.output_dir, exist_ok=True)

    def get_latest_video_folder(self) -> Optional[str]:
        """Get the most recent video output folder from isaacgym"""
        output_dir = "outputs/train"
        folders = glob.glob(f"{output_dir}/*/")
        if not folders:
            print("Warning: No output folders found in outputs/train/")
            return None
        latest = max(folders, key=os.path.getctime)
        print(f"Found latest output folder: {latest}")
        return latest

    def get_video_path(self, results_name: str, iter_num: Optional[int] = None) -> str:
        """Get the path where a video should be saved"""
        video_name = f"{results_name}_iter{iter_num}.mp4" if iter_num is not None else f"{results_name}_eval.mp4"
        return os.path.join(self.output_dir, video_name)

    def animate_policy(self, task_name: str, checkpoint_path: str):
        """Animate a policy using isaacgym"""
        cmd = [
            "python", "isaacgymenvs/isaacgymenvs/train.py",
            "test=True", "headless=False",
            f"task={task_name}",
            f"checkpoint={checkpoint_path}",
            "capture_video=True"
        ]
        if self.num_envs:
            cmd.append(f"num_envs={self.num_envs}")
        # if self.virtual_screen_capture:
        #     cmd.append("virtual_screen_capture=True")
        if self.force_render:
            cmd.append("force_render=True")
        if self.capture_video_freq:
            cmd.append(f"capture_video_freq={self.capture_video_freq}")
        if self.capture_video_len:
            cmd.append(f"capture_video_len={self.capture_video_len}")
            
        process = subprocess.Popen(cmd)
        process.wait()

    def save_video(self, results_name: str, iter_num: Optional[int] = None):
        """Save generated video to results folder"""
        latest_folder = self.get_latest_video_folder()
        if not latest_folder:
            return
            
        # Videos are in outputs/train/DATETIME/videos/TASKNAME_DATETIME2/rl-video-step-0.mp4
        videos_base = os.path.join(latest_folder, "videos")
        print(f"Looking for videos in: {videos_base}")
        
        if not os.path.exists(videos_base):
            print(f"Warning: Videos directory not found at {videos_base}")
            return
            
        # Get the most recent task video folder
        task_video_folders = glob.glob(os.path.join(videos_base, "*/"))
        if not task_video_folders:
            print(f"Warning: No task video folders found in {videos_base}")
            return
            
        latest_task_folder = max(task_video_folders, key=os.path.getctime)
        print(f"Found task video folder: {latest_task_folder}")
        
        # Look for mp4 files in this folder
        video_files = glob.glob(os.path.join(latest_task_folder, "*.mp4"))
        if not video_files:
            print(f"Warning: No mp4 files found in {latest_task_folder}")
            return
            
        video_dest = self.get_video_path(results_name, iter_num)
        print(f"Copying video from {video_files[0]} to {video_dest}")
        shutil.copy(video_files[0], video_dest)

        # Save task folder info if enabled
        if self.save_task_folder:
            info_file = video_dest.replace('.mp4', '_task_folder.txt')
            with open(info_file, 'w') as f:
                f.write(f"Task video folder: {latest_task_folder}\n")
                f.write(f"Original video: {video_files[0]}\n")

def generate_policy_video(processor: EurekaTaskProcessor, video_prefix: str, video_generator: VideoGenerator, iter_num: Optional[int] = None) -> None:
    """Generate video for a specific policy iteration or eval run
    
    Args:
        processor: The task processor instance
        video_prefix: Base name for the video
        video_generator: Video generator instance to use
        iter_num: If provided, generates video for that iteration. If None, generates eval video.
    """
    video_dest = video_generator.get_video_path(video_prefix, iter_num)
    stage = f"iteration {iter_num}" if iter_num is not None else "evaluation"
    
    if os.path.exists(video_dest):
        print(f"Video for {stage} already exists, skipping...")
        return
        
    policies = processor.get_iteration_policies(iter_num)
    if not policies:
        if iter_num is not None:  # Only print warning for iterations, not eval
            print(f"Warning: Skipping video generation for iteration {iter_num} - no valid policies")
        return
        
    policy_folder, _ = processor.get_best_policy(policies)
    if not policy_folder:
        return
        
    checkpoint = processor.get_checkpoint_path(policy_folder)
    if not checkpoint:
        return
        
    print(f"Animating {stage}")
    video_generator.animate_policy(processor.task_name, checkpoint)
    video_generator.save_video(video_prefix, iter_num)
    if iter_num is not None:
        time.sleep(1)  # Small delay between animations for iterations only

def generate_iteration_videos(processor: EurekaTaskProcessor, video_prefix: str, video_generator: VideoGenerator) -> None:
    """Generate videos for all training iterations"""
    for iter_num in range(processor.iteration):
        generate_policy_video(processor, video_prefix, video_generator, iter_num)

def generate_eval_video(processor: EurekaTaskProcessor, video_prefix: str, video_generator: VideoGenerator) -> None:
    """Generate video for best evaluation run"""
    generate_policy_video(processor, video_prefix, video_generator, None)

def prepare_vid_gen(task_folder: str) -> Tuple[EurekaTaskProcessor, str]:
    processor = EurekaTaskProcessor(task_folder)
    video_prefix = f"{processor.task_name}{processor.suffix}"
    # suffix implies GPT/Eureka, no suffix is human baseline
    if processor.suffix:
        video_prefix += f"_epoch_{processor.max_iterations}"
    return processor, video_prefix

def generate_videos(task_folder: str):
    """Generate videos for best policies of each iteration"""
    try:
        print(f"Generating videos for task folder: {task_folder}")
        processor, video_prefix = prepare_vid_gen(task_folder)
        
        # Create a single video generator instance to be used throughout
        video_generator = VideoGenerator()
        
        if processor.suffix:
            # Generate videos for training iterations
            generate_iteration_videos(processor, video_prefix, video_generator)
        
        # Generate video for best eval run
        generate_eval_video(processor, video_prefix, video_generator)
                    
    except Exception as e:
        print(f"Error generating videos for {task_folder}: {e}")


def main():
    # Process both eureka and human baseline folders
    for exp_type in ['eureka', 'human_baseline']:
        task_folders = glob.glob(os.path.join(CHECKPOINTS_DIR, exp_type, "*/"))
        for task_folder in task_folders:
            generate_videos(task_folder)

if __name__ == "__main__":
    main()
