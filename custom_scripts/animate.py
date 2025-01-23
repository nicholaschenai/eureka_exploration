"""
animates the trained policies in custom_checkpoints


TODO: copy vid from isaacgymenvs/isaacgymenvs/outputs/DATETIME/videos to the subfolder of custom_checkpoints where it came from. the DATETIME is the datetime of the start of animation,
which is not the same as the datetime of the parent folder.
so we need to loop thru the runs, take the most recent video folder and copy it before animating the next policy.
"""

import os
import glob
import shutil
import subprocess
import time
from typing import Optional

from eureka_task_processor import EurekaTaskProcessor

# Define results directory structure
RESULTS_DIR = "results"
VIDEOS_DIR = os.path.join(RESULTS_DIR, "videos")

def ensure_results_dirs():
    """Create results directory structure if it doesn't exist"""
    for d in [RESULTS_DIR, VIDEOS_DIR]:
        os.makedirs(d, exist_ok=True)


class VideoGenerator:
    @staticmethod
    def get_latest_video_folder() -> Optional[str]:
        """Get the most recent video output folder from isaacgym"""
        output_dir = "outputs/train"
        folders = glob.glob(f"{output_dir}/*/")
        if not folders:
            print("Warning: No output folders found in outputs/train/")
            return None
        latest = max(folders, key=os.path.getctime)
        print(f"Found latest output folder: {latest}")
        return latest

    @staticmethod
    def animate_policy(task_name: str, checkpoint_path: str):
        """Animate a policy using isaacgym"""
        cmd = [
            "python", "isaacgymenvs/isaacgymenvs/train.py",
            "test=True", "headless=False",
            f"task={task_name}",
            f"checkpoint={checkpoint_path}",
            "capture_video=True"
        ]
        process = subprocess.Popen(cmd)
        process.wait()

    @staticmethod
    def save_video(results_name: str, iter_num: Optional[int] = None):
        """Save generated video to results folder"""
        latest_folder = VideoGenerator.get_latest_video_folder()
        if latest_folder:
            # Videos are in outputs/train/DATETIME/videos/TASKNAME_DATETIME2/rl-video-step-0.mp4
            videos_base = os.path.join(latest_folder, "videos")
            print(f"Looking for videos in: {videos_base}")
            
            if os.path.exists(videos_base):
                # Get the most recent task video folder
                task_video_folders = glob.glob(os.path.join(videos_base, "*/"))
                if task_video_folders:
                    latest_task_folder = max(task_video_folders, key=os.path.getctime)
                    print(f"Found task video folder: {latest_task_folder}")
                    
                    # Look for mp4 files in this folder
                    video_files = glob.glob(os.path.join(latest_task_folder, "*.mp4"))
                    if video_files:
                        # Name format: taskname_iter{X}.mp4 or taskname_eval.mp4
                        video_name = f"{results_name}_iter{iter_num}.mp4" if iter_num is not None else f"{results_name}_eval.mp4"
                        video_dest = os.path.join(VIDEOS_DIR, video_name)
                        print(f"Copying video from {video_files[0]} to {video_dest}")
                        shutil.copy(video_files[0], video_dest)
                    else:
                        print(f"Warning: No mp4 files found in {latest_task_folder}")
                else:
                    print(f"Warning: No task video folders found in {videos_base}")
            else:
                print(f"Warning: Videos directory not found at {videos_base}")

def generate_videos(task_folder: str):
    """Generate videos for best policies of each iteration"""
    try:
        processor = EurekaTaskProcessor(task_folder)
        print(f"Generating videos for task folder: {task_folder}")
        video_prefix = f"{processor.task_name}{processor.suffix}"
        # Generate videos for training iterations
        for iter_num in range(processor.iteration):
            # Check if video already exists
            video_name = f"{video_prefix}_iter{iter_num}.mp4"
            video_dest = os.path.join(VIDEOS_DIR, video_name)
            if os.path.exists(video_dest):
                print(f"Video for iteration {iter_num} already exists, skipping...")
                continue
            
            iter_policies = processor.get_iteration_policies(iter_num)
            if not iter_policies:
                print(f"Warning: Skipping video generation for iteration {iter_num} - no valid policies")
                continue
                
            policy_folder, _ = processor.get_best_policy(iter_policies)
            if not policy_folder:
                continue
                
            checkpoint = processor.get_checkpoint_path(policy_folder)
            if not checkpoint:
                continue
                
            print(f"Animating iteration {iter_num}")
            VideoGenerator.animate_policy(processor.task_name, checkpoint)
            VideoGenerator.save_video(video_prefix, iter_num)
            time.sleep(1)  # Small delay between animations
        
        # Generate video for best eval run
        eval_video_name = f"{video_prefix}_eval.mp4"
        eval_video_dest = os.path.join(VIDEOS_DIR, eval_video_name)
        if os.path.exists(eval_video_dest):
            print("Evaluation video already exists, skipping...")
            return
            
        eval_policies = processor.get_iteration_policies(None)
        if eval_policies:
            policy_folder, _ = processor.get_best_policy(eval_policies)
            if policy_folder:
                checkpoint = processor.get_checkpoint_path(policy_folder)
                if checkpoint:
                    print("Animating evaluation")
                    VideoGenerator.animate_policy(processor.task_name, checkpoint)
                    VideoGenerator.save_video(video_prefix, None)
                    
    except Exception as e:
        print(f"Error generating videos for {task_folder}: {e}")


def main():
    # Ensure results directories exist
    ensure_results_dirs()
    
    task_folders = glob.glob("custom_checkpoints/eureka/*/")
    
    for task_folder in task_folders:
        generate_videos(task_folder)

if __name__ == "__main__":
    main()
