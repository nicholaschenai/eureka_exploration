import glob
import os
import shutil
import subprocess
import time

from typing import Optional, Tuple

from .eureka_task_processor import EurekaTaskProcessor
from .artifact_manager import ArtifactManager
from .policy_processor import PolicyProcessor

VIDEOS_DIR = os.path.join("eureka_artifacts", "videos")


class VideoGenerator(ArtifactManager):
    def __init__(self, output_dir: Optional[str] = None, num_envs: Optional[int] = None,
                 virtual_screen_capture: bool = True, force_render: bool = False,
                 capture_video_freq: Optional[int] = None, capture_video_len: Optional[int] = None,
                 save_task_folder: bool = False):
        super().__init__(output_dir or VIDEOS_DIR, save_metadata=save_task_folder)
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
        return self.get_artifact_path(results_name, iter_num, '.mp4')

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

        # Save metadata about source
        source_info = {
            "Task video folder": latest_task_folder,
            "Original video": video_files[0]
        }
        self.save_source_metadata(video_dest, source_info)

    def process_policy(self, processor: EurekaTaskProcessor, video_prefix: str, iter_num: Optional[int] = None) -> None:
        """Process a policy by generating and saving its video"""
        video_dest = self.get_video_path(video_prefix, iter_num)
        checkpoint, stage, should_skip = PolicyProcessor.get_best_policy_checkpoint(
            processor, video_prefix, iter_num, video_dest)

        if should_skip or not checkpoint:
            return

        print(f"Animating {stage}")
        self.animate_policy(processor.task_name, checkpoint)
        self.save_video(video_prefix, iter_num)
        if iter_num is not None:
            time.sleep(1)  # Small delay between animations for iterations only
