"""
animates the trained policies in eureka_artifacts for both eureka and human baseline experiments
"""

import os
import glob
from typing import Optional

from custom_utils import prepare_gen
from custom_utils.video_generator import VideoGenerator
from custom_utils.eureka_task_processor import EurekaTaskProcessor

# Define results directory structure
# RESULTS_DIR = "results"
# CHECKPOINTS_DIR = "custom_checkpoints"
CHECKPOINTS_DIR = "eureka_artifacts"


def generate_policy_video(processor: EurekaTaskProcessor, video_prefix: str, video_generator: VideoGenerator, iter_num: Optional[int] = None) -> None:
    """Generate video for a specific policy iteration or eval run
    
    Args:
        processor: The task processor instance
        video_prefix: Base name for the video
        video_generator: Video generator instance to use
        iter_num: If provided, generates video for that iteration. If None, generates eval video.
    """
    video_generator.process_policy(processor, video_prefix, iter_num)

def generate_iteration_videos(processor: EurekaTaskProcessor, video_prefix: str, video_generator: VideoGenerator) -> None:
    """Generate videos for all training iterations"""
    for iter_num in range(processor.iteration):
        generate_policy_video(processor, video_prefix, video_generator, iter_num)

def generate_eval_video(processor: EurekaTaskProcessor, video_prefix: str, video_generator: VideoGenerator) -> None:
    """Generate video for best evaluation run"""
    generate_policy_video(processor, video_prefix, video_generator, None)

def generate_videos(task_folder: str):
    """Generate videos for best policies of each iteration"""
    try:
        print(f"Generating videos for task folder: {task_folder}")
        processor, video_prefix = prepare_gen(task_folder)
        
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
