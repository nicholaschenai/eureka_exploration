"""
Manages checkpoint copying and organization
"""

import os
import shutil
from typing import Optional

from .artifact_manager import ArtifactManager
from .policy_processor import PolicyProcessor
from .eureka_task_processor import EurekaTaskProcessor

class CheckpointManager(ArtifactManager):
    def __init__(self, output_dir: str, save_metadata: bool = False):
        super().__init__(output_dir, save_metadata)

    def get_checkpoint_path(self, results_name: str, iter_num: Optional[int] = None) -> str:
        """Get the path where a checkpoint should be saved"""
        return self.get_artifact_path(results_name, iter_num, '.pth')

    def process_policy(self, processor: EurekaTaskProcessor, checkpoint_prefix: str, iter_num: Optional[int] = None):
        """Process a policy by copying its checkpoint"""
        checkpoint_dest = self.get_checkpoint_path(checkpoint_prefix, iter_num)
        checkpoint, stage, should_skip = PolicyProcessor.get_best_policy_checkpoint(
            processor, checkpoint_prefix, iter_num, checkpoint_dest)
            
        if should_skip or not checkpoint:
            return
            
        print(f"Copying checkpoint for {stage}")
        self.copy_checkpoint(checkpoint, checkpoint_prefix, iter_num)

    def copy_checkpoint(self, source_path: str, results_name: str, iter_num: Optional[int] = None):
        """Copy checkpoint to results folder with standardized naming"""
        if not os.path.exists(source_path):
            print(f"Warning: Source checkpoint not found: {source_path}")
            return

        checkpoint_dest = self.get_checkpoint_path(results_name, iter_num)
        print(f"Copying checkpoint from {source_path} to {checkpoint_dest}")
        shutil.copy(source_path, checkpoint_dest)

        # Save metadata about source
        source_info = {
            "Original checkpoint": source_path,
            "Policy folder": os.path.dirname(os.path.dirname(source_path))
        }
        self.save_source_metadata(checkpoint_dest, source_info)
