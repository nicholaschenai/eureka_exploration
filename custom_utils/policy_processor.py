"""
Base class for processing policies and handling common workflow
"""

import os
from typing import Optional, Tuple

from .eureka_task_processor import EurekaTaskProcessor

class PolicyProcessor:
    @staticmethod
    def get_best_policy_checkpoint(processor: EurekaTaskProcessor, 
                                 prefix: str, 
                                 iter_num: Optional[int] = None,
                                 artifact_path: Optional[str] = None) -> Tuple[Optional[str], str, bool]:
        """
        Get the best policy checkpoint for a given iteration or eval run
        
        Returns:
            Tuple of (checkpoint_path, stage_description, should_skip)
        """
        stage = f"iteration {iter_num}" if iter_num is not None else "evaluation"
        
        # Skip if artifact already exists (video or checkpoint)
        if artifact_path and os.path.exists(artifact_path):
            print(f"Artifact for {stage} already exists, skipping...")
            return None, stage, True
            
        policies = processor.get_iteration_policies(iter_num)
        if not policies:
            if iter_num is not None:
                print(f"Warning: Skipping {stage} - no valid policies")
            return None, stage, True
            
        policy_folder, _ = processor.get_best_policy(policies)
        if not policy_folder:
            return None, stage, True
            
        checkpoint = processor.get_checkpoint_path(policy_folder)
        if not checkpoint:
            return None, stage, True
            
        return checkpoint, stage, False
