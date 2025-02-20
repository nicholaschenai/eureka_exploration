"""
Base class for managing artifacts (videos, checkpoints, etc) from training runs
"""

import os
import glob
from typing import Optional

class ArtifactManager:
    def __init__(self, 
                 output_dir: str,
                 save_metadata: bool = False):
        self.output_dir = output_dir
        self.save_metadata = save_metadata
        os.makedirs(self.output_dir, exist_ok=True)

    def save_source_metadata(self, dest_path: str, source_info: dict):
        """Save metadata about the source of an artifact"""
        if not self.save_metadata:
            return
            
        base, ext = os.path.splitext(dest_path)
        info_file = f"{base}_task_folder.txt"
        
        with open(info_file, 'w') as f:
            for key, value in source_info.items():
                f.write(f"{key}: {value}\n")

    def get_artifact_path(self, results_name: str, iter_num: Optional[int] = None, extension: str = '') -> str:
        """Get the path where an artifact should be saved"""
        filename = (f"{results_name}_iter{iter_num}{extension}" if iter_num is not None 
                   else f"{results_name}_eval{extension}")
        return os.path.join(self.output_dir, filename)
