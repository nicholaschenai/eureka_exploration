"""
Copies files and folders from the eureka/outputs folder into the custom_checkpoints folder with sanitized data:
- for tensorboard files (events.out.tfevents.TIMESTAMP.DESKTOP_NAME), the desktop name is sanitized
- for .log files, these are sanitized: 
    - desktop name in mentions of the tensorboard files
    - HTTP requests
"""

import os
import shutil
import re
from pathlib import Path
import argparse

# Constants
SOURCE_DIR = Path("eureka/outputs")
# DEST_DIR = Path("custom_checkpoints")
DEST_DIR = Path("eureka_artifacts")
DESKTOP_PATTERN = r'DESKTOP-[A-Z0-9]{7}'
SANITIZED_NAME = 'SERVER'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_folder", default="", help="Optional: Folder path relative to SOURCE_DIR (e.g., 'eureka/2024-02-13')")
    return parser.parse_args()

def sanitize_filename(filename: str) -> str:
    """Sanitize tensorboard filename by replacing desktop name."""
    if 'tfevents' in filename:
        return re.sub(DESKTOP_PATTERN, SANITIZED_NAME, filename)
    return filename

def sanitize_log_content(content: str) -> str:
    """Sanitize log file content."""
    # Sanitize desktop names in tensorboard file mentions
    content = re.sub(DESKTOP_PATTERN, SANITIZED_NAME, content)
    # Sanitize HTTP requests
    content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[REDACTED_URL]', content)
    return content

def copy_and_sanitize(args):
    """Main function to copy and sanitize files."""
    # If no target folder specified (empty string), use original behavior
    root_path = SOURCE_DIR
    if args.target_folder:
        root_path = SOURCE_DIR / args.target_folder
    if not root_path.exists():
        raise FileNotFoundError(f"Target directory {root_path} not found")
    
    DEST_DIR.mkdir(exist_ok=True)
    
    for root, _, files in os.walk(root_path):
        # Skip if the path is SOURCE_DIR/old
        if Path(root).relative_to(SOURCE_DIR).parts[:1] == ('old',):
            continue
            
        rel_path = Path(root).relative_to(SOURCE_DIR)
        dest_root = DEST_DIR / rel_path
        dest_root.mkdir(exist_ok=True)
        
        for file in files:
            source_file = Path(root) / file
            dest_file = dest_root / sanitize_filename(file)
            print(f"Copying {source_file} to {dest_file}")

            if file.endswith('.log'):
                # Sanitize log files
                with open(source_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                with open(dest_file, 'w', encoding='utf-8') as f:
                    f.write(sanitize_log_content(content))
            else:
                # Copy other files directly
                shutil.copy2(source_file, dest_file)

if __name__ == "__main__":
    args = parse_args()
    try:
        copy_and_sanitize(args)
        print("Files copied and sanitized successfully")
    except Exception as e:
        print(f"Error: {e}")
