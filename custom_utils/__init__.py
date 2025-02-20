from typing import Tuple

from custom_utils.eureka_task_processor import EurekaTaskProcessor


def prepare_gen(task_folder: str) -> Tuple[EurekaTaskProcessor, str]:
    processor = EurekaTaskProcessor(task_folder)
    video_prefix = f"{processor.task_name}{processor.suffix}"
    # suffix implies GPT/Eureka, no suffix is human baseline
    if processor.suffix:
        video_prefix += f"_epoch_{processor.max_iterations}"
    return processor, video_prefix