import subprocess
import time

def wait_for_free_vram(required_gb=8, check_interval=60):
    """
    Block until there's at least required_gb of VRAM free on any GPU.
    
    Args:
        required_gb (float): Required free VRAM in GB
        check_interval (int): How often to check in seconds
        
    Returns:
        int: GPU index with sufficient free memory
    """
    while True:
        try:
            # Query nvidia-smi for memory info
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,nounits,noheader'],
                encoding='utf-8'
            )
            
            # Parse the output
            for line in result.strip().split('\n'):
                gpu_id, free_memory = map(int, line.split(','))
                free_gb = free_memory / 1024.0  # Convert MB to GB
                
                if free_gb >= required_gb:
                    return gpu_id
                    
        except subprocess.CalledProcessError:
            print("Error querying nvidia-smi, will retry...")
            
        time.sleep(check_interval)
