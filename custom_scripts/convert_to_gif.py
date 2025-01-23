import os
from pathlib import Path
from moviepy.editor import VideoFileClip

def convert_videos_to_gifs():
    # Define paths
    videos_dir = Path("results/videos")
    gifs_dir = Path("results/gifs")
    
    # Create gifs directory if it doesn't exist
    gifs_dir.mkdir(exist_ok=True)
    
    # Find all MP4 files
    mp4_files = list(videos_dir.glob("*.mp4"))
    
    print(f"Found {len(mp4_files)} MP4 files")
    
    # Convert each video
    for video_path in mp4_files:
        gif_path = gifs_dir / f"{video_path.stem}.gif"
        
        # Skip if GIF already exists
        if gif_path.exists():
            print(f"Skipping {video_path.name} - GIF already exists")
            continue
        
        print(f"Converting {video_path.name} to GIF...")
        
        try:
            # Load video and convert to GIF
            clip = VideoFileClip(str(video_path))
            
            # Optimize for size while maintaining quality:
            # 1. Reduce dimensions (keep aspect ratio)
            width = min(clip.w, 320)  # cap width at 320px
            clip = clip.resize(width=width)
            
            # 2. Reduce fps for smoother appearance but smaller size
            target_fps = min(clip.fps, 10)  # cap fps
            
            # 3. Write GIF with optimizations
            clip.write_gif(
                str(gif_path),
                fps=target_fps,
                program='ffmpeg',  # Using ffmpeg for better compression
                opt='optimizeplus'  # Additional optimization
            )
            
            print(f"Created {gif_path.name}")
            print(f"Original video size: {os.path.getsize(video_path) / 1024 / 1024:.2f}MB")
            print(f"GIF size: {os.path.getsize(gif_path) / 1024 / 1024:.2f}MB")
            
        except Exception as e:
            print(f"Error converting {video_path.name}: {e}")
        
        finally:
            # Clean up
            if 'clip' in locals():
                clip.close()

if __name__ == "__main__":
    convert_videos_to_gifs()
