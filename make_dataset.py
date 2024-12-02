import os
import shutil

# Define source and destination directories
source_labels_dir = '/home/robotics/CogVideo/dataset/labels'
source_videos_dir = '/home/robotics/CogVideo/dataset/videos'
destination_dir = '/home/robotics/cogvideox-factory/cogx-dataset'
destination_videos_dir = os.path.join(destination_dir, 'videos')

# Create destination directories if they don't exist
os.makedirs(destination_videos_dir, exist_ok=True)

# Create and write to prompts.txt
with open(os.path.join(destination_dir, 'prompt.txt'), 'w') as prompts_file:
    for filename in sorted(os.listdir(source_labels_dir)):
        if filename.endswith('.txt'):
            with open(os.path.join(source_labels_dir, filename), 'r') as f:
                prompt = f.read().strip()
                prompts_file.write(f"{prompt}\n")

# Create and write to videos.txt
with open(os.path.join(destination_dir, 'videos.txt'), 'w') as videos_file:
    for filename in sorted(os.listdir(source_videos_dir)):
        if filename.endswith('.mp4'):
            source_video_path = os.path.join(source_videos_dir, filename)
            destination_video_path = os.path.join(destination_videos_dir, filename)
            shutil.copy(source_video_path, destination_video_path)
            videos_file.write(f"videos/{filename}\n")

