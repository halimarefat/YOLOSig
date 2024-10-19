import os
import random
import shutil

def move_random_files(source_dir, dest_dir, percentage=10):
    # Ensure destination directory exists
    os.makedirs(dest_dir, exist_ok=True)
    
    # List all files in the source directory
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    
    # Calculate the number of files to move
    num_files_to_move = max(1, int(len(files) * percentage / 100))
    
    # Select random files to move
    files_to_move = random.sample(files, num_files_to_move)
    
    # Move each selected file to the destination directory
    for file in files_to_move:
        src_path = os.path.join(source_dir, file)
        dest_path = os.path.join(dest_dir, file)
        shutil.move(src_path, dest_path)
        print(f"Moved: {src_path} -> {dest_path}")

# Define source and destination directories
source_dir = '/Users/ali/Desktop/YOLOSeg/BCSD_YOLO/images/train'
dest_dir = '/Users/ali/Desktop/YOLOSeg/BCSD_YOLO/images/val'

# Move 10% of files randomly
move_random_files(source_dir, dest_dir, percentage=10)

# Define source and destination directories for labels
source_label_dir = '/Users/ali/Desktop/YOLOSeg/BCSD_YOLO/labels/train'
dest_label_dir = '/Users/ali/Desktop/YOLOSeg/BCSD_YOLO/labels/val'

# Move 10% of label files randomly
move_random_files(source_label_dir, dest_label_dir, percentage=10)
