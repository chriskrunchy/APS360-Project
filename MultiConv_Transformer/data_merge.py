import os
import shutil

# Paths to the classified folders
source_dir1 = '/home/adam/final_project/APS360-Project/MultiConv_Transformer/data/classified_images_chest'
source_dir2 = '/home/adam/final_project/APS360-Project/MultiConv_Transformer/data/classified_images_cheXpert'

# Destination directory
dest_dir = '/home/adam/final_project/APS360-Project/MultiConv_Transformer/data/images'

# Ensure the destination directory exists
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir, exist_ok=True)

# Function to merge directories
def merge_dirs(source_dir, dest_dir):
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            class_name = os.path.basename(root)
            dest_class_dir = os.path.join(dest_dir, class_name)
            if not os.path.exists(dest_class_dir):
                os.makedirs(dest_class_dir, exist_ok=True)
            shutil.copy(os.path.join(root, file), os.path.join(dest_class_dir, file))

# Merge the directories
merge_dirs(source_dir1, dest_dir)
merge_dirs(source_dir2, dest_dir)

print("Merge complete.")
