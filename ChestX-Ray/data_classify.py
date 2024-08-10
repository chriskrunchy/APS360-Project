import os
import pandas as pd
from shutil import copy2

# Base path for dataset
base_dir = '/home/adam/final_project/data/nih-chest-xray-dataset'

# Metadata CSV path
metadata_path = os.path.join(base_dir, 'Data_Entry_2017.csv')

# Output directory for classified images
output_dir = os.path.join(base_dir, 'classified_images')

# Load metadata
metadata = pd.read_csv(metadata_path)
metadata.set_index('Image Index', inplace=True)

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# Process each image file found in the given directory
def process_image(folder, file_name):
    if file_name.endswith('.png'):  # Only process PNG files
        if file_name in metadata.index:
            labels = metadata.loc[file_name, 'Finding Labels'].split('|')
            for label in labels:
                label_dir = os.path.join(output_dir, label)
                if not os.path.exists(label_dir):
                    os.makedirs(label_dir, exist_ok=True)
                src_path = os.path.join(folder, file_name)
                dst_path = os.path.join(label_dir, file_name)
                copy2(src_path, dst_path)
                # print(f"Copied {src_path} to {dst_path}")
    #     else:
    #         print(f"File {file_name} not found in metadata.")
    # else:
    #     print(f"Skipped non-image file: {file_name}")

# Find all image folders
image_folders = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if d.startswith('images_')]
# print("Image folders found:", image_folders)

# Process each image directory
for folder in image_folders:
    # print(f"Processing folder: {folder}")
    inner_folder = os.path.join(folder, "images")  # Assuming all images are inside a subfolder named 'images'
    if os.path.exists(inner_folder):
        for file_name in os.listdir(inner_folder):
            process_image(inner_folder, file_name)

print("Classification complete.")