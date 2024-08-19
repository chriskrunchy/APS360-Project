import os

# Path to the balanced folder
balanced_path = '/home/adam/final_project/APS360-Project/ChestX-Ray/data/nih-chest-xray-dataset/balanced'

# Dictionary to store the count of images per folder
folder_counts = {}
total = 0

# Iterate over each folder in the balanced directory
for folder in os.listdir(balanced_path):
    folder_path = os.path.join(balanced_path, folder)
    if os.path.isdir(folder_path):
        # Count the number of images in the folder
        num_images = len(os.listdir(folder_path))
        folder_counts[folder] = num_images
        total += num_images

# Print the number of images in each folder
for folder, count in folder_counts.items():
    print(f"Folder '{folder}' contains {count} images.")
print(f"Total: {total}")