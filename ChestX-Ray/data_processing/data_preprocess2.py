import os
import random
import shutil

# Path to the balanced folder
balanced_path = '/home/adam/final_project/APS360-Project/ChestX-Ray/data/nih-chest-xray-dataset/balanced'

# Minimum images per folder
min_images = 2000

# Step 1: Remove folders with fewer than 2,000 images
folders_to_remove = []

for folder in os.listdir(balanced_path):
    folder_path = os.path.join(balanced_path, folder)
    if os.path.isdir(folder_path):
        num_images = len(os.listdir(folder_path))
        if num_images < min_images:
            folders_to_remove.append(folder_path)

# Remove the folders
for folder in folders_to_remove:
    shutil.rmtree(folder)
    print(f"Removed folder {folder} with fewer than {min_images} images.")

# Recalculate the remaining folders and image counts
remaining_folders = [folder for folder in os.listdir(balanced_path) if os.path.isdir(os.path.join(balanced_path, folder))]
image_counts = [len(os.listdir(os.path.join(balanced_path, folder))) for folder in remaining_folders]

# Step 2: Calculate the average number of images in the remaining folders
average_images = sum(image_counts) // len(image_counts)
print(f"Average number of images per folder: {average_images}")

# Step 3: Oversample or undersample the folders to match the average
for folder in remaining_folders:
    folder_path = os.path.join(balanced_path, folder)
    image_files = os.listdir(folder_path)
    num_images = len(image_files)
    
    if num_images < average_images:
        # Oversample
        images_to_add = average_images - num_images
        while images_to_add > 0:
            for img in image_files:
                if images_to_add <= 0:
                    break
                new_image_name = f"copy_{images_to_add}_{img}"
                new_image_path = os.path.join(folder_path, new_image_name)
                shutil.copy(os.path.join(folder_path, img), new_image_path)
                images_to_add -= 1
        print(f"Oversampled {folder} to {average_images} images.")
    
    elif num_images > average_images:
        # Undersample
        images_to_remove = num_images - average_images
        images_to_remove = random.sample(image_files, images_to_remove)
        for img in images_to_remove:
            os.remove(os.path.join(folder_path, img))
        print(f"Undersampled {folder} to {average_images} images.")

# Step 4: Adjust the "No Finding" folder to have the same number of images as the other folders
no_finding_folder = os.path.join(balanced_path, 'No Finding')
if os.path.exists(no_finding_folder):
    image_files = os.listdir(no_finding_folder)
    num_images = len(image_files)
    
    if num_images > average_images:
        images_to_remove = num_images - average_images
        images_to_remove = random.sample(image_files, images_to_remove)
        for img in images_to_remove:
            os.remove(os.path.join(no_finding_folder, img))
        print(f"Adjusted 'No Finding' folder to {average_images} images.")

print("Folder balancing complete.")
