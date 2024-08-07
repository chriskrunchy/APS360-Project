import os

total = []

def count_images_in_folders(root_dir):
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            image_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
            total.append(image_count)
            print(f"{folder_name}: {image_count}")

# Provide the path to the directory containing the classified images
root_directory = '/home/adam/final_project/APS360-Project/MultiConv_Transformer/data/images'
count_images_in_folders(root_directory)
print(f"Total: {sum(total)}")
