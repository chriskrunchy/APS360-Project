import os
import csv

# Path to the balanced folder
balanced_path = '/home/adam/final_project/APS360-Project/ChestX-Ray/data/nih-chest-xray-dataset/balanced'
csv_output_path = '/home/adam/final_project/APS360-Project/ChestX-Ray/data/image_labels.csv'

# Create a list to store image filenames and their corresponding class names
image_data = []

# Iterate over each folder in the balanced directory
for folder in os.listdir(balanced_path):
    folder_path = os.path.join(balanced_path, folder)
    if os.path.isdir(folder_path):
        for image_file in os.listdir(folder_path):
            image_id = image_file
            class_name = folder
            image_data.append([image_id, class_name])

# Write the collected data to a CSV file
with open(csv_output_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image ID', 'Class Name'])  # Header
    writer.writerows(image_data)

print(f"CSV file created successfully at {csv_output_path}.")
