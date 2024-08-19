import os
import shutil
import pandas as pd

# Define the path to the dataset and the new 'balanced' folder
data_path = '/home/adam/final_project/APS360-Project/ChestX-Ray/data/nih-chest-xray-dataset'
csv_path = '/home/adam/final_project/APS360-Project/ChestX-Ray/data/nih-chest-xray-dataset/Data_Entry_2017.csv'
balanced_path = '/home/adam/final_project/APS360-Project/ChestX-Ray/data/nih-chest-xray-dataset/balanced'

# Create the 'balanced' folder if it doesn't exist
if not os.path.exists(balanced_path):
    os.makedirs(balanced_path)

# Load the Data_Entry_2017.csv
df = pd.read_csv(csv_path)

# Filter for rows with a single label
df['Finding Labels'] = df['Finding Labels'].str.split('|')
single_label_df = df[df['Finding Labels'].apply(lambda x: len(x) == 1)]

# Function to search for the image across all subfolders
def find_image(image_name, data_path):
    for root, dirs, files in os.walk(data_path):
        if image_name in files:
            return os.path.join(root, image_name)
    return None

# Map each image to its respective folder (disease label)
for index, row in single_label_df.iterrows():
    image_name = row['Image Index']
    label = row['Finding Labels'][0]

    # Find the image in the dataset directory
    src_path = find_image(image_name, data_path)
    
    if not src_path:
        print(f"Image {image_name} not found in any folder within {data_path}")
        continue

    # Create the destination folder if it doesn't exist
    dest_folder = os.path.join(balanced_path, label)
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Move the file to the respective folder
    dest_path = os.path.join(dest_folder, image_name)
    shutil.copy(src_path, dest_path)  # Use shutil.move() if you want to move instead of copy

    print(f'Copied {image_name} to {dest_folder}')

print("Processing complete.")