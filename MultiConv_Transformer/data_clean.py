import os
import shutil
from collections import defaultdict
from random import sample

def clean_data(data_dir, output_dir, threshold=5000):
    class_counts = {
        'Nodule': 6331,
		'Pneumothorax': 5302,
		'Mass': 5782,
		'Infiltration': 19894,
		'Atelectasis': 11559,
		'Effusion': 13317,
		'Consolidation': 4667,
		'Edema': 2303,
		'Emphysema': 2516,
		'Pneumonia': 1431,
		'No Finding': 60361,
		'Fibrosis': 1686,
		'Cardiomegaly': 2776,
		'Hernia': 227,
		'Pleural_Thickening': 3385,

    }

    # Filter classes with more than 10k images and find the minimum count among them
    valid_classes = {cls: count for cls, count in class_counts.items() if count >= threshold}
    min_count = min(valid_classes.values())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for cls, count in valid_classes.items():
        class_dir = os.path.join(data_dir, cls)
        if count > min_count:
            # Select random images to match the smallest class size
            selected_images = sample(os.listdir(class_dir), min_count)
        else:
            selected_images = os.listdir(class_dir)

        output_class_dir = os.path.join(output_dir, cls)
        os.makedirs(output_class_dir, exist_ok=True)

        for img in selected_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(output_class_dir, img))

if __name__ == "__main__":
    data_dir = "/home/adam/final_project/APS360-Project/MultiConv_Transformer/data/images"
    output_dir = "/home/adam/final_project/APS360-Project/MultiConv_Transformer/data/cleaned_images"
    clean_data(data_dir, output_dir)
