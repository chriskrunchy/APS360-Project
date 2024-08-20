import os

# Organize images into class-specific folders
for idx, row in balanced_df.iterrows():
    class_dir = os.path.join('Balanced_Dataset', row['Finding Labels'])
    os.makedirs(class_dir, exist_ok=True)
    src_path = os.path.join('Images', row['Image Index'])
    dst_path = os.path.join(class_dir, row['Image Index'])
    os.rename(src_path, dst_path)

# Create new CSV file with accurate labels
balanced_df['Image Path'] = balanced_df.apply(lambda row: os.path.join('Balanced_Dataset', row['Finding Labels'], row['Image Index']), axis=1)
balanced_df.to_csv('balanced_NIH_Chest_Xray_with_paths.csv', index=False)
