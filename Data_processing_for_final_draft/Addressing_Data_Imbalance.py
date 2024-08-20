import numpy as np
from sklearn.utils import resample

# Calculate average number of images per class
avg_images_per_class = 12159

# Classes with fewer than 2,000 images are excluded
filtered_classes = df['Finding Labels'].value_counts()
filtered_classes = filtered_classes[filtered_classes >= 2000].index.tolist()
df = df[df['Finding Labels'].isin(filtered_classes)]

# Undersample "No Finding" class and oversample other classes to balance dataset
balanced_dfs = []
for class_label in filtered_classes:
    class_df = df[df['Finding Labels'] == class_label]
    if len(class_df) > avg_images_per_class:
        class_df = resample(class_df, replace=False, n_samples=avg_images_per_class, random_state=42)
    else:
        class_df = resample(class_df, replace=True, n_samples=avg_images_per_class, random_state=42)
    balanced_dfs.append(class_df)

# Combine balanced classes into one DataFrame
balanced_df = pd.concat(balanced_dfs)

# Save the balanced dataset
balanced_df.to_csv('balanced_NIH_Chest_Xray.csv', index=False)
