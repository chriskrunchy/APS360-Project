# Map class names to integer indices
class_mapping = {label: idx for idx, label in enumerate(balanced_df['Finding Labels'].unique())}
balanced_df['Label Index'] = balanced_df['Finding Labels'].map(class_mapping)

# Save the updated CSV with label indices
balanced_df.to_csv('balanced_NIH_Chest_Xray_mapped.csv', index=False)
