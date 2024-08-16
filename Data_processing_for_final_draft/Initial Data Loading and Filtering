import pandas as pd

# Load the NIH dataset CSV
df = pd.read_csv('NIH_Chest_Xray.csv')

# Filter rows with a single disease label
df = df[df['Finding Labels'].apply(lambda x: len(x.split('|')) == 1)]

# Further filter to keep only necessary columns (e.g., Image Path and Label)
df = df[['Image Index', 'Finding Labels']]

# Save the filtered dataset
df.to_csv('filtered_NIH_Chest_Xray.csv', index=False)
