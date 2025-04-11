import pandas as pd
from sklearn.model_selection import train_test_split

# Load your data
df = pd.read_csv('12_prueba.csv')

# Get the unique protein names
unique_proteins = df.iloc[:, 0].unique()

# Split proteins into train and test (e.g., 80% train, 20% test)
train_proteins, test_proteins = train_test_split(unique_proteins, test_size=0.2, random_state=42)

# Create train and test DataFrames by filtering
train_df = df[df.iloc[:, 0].isin(train_proteins)]
test_df = df[df.iloc[:, 0].isin(test_proteins)]

# Optional: Save the results
train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)
