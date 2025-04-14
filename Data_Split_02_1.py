import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('binding_features.csv')

unique_proteins = df.iloc[:, 0].unique()

train_proteins, test_proteins = train_test_split(unique_proteins, test_size=0.2, random_state=42)

train_df = df[df.iloc[:, 0].isin(train_proteins)]
test_df = df[df.iloc[:, 0].isin(test_proteins)]

train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)
