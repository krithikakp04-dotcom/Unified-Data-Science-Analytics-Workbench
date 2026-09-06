import pandas as pd

# Load the dataset
df = pd.read_csv("housing.csv")

# Display first 5 rows
print("FIRST 5 ROWS OF THE DATASET:")
print(df.head())

# Display dataset information
print("\nDATASET INFORMATION:")
print(df.info())

# Display dataset shape
print("\nDATASET SHAPE:")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Display column names
print("\nCOLUMN NAMES:")
print(df.columns.tolist())