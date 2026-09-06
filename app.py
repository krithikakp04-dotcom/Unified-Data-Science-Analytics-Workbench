import pandas as pd
from preprocessing import preprocess_data

# ==================================
# LOAD DATASET
# ==================================
df = pd.read_csv("housing.csv")

print("=" * 50)
print("UNIFIED DATA SCIENCE WORKBENCH")
print("=" * 50)

print("\nDATASET LOADED SUCCESSFULLY!")

print("\nOriginal Dataset Shape:")
print(df.shape)

# ==================================
# DATA PROFILING
# ==================================
print("\nCOLUMN DATA TYPES:")
print(df.dtypes)

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nDUPLICATE ROWS:")
print(df.duplicated().sum())

# ==================================
# DATA PREPROCESSING
# ==================================
df_cleaned = preprocess_data(df)

# ==================================
# SAVE CLEANED DATASET
# ==================================
df_cleaned.to_csv("cleaned_housing.csv", index=False)

print("\nCleaned dataset saved as: cleaned_housing.csv")

print("\nFirst 5 rows of cleaned dataset:")
print(df_cleaned.head())