import pandas as pd
from preprocessing import preprocess_data
from eda import perform_eda
from statistics_analysis import statistical_analysis, generate_insights

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

# ==================================
# EXPLORATORY DATA ANALYSIS
# ==================================
correlation = perform_eda(df_cleaned)

print("\nCORRELATION MATRIX:")
print(correlation)
# ==================================
# STATISTICAL ANALYSIS
# ==================================
statistics = statistical_analysis(df_cleaned)

# ==================================
# AUTOMATIC INSIGHT GENERATION
# ==================================
insights = generate_insights(df_cleaned)