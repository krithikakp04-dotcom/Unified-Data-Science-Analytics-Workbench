import pandas as pd


def preprocess_data(df):

    print("\n" + "=" * 50)
    print("STARTING DATA PREPROCESSING")
    print("=" * 50)

    # Store original shape
    original_rows = df.shape[0]

    # Check missing values before cleaning
    print("\nMissing values BEFORE cleaning:")
    print(df.isnull().sum())

    # Fill missing values in numerical columns with median
    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    for column in numerical_columns:
        if df[column].isnull().sum() > 0:
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)

    # Fill missing values in categorical columns with mode
    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for column in categorical_columns:
        if df[column].isnull().sum() > 0:
            mode_value = df[column].mode()[0]
            df[column] = df[column].fillna(mode_value)

    # Remove duplicate rows
    duplicates = df.duplicated().sum()
    df = df.drop_duplicates()

    # Check missing values after cleaning
    print("\nMissing values AFTER cleaning:")
    print(df.isnull().sum())

    print("\nDuplicate rows removed:", duplicates)
    print("Original number of rows:", original_rows)
    print("Rows after preprocessing:", df.shape[0])

    print("\nDATA PREPROCESSING COMPLETED!")

    return df