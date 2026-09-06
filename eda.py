import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def perform_eda(df):

    print("\n" + "=" * 50)
    print("STARTING EXPLORATORY DATA ANALYSIS")
    print("=" * 50)

    # Create folder for graphs
    os.makedirs("reports", exist_ok=True)

    # Select numerical columns
    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    # ==================================
    # 1. HISTOGRAMS
    # ==================================
    print("\nGenerating histograms...")

    for column in numerical_columns:

        plt.figure(figsize=(8, 5))
        plt.hist(df[column].dropna(), bins=30)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(f"reports/{column}_histogram.png")
        plt.close()

    # ==================================
    # 2. BOXPLOTS
    # ==================================
    print("Generating box plots...")

    for column in numerical_columns:

        plt.figure(figsize=(8, 5))
        sns.boxplot(x=df[column].dropna())

        plt.title(f"Box Plot of {column}")

        plt.tight_layout()
        plt.savefig(f"reports/{column}_boxplot.png")
        plt.close()

    # ==================================
    # 3. CORRELATION HEATMAP
    # ==================================
    print("Generating correlation heatmap...")

    correlation = df[numerical_columns].corr()

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()
    plt.savefig("reports/correlation_heatmap.png")
    plt.close()

    print("\nEDA COMPLETED!")
    print("Graphs saved inside the reports folder.")

    return correlation