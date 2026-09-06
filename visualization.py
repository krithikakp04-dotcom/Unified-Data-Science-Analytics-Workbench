import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_model_comparison(results_df):

    os.makedirs("reports", exist_ok=True)

    # R2 Score comparison
    plt.figure(figsize=(8, 5))

    plt.bar(
        results_df["Model"],
        results_df["R2 Score"]
    )

    plt.title("Model R2 Score Comparison")
    plt.xlabel("Model")
    plt.ylabel("R2 Score")
    plt.xticks(rotation=15)

    plt.tight_layout()
    plt.savefig("reports/model_r2_comparison.png")
    plt.close()

    # RMSE comparison
    plt.figure(figsize=(8, 5))

    plt.bar(
        results_df["Model"],
        results_df["RMSE"]
    )

    plt.title("Model RMSE Comparison")
    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.xticks(rotation=15)

    plt.tight_layout()
    plt.savefig("reports/model_rmse_comparison.png")
    plt.close()

    print("\nModel comparison graphs created successfully!")

def plot_actual_vs_predicted(y_test, predictions):

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        predictions
    )

    # Perfect prediction reference line
    minimum = min(y_test.min(), predictions.min())
    maximum = max(y_test.max(), predictions.max())

    plt.plot(
        [minimum, maximum],
        [minimum, maximum]
    )

    plt.title("Actual vs Predicted House Values")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")

    plt.tight_layout()
    plt.savefig("reports/actual_vs_predicted.png")
    plt.close()

    print("Actual vs Predicted graph created successfully!")    