import pandas as pd
import os
from datetime import datetime

def monitor_model(model_results):

    # Create reports folder if it does not exist
    os.makedirs("reports", exist_ok=True)

    # Get best model based on highest R2 score
    best_model = model_results.loc[
        model_results["R2 Score"].idxmax()
    ]

    # Create monitoring data
    monitoring_data = pd.DataFrame({
        "Date": [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ],
        "Best Model": [
            best_model["Model"]
        ],
        "MAE": [
            best_model["MAE"]
        ],
        "MSE": [
            best_model["MSE"]
        ],
        "RMSE": [
            best_model["RMSE"]
        ],
        "R2 Score": [
            best_model["R2 Score"]
        ]
    })

    monitoring_file = "reports/model_monitoring.csv"

    # Append new monitoring result if file already exists
    if os.path.exists(monitoring_file):

        old_data = pd.read_csv(monitoring_file)

        monitoring_data = pd.concat(
            [old_data, monitoring_data],
            ignore_index=True
        )

    # Save monitoring history
    monitoring_data.to_csv(
        monitoring_file,
        index=False
    )

    print("\n" + "=" * 60)
    print("MODEL MONITORING")
    print("=" * 60)

    print("Best Model:", best_model["Model"])
    print("MAE:", round(best_model["MAE"], 2))
    print("RMSE:", round(best_model["RMSE"], 2))
    print("R2 Score:", round(best_model["R2 Score"], 4))

    print("\nMonitoring results saved to:")
    print("reports/model_monitoring.csv")

    return monitoring_data

