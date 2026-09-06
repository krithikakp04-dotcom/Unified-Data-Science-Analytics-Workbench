import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def train_and_compare_models(df):

    print("\n" + "=" * 60)
    print("AUTOMATIC MACHINE LEARNING")
    print("=" * 60)

    # ==================================
    # 1. SELECT TARGET
    # ==================================
    target = "median_house_value"

    X = df.drop(columns=[target])
    y = df[target]

    print("\nTarget column:", target)
    print("Number of features:", X.shape[1])

    # ==================================
    # 2. IDENTIFY COLUMN TYPES
    # ==================================
    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    print("\nNumerical columns:")
    print(numerical_columns)

    print("\nCategorical columns:")
    print(categorical_columns)

    # ==================================
    # 3. PREPROCESSING
    # ==================================
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                "passthrough",
                numerical_columns
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_columns
            )
        ]
    )

    # ==================================
    # 4. SPLIT DATA
    # ==================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\nTraining samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    # ==================================
    # 5. MODELS
    # ==================================
    models = {
        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            random_state=42
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    }

    results = []
    best_model = None
    best_model_name = None
    best_r2 = float("-inf")

    # ==================================
    # 6. TRAIN AND EVALUATE MODELS
    # ==================================
    for name, model in models.items():

        print("\nTraining:", name)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        predictions = pipeline.predict(X_test)

        # Metrics
        mae = mean_absolute_error(
            y_test,
            predictions
        )

        mse = mean_squared_error(
            y_test,
            predictions
        )

        rmse = mse ** 0.5

        r2 = r2_score(
            y_test,
            predictions
        )

        print("MAE:", round(mae, 2))
        print("MSE:", round(mse, 2))
        print("RMSE:", round(rmse, 2))
        print("R2 Score:", round(r2, 4))

        # Store results
        results.append({
            "Model": name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2 Score": r2
        })

        # Find best model
        if r2 > best_r2:
            best_r2 = r2
            best_model = pipeline
            best_model_name = name

    # ==================================
    # 7. CREATE RESULTS DATAFRAME
    # ==================================
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="R2 Score",
        ascending=False
    )

    # Save results
    results_df.to_csv(
        "reports/model_comparison.csv",
        index=False
    )

    # ==================================
    # 8. SAVE BEST MODEL
    # ==================================
    joblib.dump(
        best_model,
        "best_model.pkl"
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)

    print(results_df)

    print("\nBEST MODEL:", best_model_name)
    print("BEST R2 SCORE:", round(best_r2, 4))

    print("\nBest model saved as: best_model.pkl")
    print("Results saved as: reports/model_comparison.csv")

    return best_model, results_df