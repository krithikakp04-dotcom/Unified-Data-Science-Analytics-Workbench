import streamlit as st
import pandas as pd
import os
import joblib

# ==================================

# PAGE CONFIGURATION

# ==================================

st.set_page_config(
page_title="Unified Data Science Workbench",
page_icon="📊",
layout="wide"
)

# ==================================
# LANDING PAGE
# ==================================

st.title("📊 Unified Data Science & Analytics Workbench")

st.subheader("Project Details")

col1, col2 = st.columns(2)

with col1:
    st.info(
        """
        **Project Topic**

        Unified Data Science & Analytics Workbench
        """
    )

with col2:
    st.info(
        """
        **Student Details**

        **Full Name:** Krithika K P

        **Registered Email ID:** krithikakp04@gmail.com
        """
    )

st.markdown("---")

st.write("An End-to-End Automated Data Science Platform")

# ==================================

# LOAD DATA

# ==================================

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_housing.csv")


try:
    df = load_data()
except FileNotFoundError:
    st.error("cleaned_housing.csv not found. Please run app.py first.")
    st.stop()

# ==================================

# SIDEBAR

# ==================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
"Select a Module",
[
"Dataset Overview",
"Data Profiling",
"Statistical Analysis",
"EDA Visualizations",
"Model Comparison",
"Predictions",
"Model Monitoring"
]
)

# ==================================

# DATASET OVERVIEW

# ==================================

if page == "Dataset Overview":

    st.header("Dataset Overview")

    col1, col2 = st.columns(2)

    col1.metric("Number of Rows", df.shape[0])
    col2.metric("Number of Columns", df.shape[1])

    st.subheader("First 10 Rows")
    st.dataframe(df.head(10))

    st.subheader("Column Names")
    st.write(df.columns.tolist())


# ==================================

# DATA PROFILING

# ==================================

elif page == "Data Profiling":

    st.header("Data Profiling")

    # Data Types
    st.subheader("Data Types")

    data_types = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values
    })

    st.dataframe(data_types)

    # Missing Values
    st.subheader("Missing Values")

    missing_values = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values
    })

    st.dataframe(missing_values)

    # Duplicate Rows
    st.metric(
        "Duplicate Rows",
        int(df.duplicated().sum())
    )

# ==================================

# STATISTICAL ANALYSIS

# ==================================

elif page == "Statistical Analysis":

    st.header("Statistical Analysis")

    if os.path.exists("reports/statistical_summary.csv"):

        statistics = pd.read_csv(
            "reports/statistical_summary.csv"
        )

        st.dataframe(statistics)

    else:
        st.warning(
            "Statistical summary file not found. Run app.py first."
        )

# ==================================

# EDA VISUALIZATIONS

# ==================================

elif page == "EDA Visualizations":

    st.header("Exploratory Data Analysis")

    graph_options = [
        "Correlation Heatmap",
        "Model R2 Comparison",
        "Model RMSE Comparison",
        "Actual vs Predicted"
    ]

    selected_graph = st.selectbox(
        "Select a Graph",
        graph_options
    )

    graph_files = {
        "Correlation Heatmap":
            "reports/correlation_heatmap.png",

        "Model R2 Comparison":
            "reports/model_r2_comparison.png",

        "Model RMSE Comparison":
            "reports/model_rmse_comparison.png",

        "Actual vs Predicted":
            "reports/actual_vs_predicted.png"
    }

    graph_path = graph_files[selected_graph]

    if os.path.exists(graph_path):

        st.image(
            graph_path,
            caption=selected_graph
        )

    else:
        st.warning(
            "Graph not found. Run app.py first."
        )

# ==================================

# MODEL COMPARISON

# ==================================

elif page == "Model Comparison":

    st.header("Machine Learning Model Comparison")

    if os.path.exists("reports/model_comparison.csv"):

        results = pd.read_csv(
            "reports/model_comparison.csv"
        )

        st.dataframe(results)

        st.subheader("R² Score Comparison")

        st.bar_chart(
            results.set_index("Model")["R2 Score"]
        )

        st.subheader("RMSE Comparison")

        st.bar_chart(
            results.set_index("Model")["RMSE"]
        )

        best_model = results.loc[
            results["R2 Score"].idxmax()
        ]

        st.success(
            f"Best Model: {best_model['Model']} | "
            f"R² Score: {best_model['R2 Score']:.4f}"
        )

    else:
        st.warning(
            "Model comparison results not found. Run app.py first."
        )
# ==================================

# MODEL MONITORING

# ==================================

elif page == "Model Monitoring":

    st.header("Model Monitoring & Performance Tracking")

    monitoring_file = "reports/model_monitoring.csv"

    if os.path.exists(monitoring_file):

        monitoring_data = pd.read_csv(monitoring_file)

        st.subheader("Monitoring History")
        st.dataframe(monitoring_data)

        latest = monitoring_data.iloc[-1]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Latest MAE",
            f"{latest['MAE']:.2f}"
        )

        col2.metric(
            "Latest RMSE",
            f"{latest['RMSE']:.2f}"
        )

        col3.metric(
            "Latest R² Score",
            f"{latest['R2 Score']:.4f}"
        )

        st.subheader("R² Score Over Time")

        chart_data = monitoring_data.copy()
        chart_data["Date"] = pd.to_datetime(
            chart_data["Date"]
        )

        st.line_chart(
            chart_data.set_index("Date")["R2 Score"]
        )

    else:

        st.warning(
            "Monitoring data not found. Run app.py first."
        )


# ==================================

# PREDICTIONS

# ==================================

elif page == "Predictions":

    st.header("House Price Prediction")

    st.info(
        "Enter housing details to predict the median house value."
    )

    col1, col2 = st.columns(2)

    with col1:

        longitude = st.number_input(
            "Longitude",
            value=-122.23
        )

        latitude = st.number_input(
            "Latitude",
            value=37.88
        )

        housing_median_age = st.number_input(
            "Housing Median Age",
            value=30.0
        )

        total_rooms = st.number_input(
            "Total Rooms",
            value=2000.0
        )

        total_bedrooms = st.number_input(
            "Total Bedrooms",
            value=400.0
        )

    with col2:

        population = st.number_input(
            "Population",
            value=1000.0
        )

        households = st.number_input(
            "Households",
            value=350.0
        )

        median_income = st.number_input(
            "Median Income",
            value=3.5
        )

        ocean_proximity = st.selectbox(
            "Ocean Proximity",
            [
                "<1H OCEAN",
                "INLAND",
                "NEAR OCEAN",
                "NEAR BAY",
                "ISLAND"
            ]
        )

    if st.button("Predict House Value"):

        if os.path.exists("best_model.pkl"):

            model = joblib.load("best_model.pkl")

            input_data = pd.DataFrame({
                "longitude": [longitude],
                "latitude": [latitude],
                "housing_median_age": [
                    housing_median_age
                ],
                "total_rooms": [total_rooms],
                "total_bedrooms": [
                    total_bedrooms
                ],
                "population": [population],
                "households": [households],
                "median_income": [
                    median_income
                ],
                "ocean_proximity": [
                    ocean_proximity
                ]
            })

            prediction = model.predict(input_data)[0]

            st.success(
                f"Predicted House Value: ${prediction:,.2f}"
            )

        else:
            st.error(
                "best_model.pkl not found. "
                "Run app.py first to create the model."
            )

