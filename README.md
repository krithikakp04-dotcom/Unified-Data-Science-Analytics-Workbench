# Unified Data Science & Analytics Workbench

## Project Overview

The Unified Data Science & Analytics Workbench is an end-to-end Data Science platform designed to automate major stages of the Data Science lifecycle. The system performs data preprocessing, exploratory data analysis, statistical analysis, automated machine learning, model comparison, visualization, prediction, and model performance monitoring.

The project uses the California Housing dataset to demonstrate a complete Data Science workflow.

---

## Features

* Automated data loading and preprocessing
* Missing value and duplicate data handling
* Exploratory Data Analysis (EDA)
* Correlation analysis and visualization
* Statistical summary generation
* Automatic insight generation
* Training of multiple Machine Learning models
* Automatic model comparison
* Best model selection based on R² score
* Model performance visualization
* House price prediction
* Interactive Streamlit dashboard
* Model monitoring and performance tracking

---

## Project Modules

### 1. Data Engineering

The dataset is loaded and cleaned automatically.

Functions include:

* Loading CSV data
* Handling missing values
* Removing duplicate records
* Saving the cleaned dataset

### 2. Exploratory Data Analysis

EDA is performed to understand the structure and relationships in the dataset.

Generated analysis includes:

* Dataset information
* Statistical summaries
* Missing value analysis
* Correlation heatmap

### 3. Statistical Analysis

Statistical information is generated automatically, including:

* Mean
* Median
* Standard deviation
* Minimum and maximum values
* Correlations between numerical features

### 4. Automated Machine Learning

The system automatically trains and compares multiple regression models:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

The models are evaluated using:

* MAE
* MSE
* RMSE
* R² Score

The best-performing model is selected automatically.

### 5. Data Visualization

The following visualizations are generated:

* Correlation Heatmap
* Model R² Score Comparison
* Model RMSE Comparison
* Actual vs Predicted Values

### 6. Interactive Dashboard

A Streamlit dashboard provides the following modules:

* Dataset Overview
* Data Profiling
* Statistical Analysis
* EDA Visualizations
* Model Comparison
* Predictions
* Model Monitoring

### 7. Model Monitoring

The system stores model performance history and tracks:

* Best Model
* MAE
* MSE
* RMSE
* R² Score
* Date and Time of Evaluation

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Streamlit
* Joblib
* Git
* GitHub

---

## Project Structure

```text
unified_data_science_workbench/
│
├── app.py
├── preprocessing.py
├── eda.py
├── statistics_analysis.py
├── automl.py
├── visualization.py
├── monitoring.py
├── dashboard.py
├── housing.csv
│
├── cleaned_housing.csv
├── best_model.pkl
│
└── reports/
    ├── correlation_heatmap.png
    ├── statistical_summary.csv
    ├── automatic_insights.txt
    ├── model_comparison.csv
    ├── model_r2_comparison.png
    ├── model_rmse_comparison.png
    ├── actual_vs_predicted.png
    └── model_monitoring.csv
```

---

## Installation

Install the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib streamlit joblib
```

---

## How to Run the Project

### Step 1: Run the Data Science Pipeline

```bash
python app.py
```

This will perform:

```text
Data Loading
    ↓
Data Preprocessing
    ↓
EDA
    ↓
Statistical Analysis
    ↓
Automatic Insight Generation
    ↓
Automated Machine Learning
    ↓
Model Comparison
    ↓
Best Model Selection
    ↓
Visualization
    ↓
Model Monitoring
```

### Step 2: Run the Dashboard

```bash
streamlit run dashboard.py
```

The Streamlit dashboard will open in your web browser.

---

## Machine Learning Workflow

```text
Housing Dataset
       ↓
Data Preprocessing
       ↓
Feature and Target Selection
       ↓
Train-Test Split
       ↓
Linear Regression
Decision Tree
Random Forest
       ↓
Performance Evaluation
       ↓
MAE | MSE | RMSE | R²
       ↓
Best Model Selection
       ↓
Model Saving
       ↓
Prediction and Monitoring
```

---

## Dataset

The project uses the California Housing dataset.

Target variable:

```text
median_house_value
```

Features include:

* longitude
* latitude
* housing_median_age
* total_rooms
* total_bedrooms
* population
* households
* median_income
* ocean_proximity

---

## Output

After running the project, the system generates:

```text
cleaned_housing.csv
reports/statistical_summary.csv
reports/automatic_insights.txt
reports/correlation_heatmap.png
reports/model_comparison.csv
reports/model_r2_comparison.png
reports/model_rmse_comparison.png
reports/actual_vs_predicted.png
reports/model_monitoring.csv
best_model.pkl
```

---

## Future Enhancements

* Add more Machine Learning algorithms
* Implement hyperparameter optimization
* Add real-time data ingestion
* Implement data drift detection
* Add automated model retraining
* Deploy the application to the cloud
* Add user authentication
* Support multiple datasets
* Implement database integration

---

## Outcome

This project demonstrates a complete end-to-end Data Science lifecycle platform covering Data Engineering, Exploratory Data Analysis, Statistics, Machine Learning, Dashboarding, and MLOps.

The platform provides an automated workflow that transforms raw data into analytical insights, predictive models, visualizations, predictions, and monitored model performance.
