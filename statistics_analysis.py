import pandas as pd


def statistical_analysis(df):

    print("\n" + "=" * 50)
    print("STATISTICAL ANALYSIS")
    print("=" * 50)

    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    # Store results
    statistics = pd.DataFrame({
        "Mean": df[numerical_columns].mean(),
        "Median": df[numerical_columns].median(),
        "Standard Deviation": df[numerical_columns].std(),
        "Minimum": df[numerical_columns].min(),
        "Maximum": df[numerical_columns].max(),
        "Skewness": df[numerical_columns].skew()
    })

    print("\nSTATISTICAL SUMMARY:")
    print(statistics)

    # Save statistics
    statistics.to_csv(
        "reports/statistical_summary.csv"
    )

    return statistics


def generate_insights(df):

    print("\n" + "=" * 50)
    print("AUTOMATIC DATA INSIGHTS")
    print("=" * 50)

    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    insights = []

    # Mean and median insights
    for column in numerical_columns:

        mean = df[column].mean()
        median = df[column].median()

        insight = (
            f"{column}: Mean = {mean:.2f}, "
            f"Median = {median:.2f}"
        )

        insights.append(insight)

    # Correlation analysis
    correlation = df[numerical_columns].corr()

    print("\nStrong Relationships:")

    for i in range(len(correlation.columns)):
        for j in range(i + 1, len(correlation.columns)):

            value = correlation.iloc[i, j]

            if abs(value) >= 0.5:

                column1 = correlation.columns[i]
                column2 = correlation.columns[j]

                relationship = (
                    f"{column1} and {column2} "
                    f"have correlation = {value:.2f}"
                )

                insights.append(relationship)

    # Print insights
    for number, insight in enumerate(insights, start=1):
        print(f"{number}. {insight}")

    # Save insights as a text file
    with open(
        "reports/automatic_insights.txt",
        "w",
        encoding="utf-8"
    ) as file:

        for number, insight in enumerate(insights, start=1):
            file.write(f"{number}. {insight}\n")

    print("\nInsights saved in:")
    print("reports/automatic_insights.txt")

    return insights