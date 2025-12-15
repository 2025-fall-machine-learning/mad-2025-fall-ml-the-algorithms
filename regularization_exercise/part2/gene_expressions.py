import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def main():
    # Load the dataset
    df = pd.read_csv('regularization_exercise/part2/gene_expressions.csv')
    # print("First five rows:\n", df.head())
    # print("Last five rows:\n", df.tail())

    # Separate predictors and response
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    # print("Predictors (X):\n", X)
    # print("Response (y):\n", y)

    # Diagnostic prints
    # print("X shape:", X.shape, "y shape:", y.shape)
    # print("Missing X total:", X.isna().sum().sum(), "Missing y total:", y.isna().sum())
    # print("y summary:\n", y.describe())

    # Inspect the Pearson's R values.
    pearson_r = X.corrwith(y, method='pearson')
    # print("Pearson's R values:\n", pearson_r)
    # print("\nTop 20 positive correlations:\n", pearson_r.sort_values(ascending=False).head(20))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Fit a Linear Regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    prediction = lr.predict(X_test)

    # Compute and print RMSE & response summary statistics
    rmse = np.sqrt(mean_squared_error(y_test, prediction))
    print("Linear Regression RMSE:\n", rmse)
    # print("\nResponse summary statistics:\n", y.describe())

    # Fit a Lasso Regression model
    lasso = LassoCV(cv=5, max_iter=10000)
    lasso.fit(X_train, y_train)
    # Predict and compute RMSE for Lasso
    lasso_prediction = lasso.predict(X_test)
    lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_prediction))
    print("Lasso Regression RMSE:\n", lasso_rmse)

    # Fit a Ridge regression model
    ridge = RidgeCV(cv=5)
    ridge.fit(X_train, y_train)
    # Predict and compute RMSE for Ridge
    ridge_prediction = ridge.predict(X_test)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_prediction))
    print("Ridge Regression RMSE:\n", ridge_rmse)

    # Save RMSE results to text file "rmse_results.txt"
    results_path = Path(__file__).parent / "gene_expressions_rmse_results.txt"
    with results_path.open("w", encoding="utf-8") as f:
        f.write(f"Linear_RMSE: {rmse}\n")
        f.write(f"Lasso_RMSE: {lasso_rmse}\n")
        f.write(f"Ridge_RMSE: {ridge_rmse}\n")


if __name__ == "__main__":
    main()
