import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def main():
    # Load the dataset
    df = pd.read_csv('regularization_exercise/part1/marketing_buckets.csv')
    # print(df)

    # Separate predictors and response
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    # print("Predictors (X):\n", X)
    # print("Response (y):\n", y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # print("X_train:\n", X_train)
    # print("X_test:\n", X_test)
    # print("y_train:\n", y_train)
    # print("y_test:\n", y_test)

    # Fit a Linear Regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    prediction = lr.predict(X_test)

    # Compute and print RMSE & response summary statistics
    rmse = np.sqrt(mean_squared_error(y_test, prediction))
    print("Linear Regression RMSE:\n", rmse)
    # print("\nResponse summary statistics:\n", y.describe())

    # Scale predictors
    scale = StandardScaler()
    X_train_scaled = scale.fit_transform(X_train)
    X_test_scaled = scale.transform(X_test)

    # Fit a Lasso Regression model
    lasso = LassoCV(cv=5)
    lasso.fit(X_train_scaled, y_train)
    # Predict and compute RMSE for Lasso
    lasso_prediction = lasso.predict(X_test_scaled)
    lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_prediction))
    print("Lasso Regression RMSE:\n", lasso_rmse)

    # Fit a Ridge regression model
    # I see in step 8 ridge is supposed to give the same results as linear, but theres a ~4,000,000 difference between them. 
    # Not sure if they're actually supposed to be identical but my Linear was had a ~5% worse RMSE. I think it has to do with my predictor scaling. But without scaling the numbers are wayyyyy off.
    ridge = RidgeCV(cv=5)
    ridge.fit(X_train_scaled, y_train)
    # Predict and compute RMSE for Ridge
    ridge_prediction = ridge.predict(X_test_scaled)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_prediction))
    print("Ridge Regression RMSE:\n", ridge_rmse)

    # Save RMSE results to text file "rmse_results.txt"
    results_path = Path(__file__).parent / "marketing_buckets_rmse_results.txt"
    with results_path.open("w", encoding="utf-8") as f:
        f.write(f"Linear_RMSE: {rmse}\n")
        f.write(f"Lasso_RMSE: {lasso_rmse}\n")
        f.write(f"Ridge_RMSE: {ridge_rmse}\n")


if __name__ == "__main__":
    main()