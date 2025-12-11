import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics


# Load the CSV file into a DataFrame
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# Display the dataset and basic structure
def show_data(df: pd.DataFrame) -> None:
    print("\n=== Dataset ===")
    print(df)
    print("\n=== Info ===")
    print(df.info())


# Split predictors and response into training and testing sets
def split_data(df: pd.DataFrame):
    X = df.drop(columns=["LVT"])       # All predictors
    y = df["LVT"]                      # Response variable

    # Fixed split for Tasks 5–8
    X_train, X_test, y_train, y_test = ms.train_test_split(
        X, y, test_size=0.3 # random_state=1
    )

    # For Task 9, remove random_state above and use this:
    # X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.3)

    return X_train, X_test, y_train, y_test, y


# Fit a linear regression model and compute RMSE
def run_linear_regression(X_train, X_test, y_train, y_test):
    model = lm.LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print(f"\nLinear RMSE: {rmse:,.2f}")
    return rmse


# Fit LassoCV and RidgeCV models and compute RMSE for each
def run_lasso_ridge(X_train, X_test, y_train, y_test):
    lasso = lm.LassoCV(cv=5)
    lasso.fit(X_train, y_train.values.ravel())
    y_pred_lasso = lasso.predict(X_test)
    rmse_lasso = np.sqrt(metrics.mean_squared_error(y_test, y_pred_lasso))

    ridge = lm.RidgeCV(cv=5)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    rmse_ridge = np.sqrt(metrics.mean_squared_error(y_test, y_pred_ridge))

    print(f"Lasso RMSE: {rmse_lasso:,.2f}")
    print(f"Ridge RMSE: {rmse_ridge:,.2f}")

    return rmse_lasso, rmse_ridge, y_pred_lasso, y_pred_ridge


# Create a correlation matrix (Pearson)
def make_correlation_matrix(df: pd.DataFrame):
    corr = df.corr(method="pearson")      # Compute correlations
    print("\n=== Correlation Matrix ===")
    print(corr)
    return corr


# Create a heatmap from the correlation matrix
def make_heatmap(corr_df: pd.DataFrame):
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_df, cmap="coolwarm", annot=False, vmin=-1, vmax=1)
    plt.title("Correlation Matrix Heatmap")
    plt.show()


# Main program controlling function calls
def main():
    df = load_data("marketing_buckets.csv")                  # Load dataset
    show_data(df)                                            # Display dataset info
    X_train, X_test, y_train, y_test, y = split_data(df)     # Train/test split

    print("\nResponse describe() for LVT:")
    print(y.describe())                                      # Show response stats

    run_linear_regression(X_train, X_test, y_train, y_test)  # Linear model

    rmse_lasso, rmse_ridge, y_pred_lasso, y_pred_ridge = run_lasso_ridge(
        X_train, X_test, y_train, y_test
    )                                                        # Lasso & Ridge

    corr = make_correlation_matrix(df)                       # Correlation matrix
    make_heatmap(corr)                                       # Heatmap visualization


# Run the program
if __name__ == "__main__":
    main()
