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


# Display the first few and last few rows of the dataset
def show_head_and_tail(df: pd.DataFrame) -> None:
    print("\n=== First few rows ===")
    print(df.head())
    print("\n=== Last few rows ===")
    print(df.tail())


# Inspect Pearson correlations of all predictors with the response y
def inspect_pearson_correlations(df: pd.DataFrame) -> None:
    corr = df.corr(method="pearson")
    corr_with_y = corr["y"].sort_values(ascending=False)

    print("\n=== Pearson Correlation of each feature with y ===")
    print(corr_with_y)


# Create a full correlation matrix and return it
def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    corr = df.corr(method="pearson")
    print("\n=== Full Correlation Matrix ===")
    print(corr)
    return corr


# Create a heatmap from the correlation matrix
def show_correlation_heatmap(corr_df: pd.DataFrame):
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_df, cmap="coolwarm", annot=False)
    plt.title("Gene Expression Correlation Matrix Heatmap")
    plt.show()


# Split predictors and response into training and testing sets
def split_data(df: pd.DataFrame):
    X = df.drop(columns=["y"])        # All predictors
    y = df["y"]                       # Response variable

    # Use 20% of data for testing and set random_state=0
    X_train, X_test, y_train, y_test = ms.train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    return X_train, X_test, y_train, y_test


# Fit a linear regression model and compute RMSE
def run_linear_regression(X_train, X_test, y_train, y_test):
    model = lm.LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print(f"\nLinear Regression RMSE: {rmse:,.3f}")
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

    print(f"Lasso Regression RMSE: {rmse_lasso:,.3f}")
    print(f"Ridge Regression RMSE: {rmse_ridge:,.3f}")

    return rmse_lasso, rmse_ridge


# Main program controlling function calls
def main():
    df = load_data("gene_expressions.csv")                 # Load dataset
    show_head_and_tail(df)                                 # Show head/tail

    inspect_pearson_correlations(df)                       # Pearson correlation with y

    corr = compute_correlation_matrix(df)                  # Full correlation matrix
    show_correlation_heatmap(corr)                         # Heatmap visualization

    X_train, X_test, y_train, y_test = split_data(df)      # Train/test split

    run_linear_regression(X_train, X_test, y_train, y_test)  # Linear model

    run_lasso_ridge(X_train, X_test, y_train, y_test)         # Lasso & Ridge models
    # Expected RMSE values:
    # Linear ≈ 1184
    # Lasso  ≈ 460   (best)
    # Ridge  ≈ 1184  (similar to linear)


# Run the program
if __name__ == "__main__":
    main()
