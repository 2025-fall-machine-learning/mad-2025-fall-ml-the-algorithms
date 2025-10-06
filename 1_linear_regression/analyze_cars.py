import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import os

# --- Utility functions ---

def run_simple_linear_regression(df, predictor, response):
    # Run simple regression and check linearity.
    X, y = df[[predictor]], df[response]
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    print("Simple Linear Regression:")
    print(f"Predictor: {predictor}, Response: {response}")
    print(f"R^2: {model.score(X, y):.3f}")

    # Linearity check
    plt.scatter(X, y, color="blue", label="True values")
    plt.plot(X, y_pred, color="red", linestyle="dotted", label="Fit line")
    plt.xlabel(predictor); plt.ylabel(response)
    plt.title(f"{predictor} vs {response}")
    plt.legend(); plt.show()


def calculate_vif(X):
    # Check independence with Variance Inflation Factor (VIF).
    vif = pd.DataFrame()
    vif["feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif.sort_values("VIF", ascending=False)


def run_multiple_linear_regression(df, response, drop_cols=None):
    # Run multiple regression with one-hot encoding and VIF-based feature selection.
    drop_cols = drop_cols or []
    X = pd.get_dummies(df.drop(columns=drop_cols + [response]), drop_first=True)
    y = df[response]

    print("Original columns:", df.columns.tolist())
    print("Columns after one-hot encoding:", X.columns.tolist())
    print("Number of features after encoding:", X.shape[1])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Ensure numeric for VIF
    X_train = X_train.astype(float)  # Make sure all predictors are numeric

    # VIF check for multicollinearity
    vif = calculate_vif(X_train)
    high_vif = vif[vif["VIF"] > 10]["feature"].tolist()
    if high_vif:
        print("\033[1mFeatures Dropped (High VIF > 10):\033[0m", high_vif) # Bold output
        X_train, X_test = X_train.drop(columns=high_vif), X_test.drop(columns=high_vif)

    # Show features kept for modeling (low/normal VIF)
    kept_features = list(X_train.columns)
    print("\033[1mFeatures kept for regression (low/normal VIF):\033[0m", kept_features) # Bold output

    # After one-hot encoding
    num_original = len(df.columns) - len(drop_cols) - 1  # minus response and dropped
    num_encoded = X.shape[1]

    # ... (after dropping high VIF features)
    num_dropped = len(high_vif)
    num_kept = len(X_train.columns)

    # Print summary table (was just curious to see the numbers before and after one-hot)
    print("\n\033[1mFeature Summary Table\033[0m")
    print(f"{'Step':35} | {'Number of Features'}")
    print("-" * 55)
    print(f"{'Original dataset (excluding drops/response)':35} | {num_original}")
    print(f"{'After one-hot encoding':35} | {num_encoded}")
    print(f"{'Dropped (high VIF > 10)':35} | {num_dropped}")
    print(f"{'Kept for regression (low/normal VIF)':35} | {num_kept}")
    print("-" * 55)

    # Fit model
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nMultiple Linear Regression:")
    print(f"R^2 train: {model.score(X_train, y_train):.3f}")
    print(f"R^2 test: {r2_score(y_test, y_pred):.3f}")
    print("Remaining features:", X_train.shape[1])

    # Linearity check: predicted vs actual
    plt.scatter(y_test, y_pred, color="green", alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Price"); plt.ylabel("Predicted Price")
    plt.title("Predicted vs Actual Price"); plt.show()

    return model, X_train.columns


# --- Main workflow ---
def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "cars.csv")

    df = pd.read_csv(csv_path)

    # Simple regression: enginesize vs price
    run_simple_linear_regression(df, predictor="enginesize", response="price")

    # Multiple regression: drop ID & CarName
    run_multiple_linear_regression(
        df,
        response="price",
        drop_cols=["car_ID", "CarName"]
    )

if __name__ == "__main__":
    main()
