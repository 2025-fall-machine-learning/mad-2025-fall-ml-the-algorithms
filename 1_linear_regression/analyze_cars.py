import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

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
    df = pd.read_csv("cars.csv")


    # Simple regression: enginesize vs price
    run_simple_linear_regression(df, predictor="enginesize", response="price")

    # Multiple regression: drop ID & CarName
    run_multiple_linear_regression(
        df,
        response="price",
        drop_cols=["car_ID", "CarName"]
    )

    #(car_ID is just a unique identifier for each row (like a serial number). 
            #It does not contain any information useful for predicting price)

#CarName is a text label (the name of the car). 
# As a string, it is not directly useful for regression, 
# and one-hot encoding it would create a huge number of unnecessary columns (Possible overfitting).

if __name__ == "__main__":
    main()
