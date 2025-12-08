
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def main():
    # --- Load data ---
    df = pd.read_csv("marketing_buckets.csv")
    print("\n=== Shape (rows, cols) ===")
    print(df.shape)

    # Confirm response column is the last one (LVT)
    assert "LVT" in df.columns, "Expected 'LVT' as the response column."
    y = df["LVT"]
    X = df.drop(columns=["LVT"])

    print("\n=== First few predictor columns ===")
    print(list(X.columns)[:10], "... (total predictors:", X.shape[1], ")")
    print("\n=== Response 'LVT'.describe() ===")
    print(y.describe())

    # --- Split: 30% test, random_state=1 ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )
    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

    # --- Linear Regression ---
    lin = LinearRegression().fit(X_train, y_train)
    lin_pred = lin.predict(X_test)
    lin_rmse = rmse(y_test, lin_pred)
    print(f"\nLinear RMSE: {lin_rmse:,.2f}")

    # --- RidgeCV (cv=5) with scaling ---
    ridge = Pipeline([("scaler", StandardScaler()), ("ridge", RidgeCV(cv=5))]).fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    ridge_rmse = rmse(y_test, ridge_pred)
    print(f"RidgeCV (cv=5) RMSE: {ridge_rmse:,.2f}")

    # --- LassoCV (cv=5) with scaling ---
    lasso = Pipeline([("scaler", StandardScaler()), ("lasso", LassoCV(cv=5, max_iter=10000))]).fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)
    lasso_rmse = rmse(y_test, lasso_pred)
    print(f"LassoCV (cv=5) RMSE: {lasso_rmse:,.2f}")

    # --- Hyperparameters chosen by CV ---
    print("\nLasso alpha (CV):", lasso.named_steps["lasso"].alpha_)
    try:
        print("Ridge alpha (CV):", ridge.named_steps["ridge"].alpha_)
    except AttributeError:
        pass

    # --- Compare RMSE to LVT.describe() figures ---
    print("\n=== RMSE vs. LVT summary ===")
    print(f"Mean: {y.mean():,.2f} | Std: {y.std():,.2f} | "
          f"IQR (~25%..75%): {y.quantile(0.25):,.2f} .. {y.quantile(0.75):,.2f} | "
          f"Min: {y.min():,.2f} | Max: {y.max():,.2f}")

    # --- Rerun without random_state (one demo) ---
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.3)  # no random_state
    lin2 = LinearRegression().fit(X_train2, y_train2)
    ridge2 = Pipeline([("scaler", StandardScaler()), ("ridge", RidgeCV(cv=5))]).fit(X_train2, y_train2)
    lasso2 = Pipeline([("scaler", StandardScaler()), ("lasso", LassoCV(cv=5, max_iter=10000))]).fit(X_train2, y_train2)

    print("\n=== Rerun (no random_state) RMSEs ===")
    print(f"Linear: {rmse(y_test2, lin2.predict(X_test2)):,.2f} | "
          f"RidgeCV: {rmse(y_test2, ridge2.predict(X_test2)):,.2f} | "
          f"LassoCV: {rmse(y_test2, lasso2.predict(X_test2)):,.2f}")

if __name__ == "__main__":
    main()