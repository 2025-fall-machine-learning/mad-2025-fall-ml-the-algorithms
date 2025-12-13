

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error
import sys

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def main():

    # --- Output to both console and file ---
    output_file = open("output.txt", "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, output_file)


    # === PART 1: marketing_buckets.csv ===
    print("\n================ PART 1: marketing_buckets.csv ================".upper())
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

    print(f"\n=== Rerun (no random_state) RMSEs ===")
    print(f"Linear: {rmse(y_test2, lin2.predict(X_test2)):,.2f} | "
          f"RidgeCV: {rmse(y_test2, ridge2.predict(X_test2)):,.2f} | "
          f"LassoCV: {rmse(y_test2, lasso2.predict(X_test2)):,.2f}")



    # === PART 2: gene_expressions.csv ===
    print("\n================ PART 2: gene_expressions.csv ================".upper())
    df2 = pd.read_csv("gene_expressions.csv")
    print("\n=== First few rows ===")
    print(df2.head())
    print("\n=== Last few rows ===")
    print(df2.tail())

    # Last column is response 'y', rest are predictors
    y2 = df2.iloc[:, -1]
    X2 = df2.iloc[:, :-1]

    print("\n=== Pearson's R values (predictors vs y) ===")
    pearson_r = X2.apply(lambda col: col.corr(y2))
    print(pearson_r.describe())
    print("(Full Pearson's R array omitted for brevity)")

    # --- Split: 20% test, random_state=0 ---
    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y2, test_size=0.2, random_state=0
    )
    print(f"\nTrain shape: {X2_train.shape}, Test shape: {X2_test.shape}")

    # --- Linear Regression ---
    lin2 = LinearRegression().fit(X2_train, y2_train)
    lin2_pred = lin2.predict(X2_test)
    lin2_rmse = rmse(y2_test, lin2_pred)
    print(f"\nLinear RMSE: {lin2_rmse:,.2f}")

    # --- RidgeCV (cv=5) with scaling ---
    ridge2 = Pipeline([("scaler", StandardScaler()), ("ridge", RidgeCV(cv=5))]).fit(X2_train, y2_train)
    ridge2_pred = ridge2.predict(X2_test)
    ridge2_rmse = rmse(y2_test, ridge2_pred)
    print(f"RidgeCV (cv=5) RMSE: {ridge2_rmse:,.2f}")

    # --- LassoCV (cv=5) with scaling ---
    lasso2 = Pipeline([("scaler", StandardScaler()), ("lasso", LassoCV(cv=5, max_iter=10000))]).fit(X2_train, y2_train)
    lasso2_pred = lasso2.predict(X2_test)
    lasso2_rmse = rmse(y2_test, lasso2_pred)
    print(f"LassoCV (cv=5) RMSE: {lasso2_rmse:,.2f}")

    # --- Hyperparameters chosen by CV ---
    print("\nLasso alpha (CV):", lasso2.named_steps["lasso"].alpha_)
    try:
        print("Ridge alpha (CV):", ridge2.named_steps["ridge"].alpha_)
    except AttributeError:
        pass

    # --- Compare RMSE to y.describe() figures ---
    print("\n=== RMSE vs. y summary ===")
    print(f"Mean: {y2.mean():,.2f} | Std: {y2.std():,.2f} | "
          f"IQR (~25%..75%): {y2.quantile(0.25):,.2f} .. {y2.quantile(0.75):,.2f} | "
          f"Min: {y2.min():,.2f} | Max: {y2.max():,.2f}")

    # --- Restore stdout and close file ---
    sys.stdout = original_stdout
    output_file.close()

if __name__ == "__main__":
    main()