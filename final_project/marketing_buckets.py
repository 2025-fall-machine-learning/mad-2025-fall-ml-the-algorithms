import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error
import os


def main():

    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "marketing_buckets.csv"))

    print(df)
    print("shape:", df.shape)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    print("X columns:", list(X.columns))
    print("y name:", y.name)
    print(y.describe())

    print("\n" + "="*80 + "\n")

    rmse_results = []

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )

    lin = LinearRegression()
    lin.fit(X_train, y_train)
    y_pred_lin = lin.predict(X_test)
    rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))
    print("\n")
    print("Linear RMSE:", rmse_lin)
    rmse_results.append(("Linear (rs=1)", rmse_lin))

    ridge = RidgeCV(cv=5)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
    print("Ridge RMSE:", rmse_ridge)
    rmse_results.append(("Ridge (rs=1)", rmse_ridge))

    lasso = LassoCV(cv=5)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
    print("Lasso RMSE:", rmse_lasso)
    rmse_results.append(("Lasso (rs=1)", rmse_lasso))
    print("\n" + "="*80 + "\n")

    for i in range(5):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        lin2 = LinearRegression()
        lin2.fit(X_train, y_train)
        y_pred_lin2 = lin2.predict(X_test)
        rmse_lin2 = np.sqrt(mean_squared_error(y_test, y_pred_lin2))
        print("Linear run", i+1, "RMSE:", rmse_lin2)
        rmse_results.append((f"Linear run {i+1}", rmse_lin2))

        ridge2 = RidgeCV(cv=5)
        ridge2.fit(X_train, y_train)
        y_pred_ridge2 = ridge2.predict(X_test)
        rmse_ridge2 = np.sqrt(mean_squared_error(y_test, y_pred_ridge2))
        print("Ridge run", i+1, "RMSE:", rmse_ridge2)
        rmse_results.append((f"Ridge run {i+1}", rmse_ridge2))

        lasso2 = LassoCV(cv=5)
        lasso2.fit(X_train, y_train)
        y_pred_lasso2 = lasso2.predict(X_test)
        rmse_lasso2 = np.sqrt(mean_squared_error(y_test, y_pred_lasso2))
        print("Lasso run", i+1, "RMSE:", rmse_lasso2)
        rmse_results.append((f"Lasso run {i+1}", rmse_lasso2))
        print()

    print("\n" + "="*80 + "\n")

    lin_vals = [x[1] for x in rmse_results if "Linear" in x[0]]
    ridge_vals = [x[1] for x in rmse_results if "Ridge" in x[0]]
    lasso_vals = [x[1] for x in rmse_results if "Lasso" in x[0]]

    avg_lin = np.mean(lin_vals)
    avg_ridge = np.mean(ridge_vals)
    avg_lasso = np.mean(lasso_vals)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80 + "\n")
    print("avg linear:", avg_lin)
    print("avg ridge:", avg_ridge)
    print("avg lasso:", avg_lasso)
    print()

    if avg_lasso < avg_lin:
        note = "lasso better"
    else:
        note = "lasso not better"

    with open(os.path.join(os.path.dirname(__file__), "rmse_results.txt"), "w") as f:
        for name, val in rmse_results:
            f.write(f"{name}: {val}\n")

        f.write("\n")
        f.write(f"Linear avg: {avg_lin}\n")
        f.write(f"Ridge avg: {avg_ridge}\n")
        f.write(f"Lasso avg: {avg_lasso}\n")
        f.write(note)

    print("saved in rmse_results.txt")


if __name__ == "__main__":
    main()
