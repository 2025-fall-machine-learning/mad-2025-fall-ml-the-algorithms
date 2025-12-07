import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error
import os


def main():

    # load csv
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "gene_expressions.csv"))

    print(df.head())
    print(df.tail())
    print("shape:", df.shape)

    # split X and y
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    print("features:", X.shape[1])
    print("target:", y.name)

    # quick Pearson check
    corrs = X.corrwith(y)
    print("mean corr:", round(corrs.mean(), 4))
    print("max corr:", round(corrs.max(), 4))
    print("min corr:", round(corrs.min(), 4))

    # train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    rmse_results = []

    # linear
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    y_pred_lin = lin.predict(X_test)
    rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))
    print("Linear RMSE:", rmse_lin)
    rmse_results.append(("Linear (genes)", rmse_lin))

    # lasso
    lasso = LassoCV(cv=5, max_iter=10000)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
    print("Lasso RMSE:", rmse_lasso)
    rmse_results.append(("Lasso (genes)", rmse_lasso))

    # ridge
    ridge = RidgeCV(cv=5)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
    print("Ridge RMSE:", rmse_ridge)
    rmse_results.append(("Ridge (genes)", rmse_ridge))

    print("\nSummary:")
    for name, val in rmse_results:
        print(name, "->", round(val, 3))

    # append results to same txt file
    with open(os.path.join(os.path.dirname(__file__), "rmse_results.txt"), "a") as f:
        f.write("\n\n=== PART 2 - gene_expressions ===\n")
        for name, val in rmse_results:
            f.write(f"{name}: {val:.4f}\n")

    print("saved to rmse_results.txt")


if __name__ == "__main__":
    main()
