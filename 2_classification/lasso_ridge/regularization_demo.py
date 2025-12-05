
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import sklearn.metrics as metrics

def rmse(y_true, y_pred):
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))

def predict(marketing_df):
    # 1) Inspect the data
    print("\n=== Head ===")
    print(marketing_df.head())
    print("\n=== Describe (all columns) ===")
    print(marketing_df.describe(include='all'))
    print("\n=== Columns ===")
    print(marketing_df.columns)

    # 2) Define predictors and response
    assert "LVT" in marketing_df.columns, "Expected a column named 'LVT' as the response."
    response_df = marketing_df[["LVT"]]
    predictors_df = marketing_df.drop(columns=["LVT"])

    # Ensure predictors are numeric (if there are object columns, one-hot encode them)
    if predictors_df.select_dtypes(include=['object']).shape[1] > 0:
        predictors_df = pd.get_dummies(predictors_df, drop_first=True)

    # 3) Split: 30% test, random_state=1
    X_train, X_test, y_train, y_test = ms.train_test_split(
        predictors_df, response_df, test_size=0.3, random_state=1
    )

    # 4) Train models
    # Linear Regression
    lr_algorithm = lm.LinearRegression()
    lr_model = lr_algorithm.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)

    # LassoCV (cv=5); Lasso requires 1D target
    lasso_algorithm = lm.LassoCV(cv=5, max_iter=10000)
    lasso_model = lasso_algorithm.fit(X_train, y_train.values.ravel())
    lasso_pred = lasso_model.predict(X_test)

    # RidgeCV (cv=5)
    ridge_algorithm = lm.RidgeCV(cv=5)
    ridge_model = ridge_algorithm.fit(X_train, y_train.values.ravel())
    ridge_pred = ridge_model.predict(X_test)

    # 5) Report R^2 on training and RMSE on test
    print("\n=== Training R-squared ===")
    print(f"Linear R^2: {lr_model.score(X_train, y_train):.4f}")
    print(f"Lasso  R^2: {lasso_model.score(X_train, y_train):.4f}")
    print(f"Ridge  R^2: {ridge_model.score(X_train, y_train):.4f}")

    print("\n=== LVT.describe() (for RMSE comparison) ===")
    print(response_df.describe())

    lr_rmse = rmse(y_test, lr_pred)
    lasso_rmse = rmse(y_test, lasso_pred)
    ridge_rmse = rmse(y_test, ridge_pred)

    print("\n=== Test RMSEs ===")
    print(f"Linear RMSE: {lr_rmse:,.2f}")
    print(f"Lasso  RMSE: {lasso_rmse:,.2f}")
    print(f"Ridge  RMSE: {ridge_rmse:,.2f}")

    # Show chosen CV hyperparameters
    print("\nLasso alpha (chosen via CV):", lasso_model.alpha_)
    # RidgeCV stores alphas tried; chosen alpha is in ridge_model.alpha_ (for some versions)
    try:
        print("Ridge alphas tried:", ridge_model.alphas)
    except AttributeError:
        pass
    try:
        print("Ridge alpha (chosen via CV):", ridge_model.alpha_)
    except AttributeError:
        pass

    # 6) Rerun without random_state to illustrate variability
    X_train2, X_test2, y_train2, y_test2 = ms.train_test_split(
        predictors_df, response_df, test_size=0.3  # random_state omitted
    )
    lr_model2 = lm.LinearRegression().fit(X_train2, y_train2)
    lasso_model2 = lm.LassoCV(cv=5, max_iter=10000).fit(X_train2, y_train2.values.ravel())
    ridge_model2 = lm.RidgeCV(cv=5).fit(X_train2, y_train2.values.ravel())

    lr_rmse2 = rmse(y_test2, lr_model2.predict(X_test2))
    lasso_rmse2 = rmse(y_test2, lasso_model2.predict(X_test2))
    ridge_rmse2 = rmse(y_test2, ridge_model2.predict(X_test2))

    print("\n=== Rerun (no random_state) RMSEs ===")
    print(f"Linear: {lr_rmse2:,.2f} | Lasso: {lasso_rmse2:,.2f} | Ridge: {ridge_rmse2:,.2f}")

def main():
    filename = "marketing_buckets.csv"
    df = pd.read_csv(filename)
    predict(df)

if __name__ == "__main__":
    main()