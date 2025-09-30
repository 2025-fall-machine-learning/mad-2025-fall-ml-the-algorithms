# Helper to robustly find column names
import re
import os
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import math
import sklearn.metrics as metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


def _canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

def find_col(df, candidates):
    canon_map = {_canon(c): c for c in df.columns}
    for name in candidates:
        key = _canon(name)
        if key in canon_map:
            return canon_map[key]
    raise KeyError(f"None of the columns {candidates} were found in {list(df.columns)}")

# Utility to print summary of 1D data
def print_1d_data_summary(data_1d):
    numpified_data = np.array(data_1d)
    if numpified_data.ndim == 2 and numpified_data.shape[1] == 1:
        flattened_numpified_data = numpified_data.flatten()
    else:
        flattened_numpified_data = numpified_data
    first_five = ", ".join(f"{x:7.3f}" for x in flattened_numpified_data[:5])
    last_five = ", ".join(f"{x:7.3f}" for x in flattened_numpified_data[-5:])
    print(f"[{first_five}, ..., {last_five}]")

# Helper to print inspection data
def print_inspection_data(predictors=None, prediction=None, response=None, prefix=None):
    if prefix:
        print(prefix)
    if predictors is not None:
        print_1d_data_summary(predictors)
    if prediction is not None:
        print_1d_data_summary(prediction)
    if response is not None:
        print_1d_data_summary(response)

# Helper to plot regression results
def plot_regression(x, y, y_pred, xlabel, ylabel, title, color_data, color_line, label_data, label_line):
    # ensure 1D arrays for plotting
    x_arr = np.array(x).flatten()
    y_arr = np.array(y).flatten()
    ypred_arr = np.array(y_pred).flatten()
    # truncate to common length
    n = min(len(x_arr), len(y_arr), len(ypred_arr))
    x_arr = x_arr[:n]
    y_arr = y_arr[:n]
    ypred_arr = ypred_arr[:n]
    # scatter and line
    plt.scatter(x_arr, y_arr, color=color_data, label=label_data)
    # sort for a clean line
    order = np.argsort(x_arr)
    plt.plot(x_arr[order], ypred_arr[order], color=color_line, label=label_line)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()


# --- Diagnostics helpers -------------------------------------------------
def rmse(y_true, y_pred):
    return math.sqrt(metrics.mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return metrics.mean_absolute_error(y_true, y_pred)

def r_squared(y_true, y_pred):
    return metrics.r2_score(y_true, y_pred)

def resid_pred_correlation(y_true, y_pred):
    resid = np.array(y_true).flatten() - np.array(y_pred).flatten()
    pred = np.array(y_pred).flatten()
    if len(resid) < 2:
        return float('nan')
    return float(np.corrcoef(pred, resid)[0,1])

def durbin_watson(y_true, y_pred):
    resid = np.array(y_true).flatten() - np.array(y_pred).flatten()
    # DW = sum((e_t - e_{t-1})^2) / sum(e_t^2)
    if len(resid) < 2:
        return float('nan')
    diff = np.diff(resid)
    return float((diff**2).sum() / (resid**2).sum())


def compute_and_print_diagnostics(y_true, y_pred, label=""):
    print(f"Diagnostics {label}:")
    print(f"  R^2 : {r_squared(y_true, y_pred):.4f}")
    print(f"  RMSE: {rmse(y_true, y_pred):.2f}")
    print(f"  MAE : {mae(y_true, y_pred):.2f}")
    print(f"  resid vs pred corr (linearity): {resid_pred_correlation(y_true, y_pred):.4f}")
    print(f"  Durbin-Watson (independence): {durbin_watson(y_true, y_pred):.4f}")


def vif_feature_selection(df_numeric, thresh=5.0):
    """Simple VIF-based feature selection.
    df_numeric: DataFrame of numeric predictors (not numpy array).
    Returns: DataFrame with selected columns (subset of input) and list of dropped columns.
    """
    # work on a copy and ensure numeric columns only
    X = df_numeric.copy()
    X = X.select_dtypes(include=[np.number]).copy()
    # fill NA with zeros to avoid singular matrices caused by NA
    X = X.fillna(0)
    dropped = []
    # drop zero-variance columns first
    variances = X.var(axis=0)
    zero_var_cols = variances[variances == 0].index.tolist()
    for c in zero_var_cols:
        dropped.append(c)
        print(f"Dropping '{c}' because it has zero variance")
    X = X.drop(columns=zero_var_cols)

    while True:
        # stop if no features or only one feature left
        if X.shape[1] <= 1:
            break
        vif = []
        # add constant for VIF calculation
        X_const = sm.add_constant(X)
        for i in range(1, X_const.shape[1]):
            try:
                v = variance_inflation_factor(X_const.values, i)
            except Exception:
                v = float('inf')
            vif.append(v)
        max_vif = max(vif)
        if max_vif > thresh:
            drop_ix = int(np.argmax(vif))
            col_to_drop = X.columns[drop_ix]
            dropped.append(col_to_drop)
            print(f"Dropping '{col_to_drop}' with VIF={max_vif:.2f}")
            X = X.drop(columns=[col_to_drop])
        else:
            break
    return X, dropped

# Simple Linear Regression: engine size vs price
def simple_linear_regression_cars(cars_df, create_testing_set):
    eng_col = find_col(cars_df, ['engine-size', 'enginesize', 'engine_size', 'Engine Size'])
    price_col = find_col(cars_df, ['price', 'msrp', 'saleprice', 'Price'])
    predictors = cars_df[[eng_col]].values
    response = cars_df[price_col].values

    training_predictors = predictors
    training_response = response

    if create_testing_set:
        training_predictors, testing_predictors, training_response, testing_response = ms.train_test_split(
            predictors, response, test_size=0.25, random_state=42)

    model = lm.LinearRegression()
    model.fit(training_predictors, training_response)
    prediction = model.predict(training_predictors)
    print("Simple Linear Regression: engine-size vs price (Training Data)")
    print_inspection_data(training_predictors, prediction, training_response)
    compute_and_print_diagnostics(training_response, prediction, label="(Training)")
    plot_regression(training_predictors, training_response, prediction,
                   xlabel='engine-size', ylabel='price',
                   title='Linear Regression: engine-size vs price (Training Data)',
                   color_data='blue', color_line='red',
                   label_data='Training Data', label_line='Best Fit Line')

    if create_testing_set:
        prediction = model.predict(testing_predictors)
        print("Simple Linear Regression: engine-size vs price (Testing Data)")
        print_inspection_data(testing_predictors, prediction, testing_response)
    compute_and_print_diagnostics(testing_response, prediction, label="(Testing)")
    plot_regression(testing_predictors, testing_response, prediction,
               xlabel='engine-size', ylabel='price',
               title='Linear Regression: engine-size vs price (Testing Data)',
               color_data='green', color_line='red',
               label_data='Testing Data', label_line='Best Fit Line')

# Multiple Linear Regression: use all numeric features + one-hot encoding for categoricals
def multiple_linear_regression_cars(cars_df, create_testing_set, one_hot_encode=True):
    price_col = find_col(cars_df, ['price', 'msrp', 'saleprice', 'Price'])
    predictors = cars_df.drop(columns=[price_col])
    response = cars_df[price_col].values

    # One-hot encode categorical columns if requested
    if one_hot_encode:
        predictors = pd.get_dummies(predictors, drop_first=True)
    else:
        predictors = predictors.select_dtypes(include=[np.number])

    # Apply VIF-based feature selection on the numeric DataFrame
    try:
        reduced_predictors_df, dropped = vif_feature_selection(predictors)
        if dropped:
            print("VIF-based feature selection dropped:", dropped)
        predictors = reduced_predictors_df
        # if VIF selection removed all columns, fall back to a minimal numeric set
        if predictors.shape[1] == 0:
            print("VIF selection removed all features; falling back to a minimal numeric set.")
            # prefer engine size if available
            try:
                eng = find_col(df=cars_df, candidates=['engine-size', 'enginesize', 'engine_size', 'Engine Size'])
                predictors = cars_df[[eng]].select_dtypes(include=[np.number]).copy()
                print(f"Falling back to predictor: {eng}")
            except Exception:
                # last resort: pick first numeric column
                nums = cars_df.select_dtypes(include=[np.number]).columns.tolist()
                if nums:
                    predictors = cars_df[[nums[0]]].copy()
                    print(f"Falling back to predictor: {nums[0]}")
                else:
                    raise RuntimeError("No numeric predictors available after VIF fallback.")
    except Exception as e:
        # If statsmodels or VIF calc fails, continue with full set but warn
        print("VIF selection skipped due to error:", e)

    predictors = predictors.values

    training_predictors = predictors
    training_response = response

    if create_testing_set:
        training_predictors, testing_predictors, training_response, testing_response = ms.train_test_split(
            predictors, response, test_size=0.25, random_state=42)

    model = lm.LinearRegression()
    model.fit(training_predictors, training_response)
    prediction = model.predict(training_predictors)

    print("Multiple Linear Regression: all features (Training Data)")
    print_inspection_data(prediction=prediction, response=training_response)
    # For multiple regression, plot prediction vs actual
    plt.scatter(training_response, prediction, color='blue', label='Training Data')
    plt.plot([min(training_response), max(training_response)], [min(training_response), max(training_response)], color='red', label='Ideal Fit')
    plt.xlabel('Actual price')
    plt.ylabel('Predicted price')
    plt.title('Multiple Regression: Predicted vs Actual (Training Data)')
    plt.legend()
    plt.show()

    if create_testing_set:
        prediction = model.predict(testing_predictors)
        print("Multiple Linear Regression: all features (Testing Data)")
        print_inspection_data(prediction=prediction, response=testing_response)
        plt.scatter(testing_response, prediction, color='green', label='Testing Data')
        plt.plot([min(testing_response), max(testing_response)], [min(testing_response), max(testing_response)], color='red', label='Ideal Fit')
        plt.xlabel('Actual price')
        plt.ylabel('Predicted price')
        plt.title('Multiple Regression: Predicted vs Actual (Testing Data)')
        plt.legend()
        plt.show()

# Main function
def main():
    # Try several likely locations for cars.csv to avoid FileNotFoundError when running from
    # different working directories.
    candidates = [
        'cars.csv',
        os.path.join('1_linear_regression', 'cars.csv'),
        os.path.join('data', 'cars.csv'),
        os.path.join('1_linear_regression', 'cars.csv'),
        os.path.abspath('cars.csv')
    ]
    cars_df = None
    for p in candidates:
        try:
            if os.path.exists(p):
                cars_df = pd.read_csv(p)
                print(f"Loaded cars.csv from: {p}")
                break
        except Exception:
            # ignore and try next
            pass
    if cars_df is None:
        tried = '\n'.join(candidates)
        raise FileNotFoundError(f"Could not find 'cars.csv'. Tried:\n{tried}\nPlease place cars.csv in the project root or specify its path.")
    simple_linear_regression_cars(cars_df, create_testing_set=True)
    multiple_linear_regression_cars(cars_df, create_testing_set=True, one_hot_encode=True)

if __name__ == "__main__":
    main()
