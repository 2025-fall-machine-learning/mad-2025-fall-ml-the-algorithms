import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


# Global constant to control reproducibility of train/test splits.
RANDOM_STATE: int = 42


def _canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def find_col(df: pd.DataFrame, candidates: List[str]) -> str:
    canon_map = {_canon(col): col for col in df.columns}
    for name in candidates:
        key = _canon(name)
        if key in canon_map:
            return canon_map[key]
    raise KeyError(f"None of the columns {candidates} were found in {list(df.columns)}")


def print_1d_data_summary(data_1d: np.ndarray) -> None:
    arr = np.asarray(data_1d)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.flatten()
    first_five = ", ".join(f"{x:7.3f}" for x in arr[:5])
    last_five = ", ".join(f"{x:7.3f}" for x in arr[-5:])
    print(f"[{first_five}, ..., {last_five}]")


def print_inspection_data(predictors: np.ndarray | None = None,
                          prediction: np.ndarray | None = None,
                          response: np.ndarray | None = None,
                          prefix: str | None = None) -> None:
    if prefix:
        print(prefix)
    if predictors is not None:
        print("Predictors:")
        print_1d_data_summary(predictors)
    if prediction is not None:
        print("Predictions:")
        print_1d_data_summary(prediction)
    if response is not None:
        print("Actual:")
        print_1d_data_summary(response)


def plot_regression(x: np.ndarray, y: np.ndarray, y_pred: np.ndarray,
                    xlabel: str, ylabel: str, title: str,
                    label_data: str, label_line: str) -> None:
    x_arr = np.asarray(x).flatten()
    y_arr = np.asarray(y).flatten()
    pred_arr = np.asarray(y_pred).flatten()
    # sort by predictor for a smooth line
    order = np.argsort(x_arr)
    plt.scatter(x_arr, y_arr, label=label_data)
    plt.plot(x_arr[order], pred_arr[order], label=label_line)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the coefficient of determination (R²) for a set of predictions."""
    return r2_score(y_true, y_pred)


def compute_and_print_diagnostics(y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  label: str = "") -> None:
    """Print diagnostic metrics for regression performance.

    Parameters
    ----------
    y_true : array-like
        Actual response values.
    y_pred : array-like
        Predicted response values.
    """
    print(f"Diagnostics {label}:")
    print(f"  R^2: {r_squared(y_true, y_pred):.4f}")


def vif_feature_selection(df_numeric: pd.DataFrame,
                          thresh: float = 5.0) -> Tuple[pd.DataFrame, List[str]]:
    """Remove multicollinear features using iterative VIF-based selection.

    This function computes the variance inflation factor (VIF) for each
    numeric predictor and iteratively removes the column with the highest VIF
    exceeding the specified threshold. It also drops zero-variance columns
    and fills missing values with zeros to avoid singular matrices.

    Parameters
    ----------
    df_numeric : pandas.DataFrame
        DataFrame containing only numeric predictor columns.
    thresh : float, default 5.0
        Threshold above which a feature is considered to exhibit high
        multicollinearity and should be removed.

    Returns
    -------
    reduced_df : pandas.DataFrame
        The DataFrame after removing highly collinear features.
    dropped : list of str
        Names of columns that were dropped due to high VIF or zero variance.
    """
    # Work on a copy to avoid modifying the original DataFrame
    X = df_numeric.select_dtypes(include=[np.number]).copy()
    # Replace NaN with 0 to avoid issues in VIF calculations
    X = X.fillna(0)
    dropped: List[str] = []

    # Drop zero variance columns first
    variances = X.var(axis=0)
    zero_var_cols = variances[variances == 0].index.tolist()
    for col in zero_var_cols:
        dropped.append(col)
    X = X.drop(columns=zero_var_cols)

    # Iteratively remove columns with highest VIF above threshold
    while True:
        if X.shape[1] <= 1:
            break  # Cannot compute VIF with fewer than 2 predictors
        X_const = sm.add_constant(X, has_constant='add')
        vif_values = []
        # Skip index 0 (constant) when computing VIF values
        for i in range(1, X_const.shape[1]):
            try:
                vif = variance_inflation_factor(X_const.values, i)
            except Exception:
                vif = np.inf
            vif_values.append(vif)
        max_vif = max(vif_values)
        if max_vif > thresh:
            drop_index = int(np.argmax(vif_values))
            col_to_drop = X.columns[drop_index]
            dropped.append(col_to_drop)
            X = X.drop(columns=[col_to_drop])
        else:
            break
    return X, dropped


def simple_linear_regression_cars(cars_df: pd.DataFrame,
                                  create_testing_set: bool = True) -> None:
    """Perform simple linear regression predicting price from engine size.

    The function locates the ``engine size`` and ``price`` columns using
    flexible matching, fits a linear regression model, and optionally
    evaluates it on a separate testing set. Diagnostic metrics and
    inspection summaries are printed, and a scatter plot with the fitted
    regression line is shown.

    Parameters
    ----------
    cars_df : pandas.DataFrame
        DataFrame containing car features and target prices.
    create_testing_set : bool, default True
        Whether to split the data into training and testing subsets.
    """
    # Locate the predictor and response columns in a flexible way
    eng_col = find_col(cars_df, ['engine-size', 'enginesize', 'engine_size', 'Engine Size'])
    price_col = find_col(cars_df, ['price', 'msrp', 'saleprice', 'Price'])

    X = cars_df[[eng_col]].values
    y = cars_df[price_col].values

    # Split into train/test if requested
    if create_testing_set:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=RANDOM_STATE
        )
    else:
        X_train, y_train = X, y

    # Fit the model on training data
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)

    print("\n=== Simple Linear Regression ===")
    print(f"Using predictor column: '{eng_col}' and response column: '{price_col}'")
    print_inspection_data(predictors=X_train, prediction=y_pred_train, response=y_train,
                          prefix="Training data snapshot:")
    compute_and_print_diagnostics(y_train, y_pred_train, label="(Training)")

    # Plot the training data and fitted line
    plot_regression(
        X_train, y_train, y_pred_train,
        xlabel=eng_col, ylabel=price_col,
        title='Simple Linear Regression (Training Data)',
        label_data='Training Data', label_line='Fitted Line'
    )

    # Evaluate on test data only if a test set was created
    if create_testing_set:
        y_pred_test = model.predict(X_test)
        print("Testing data snapshot:")
        print_inspection_data(predictors=X_test, prediction=y_pred_test, response=y_test)
        compute_and_print_diagnostics(y_test, y_pred_test, label="(Testing)")
        plot_regression(
            X_test, y_test, y_pred_test,
            xlabel=eng_col, ylabel=price_col,
            title='Simple Linear Regression (Testing Data)',
            label_data='Testing Data', label_line='Fitted Line'
        )


def multiple_linear_regression_cars(cars_df: pd.DataFrame,
                                    create_testing_set: bool = True,
                                    one_hot_encode: bool = True,
                                    vif_thresh: float = 5.0) -> None:
    """Perform multiple linear regression predicting price using all features.

    Categorical variables can be one-hot encoded. Multicollinearity is
    mitigated by iteratively dropping predictors with high VIF. Fallbacks
    ensure at least one predictor remains. Results are reported for
    training (and optionally testing) data.

    Parameters
    ----------
    cars_df : pandas.DataFrame
        DataFrame containing car features and target prices.
    create_testing_set : bool, default True
        Whether to split the data into training and testing subsets.
    one_hot_encode : bool, default True
        Whether to apply one-hot encoding to categorical variables.
    vif_thresh : float, default 5.0
        Threshold for the VIF-based feature selection procedure.
    """
    price_col = find_col(cars_df, ['price', 'msrp', 'saleprice', 'Price'])
    predictors_df = cars_df.drop(columns=[price_col])

    # Apply one-hot encoding to categorical features if requested
    if one_hot_encode:
        predictors_df = pd.get_dummies(predictors_df, drop_first=True)
    else:
        predictors_df = predictors_df.select_dtypes(include=[np.number])

    # Feature selection using VIF
    try:
        reduced_df, dropped_cols = vif_feature_selection(predictors_df, thresh=vif_thresh)
    except Exception as exc:
        print(f"VIF selection skipped due to error: {exc}")
        reduced_df = predictors_df
        dropped_cols = []

    # Fallback if all predictors are removed
    if reduced_df.shape[1] == 0:
        print("All predictors were removed by VIF selection. Falling back to simpler predictors.")
        # Attempt to fall back to engine size column if present
        try:
            eng_col = find_col(cars_df, ['engine-size', 'enginesize', 'engine_size', 'Engine Size'])
            reduced_df = cars_df[[eng_col]].select_dtypes(include=[np.number]).copy()
            print(f"Fallback predictor used: {eng_col}")
        except Exception:
            # If engine size isn't available, use the first numeric column in the original DataFrame
            numeric_cols = cars_df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                reduced_df = cars_df[[numeric_cols[0]]].copy()
                print(f"Fallback predictor used: {numeric_cols[0]}")
            else:
                raise RuntimeError("No numeric predictors are available to build a model.")

    # Prepare arrays for regression
    X = reduced_df.values
    y = cars_df[price_col].values

    # Optionally split into training and testing sets
    if create_testing_set:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=RANDOM_STATE
        )
    else:
        X_train, y_train = X, y

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)

    # Reporting
    print("\n=== Multiple Linear Regression ===")
    print(f"Total predictors after encoding: {predictors_df.shape[1]}")
    if dropped_cols:
        print("Features dropped due to high VIF (> {}):".format(vif_thresh), dropped_cols)
    print("Features used in final model:", list(reduced_df.columns))
    compute_and_print_diagnostics(y_train, y_pred_train, label="(Training)")

    # Plot predicted vs actual for training data
    plt.scatter(y_train, y_pred_train, label='Training Data', alpha=0.6)
    min_val = min(y_train.min(), y_pred_train.min())
    max_val = max(y_train.max(), y_pred_train.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', label='Ideal Fit')
    plt.xlabel('Actual price')
    plt.ylabel('Predicted price')
    plt.title('Multiple Regression: Predicted vs Actual (Training Data)')
    plt.legend()
    plt.show()

    # Evaluate on test data if split
    if create_testing_set:
        y_pred_test = model.predict(X_test)
        compute_and_print_diagnostics(y_test, y_pred_test, label="(Testing)")
        plt.scatter(y_test, y_pred_test, label='Testing Data', alpha=0.6)
        min_val = min(y_test.min(), y_pred_test.min())
        max_val = max(y_test.max(), y_pred_test.max())
        plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', label='Ideal Fit')
        plt.xlabel('Actual price')
        plt.ylabel('Predicted price')
        plt.title('Multiple Regression: Predicted vs Actual (Testing Data)')
        plt.legend()
        plt.show()


def main() -> None:
    # Likely locations of the dataset; expand if additional directories are used
    candidates = [
        'cars.csv',
        os.path.join('1_linear_regression', 'cars.csv'),
        os.path.join('data', 'cars.csv'),
        os.path.abspath('cars.csv'),
    ]
    cars_df: pd.DataFrame | None = None
    for path in candidates:
        if os.path.exists(path):
            try:
                cars_df = pd.read_csv(path)
                print(f"Loaded cars.csv from: {path}")
                break
            except Exception:
                # if reading fails, try the next path
                continue
    # If no file was loaded, raise an informative error
    if cars_df is None:
        tried_paths = "\n".join(candidates)
        raise FileNotFoundError(
            f"Could not find 'cars.csv'. Tried the following locations:\n{tried_paths}\n"
            "Please place cars.csv in the project root or specify its path."
        )
    # Preview key columns to reassure the user
    try:
        eng_col = find_col(cars_df, ['engine-size', 'enginesize', 'engine_size', 'Engine Size'])
        price_col = find_col(cars_df, ['price', 'msrp', 'saleprice', 'Price'])
        print("\nPreview of key columns (first 5 rows):")
        print(cars_df[[eng_col, price_col]].head())
    except Exception:
        # If preview fails, silently continue
        pass

    # Run simple and multiple regression analyses
    simple_linear_regression_cars(cars_df, create_testing_set=True)
    multiple_linear_regression_cars(cars_df, create_testing_set=True, one_hot_encode=True, vif_thresh=5.0)


if __name__ == "__main__":
    main()