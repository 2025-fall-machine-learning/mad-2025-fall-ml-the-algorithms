import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import sklearn.model_selection as ms

# ---------- Helper functions ----------
# These helpers keep the two regression workflows compact and consistent:
# - print_1d_data_summary: uniform, readable numeric previews
# - create_linear_regression_model / perform_linear_regression_prediction: one place for fit/predict
# - make_sets: one place to create train/test or "all-as-train"
# - summarize_set: one place to print the same three summaries in the same order
# - plot_*: one place to make consistent plots for simple vs multi-dimensional predictors

def print_1d_data_summary(data_1d, name=None):
    """
    Show a compact preview of a 1D numeric array/Series:
      - Handles both shape (n,) and column vectors of shape (n,1)
      - Prints first 5 and last 5 values to help spot scale, outliers, or NaNs quickly
    """
    arr = np.array(data_1d)                           # accept lists, Series, ndarrays
    if arr.ndim == 2 and arr.shape[1] == 1:           # if it's a column vector, flatten to 1D
        arr = arr.flatten()
    # NOTE: If n < 10, "first 5" and "last 5" overlap—still fine, just a quick peek.
    first = ", ".join(f"{x:7.3f}" for x in arr[:5])   # format for aligned, readable columns
    last  = ", ".join(f"{x:7.3f}" for x in arr[-5:])
    label = f"{name}: " if name else ""
    print(f"{label}[{first}, ..., {last}]")

def create_linear_regression_model(predictors, response):
    """
    Fit a standard OLS linear regression:
      predictors (X): shape (n_samples, n_features)
      response   (y): shape (n_samples,)
    Returns a fitted scikit-learn model with learned coef_ and intercept_.
    """
    model = lm.LinearRegression()
    model.fit(predictors, response)                   # OLS closed-form (or SVD-based) solution
    return model

def perform_linear_regression_prediction(model, predictors):
    """
    Predict using a fitted model:
      predictors (X): shape (n_samples, n_features)
    Returns y_hat (predictions) of shape (n_samples,).
    """
    return model.predict(predictors)

def make_sets(predictors, response, create_testing_set):
    """
    Create train/test splits *consistently* across experiments.
      - If create_testing_set is True, returns a 75/25 split with a fixed random_state for reproducibility.
      - Else, returns all data as the training set and None for the testing set (avoids two code paths downstream).
    Returns:
      X_tr, y_tr, X_te, y_te
    """
    if create_testing_set:
        # NOTE: random_state ensures deterministic splits so plots & metrics are comparable run-to-run.
        return ms.train_test_split(predictors, response, test_size=0.25, random_state=42)
    else:
        return predictors, response, None, None       # "all as train"; testing is disabled downstream

def summarize_set(name, X=None, y_pred=None, y_true=None):
    """
    Print a labeled, uniform summary of whichever components exist for a split:
      - predictors (X)
      - predictions (y_pred)
      - true values (y_true)
    This keeps the main workflows concise and the output standardized.
    """
    print(f"{name}:")
    if X is not None:       print_1d_data_summary(X, "predictors")  # quick sanity check on scales/ranges
    if y_pred is not None:  print_1d_data_summary(y_pred, "prediction")
    if y_true is not None:  print_1d_data_summary(y_true, "response")

def plot_simple_regression(X, y_true, y_pred, xlab, ylab, title):
    """
    Plot for *1D predictor* regression:
      - Scatter of raw (X, y_true)
      - Fitted line overlay y_pred vs X (sorted by X to draw a clean line)
    Sorting by X ensures the line is drawn left→right instead of zigzagging.
    """
    X = np.array(X).reshape(-1)                        # collapse (n,1) to (n,)
    order = np.argsort(X)                              # sort to connect the line monotonically in x
    plt.figure()
    plt.scatter(X, y_true, label="Data", color="blue") # raw points
    plt.plot(X[order], np.array(y_pred)[order],        # fitted line
             label="Best Fit Line", color="red")
    plt.xlabel(xlab); plt.ylabel(ylab)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_pred_vs_true(y_true, y_pred, title):
    """
    Plot for *multi-dimensional predictor* regression:
      - x-axis = true y, y-axis = predicted y_hat
      - dashed y=x reference line (perfect predictions would lie on this)
    This directly visualizes calibration and spread without requiring 1D X.
    """
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.7, color="blue")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "--", color="red", label="Ideal")  # 45° line
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------- Regression workflows ----------
# Each workflow:
#   1) selects columns -> X, y
#   2) creates train/test sets in a uniform way
#   3) fits, predicts, summarizes outputs
#   4) plots using the appropriate visual for the dimensionality

def simple_linear_regression(df, create_testing_set):
    """
    Simple regression: Diam -> Height
    """
    # Select feature and target columns
    features = df[['Diam']].values
    targets = df['Height'].values

    # Train/test split
    train_features, train_targets, test_features, test_targets = make_sets(features, targets, create_testing_set)

    # Fit model on training set
    model = create_linear_regression_model(train_features, train_targets)

    # Predict on training
    train_predictions = perform_linear_regression_prediction(model, train_features)

    # Summaries + plot
    summarize_set("Training", X=train_features, y_pred=train_predictions, y_true=train_targets)
    plot_simple_regression(train_features, train_targets, train_predictions,
                           "Diam", "Height", "Diam vs Height (Training)")

    # Optional: evaluate on test
    if test_features is not None:
        test_predictions = perform_linear_regression_prediction(model, test_features)
        summarize_set("Testing", X=test_features, y_pred=test_predictions, y_true=test_targets)
        plot_simple_regression(test_features, test_targets, test_predictions,
                               "Diam", "Height", "Diam vs Height (Testing)")


def multiple_linear_regression(df, create_testing_set, one_hot_encode):
    """
    Multiple regression: Diam, Height (+ optional Season) -> Volume
    """
    # Build feature matrix
    if one_hot_encode:
        season_dummies = pd.get_dummies(df['Season'], prefix='Season')
        features = pd.concat([df[['Diam', 'Height']], season_dummies], axis=1).values
    else:
        features = df[['Diam', 'Height']].values
    targets = df['Volume'].values

    # Train/test split
    train_features, train_targets, test_features, test_targets = make_sets(features, targets, create_testing_set)

    # Fit model on training set
    model = create_linear_regression_model(train_features, train_targets)

    # Predict on training
    train_predictions = perform_linear_regression_prediction(model, train_features)

    # Summaries + plot
    summarize_set("Training", y_pred=train_predictions, y_true=train_targets)
    plot_pred_vs_true(train_targets, train_predictions, "Volume: True vs Predicted (Training)")

    # Optional: evaluate on test
    if test_features is not None:
        test_predictions = perform_linear_regression_prediction(model, test_features)
        summarize_set("Testing", y_pred=test_predictions, y_true=test_targets)
        plot_pred_vs_true(test_targets, test_predictions, "Volume: True vs Predicted (Testing)")

# ---------- Main ----------
# The entry point wires everything together:
#   - loads the dataset
#   - chooses which experiments to run
#   - leaves toggles commented for quick exploration

def main():
    # Load the cherry tree dataset.
    # TIP: If you ever hit FileNotFoundError, confirm your working directory or use an absolute path.
    cherry_tree_df = pd.read_csv('1_linear_regression/CherryTree.csv')

    # Run the simple regression without a test split by default.
    # Toggle others on as needed for exploration.
    simple_linear_regression(cherry_tree_df, False)
    # simple_linear_regression(cherry_tree_df, True)                 # enable held-out test
    # multiple_linear_regression(cherry_tree_df, False, False)       # multi-D without Season
    # multiple_linear_regression(cherry_tree_df, False, True)        # multi-D with Season one-hot
    # multiple_linear_regression(cherry_tree_df, True, False)        # multi-D + test split

if __name__ == "__main__":
    main()
