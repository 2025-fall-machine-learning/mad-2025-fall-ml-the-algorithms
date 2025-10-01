import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
import numpy as np

# --- Main workflow ---
def main():
    df = pd.read_csv("C:/Users/student/OneDrive - Madison College/Mac-Learning-F25/mad-2025-fall-ml-the-algorithms/1_linear_regression/cars.csv")

    # Show a quick heatmap for numeric columns to guide predictor choice
    try:
        run_heatmap(df, response='price')
    except Exception as e:
        print("Heatmap skipped (error):", e)

    # Simple regression: enginesize vs price
    run_simple_linear_regression(df, predictor="enginesize", response="price")

    # Multiple regression: drop ID & CarName
    run_multiple_linear_regression(
        df,
        response="price",
        drop_cols=["car_ID", "CarName"]
    )

# Simple heatmap helper
def run_heatmap(df, response=None):
    """Plot a simple correlation heatmap for numeric columns and print top correlations to response."""
    num = df.select_dtypes(include=[np.number]).copy()
    if num.shape[1] < 2:
        print("Not enough numeric columns for a heatmap.")
        return

    corr = num.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(corr, cmap='RdBu', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(corr.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    plt.title('Correlation matrix (numeric columns)')
    plt.tight_layout()
    plt.show()

    if response and response in corr.columns:
        print("Top correlations with response:")
        print(corr[response].abs().sort_values(ascending=False).head(10))

# --- Utility functions ---

def run_simple_linear_regression(df, predictor, response):
    # Run simple regression and check linearity & independence.
    X, y = df[[predictor]], df[response]
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    print("Simple Linear Regression:")
    print(f"Predictor: {predictor}, Response: {response}")
    print(f"R^2: {model.score(X, y):.3f}")

    # Linearity: residuals vs fitted
    residuals = np.asarray(y) - np.asarray(y_pred)
    dw = durbin_watson(residuals)
    corr = np.corrcoef(np.ravel(y_pred), np.ravel(residuals))[0, 1]
    print(f"Durbin-Watson (independence): {dw:.3f}")
    print(f"Correlation (fitted vs residuals) — linearity check: {corr:.3f}")

    # Plot fit
    plt.scatter(X, y, color="blue", label="True values")
    plt.plot(np.sort(X.values.ravel()), np.sort(y_pred.ravel()), color="red", linestyle="dotted", label="Fit line")
    plt.xlabel(predictor); plt.ylabel(response)
    plt.title(f"{predictor} vs {response}")
    plt.legend(); plt.show()

    # Residuals vs fitted plot
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Fitted values"); plt.ylabel("Residuals")
    plt.title(f"Residuals vs Fitted ({predictor} -> {response})")
    plt.show()


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

    # Linearity check: predicted vs actual (visual)
    # If there are exactly 2 kept numeric predictors, plot in 3D (scatter + fitted surface)
    if len(kept_features) == 2:
        # attempt 3D scatter + surface using test set predictors
        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            f1, f2 = kept_features[0], kept_features[1]
            X_test_sub = X_test[[f1, f2]].astype(float)
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_test_sub[f1], X_test_sub[f2], y_test, c='b', marker='o', alpha=0.6)

            # grid for surface
            f1_lin = np.linspace(X_test_sub[f1].min(), X_test_sub[f1].max(), 20)
            f2_lin = np.linspace(X_test_sub[f2].min(), X_test_sub[f2].max(), 20)
            F1, F2 = np.meshgrid(f1_lin, f2_lin)
            grid = np.column_stack([F1.ravel(), F2.ravel()])

            # predict on grid if possible
            try:
                Z = model.predict(grid).reshape(F1.shape)
                ax.plot_surface(F1, F2, Z, color='r', alpha=0.3)
            except Exception:
                # model may expect full feature set; skip surface then
                pass

            ax.set_xlabel(f1); ax.set_ylabel(f2); ax.set_zlabel(response)
            plt.title('3D scatter and fitted surface (test set)')
            plt.tight_layout(); plt.show()
        except Exception:
            # fallback to 2D predicted vs actual
            plt.scatter(y_test, y_pred, color="green", alpha=0.6)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            plt.xlabel("Actual Price"); plt.ylabel("Predicted Price")
            plt.title("Predicted vs Actual Price"); plt.show()
    else:
        plt.scatter(y_test, y_pred, color="green", alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel("Actual Price"); plt.ylabel("Predicted Price")
        plt.title("Predicted vs Actual Price"); plt.show()

    # Residuals & independence checks
    residuals = np.asarray(y_test) - np.asarray(y_pred)
    dw = durbin_watson(residuals)
    corr = np.corrcoef(np.ravel(y_pred), np.ravel(residuals))[0, 1]
    print(f"Durbin-Watson (independence): {dw:.3f}")
    print(f"Correlation (fitted vs residuals) — linearity check: {corr:.3f}")

    # Residuals vs fitted plot for multiple regression
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Fitted values"); plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted (Multiple Regression)")
    plt.show()

    return model, X_train.columns

if __name__ == "__main__":
    main()
