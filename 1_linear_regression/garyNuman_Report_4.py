"""
cars.csv — Price Prediction Analysis

This script demonstrates:
  1) Simple Linear Regression (engine size → price)
  2) Multiple Linear Regression (numeric + one-hot encoded categorical features → price)

Outputs:
  - R² (proportion of variance in the dependent variable)
  - Root Mean Squared Error metrics for both models
  - Simple LR: separate TRAIN and TEST scatter+regression-line plots
  - Multiple LR: separate TRAIN and TEST Predicted vs Actual plots
"""

# =======================
# 1. Import libraries
# =======================
# Brings in data tools (pandas/numpy), plotting (matplotlib/seaborn),
# and modeling utilities (scikit-learn). The warnings line silences a minor sklearn message.
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore", message="X has feature names")

# -----------------------
# Small plotting helpers
# -----------------------
# Helper for Simple LR: given a subset (TRAIN/TEST), draw scatter of (enginesize, price)
# and overlay the fitted regression line predicted by the provided model.
def plot_simple_lr(ax, X_subset, y_subset, model, title):
    """Scatter of (engine size, price) with fitted regression line over the subset range."""
    tmp = pd.DataFrame({"enginesize": X_subset["enginesize"], "price": y_subset}).sort_values("enginesize")
    ax.scatter(tmp["enginesize"], tmp["price"], alpha=0.6)
    ax.plot(tmp["enginesize"], model.predict(tmp[["enginesize"]]), linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Engine Size")
    ax.set_ylabel("Price")

# Helper for Multiple LR: Predicted vs Actual plot with a 45° reference line.
# This helps you visually assess calibration (perfect predictions lie on the dashed line).
def plot_pred_vs_actual(ax, y_true, y_pred, title):
    """Predicted vs Actual with 45° reference line."""
    ax.scatter(y_true, y_pred, alpha=0.6)
    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(title)
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")

# =======================
# 2. Load and clean data
# =======================
# Reads the CSV, normalizes column names (lowercase, no spaces/dashes),
# coerces key numeric columns, and drops rows lacking required fields for Simple LR.
# After this block, df is ready for modeling steps below.
# (Minor robustness) Use a raw string for Windows path backslashes
df = pd.read_csv(r"1_linear_regression\cars.csv")

# Standardize column names
df.columns = df.columns.str.lower().str.replace("-", "", regex=False).str.replace(" ", "", regex=False)

# Coerce key numeric columns
for col in ["price", "enginesize", "horsepower", "curbweight", "highwaympg"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Require price and enginesize for simple model
df = df.dropna(subset=["price", "enginesize"])

# ===================================================
# 3. SIMPLE LINEAR REGRESSION (engine size → price)
# ===================================================
# Splits data into TRAIN/TEST, fits a one-feature LinearRegression on TRAIN,
# evaluates on both splits (R² and RMSE), and then shows two plots:
# (1) TRAIN scatter + fitted line, (2) TEST scatter + fitted line.
X = df[["enginesize"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=99
)

linreg = LinearRegression()
linreg.fit(X_train, y_train)

# Metrics (now for both TRAIN and TEST)
# These four values let you compare fit quality within sample (TRAIN) vs generalization (TEST).
y_pred_train = linreg.predict(X_train)
y_pred_test  = linreg.predict(X_test)

r2_simple_train  = r2_score(y_train, y_pred_train)
rmse_simple_train = mean_squared_error(y_train, y_pred_train) ** 0.5
r2_simple_test   = r2_score(y_test, y_pred_test)
rmse_simple_test  = mean_squared_error(y_test, y_pred_test) ** 0.5

# Console summary for Simple LR: shows R² (higher is better) and RMSE (lower is better) for each split.
print("\n=== Simple Linear Regression (Engine Size → Price) ===")
print(f"TRAIN  R2: {r2_simple_train:.3f} | RMSE: ${rmse_simple_train:,.2f}")
print(f"TEST   R2: {r2_simple_test:.3f}  | RMSE: ${rmse_simple_test:,.2f}")

# --- Plots: TRAIN then TEST ---
# Expect two figures: TRAIN plot, then TEST plot. Each shows the same fitted line,
# but drawn over the respective subset’s x-range.
fig, ax = plt.subplots(figsize=(6, 4))
plot_simple_lr(ax, X_train, y_train, linreg, "Simple LR — TRAIN: Engine Size vs Price")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(6, 4))
plot_simple_lr(ax, X_test, y_test, linreg, "Simple LR — TEST: Engine Size vs Price")
plt.tight_layout()
plt.show()

# ============================================================
# 4. MULTIPLE LINEAR REGRESSION (numeric + categorical → price)
# ============================================================
# Constructs a richer feature set: numeric predictors (selected below)
# plus one-hot encoded categoricals (created later in 4B).
numeric_feats = ["enginesize", "horsepower", "curbweight", "highwaympg"]
numeric_feats = [c for c in numeric_feats if c in df.columns]

if not numeric_feats:
    # If no numeric predictors survive the presence check, skip the assumption checks (linearity/independence).
    print("\nNo numeric predictors available after filtering; skipping 4A checks.")
else:
    # -------------------------------
    # 4A. FEATURE SELECTION & CHECKS
    # -------------------------------
    # Linearity check: compute Pearson r(feature, price) and separate into “strong” vs “weak”
    # relative to the threshold. This informs which numeric features to keep.
    # Linearity: Pearson r (feature vs price)
    linearity_threshold = 0.30
    correlations_with_price = (
        df[numeric_feats + ["price"]]
        .corr()["price"]
        .drop("price")
        .sort_values(ascending=False)
    )

    print("\n=== Linearity Check: Pearson Correlation with Price ===")
    print(correlations_with_price)

    strong_predictors = correlations_with_price[correlations_with_price.abs() >= linearity_threshold].index.tolist()
    weak_predictors = correlations_with_price[correlations_with_price.abs() < linearity_threshold].index.tolist()

    # The next two prints tell you which features are recommended to keep/drop for linearity.
    print(f"\nPredictors meeting linearity threshold (|r| ≥ {linearity_threshold}): {strong_predictors}")
    print(f"Predictors below threshold (|r| < {linearity_threshold}): {weak_predictors}")

    if strong_predictors:
        # If at least one strong predictor exists, we retain only those to strengthen the model’s linear signal.
        numeric_feats = strong_predictors
    else:
        # If none are strong, keep the originals so the rest of the pipeline still runs.
        print("No predictors meet the linearity threshold; retaining original numeric features to proceed.")

    # Independence check: compute predictor–predictor correlation matrix and visualize as a heatmap.
    # Look for |r| > 0.80 pairs (potential redundancy), which the following block prints explicitly.
    # Independence: predictor–predictor correlation matrix + heatmap
    corr_pred = df[numeric_feats].corr()
    print("\n=== Predictor–Predictor Correlation Matrix ===")
    print(corr_pred)

    mask = np.triu(np.ones_like(corr_pred, dtype=bool))
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        corr_pred, mask=mask, annot=True, fmt=".2f",
        vmin=-1, vmax=1, center=0, cmap="coolwarm"
    )
    plt.title("Predictor–Predictor Correlation (Independence)")
    plt.tight_layout()
    plt.show()

    # Identify highly correlated pairs for potential manual removal (you choose which to drop).
    # A common policy is to keep the one more correlated with price, but no automatic drop is required.
    # Identify highly correlated pairs
    high_corr_pairs = [
        (c1, c2) for c1 in corr_pred.columns for c2 in corr_pred.columns
        if c1 < c2 and abs(corr_pred.loc[c1, c2]) > 0.8
    ]

    if high_corr_pairs:
        print("\nHighly correlated predictor pairs (|r| > 0.80):")
        for pair in high_corr_pairs:
            print(f"  {pair[0]} ↔ {pair[1]} (r = {corr_pred.loc[pair[0], pair[1]]:.2f})")
        print("Consider removing one predictor from each pair to improve independence.")

        # Optional auto-drop logic (keeps the stronger-to-price predictor) is included here.
        # If enabled by your data, the list 'numeric_feats' will be reduced accordingly.
        # Optional: auto-drop the weaker-to-price feature of each high-corr pair
        r_to_price_abs = correlations_with_price.abs()
        to_drop = set()
        for a, b in high_corr_pairs:
            a_r = r_to_price_abs.get(a, 0.0)
            b_r = r_to_price_abs.get(b, 0.0)
            drop = a if a_r < b_r else b
            to_drop.add(drop)

        if to_drop:
            numeric_feats = [c for c in numeric_feats if c not in to_drop]
            print(f"Dropping (independence step; kept stronger-to-price feature): {sorted(to_drop)}")

        # If everything ended up dropped (edge case), we keep the single strongest predictor to continue.
        if not numeric_feats:
            best = correlations_with_price.abs().idxmax()
            numeric_feats = [best]
            print(f"No predictors left after independence drop; keeping strongest predictor: {best}")
    else:
        print("\nNo highly correlated predictor pairs found — predictors appear independent.")

# -------------------------------
# 4B. MODELING (after checks)
# -------------------------------
# One-hot encodes categoricals, assembles the modeling DataFrame (X with numeric+dummies, y=price),
# splits into TRAIN/TEST, fits a LinearRegression, prints metrics for both splits,
# and then shows Predicted vs Actual plots (TRAIN, then TEST).
categorical_feats = ["fueltype", "aspiration", "doornumber", "drivewheels", "bodystyle"]
categorical_feats = [c for c in categorical_feats if c in df.columns]

df_encoded = pd.get_dummies(
    df[numeric_feats + categorical_feats + ["price"]],
    columns=categorical_feats,
    drop_first=True,
    dtype=int
)

X_multi = df_encoded.drop("price", axis=1)
y_multi = df_encoded["price"]

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.25, random_state=99
)

multi_reg = LinearRegression()
multi_reg.fit(X_train_m, y_train_m)

# Metrics for TRAIN & TEST
# Again, you get R² (explained variance) and RMSE (average absolute error size),
# now for the multivariate model that includes numeric and one-hot features.
y_pred_train_m = multi_reg.predict(X_train_m)
y_pred_test_m  = multi_reg.predict(X_test_m)

r2_multi_train  = r2_score(y_train_m, y_pred_train_m)
rmse_multi_train = mean_squared_error(y_train_m, y_pred_train_m) ** 0.5
r2_multi_test   = r2_score(y_test_m, y_pred_test_m)
rmse_multi_test  = mean_squared_error(y_test_m, y_pred_test_m) ** 0.5

# Console summary for Multiple LR: compare TRAIN vs TEST to assess generalization.
print("\n=== Multiple Linear Regression (Numeric + One-Hot Encoded Categorical) ===")
print(f"TRAIN  R2: {r2_multi_train:.3f} | RMSE: ${rmse_multi_train:,.2f}")
print(f"TEST   R2: {r2_multi_test:.3f}  | RMSE: ${rmse_multi_test:,.2f}")

# --- Plots: TRAIN then TEST (Predicted vs Actual) ---
# Expect two figures: TRAIN Predicted vs Actual, then TEST Predicted vs Actual.
# Points close to the dashed line indicate accurate predictions.
fig, ax = plt.subplots(figsize=(6, 4))
plot_pred_vs_actual(ax, y_train_m, y_pred_train_m, "Multiple LR — TRAIN: Predicted vs Actual")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(6, 4))
plot_pred_vs_actual(ax, y_test_m, y_pred_test_m, "Multiple LR — TEST: Predicted vs Actual")
plt.tight_layout()
plt.show()