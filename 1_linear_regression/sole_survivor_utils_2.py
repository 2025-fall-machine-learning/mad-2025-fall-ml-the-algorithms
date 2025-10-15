# sole_survivor_utils.py
# Utilities for analyzing Sole Survivor data and training a regression model
# to evaluate expert initial ratings and predict overall survival scores.

from __future__ import annotations

# -----------------------------
# Imports
# -----------------------------
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import (
    SelectKBest,
    f_regression,
    mutual_info_regression,
    RFE,
    SelectFromModel,
)

# -----------------------------
# Small helpers
# -----------------------------

# Normalize headers so downstream code can rely on consistent names (strip, lowercase, remove spaces/hyphens).
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace("-", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    return df

# Compute Root Mean Squared Error; handy for regression evaluation alongside R^2.
def rmse(y_true, y_pred) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)

# -----------------------------
# Feature diagnostics
# -----------------------------

# Split columns into numeric vs categorical (excluding the target) for encoding strategy.
def detect_feature_types(df: pd.DataFrame, target: str):
    numeric_feats, categorical_feats = [], []
    for c in df.columns:
        if c == target:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_feats.append(c)
        elif (
            pd.api.types.is_bool_dtype(df[c])
            or pd.api.types.is_categorical_dtype(df[c])
            or df[c].dtype == object
        ):
            categorical_feats.append(c)
    return numeric_feats, categorical_feats

# Compute Pearson correlations of numeric features vs the target and sort by |r|.
def correlations_with_target(df: pd.DataFrame, target: str, features: list[str]) -> pd.Series:
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        return pd.Series(dtype=float)
    return (
        df[numeric + [target]]
        .corr()[target]
        .drop(target)
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )

# Build a numeric predictor↔predictor correlation matrix to check multicollinearity.
def predictor_independence_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        return pd.DataFrame()
    return df[numeric].corr()

# List pairs of numeric predictors whose absolute correlation exceeds a threshold.
def find_high_corr_pairs(corr_mat: pd.DataFrame, threshold: float = 0.8):
    pairs = []
    if corr_mat.empty:
        return pairs
    cols = corr_mat.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            val = float(corr_mat.loc[a, b])
            if abs(val) > threshold:
                pairs.append((a, b, val))
    return pairs

# -----------------------------
# Plotting helpers
# -----------------------------

# Visualize predictor↔predictor correlations (upper triangle) to spot clusters/collinearity.
def plot_corr_heatmap(corr_mat: pd.DataFrame, title: str):
    if corr_mat.empty:
        print("No numeric predictors available for independence heatmap.")
        return
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        corr_mat, mask=mask, annot=True, fmt=".2f",
        vmin=-1, vmax=1, center=0, cmap="coolwarm"
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Show predicted vs actual with a 45° line to assess calibration and fit quality.
def plot_pred_vs_actual(y_true, y_pred, title: str, target_name: str = "Survival Score"):
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    plt.figure(figsize=(6, 4))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors="w", linewidth=0.5)
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="gray")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.title(title)
    plt.xlabel(f"Actual {target_name}")
    plt.ylabel(f"Predicted {target_name}")
    plt.tight_layout()
    plt.show()

# Plot residuals vs predicted to check heteroscedasticity and model misspecification.
def plot_residuals(y_true, y_pred, title: str, target_name: str = "Survival Score"):
    resid = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, resid, alpha=0.6, edgecolors="w", linewidth=0.5)
    plt.axhline(0, linestyle="--", color="gray")
    plt.title(title)
    plt.xlabel(f"Predicted {target_name}")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    plt.show()

# -----------------------------
# Modeling pipeline
# -----------------------------

# Hold all training outputs: model, metrics, features, predictions, and selector info.
@dataclass
class ModelArtifacts:
    model: LinearRegression
    train_metrics: dict
    test_metrics: dict
    feature_columns: list            # post-encoding columns (pre-selection)
    y_train: np.ndarray
    y_train_pred: np.ndarray
    y_test: np.ndarray
    y_test_pred: np.ndarray
    selector: object | None = None   # fitted selector (if any)
    selected_features: list | None = None  # selected column names (if available)

# Build model-ready X (numeric + one-hot categoricals) and numeric y; drop ID-like categoricals to avoid leakage.
def build_design_matrices(
    df: pd.DataFrame,
    target: str,
    numeric_feats: list[str] | None = None,
    categorical_feats: list[str] | None = None
):
    if numeric_feats is None or categorical_feats is None:
        auto_num, auto_cat = detect_feature_types(df, target)
        if numeric_feats is None:
            numeric_feats = auto_num
        if categorical_feats is None:
            categorical_feats = auto_cat

    safe_cat_feats = [c for c in categorical_feats if c.lower() not in ("name", "id", "player", "contestant")]

    dropped = set(categorical_feats) - set(safe_cat_feats)
    if dropped:
        print(f"⚠️ Dropped identifier-like categorical columns: {list(dropped)}")

    cols = [c for c in (numeric_feats + safe_cat_feats + [target]) if c in df.columns]
    data = df[cols].copy()

    X = pd.get_dummies(
        data[numeric_feats + safe_cat_feats],
        columns=safe_cat_feats, drop_first=True, dtype=int
    )
    y = pd.to_numeric(data[target], errors="coerce")

    return X, y, numeric_feats, safe_cat_feats

# Build an unfitted selector to reduce features; fit on TRAIN ONLY in the modeling step.
def make_feature_selector(method: str = "kbest", k: int = 10):
    m = method.lower()
    if m == "kbest":
        return SelectKBest(score_func=f_regression, k=k)
    if m == "kbest_mi":
        return SelectKBest(score_func=mutual_info_regression, k=k)
    if m == "rfe":
        return RFE(estimator=LinearRegression(), n_features_to_select=k, step=1)
    if m == "lasso_sfmodel":
        return SelectFromModel(estimator=LassoCV(cv=5, random_state=99), threshold="median")
    raise ValueError(f"Unknown feature selection method: {method}")

# Split, (optionally) select features on TRAIN ONLY, fit LinearRegression, and compute metrics.
def fit_evaluate_linear_regression(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    random_state: int = 99,
    selector=None,
) -> ModelArtifacts:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    selected_columns = None
    if selector is not None:
        selector.fit(X_train, y_train)          # fit on TRAIN ONLY to prevent leakage
        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)
        try:
            support = selector.get_support()
            selected_columns = X.columns[support].tolist()
        except Exception:
            selected_columns = None

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    return ModelArtifacts(
        model=model,
        train_metrics={"r2": float(r2_score(y_train, y_pred_train)), "rmse": rmse(y_train, y_pred_train)},
        test_metrics={"r2": float(r2_score(y_test, y_pred_test)), "rmse": rmse(y_test, y_pred_test)},
        feature_columns=X.columns.tolist() if hasattr(X, "columns") else [],
        y_train=y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.asarray(y_train),
        y_train_pred=y_pred_train,
        y_test=y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.asarray(y_test),
        y_test_pred=y_pred_test,
        selector=selector,
        selected_features=selected_columns,
    )

# Run K-fold cross-validation (leakage-safe via a Pipeline that includes the selector).
def cross_validate_linear(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv_folds: int = 5,
    selector=None,
    random_state: int = 99,
) -> dict:
    steps = []
    if selector is not None:
        steps.append(("selector", selector))  # selector is fit inside each fold on fold-train only
    steps.append(("lr", LinearRegression()))
    pipe = Pipeline(steps)

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    r2_scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
    mse_scores = cross_val_score(pipe, X, y, cv=cv, scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(-mse_scores)

    return {
        "r2_mean": float(r2_scores.mean()),
        "r2_std": float(r2_scores.std()),
        "rmse_mean": float(rmse_scores.mean()),
        "rmse_std": float(rmse_scores.std()),
    }

# Apply the same preprocessing/column alignment (and selector) to next-season data and predict scores.
def align_and_predict_next(
    model: LinearRegression,
    train_X_cols: list[str],
    next_df: pd.DataFrame,
    numeric_feats: list[str],
    categorical_feats: list[str],
    selector=None,
) -> pd.DataFrame:
    next_df = next_df.copy()
    available_numeric = [c for c in numeric_feats if c in next_df.columns]
    available_categorical = [c for c in categorical_feats if c in next_df.columns]

    if not available_numeric and not available_categorical:
        X_next_aligned = pd.DataFrame(0, index=next_df.index, columns=train_X_cols)
    else:
        X_next = pd.get_dummies(
            next_df[available_numeric + available_categorical],
            columns=available_categorical, drop_first=True, dtype=int
        )
        X_next_aligned = X_next.reindex(columns=train_X_cols, fill_value=0)

    if selector is not None:
        try:
            X_next_aligned = selector.transform(X_next_aligned)
        except Exception as e:
            print(f"⚠️ Feature selector transform failed on next data: {e}. Skipping selector.")

    try:
        preds = model.predict(X_next_aligned)
    except Exception as e:
        raise RuntimeError(
            f"Prediction failed. X_next_aligned shape={getattr(X_next_aligned, 'shape', None)}"
        ) from e

    out = next_df.copy()
    out["predicted_survivalscore"] = preds
    return out