"""
Diabetes classifier with proper preprocessing, SMOTE for class imbalance,
and rich evaluation (confusion matrix, PR/F1, ROC-AUC, ROC curve).

Run:
    python train_diabetes.py
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# NEW: use imbalanced-learn pipeline + SMOTE
from imblearn.pipeline import Pipeline   # NOTE: from imblearn, not sklearn
from imblearn.over_sampling import SMOTE

# Pretty heatmap optional
USE_SEABORN_CM = True
try:
    import seaborn as sns  # type: ignore
except Exception:
    USE_SEABORN_CM = False


# ---- Toggles you can tweak ----
USE_SMOTE = True                 # Turn SMOTE on/off easily
USE_CLASS_WEIGHT = False         # You can set this True to also use class_weight='balanced'
SMOTE_STRATEGY = "auto"          # e.g., 0.6, 1.0, "minority", "not minority", "auto"
SMOTE_RANDOM_STATE = 489


# ---- Custom transformer to convert zeros -> NaN on specific numeric columns ----
class ZeroToNaN(BaseEstimator, TransformerMixin):
    """Replace 0 with NaN for specified columns (common in Pima-like datasets)."""
    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.columns:
            if c in X.columns:
                X[c] = X[c].replace(0, np.nan)
        return X


def load_data(csv_path: str | os.PathLike) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Outcome" not in df.columns:
        raise ValueError("CSV must contain an 'Outcome' column (0/1).")
    return df


def build_pipeline(feature_df: pd.DataFrame) -> Pipeline:
    """Build a full pipeline: Zero->NaN -> Preprocess -> (SMOTE) -> Classifier."""
    # Categorical columns in your dataset (keep them if present)
    cat_cols = [c for c in ["DNR Order", "Med Tech"] if c in feature_df.columns]
    # Numeric = everything else (except cat)
    num_cols = [c for c in feature_df.columns if c not in cat_cols]

    # In Pima-like datasets, zeros are invalid for these numeric columns:
    zero_invalid_candidates = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    zero_invalid = [c for c in zero_invalid_candidates if c in feature_df.columns]

    # Preprocess: impute + scale numeric; impute + one-hot categorical
    num_branch = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_branch = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(transformers=[
        ("num", num_branch, num_cols),
        ("cat", cat_branch, cat_cols),
    ])

    # Classifier
    clf = LogisticRegression(
        max_iter=5000,
        C=1.0,
        solver="lbfgs",
        # Either rely on SMOTE only (recommended) or combine with class_weight:
        class_weight="balanced" if USE_CLASS_WEIGHT else None,
        n_jobs=None,
        random_state=489
    )

    # Build the imblearn Pipeline so SMOTE occurs *after* preprocessing, *before* classifier
    steps = [
        ("zero_to_nan", ZeroToNaN(columns=zero_invalid)),
        ("pre", pre),
    ]

    if USE_SMOTE:
        steps.append(("smote", SMOTE(sampling_strategy=SMOTE_STRATEGY, random_state=SMOTE_RANDOM_STATE)))

    steps.append(("clf", clf))

    pipe = Pipeline(steps=steps)
    return pipe


def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix") -> None:
    """Pretty confusion matrix with labels & percentages."""
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()
    labels = np.array([
        [f"TN\n{tn}\n{tn/total:.1%}", f"FP\n{fp}\n{fp/total:.1%}"],
        [f"FN\n{fn}\n{fn/total:.1%}", f"TP\n{tp}\n{tp/total:.1%}"],
    ])

    if USE_SEABORN_CM:
        sns.heatmap(
            cm, annot=labels, fmt="", cmap="Blues", cbar=False,
            xticklabels=["Pred: No Diabetes", "Pred: Diabetes"],
            yticklabels=["Actual: No Diabetes", "Actual: Diabetes"],
            linewidths=1, linecolor="white"
        )
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()
    else:
        fig, ax = plt.subplots()
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Pred: No", "Pred: Yes"])
        ax.set_yticklabels(["Actual: No", "Actual: Yes"])
        ax.set_title(title)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for (i, j), val in np.ndenumerate(labels):
            ax.text(j, i, val, ha="center", va="center")
        plt.tight_layout()
        plt.show()


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, *, threshold: float | None = None) -> None:
    """Evaluate with confusion matrix, PR/F1, ROC-AUC, ROC curve."""
    probs = model.predict_proba(X_test)[:, 1]
    y_pred_default = model.predict(X_test)

    if threshold is not None:
        y_pred = (probs >= threshold).astype(int)
        header = f"Evaluation @ custom threshold = {threshold:.2f}"
    else:
        y_pred = y_pred_default
        header = "Evaluation @ default threshold = 0.50"

    print("\n" + "="*len(header))
    print(header)
    print("="*len(header))

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, digits=3))

    cm = confusion_matrix(y_test, y_pred)
    print("\n--- Confusion Matrix (TN FP / FN TP) ---")
    print(cm)
    plot_confusion_matrix(cm, title="Confusion Matrix")

    auc = roc_auc_score(y_test, probs)
    print(f"\nROC-AUC: {auc:.3f}")

    fpr, tpr, thresholds = roc_curve(y_test, probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 0, 1], [0, 1, 1], linestyle="--", label="Perfect model (guide)")
    plt.plot([0, 1], [0, 1], linestyle=":", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    y_all_zero = np.zeros_like(y_test)
    base_acc = accuracy_score(y_test, y_all_zero)
    model_acc = accuracy_score(y_test, y_pred_default)
    print(f"\nBaseline (all zeros) Accuracy: {base_acc:.3f}")
    print(f"Model Accuracy (0.50 threshold): {model_acc:.3f}  "
          f"(lift vs. baseline: {model_acc - base_acc:+.3f})")


def main():
    # Locate CSV
    here = Path(__file__).resolve().parent
    candidates = [
        here / "pa_diabetes.csv",
        here / "diabetes" / "pa_diabetes.csv",
        Path("diabetes/pa_diabetes.csv"),
        Path("pa_diabetes.csv"),
    ]
    csv_path = None
    for p in candidates:
        if p.exists():
            csv_path = p
            break
    if csv_path is None:
        raise FileNotFoundError(
            "Could not find 'pa_diabetes.csv'. "
            "Place it next to this script or under ./diabetes/."
        )

    print(f"Reading CSV: {csv_path}")
    df = load_data(csv_path)

    # Split
    y = df["Outcome"].astype(int)
    X = df.drop(columns=["Outcome"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=489, stratify=y
    )

    # Build pipeline & fit
    model = build_pipeline(X_train)
    model.fit(X_train, y_train)

    # Evaluate @ default 0.5 threshold
    evaluate(model, X_test, y_test)

    # OPTIONAL: try a lower threshold to boost recall
    # evaluate(model, X_test, y_test, threshold=0.35)


if __name__ == "__main__":
    main()