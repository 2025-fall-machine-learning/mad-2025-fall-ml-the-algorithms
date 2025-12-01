
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns



def preprocess_titanic(df):

    cols_to_keep = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']
    df = df.copy()
    df = df[[c for c in cols_to_keep if c in df.columns]]

    # Normalize invalid markers -> NaN, then DROP rows with any NaN in kept columns
    df.replace(['na', 'NA', 'Na', 'n/a', 'N/A', '?', ''], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Encode Sex
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # One-hot encode Embarked
    if 'Embarked' in df.columns:
        embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked', drop_first=True)
        df = pd.concat([df.drop('Embarked', axis=1), embarked_dummies], axis=1)

    # Features and target
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    return X, y

def compute_correlation_and_vif(X):
    """Print correlation matrix and VIF plus optional heatmap."""
    print("\n--- Correlation matrix (predictors) ---")
    corr = X.corr()
    print(corr)

    # Optional heatmap display (not saved)
    try:
        plt.figure(figsize=(6, 4))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Predictor Correlation Heatmap')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Could not create heatmap: {e}")

    print("\n--- VIF (predictors) ---")
    Xc = X.select_dtypes(include=[float, int]).copy()
    Xc = sm.add_constant(Xc)
    for i in range(1, Xc.shape[1]):
        feature = Xc.columns[i]
        try:
            vif = variance_inflation_factor(Xc.values, i)
        except Exception:
            vif = float('nan')
        print(f"{feature}: {vif:.4f}")

def show_metrics(prediction, actual):
    acc = accuracy_score(actual, prediction)
    print(f"Accuracy: {acc:.2%}")
    print("\nClassification Report:")
    print(classification_report(actual, prediction, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(actual, prediction))

def train_and_evaluate(df, save_model=True):
    # Preprocess
    X_all, y = preprocess_titanic(df)

    # Initial predictors
    predictors = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    X = X_all[predictors].copy()

    # Correlation and VIF
    compute_correlation_and_vif(X)

    # Split (random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    model = LogisticRegression(max_iter=100000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("\n--- Logistic Regression Results (initial predictors) ---")
    show_metrics(preds, y_test)

    # Inspect predictions
    print("\nUnique predictions:", np.unique(preds))

    # All-ones baseline
    ones_preds = np.ones_like(y_test)
    print("\n--- All-ones baseline ---")
    show_metrics(ones_preds, y_test)

    # Add Sex & Embarked (already encoded)
    extra_cols = [c for c in X_all.columns if c not in predictors]
    X2 = pd.concat([X, X_all[extra_cols]], axis=1)

    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.2, random_state=1)
    model2 = LogisticRegression(max_iter=100000)
    model2.fit(X2_train, y2_train)
    preds2 = model2.predict(X2_test)

    print("\n--- Logistic Regression Results (with Sex & Embarked) ---")
    show_metrics(preds2, y2_test)

    # Oversampling on TRAINING set only
    ros = RandomOverSampler(random_state=1)
    Xr_train, yr_train = ros.fit_resample(X2_train, y2_train)

    model_bal = LogisticRegression(max_iter=100000)
    model_bal.fit(Xr_train, yr_train)

    # Evaluate on ORIGINAL test set
    preds_bal = model_bal.predict(X2_test)
    print("\n--- Logistic Regression Results (after RandomOverSampler) ---")
    show_metrics(preds_bal, y2_test)

    # CV accuracy before/after balancing
    cv = 5
    acc_before = cross_val_score(LogisticRegression(max_iter=100000), X2, y, cv=cv, scoring='accuracy')
    X_res_all, y_res_all = ros.fit_resample(X2, y)
    acc_after = cross_val_score(LogisticRegression(max_iter=100000), X_res_all, y_res_all, cv=cv, scoring='accuracy')

    print(f"\nCross-val accuracy before balancing: {acc_before.mean():.2f} (+/- {acc_before.std():.2f})")
    print(f"Cross-val accuracy after balancing:  {acc_after.mean():.2f} (+/- {acc_after.std():.2f})")

    # Sensitivity & Specificity (TEST SET ONLY)
    tn, fp, fn, tp = confusion_matrix(y2_test, preds2).ravel()
    sens_before = tp / (tp + fn) if (tp + fn) else 0.0
    spec_before = tn / (tn + fp) if (tn + fp) else 0.0

    tn2, fp2, fn2, tp2 = confusion_matrix(y2_test, preds_bal).ravel()
    sens_after = tp2 / (tp2 + fn2) if (tp2 + fn2) else 0.0
    spec_after = tn2 / (tn2 + fp2) if (tn2 + fp2) else 0.0

    print(f"\nSensitivity (before balancing): {sens_before:.2f}")
    print(f"Specificity (before balancing): {spec_before:.2f}")
    print(f"Sensitivity (after balancing):  {sens_after:.2f}")
    print(f"Specificity (after balancing):  {spec_after:.2f}")

    # Save values
    out_text = (
        f"Sensitivity_before_ROS: {sens_before:.2f}\n"
        f"Specificity_before_ROS: {spec_before:.2f}\n"
        f"Sensitivity_after_ROS: {sens_after:.2f}\n"
        f"Specificity_after_ROS: {spec_after:.2f}\n"
    )
    with open('sensitivity_specificity.txt', 'w') as f:
        f.write(out_text)
    print("\nWrote sensitivity/specificity values to 'sensitivity_specificity.txt'.")

    return locals() 

def main():
    csv_path = 'Titanic-Dataset.csv'
    if not os.path.exists(csv_path):
        print(f"CSV file '{csv_path}' not found. Please ensure it is in the current directory.")
        return
    df = pd.read_csv(csv_path)
    train_and_evaluate(df)

if __name__ == '__main__':
    main()