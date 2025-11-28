import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import recall_score
from imblearn.over_sampling import RandomOverSampler, SMOTE
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns




def preprocess_titanic(df):
    # Keep useful features and drop noise
    cols_to_keep = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']
    df = df.copy()
    # Some datasets include extra columns; only keep what we need if available
    df = df[[c for c in cols_to_keep if c in df.columns]]

    # Normalize common invalid/missing markers to actual NaN
    df.replace(['na', 'NA', 'Na', 'n/a', 'N/A', '?', ''], np.nan, inplace=True)

    # Delete any row that contains a missing/invalid value in the kept columns
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
    print('\n--- Correlation matrix (predictors) ---')
    corr = X.corr()
    print(corr)

    # Create and save a heatmap visualization for the correlation matrix
    try:
        plt.figure(figsize=(6, 4))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Predictor Correlation Heatmap')
        plt.tight_layout()
        #plt.savefig('correlation_heatmap.png', dpi=200)
        #plt.close()
        plt.show()
    except Exception as e:
        print(f"Could not create heatmap: {e}")

    if variance_inflation_factor is None:
        print('\nstatsmodels not available — skipping VIF calculation.')
        return

    print('\n--- VIF (predictors) ---')
    # Add constant for VIF calculation
    Xc = X.copy()
    Xc = Xc.select_dtypes(include=[float, int])
    Xc = sm.add_constant(Xc)
    vif_data = []
    for i in range(1, Xc.shape[1]):
        feature = Xc.columns[i]
        try:
            vif = variance_inflation_factor(Xc.values, i)
        except Exception:
            vif = float('nan')
        vif_data.append((feature, vif))
    for feature, vif in vif_data:
        print(f"{feature}: {vif:.4f}")


def show_metrics(prediction, actual):
    acc = accuracy_score(actual, prediction)
    print(f"Accuracy: {acc:.2%}")
    print("\nClassification Report:")
    print(classification_report(actual, prediction, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(actual, prediction))


def train_and_evaluate(df, save_model=True):
    # Get preprocessed X and y where Sex and Embarked are already encoded
    X_all, y = preprocess_titanic(df)

    # Part 3: initial logistic regression using only specified predictors
    predictors = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    X = X_all[predictors].copy()

    # Check linearity/independence: correlation and VIF
    compute_correlation_and_vif(X)

    # Split using random_state=1 per instructions
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print('\n--- Logistic Regression Results (initial predictors) ---')
    show_metrics(preds, y_test)

    # Inspect prediction contents
    print('\nUnique predictions:', np.unique(preds))

    # Compare list of all 1s to response testing set
    ones_preds = np.ones_like(y_test)
    print('\n--- All-ones baseline ---')
    show_metrics(ones_preds, y_test)

    # Part 7-9: include Sex and Embarked (Sex is binary, Embarked has 3 values encoded earlier)
    extra_cols = []
    if 'Sex' in X_all.columns:
        extra_cols.append('Sex')
    extra_cols += [c for c in X_all.columns if c.startswith('Embarked_')]

    if extra_cols:
        X2 = pd.concat([X, X_all[extra_cols]], axis=1)
        # retrain with large max_iter to avoid convergence warnings
        X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.2, random_state=1)
        model2 = LogisticRegression(max_iter=100000)
        model2.fit(X2_train, y2_train)
        preds2 = model2.predict(X2_test)
        print('\n--- Logistic Regression Results (with Sex & Embarked) ---')
        show_metrics(preds2, y2_test)
    else:
        X2, X2_train, X2_test, y2_train, y2_test, model2, preds2 = None, None, None, None, None, None, None

    # Part 10: Oversample (RandomOverSampler) on training set and compare
    if RandomOverSampler is None:
        print('\nimbalanced-learn not available; skipping oversampling step.')
    else:
        # Use RandomOverSampler per instruction. SMOTE alternative is shown commented out.
        ros = RandomOverSampler(random_state=0)
        # ros = SMOTE(random_state=1)  # SMOTE alternative (commented out)
        # Oversample the training set directly (use X2_train/y2_train if available)
        if extra_cols:
            X_train_for_resample = X2_train
            y_train_for_resample = y2_train
        else:
            X_train_for_resample = X_train
            y_train_for_resample = y_train

        Xr_train, yr_train = ros.fit_resample(X_train_for_resample, y_train_for_resample)

        model_bal = LogisticRegression(max_iter=100000)
        model_bal.fit(Xr_train, yr_train)

        # Evaluate on original test set (X2_test or X_test)
        X_test_eval = X2_test if extra_cols else X_test
        y_test_eval = y2_test if extra_cols else y_test
        preds_bal = model_bal.predict(X_test_eval)

        print('\n--- Logistic Regression Results (after RandomOverSampler) ---')
        show_metrics(preds_bal, y_test_eval)

        # Cross-validation accuracy before and after balancing
        cv = 5
        X_for_resample = X2 if extra_cols else X
        acc_before = cross_val_score(LogisticRegression(max_iter=100000), X_for_resample, y, cv=cv, scoring='accuracy')
        # For cross-val after balancing, resample entire X_for_resample and compute CV on resampled set
        X_res_all, y_res_all = ros.fit_resample(X_for_resample, y)
        acc_after = cross_val_score(LogisticRegression(max_iter=100000), X_res_all, y_res_all, cv=cv, scoring='accuracy')
        print(f"\nCross-val accuracy before balancing: {acc_before.mean():.2f} (+/- {acc_before.std():.2f})")
        print(f"Cross-val accuracy after balancing:  {acc_after.mean():.2f} (+/- {acc_after.std():.2f})")

        # Part 11: sensitivity and specificity prior to balancing and after balancing
        # Prior to balancing: use model2 predictions if available else model
        if extra_cols:
            base_preds = preds2
            base_y_test = y2_test
        else:
            base_preds = preds
            base_y_test = y_test

        tn, fp, fn, tp = confusion_matrix(base_y_test, base_preds).ravel()
        sensitivity_before = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity_before = tn / (tn + fp) if (tn + fp) > 0 else 0

        tn2, fp2, fn2, tp2 = confusion_matrix(y_test_eval, preds_bal).ravel()
        sensitivity_after = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0
        specificity_after = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0

        print('\nSensitivity (before balancing):', round(sensitivity_before, 2))
        print('Specificity (before balancing):', round(specificity_before, 2))
        print('Sensitivity (after balancing):', round(sensitivity_after, 2))
        print('Specificity (after balancing):', round(specificity_after, 2))

        # Part 12: Save the two sets of values to a text file
        out_text = (
            f"Sensitivity_before_ROS: {sensitivity_before:.2f}\n"
            f"Specificity_before_ROS: {specificity_before:.2f}\n"
            f"Sensitivity_after_ROS: {sensitivity_after:.2f}\n"
            f"Specificity_after_ROS: {specificity_after:.2f}\n"
        )
        with open('sensitivity_specificity.txt', 'w') as f:
            f.write(out_text)
        print("\nWrote sensitivity/specificity values to 'sensitivity_specificity.txt'.")

    # Return last-used models for possible inspection (optional)
    return locals()


def main():
    csv_path = 'Titanic-Dataset.csv'
    if not os.path.exists(csv_path):
        print(f"CSV file '{csv_path}' not found. Please ensure it is in the current directory.")
        return

    df = pd.read_csv(csv_path)
    train_and_evaluate(df, save_model=True)


if __name__ == '__main__':
    main()
