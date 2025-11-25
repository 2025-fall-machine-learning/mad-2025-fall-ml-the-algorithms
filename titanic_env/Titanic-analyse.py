import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

from imblearn.over_sampling import RandomOverSampler


def main():
    # 1) Load the Titanic dataset
    df = pd.read_csv("C:\\Users\\mahun\\OneDrive\\Work\\mad-2025-fall-ml-the-algorithms\\titanic_env\\Titanic-Dataset.csv")


    # 2) Remove invalid values ("na") and missing values
    df = df.replace(["na", "NA", "Na", "nan", "NaN", "N/A", ""], np.nan)
    df = df.dropna()

    # Make sure numeric columns are really numeric
    cols_num = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
    df[cols_num] = df[cols_num].apply(pd.to_numeric, errors="coerce")
    df = df.dropna()

    # -----------------------------
    # 3) Logistic regression (basic)
    # -----------------------------
    X = df[["Pclass", "Age", "SibSp", "Parch", "Fare"]]
    y = df["Survived"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    model_basic = LogisticRegression(random_state=1, max_iter=1000)
    model_basic.fit(X_train, y_train)

    pred_basic = model_basic.predict(X_test)

    # 5) Predictions are only 0 and 1
    print("Unique predictions (basic):", np.unique(pred_basic))

    # 6) Compare to "all 1s" baseline
    all_ones = np.ones(len(y_test))
    acc_basic = accuracy_score(y_test, pred_basic)
    acc_all_ones = accuracy_score(y_test, all_ones)

    print("Accuracy logistic regression (basic):", acc_basic)
    print("Accuracy all-ones baseline:", acc_all_ones)

    # 11) Sensitivity & specificity BEFORE adding new columns
    cm_basic = confusion_matrix(y_test, pred_basic)
    tn, fp, fn, tp = cm_basic.ravel()
    sens_basic = tp / (tp + fn)
    spec_basic = tn / (tn + fp)

    # --------------------------------------------
    # 7) Encode Sex (2 values) and Embarked (3 values)
    # --------------------------------------------
    # Sex -> make it 0/1
    df["Sex_bin"] = df["Sex"].map({"male": 1, "female": 0})

    # Embarked -> one-hot encoding (3 columns)
    embarked_cols = pd.get_dummies(df["Embarked"], prefix="Embarked")
    df = pd.concat([df, embarked_cols], axis=1)

    # --------------------------------------------
    # 8-9) Re-run with the new columns
    # --------------------------------------------
    X2 = df[["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_bin"] + list(embarked_cols.columns)]
    y2 = df["Survived"].astype(int)

    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y2, test_size=0.2, random_state=1
    )

    model_ext = LogisticRegression(random_state=1, max_iter=100000)
    model_ext.fit(X2_train, y2_train)

    pred_ext = model_ext.predict(X2_test)
    acc_ext = accuracy_score(y2_test, pred_ext)

    cm_ext = confusion_matrix(y2_test, pred_ext)
    tn, fp, fn, tp = cm_ext.ravel()
    sens_ext = tp / (tp + fn)
    spec_ext = tn / (tn + fp)

    print("\nAccuracy logistic regression (with Sex + Embarked):", acc_ext)
    print("Sensitivity BEFORE balancing:", sens_ext)
    print("Specificity BEFORE balancing:", spec_ext)

    # --------------------------------------------
    # 10) Balance data with RandomOverSampler
    # --------------------------------------------
    ros = RandomOverSampler(random_state=1)
    X2_train_bal, y2_train_bal = ros.fit_resample(X2_train, y2_train)

    model_bal = LogisticRegression(random_state=1, max_iter=100000)
    model_bal.fit(X2_train_bal, y2_train_bal)

    pred_bal = model_bal.predict(X2_test)
    acc_bal = accuracy_score(y2_test, pred_bal)

    cm_bal = confusion_matrix(y2_test, pred_bal)
    tn, fp, fn, tp = cm_bal.ravel()
    sens_bal = tp / (tp + fn)
    spec_bal = tn / (tn + fp)

    print("\nAccuracy AFTER balancing:", acc_bal)
    print("Sensitivity AFTER balancing:", sens_bal)
    print("Specificity AFTER balancing:", spec_bal)

    # --------------------------------------------
    # 12) Save the two sets of sensitivity/specificity
    # --------------------------------------------
    results = []
    results.append("=== BEFORE balancing (basic model) ===")
    results.append(f"Sensitivity: {sens_basic:.4f}")
    results.append(f"Specificity: {spec_basic:.4f}")
    results.append("")
    results.append("=== AFTER adding Sex+Embarked (before balancing) ===")
    results.append(f"Sensitivity: {sens_ext:.4f}")
    results.append(f"Specificity: {spec_ext:.4f}")
    results.append("")
    results.append("=== AFTER balancing (RandomOverSampler) ===")
    results.append(f"Sensitivity: {sens_bal:.4f}")
    results.append(f"Specificity: {spec_bal:.4f}")

    with open("titanic_sens_spec_results.txt", "w") as f:
        f.write("\n".join(results))

    print("\nSaved results in titanic_sens_spec_results.txt")


if __name__ == "__main__":
    main()
