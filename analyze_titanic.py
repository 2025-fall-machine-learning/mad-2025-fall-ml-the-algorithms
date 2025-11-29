import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import RandomOverSampler


def main():

    file = "Titanic-Dataset.csv"
    if not os.path.exists(file):
        print("file not found")
        return

    df = pd.read_csv(file)
    print("shape at start:", df.shape)

    bad_vals = ["na", "NA", "Na", "n/a", "N/A", "?", ""]
    df = df.replace(bad_vals, np.nan)

    print("missing before fix:")
    print(df.isna().sum())

    df = df[df["Survived"].notna()].copy()

    if df["Age"].isna().sum() > 0:
        df["Age"] = df["Age"].fillna(df["Age"].median())

    if df["Fare"].isna().sum() > 0:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    if "Embarked" in df.columns and df["Embarked"].isna().sum() > 0:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    if "Sex" in df.columns and df["Sex"].isna().sum() > 0:
        df["Sex"] = df["Sex"].fillna(df["Sex"].mode()[0])

    df.reset_index(drop=True, inplace=True)
    print("shape after cleaning:", df.shape)

    need = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
    for c in need:
        if c not in df.columns:
            raise ValueError("missing column " + c)

    base_feats = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

    corr = df[base_feats + ["Survived"]].corr()["Survived"].drop("Survived")
    print("correlation:", corr)

    sel = []
    for x in base_feats:
        if corr[x] >= 0.1:
            sel.append(x)

    print("selected:", sel)

    X = df[sel]
    y = df["Survived"]

    print("class distrib:", y.value_counts())

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1)

    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("lg", LogisticRegression(max_iter=1000))
    ])

    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)

    print("unique preds:", np.unique(pred))

    ones = np.ones_like(yte)
    acc1 = accuracy_score(yte, pred)
    accb = accuracy_score(yte, ones)

    print("acc logreg r1:", acc1)
    print("acc all ones r1:", accb)
    print("report:", classification_report(yte, pred, digits=4))

    Xtr0, Xte0, ytr0, yte0 = train_test_split(X, y, test_size=0.2, random_state=0)

    pipe0 = Pipeline([
        ("sc", StandardScaler()),
        ("lg", LogisticRegression(max_iter=1000))
    ])

    pipe0.fit(Xtr0, ytr0)
    pred0 = pipe0.predict(Xte0)
    ones0 = np.ones_like(yte0)

    print("acc logreg r0:", accuracy_score(yte0, pred0))
    print("acc all ones r0:", accuracy_score(yte0, ones0))

    if "Sex" not in df.columns or "Embarked" not in df.columns:
        raise ValueError("Sex or Embarked missing")

    df2 = df.copy()
    df2["Sex"] = df2["Sex"].map({"male": 0, "female": 1})

    emb = pd.get_dummies(df2["Embarked"], prefix="Embarked", drop_first=True)
    df2 = pd.concat([df2.drop("Embarked", axis=1), emb], axis=1)

    emb_cols = [c for c in df2.columns if c.startswith("Embarked_")]
    feats2 = sel + ["Sex"] + emb_cols

    print("\nfeatures v2:", feats2)

    X2 = df2[feats2]
    y2 = df2["Survived"]

    X2tr, X2te, y2tr, y2te = train_test_split(X2, y2, test_size=0.2, random_state=1)

    pipe2 = Pipeline([
        ("sc", StandardScaler()),
        ("lg", LogisticRegression(max_iter=100000))
    ])

    pipe2.fit(X2tr, y2tr)
    pred2 = pipe2.predict(X2te)

    print("acc with sex+emb:", accuracy_score(y2te, pred2))

    ros = RandomOverSampler(random_state=1)
    Xbal, ybal = ros.fit_resample(X2tr, y2tr)

    pipe_bal = Pipeline([
        ("sc", StandardScaler()),
        ("lg", LogisticRegression(max_iter=100000))
    ])

    pipe_bal.fit(Xbal, ybal)
    pred_bal = pipe_bal.predict(X2te)

    print("acc balanced:", accuracy_score(y2te, pred_bal))

    tn, fp, fn, tp = confusion_matrix(y2te, pred2).ravel()
    sens_b = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec_b = tn / (tn + fp) if (tn + fp) > 0 else 0

    tn2, fp2, fn2, tp2 = confusion_matrix(y2te, pred_bal).ravel()
    sens_a = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0
    spec_a = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0

    print("\nbefore bal:")
    print("sens:", sens_b)
    print("spec:", spec_b)

    print("\nafter bal:")
    print("sens:", sens_a)
    print("spec:", spec_a)

    try:
        p = pipe2.predict_proba(X2te)[:, 1]
        auc = roc_auc_score(y2te, p)
        print("\nauc:", auc)
    except:
        print("auc failed")

    txt = ""
    txt += f"Sensitivity_before: {sens_b:.4f}\n"
    txt += f"Specificity_before: {spec_b:.4f}\n"
    txt += f"Sensitivity_after:  {sens_a:.4f}\n"
    txt += f"Specificity_after:  {spec_a:.4f}\n"

    with open("sensitivity_specificity.txt", "w") as f:
        f.write(txt)


if __name__ == "__main__":
    main()
