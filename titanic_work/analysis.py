from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler


def load_and_clean():
    csv_path = Path(__file__).resolve().parent / "Titanic-Dataset.csv"
    titanic_df = pd.read_csv(csv_path)
    # Replace literal string "na" with pandas missing value
    titanic_df.replace("na", pd.NA, inplace=True)
    # Encode `Sex` as binary (male=0, female=1)
    if 'Sex' in titanic_df.columns:
        titanic_df['Sex'] = titanic_df['Sex'].map({'male': 0, 'female': 1})

    # Encode `Embarked` into dummy columns (drop_first=False here; we'll keep all and
    # let downstream code pick which to use). This creates columns like
    # `Embarked_C`, `Embarked_Q`, `Embarked_S` depending on values present.
    if 'Embarked' in titanic_df.columns:
        embarked_dummies = pd.get_dummies(titanic_df['Embarked'], prefix='Embarked')
        titanic_df = pd.concat([titanic_df.drop(columns=['Embarked']), embarked_dummies], axis=1)

    # Drop any rows that contain missing values (after encoding)
    titanic_df.dropna(inplace=True)
    return titanic_df


def perform_logistic_regression(titanic_df, random_state=1):
    """
    Predictors: Pclass, Age, SibSp, Parch, Fare
    Response: Survived
    """
    # Base numeric predictors
    base_predictors = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex"]
    embarked_cols = [c for c in titanic_df.columns if c.startswith('Embarked_')]
    predictors = [p for p in base_predictors if p in titanic_df.columns] + embarked_cols

    X = titanic_df[predictors].copy()
    y = titanic_df["Survived"].copy().astype(int)

    # Split into train/test sets with provided random_state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    return X_train, X_test, y_train, y_test


def fit_logistic(X_train, y_train, max_iter=100000, random_state=0, oversample=False, oversample_random_state=1):
    """Create and fit a LogisticRegression with the provided settings and return it.

    Centralizing the instantiation here makes it explicit that all calls use
    `max_iter=100000` (or whatever is passed) when fitting the model.
    """
    # Optionally apply RandomOverSampler to balance the training set
    X_fit, y_fit = X_train, y_train
    if oversample:
        if RandomOverSampler is None:
            raise ImportError("imblearn is required for oversampling. Install via: pip install imbalanced-learn")
        ros = RandomOverSampler(random_state=oversample_random_state)
        X_fit, y_fit = ros.fit_resample(X_train, y_train)

    clf = LogisticRegression(max_iter=max_iter, random_state=random_state)
    clf.fit(X_fit, y_fit)
    return clf
    
    
def check_linearity_and_independence(titanic_df):
    base_predictors = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex"]
    embarked_cols = [c for c in titanic_df.columns if c.startswith('Embarked_')]
    predictors = [p for p in base_predictors if p in titanic_df.columns] + embarked_cols
    response_col = "Survived"

    # Pearson correlation between each predictor and the response
    corr_with_response = {p: titanic_df[p].corr(titanic_df[response_col]) for p in predictors}

    # Correlation matrix among predictors
    predictors_corr = titanic_df[predictors].corr()

    # Compute VIF for each predictor using R^2 from regressing the predictor on the others
    vif_dict = {}
    for p in predictors:
        others = [q for q in predictors if q != p]
        X_others = titanic_df[others].values
        y_p = titanic_df[p].values
        model = LinearRegression().fit(X_others, y_p)
        r2 = model.score(X_others, y_p)
        vif = float('inf') if r2 >= 0.9999 else 1.0 / (1.0 - r2)
        vif_dict[p] = vif

    return corr_with_response, predictors_corr, vif_dict


def print_linearity_and_independence(titanic_df):
    corr_with_response, predictors_corr, vif = check_linearity_and_independence(titanic_df)

    print('\nCorrelation with response (Survived):')
    for k, v in corr_with_response.items():
        print(f'{k}: {v:.4f}')

    print('\nPredictors correlation matrix:')
    print(predictors_corr)

    print('\nVIF values:')
    for k, v in vif.items():
        print(f'{k}: {v:.4f}')


def predictor_correlation_heatmap(titanic_df):
    """Display a correlation heatmap for the predictor columns.

    Uses the cleaned `titanic_df` and shows pairwise Pearson correlations
    for the selected predictors: Pclass, Age, SibSp, Parch, Fare, Sex and any Embarked dummies.
    """
    base_predictors = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex"]
    embarked_cols = [c for c in titanic_df.columns if c.startswith('Embarked_')]
    predictors = [p for p in base_predictors if p in titanic_df.columns] + embarked_cols
    corr = titanic_df[predictors].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Predictors Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def inspect_prediction_values(titanic_df):
    X_train, X_test, y_train, y_test = perform_logistic_regression(titanic_df)

    clf = fit_logistic(X_train, y_train, max_iter=100000, random_state=0)
    preds = clf.predict(X_test)

    unique_vals, counts = pd.unique(preds), None
    try:
        # counts per unique value
        uniques, cnts = np.unique(preds, return_counts=True)
        counts = dict(zip(uniques.tolist(), cnts.tolist()))
    except Exception:
        counts = None

    print('\nPrediction inspection:')
    print('dtype:', preds.dtype)
    print('unique values:', unique_vals.tolist() if hasattr(unique_vals, 'tolist') else list(unique_vals))
    if counts is not None:
        print('counts:', counts)
    # return for possible further inspection
    return preds


def compare_all_ones_baseline(X_train, X_test, y_train, y_test):
    """Compare a logistic regression model to an "all ones" baseline on provided splits.

    This function expects already-computed train/test splits and will
    fit a logistic regression on (X_train, y_train) then compare its
    predictions on X_test to an all-ones baseline. It returns two
    dictionaries: (model_stats, ones_stats).
    """

    # Fit logistic regression on provided splits
    clf = fit_logistic(X_train, y_train, max_iter=100000, random_state=0)
    preds_model = clf.predict(X_test)

    # All-ones baseline
    preds_ones = np.ones_like(y_test)

    def summarise(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        accuracy = accuracy_score(y_true, y_pred)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
        return {
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'accuracy': float(accuracy), 'sensitivity': float(sensitivity), 'specificity': float(specificity)
        }

    model_stats = summarise(y_test, preds_model)
    ones_stats = summarise(y_test, preds_ones)

    return model_stats, ones_stats


def split_inspect(titanic_df, random_state=1):
    """Create one split and print a compact inspection report.

    Returns the split (X_train, X_test, y_train, y_test) so callers may reuse it.
    """
    Xtr, Xte, ytr, yte = perform_logistic_regression(titanic_df, random_state=random_state)
    print('\nSplit inspect (random_state=%s):' % random_state)
    print('  predictors:', list(Xtr.columns))
    print('  X_train.shape:', Xtr.shape, 'X_test.shape:', Xte.shape)
    print('  y_train counts:', dict(ytr.value_counts()), 'y_test counts:', dict(yte.value_counts()))
    return Xtr, Xte, ytr, yte


def run_baseline_comparison(titanic_df, random_states=(1, 0)):
    """Run baseline comparisons for the provided random_states.

    For each random_state this creates a split via `perform_logistic_regression`,
    fits the logistic model, computes baseline stats, prints verbose results,
    and returns a mapping {random_state: (model_stats, ones_stats)}.
    """
    results = {}
    for rs in random_states:
        Xtr, Xte, ytr, yte = perform_logistic_regression(titanic_df, random_state=rs)
        model_stats, ones_stats = compare_all_ones_baseline(Xtr, Xte, ytr, yte)
        results[rs] = (model_stats, ones_stats)
        print(f"\nBaseline comparison (random_state={rs}):")
        print('  Logistic Regression:', model_stats)
        print('  All-ones baseline:  ', ones_stats)

    # compact summary if exactly two states provided
    if len(random_states) >= 2:
        rs0, rs1 = random_states[0], random_states[1]
        m0, o0 = results[rs0]
        m1, o1 = results[rs1]
        print(f"\nSummary: rs{rs0} m_acc={m0['accuracy']:.3f} o_acc={o0['accuracy']:.3f} | rs{rs1} m_acc={m1['accuracy']:.3f} o_acc={o1['accuracy']:.3f}")

    return results


def predict_on_extended_test(titanic_df, random_state=1, oversample=False, oversample_random_state=1):
    """Fit logistic on the training split and predict on the extended test set.

    Returns a DataFrame containing the test predictors, the true `Survived` values,
    the predicted label and the predicted probability for the positive class.
    Output is printed to the terminal (no CSV saving).
    """
    Xtr, Xte, ytr, yte = perform_logistic_regression(titanic_df, random_state=random_state)
    # Fit with oversampling option if requested by caller
    clf = fit_logistic(Xtr, ytr, max_iter=100000, random_state=0, oversample=oversample, oversample_random_state=oversample_random_state)

    preds = clf.predict(Xte)
    proba = None
    if hasattr(clf, 'predict_proba'):
        try:
            proba = clf.predict_proba(Xte)[:, 1]
        except Exception:
            proba = None

    results_df = Xte.copy().reset_index(drop=True)
    results_df['Survived_true'] = yte.reset_index(drop=True)
    results_df['Survived_pred'] = preds
    if proba is not None:
        results_df['Pred_prob_positive'] = proba

    # Print a compact summary
    uniques, counts = np.unique(preds, return_counts=True)
    print('\nPrediction inspection on extended test set:')
    print('  pred unique values:', uniques.tolist())
    print('  pred counts:', {int(k): int(v) for k, v in zip(uniques.tolist(), counts.tolist())})
    vc = {int(k): int(v) for k, v in yte.value_counts().to_dict().items()}
    y_counts = {0: vc.get(0, 0), 1: vc.get(1, 0)}
    print('  y_test counts:', y_counts)

    # Always print the first rows to the terminal for quick inspection
    print('\nFirst rows of prediction results:')
    try:
        print(results_df.head(10).to_string(index=False))
    except Exception:
        print(results_df.head(10))

    return results_df


def compute_sens_spec_before_after(titanic_df, random_state=1, oversample_random_state=1, out_name="sensitivity_specificity.txt"):
    """Compute sensitivity and specificity before and after RandomOverSampler balancing.

    Fits a logistic regression on the same train/test split twice: once normally,
    once after applying RandomOverSampler to the training set. Writes a small
    text summary to `out_name` next to this script and prints the content.
    Returns a dict with the two result rows for programmatic use.
    """
    out = {}
    Xtr, Xte, ytr, yte = perform_logistic_regression(titanic_df, random_state=random_state)

    def calc_metrics(y_true, y_pred):
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        except Exception:
            # Fallback if confusion_matrix doesn't return 4 values (e.g., only one class present)
            vals = confusion_matrix(y_true, y_pred).tolist()
            # build tn, fp, fn, tp defensively
            tn = fp = fn = tp = 0
            if len(vals) == 1:
                # only negatives or only positives
                if (y_true == 0).all():
                    tn = vals[0][0]
                else:
                    tp = vals[0][0]
            else:
                # matrix 2x2
                tn, fp = vals[0]
                fn, tp = vals[1]
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
        return {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp), 'sensitivity': float(sensitivity), 'specificity': float(specificity)}

    # Before balancing
    clf_before = fit_logistic(Xtr, ytr, max_iter=100000, random_state=0, oversample=False)
    preds_before = clf_before.predict(Xte)
    before_metrics = calc_metrics(yte, preds_before)
    out['before'] = before_metrics

    # After balancing (oversample training set)
    try:
        clf_after = fit_logistic(Xtr, ytr, max_iter=100000, random_state=0, oversample=True, oversample_random_state=oversample_random_state)
        preds_after = clf_after.predict(Xte)
        after_metrics = calc_metrics(yte, preds_after)
        out['after'] = after_metrics
    except ImportError as ie:
        # Could not import imblearn; record the error message
        out['after'] = {'error': str(ie)}

    # Write a small text file with both results
    lines = []
    lines.append(f"Sensitivity & Specificity report (split random_state={random_state})")
    lines.append("")
    b = out['before']
    lines.append("Before balancing:")
    if 'error' in b:
        lines.append(f"  error: {b['error']}")
    else:
        lines.append(f"  sensitivity: {b['sensitivity']:.4f}")
        lines.append(f"  specificity: {b['specificity']:.4f}")
        lines.append(f"  confusion: tn={b['tn']} fp={b['fp']} fn={b['fn']} tp={b['tp']}")
    lines.append("")
    a = out.get('after', {})
    lines.append("After balancing:")
    if 'error' in a:
        lines.append(f"  error: {a['error']}")
    else:
        lines.append(f"  sensitivity: {a['sensitivity']:.4f}")
        lines.append(f"  specificity: {a['specificity']:.4f}")
        lines.append(f"  confusion: tn={a['tn']} fp={a['fp']} fn={a['fn']} tp={a['tp']}")

    text = "\n".join(lines)
    out_path = Path(__file__).resolve().parent / out_name
    try:
        out_path.write_text(text)
    except Exception as e:
        print('Failed to write output file:', e)

    print('\n' + text)
    print(f"\nWrote summary to: {out_path}")
    return out


def main():
    titanic_df = load_and_clean()
    #print(titanic_df)

    # print_linearity_and_independence(titanic_df)

    # predictor_correlation_heatmap(titanic_df)

    # inspect_prediction_values(titanic_df)

    # check of the feature set and the train/test split produced by perform_logistic_regression():
    #Xtr,Xte,ytr,yte=perform_logistic_regression(titanic_df,random_state=1); print('predictors:',list(Xtr.columns),'X_train:',Xtr.shape,'X_test:',Xte.shape,'y_train:',dict(ytr.value_counts()),'y_test:',dict(yte.value_counts()))
    
    # Predict on the extended test set (fits logistic with max_iter=100000)
    # Call prediction helper with oversampling turned on (RandomOverSampler, random_state=1)
    #predict_on_extended_test(titanic_df, random_state=1, oversample=True, oversample_random_state=1)

    # Compute sensitivity & specificity before and after balancing (single-line)
    #compute_sens_spec_before_after(titanic_df, random_state=1, oversample_random_state=1, out_name="sensitivity_specificity.txt")

if __name__ == "__main__":
    main()