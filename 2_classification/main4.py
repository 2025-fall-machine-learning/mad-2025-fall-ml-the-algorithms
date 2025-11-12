import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Extra imports ADDED
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

# NEW imports for Option B (one-class training on positives only)
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def show_prediction_results(header, prediction, actual_data):
    np_prediction = np.array(prediction)
    np_actual = np.array(actual_data)
    right_counter = 0
    wrong_counter = 0
    for (pred, actual) in zip(np_prediction, np_actual):
        if pred == actual:
            right_counter += 1
        else:
            wrong_counter += 1
    print()
    print(header)
    print(f"Right predictions: {right_counter}")
    print(f"Wrong predictions: {wrong_counter}")
    print(f"Accuracy: {right_counter / (right_counter + wrong_counter):.2%}")


def train(diabetes_df):
    """Your original binary logistic regression (kept for comparison)."""
    diabetes_df = diabetes_df.drop(['DNR Order', 'Med Tech'], axis='columns', errors='ignore')
    predictors_df = diabetes_df.drop('Outcome', axis='columns')
    response = diabetes_df['Outcome'].astype(int)

    training_predictors_df, testing_predictors_df, training_response, testing_response = train_test_split(
        predictors_df, response, test_size=0.2, random_state=489, stratify=response
    )

    # Fit logistic regression model (handles imbalance via class_weight)
    model = LogisticRegression(class_weight='balanced', max_iter=2000, solver="lbfgs")
    model.fit(training_predictors_df, training_response)
    prediction = model.predict(testing_predictors_df)

    # Confusion matrix and classification report
    cm = confusion_matrix(testing_response, prediction)
    print("\n=== Binary Logistic Regression (baseline) ===")
    print("=== Confusion Matrix ===")
    print(cm)
    print("\n=== Classification Report ===")
    print(classification_report(testing_response, prediction, digits=3))

    show_prediction_results("Logistic regression", prediction, testing_response)
    show_prediction_results("All negative predictions", [0]*len(testing_response), testing_response)


def train_one_class_on_positives(diabetes_df, nu=0.10, gamma="scale"):
    """
    OPTION B: Train on positives only via One-Class SVM.
    - Trains on X_train where y_train == 1.
    - Predicts +1 (inlier) -> 1 (diabetes-like), -1 (outlier) -> 0 (not).
    """
    # Drop the categorical columns to keep numeric-only (OneClassSVM needs numeric)
    df = diabetes_df.drop(['DNR Order', 'Med Tech'], axis='columns', errors='ignore')
    X = df.drop('Outcome', axis='columns')
    y = df['Outcome'].astype(int)

    # Standard train/test split with stratification for fair eval
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=489, stratify=y
    )

    # Keep only positive (Outcome==1) rows for training the one-class model
    X_train_pos = X_train[y_train == 1]

    # One-class pipeline: scaling -> OneClassSVM
    oc_pipeline = Pipeline(steps=[
        ("scale", StandardScaler()),
        ("ocsvm", OneClassSVM(kernel="rbf", nu=nu, gamma=gamma))
    ])

    # Fit ONLY on positives
    oc_pipeline.fit(X_train_pos)

    # Predict on the test set
    # OneClassSVM.predict -> +1 (inlier), -1 (outlier)
    oc_raw = oc_pipeline.predict(X_test)
    y_pred = np.where(oc_raw == 1, 1, 0)

    print("\n=== One-Class SVM (trained only on positives) ===")
    print(f"(nu={nu}, gamma={gamma})")
    cm = confusion_matrix(y_test, y_pred)
    print("=== Confusion Matrix ===")
    print(cm)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=3))

    # Optional: AUC using decision_function scores (higher => more diabetes-like)
    try:
        scores = oc_pipeline.decision_function(X_test)  # real-valued anomaly scores
        auc = roc_auc_score(y_test, scores)
        print(f"ROC-AUC (decision_function): {auc:.3f}")
    except Exception:
        pass

    show_prediction_results("One-Class SVM (positives-only)", y_pred, y_test)


def main():
    """Main function."""
    diabetes_df = pd.read_csv(r'diabetes\pa_diabetes.csv')

    # 1) Baseline binary classifier (for reference):
    train(diabetes_df)

    # 2) Positives-only one-class detector:
    #    TUNE 'nu' to trade recall vs. precision (try 0.05 .. 0.20).
    #    'gamma' can be "scale" (default) or "auto"; or a float.
    train_one_class_on_positives(diabetes_df, nu=0.10, gamma="scale")


if __name__ == "__main__":
    main()