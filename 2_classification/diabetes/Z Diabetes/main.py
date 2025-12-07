import numpy as np
import pandas as pd
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import sklearn.metrics as metrics
import imblearn.over_sampling as ios
import matplotlib.pyplot as plt
import seaborn as sns

# Extra imports ADDED
from sklearn.metrics import confusion_matrix, classification_report



def show_prediction_results(header, prediction, actual_data):
    np_prediction = np.round(prediction).astype(int)
    np_actual = np.array(actual_data).astype(int)
    result = np_actual == np_prediction

    num_correct_predictions = np.count_nonzero(result)
    num_incorrect_predictions = np.count_nonzero(result == False)
    print('{0}: {1} correct, {2} incorrect, accuracy: {3}.'
        .format(header, num_correct_predictions, num_incorrect_predictions,
        (num_correct_predictions/(num_correct_predictions+num_incorrect_predictions))))
    return (num_correct_predictions/(num_correct_predictions+num_incorrect_predictions))


def perform_logistic_regression(diabetes_predictors_df, diabetes_response_df, balance_counter):
    balanced_str = 'unbalanced'
    if balance_counter == 1:
        balanced_str = 'balanced'

    # I just put this here because random_state 1 is realistic; the unbalanced
    # data has a higher accuracy **number**, though it's not really more
    # accurate. With random_state 2, the balanced data is actually more accurate!
    # I contend both are more accurate, but the better number is indisputable.
    for random_state in range(0, 3):
        if balance_counter == 1:
            random_over_sampler = ios.RandomOverSampler(random_state=random_state)
            diabetes_predictors_df, diabetes_response_df \
                = random_over_sampler.fit_resample(diabetes_predictors_df, diabetes_response_df)

        (diabetes_predictors_training_df, diabetes_predictors_testing_df,
            diabetes_response_training_df, diabetes_response_testing_df) \
            = ms.train_test_split(diabetes_predictors_df, diabetes_response_df, \
                test_size = 0.2, random_state=random_state)

        algorithm = lm.LogisticRegression(max_iter=100000)
        model = algorithm.fit(diabetes_predictors_training_df, diabetes_response_training_df)
        prediction = model.predict(diabetes_predictors_testing_df)

        show_prediction_results(f'Logistic regression, {balanced_str}', prediction, diabetes_response_testing_df)
        show_prediction_results("All negative predictions", [0]*len(diabetes_response_testing_df), diabetes_response_testing_df)


def predict(diabetes_df):
    diabetes_df = diabetes_df.drop(['DNR Order', 'Med Tech'], axis='columns')

<<<<<<< HEAD
    # Fit logistic regression model
    # ADDED class_weight='balanced' to handle class imbalance after finding parameters
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(training_predictors_df, training_response)
    prediction = model.predict(testing_predictors_df)
    # print(prediction)

    # Confusion matrix and classification report ADDED
    cm = confusion_matrix(testing_response, prediction)
    print("\n=== Confusion Matrix ===")
    print(cm)
# - Top row = actual negatives
#     - First column  (0,0) = True Negatives  (TN): model predicted 0, actual was 0
#     - Second column (0,1) = False Positives (FP): model predicted 1, actual was 0
#
# - Bottom row = actual positives
#     - First column  (1,0) = False Negatives (FN): model predicted 0, actual was 1
#     - Second column (1,1) = True Positives  (TP): model predicted 1, actual was 1

    print("\n=== Classification Report ===")
    print(classification_report(testing_response, prediction))

    show_prediction_results("Logistic regression", prediction, testing_response)
    show_prediction_results("All negative predictions", [0]*len(testing_response), testing_response)
=======
    all_independent_vars = diabetes_df.columns.drop('Outcome').values.tolist()

    diabetes_predictors_df = diabetes_df[all_independent_vars]
    diabetes_response_df = diabetes_df['Outcome']
>>>>>>> fb68e5a3da97b11de8bdab4b643a8a72d70b3d32

    for balance_counter in range(2):
        perform_logistic_regression(diabetes_predictors_df, diabetes_response_df, balance_counter)


def main():
<<<<<<< HEAD
    """Main function."""
    diabetes_df = pd.read_csv(f'diabetes\pa_diabetes.csv')
    # print(diabetes_df.head())

    train(diabetes_df)
=======
    diabetes_df = pd.read_csv('pa_diabetes.csv')
    predict(diabetes_df)
>>>>>>> fb68e5a3da97b11de8bdab4b643a8a72d70b3d32


if __name__ == '__main__':
    main()
