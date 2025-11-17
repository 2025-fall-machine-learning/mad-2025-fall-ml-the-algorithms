import numpy as np
import pandas as pd
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import sklearn.metrics as metrics
import imblearn.over_sampling as ios
import matplotlib.pyplot as plt
import seaborn as sns


N_FOLD_CROSS_VALIDATION = 10


def show_prediction_results(header, prediction, actual_data):
    np_prediction = np.round(prediction).astype(int)
    np_actual = np.array(actual_data).astype(int)
    result = np_actual == np_prediction

    num_correct_predictions = np.count_nonzero(result)
    num_incorrect_predictions = np.count_nonzero(result == False)
    print('{0}: {1} correct, {2} incorrect, precision: {3}.'
        .format(header, num_correct_predictions, num_incorrect_predictions,
        (num_correct_predictions/(num_correct_predictions+num_incorrect_predictions))))
    return (num_correct_predictions/(num_correct_predictions+num_incorrect_predictions))


def compute_confusion_matrix_numbers(actual_data_df, prediction):
    confusion_tuple = metrics.confusion_matrix(actual_data_df, prediction)
    command_line_display_as_accuracy_top_confusion_matrix = confusion_tuple.T
    command_line_display_as_accuracy_top_confusion_matrix = np.flip(command_line_display_as_accuracy_top_confusion_matrix, axis=0)
    command_line_display_as_accuracy_top_confusion_matrix = np.flip(command_line_display_as_accuracy_top_confusion_matrix, axis=1)
    true_negs = 0
    false_poss = 0
    false_negs = 0
    true_poss = 0
    sensitivity = 0
    specificity = 0
    if len(confusion_tuple.ravel()) == 4:
        (true_negs, false_poss, false_negs, true_poss) = confusion_tuple.ravel()
        if ((true_poss + false_negs) > 0) and ((true_negs + false_poss) > 0):
            sensitivity = true_poss / (true_poss + false_negs)
            specificity = true_negs / (true_negs + false_poss)
    return (confusion_tuple, command_line_display_as_accuracy_top_confusion_matrix, true_negs, false_poss, false_negs, true_poss, sensitivity, specificity)


def create_confusion_matrix(actual_data_df, prediction):
    (confusion_tuple, command_line_display_as_accuracy_top_confusion_matrix, true_negs, false_poss, false_negs, true_poss, sensitivity, specificity) \
        = compute_confusion_matrix_numbers(actual_data_df, prediction)
    # if (sensitivity > 0) or (specificity > 0):
    #     print(f'tp: {true_poss}, fn: {false_negs}, tn: {true_negs}, fp: {false_poss}, sensitivity: {sensitivity}, specificity: {specificity}.')
    # print(command_line_display_as_accuracy_top_confusion_matrix)
    # sns.heatmap(confusion_tuple, annot=True)
    # plt.show()

# Initialized constants for summary stats indexing
UNBALANCED_POS = 0
BALANCED_POS = 1
ACTUAL_DATA = 0
ALL_NEGATIVES = 1


def perform_logistic_regression(diabetes_predictors_df, diabetes_response_df,
                                balance_counter, summary_stats):
    balanced_str = 'unbalanced'
    if balance_counter == 1:
        balanced_str = 'balanced'

    for random_state in range(0, 3):

        # Make copies to avoid modifying original dataframes
        predictors_df = diabetes_predictors_df.copy()
        response_df = diabetes_response_df.copy()

        if balance_counter == 1:
            random_over_sampler = ios.RandomOverSampler(random_state=random_state)
            predictors_df, response_df \
                = random_over_sampler.fit_resample(predictors_df, response_df)

        (diabetes_predictors_training_df, diabetes_predictors_testing_df,
            diabetes_response_training_df, diabetes_response_testing_df) \
            = ms.train_test_split(predictors_df, response_df, \
                test_size = 0.2, random_state=random_state)

        algorithm = lm.LogisticRegression(max_iter=100000)
        model = algorithm.fit(diabetes_predictors_training_df, diabetes_response_training_df)
        prediction = model.predict(diabetes_predictors_testing_df)

        show_prediction_results(f'Logistic regression, {balanced_str}', prediction, diabetes_response_testing_df)
        # accuracy = metrics.accuracy_score(diabetes_response_testing_df, prediction)
        # print('Sklearn accuracy: {0}'.format(accuracy))
        # Seems pretty good, doesn't it? Looks can be deceiving.
        (confusion_tuple, command_line_display_as_accuracy_top_confusion_matrix, true_negs,
                false_poss, false_negs, true_poss, sensitivity, specificity) \
            = compute_confusion_matrix_numbers(diabetes_response_testing_df, prediction)

        summary_stats[balance_counter][ACTUAL_DATA].append(
                            [true_negs, false_poss, false_negs, true_poss])

        # When unbalanced, we achieve ~60+% just predicting everybody does not have diabetes! This is
        # the problem of unbalanced data. We have many more people without diabetes than with diabetes.
        all_negatives_prediction = [0]*len(diabetes_response_testing_df)

        show_prediction_results("All negatives predictions", all_negatives_prediction, diabetes_response_testing_df)
        # accuracy = metrics.accuracy_score(diabetes_response_testing_df, all_negatives_prediction)
        # print('Sklearn accuracy: {0}'.format(accuracy))
        (confusion_tuple, command_line_display_as_accuracy_top_confusion_matrix, true_negs,
                false_poss, false_negs, true_poss, sensitivity, specificity) \
            = compute_confusion_matrix_numbers(diabetes_response_testing_df, all_negatives_prediction)

        summary_stats[balance_counter][ALL_NEGATIVES].append(
                            [true_negs, false_poss, false_negs, true_poss])

    # Moved return statement outside of the random_state loop
    return summary_stats


def predict(diabetes_df):
    diabetes_df = diabetes_df.drop(['DNR Order', 'Med Tech'], axis='columns')

    all_independent_vars = diabetes_df.columns.drop('Outcome').values.tolist()

    diabetes_predictors_df = diabetes_df[all_independent_vars]
    diabetes_response_df = diabetes_df['Outcome']

    # Properly initialize summary_stats as a list of lists
    summary_stats = [
        [[], []],
        [[], []]
    ]

    for balance_counter in range(2):
        summary_stats = perform_logistic_regression(diabetes_predictors_df, diabetes_response_df,
                                                    balance_counter, summary_stats)


def main():
    diabetes_csv = 'C:\\Users\\Emogen\\OneDrive\\Documents\\GitHub\\mad-2025-fall-ml-the-algorithms\\2_classification\\diabetes\\pa_diabetes.csv'
    diabetes_df = pd.read_csv(diabetes_csv)
    predict(diabetes_df)


if __name__ == '__main__':
    main()
