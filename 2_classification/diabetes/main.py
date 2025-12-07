import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


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
    predictors_df = diabetes_df.drop('Outcome', axis='columns')
    response = diabetes_df['Outcome']
    
    training_predictors_df, testing_predictors_df, training_response, testing_response = train_test_split(
        predictors_df, response, test_size=0.2, random_state=489
    )

    # Fit logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(training_predictors_df, training_response)
    prediction = model.predict(testing_predictors_df)
    # print(prediction)

    show_prediction_results("Logistic regression", prediction, testing_response)
    show_prediction_results("All negative predictions", [0]*len(testing_response), testing_response)

    # print(predictors_df.head())
    # print(response.head())


def main():
    """Main function."""
    diabetes_df = pd.read_csv("pa_diabetes.csv")
    # print(diabetes_df.head())

    train(diabetes_df)


if __name__ == '__main__':
    main()
