import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def show_prediction_results(header, prediction, actual_data):
    np_prediction = np.array(prediction)
    np_actual = np.array(actual_data)
    print(np_prediction)
    print(np_actual)
    right_counter = 0
    wrong_counter = 0
    for (pred, actual) in zip(np_prediction, np_actual):
        # print(f"Predicted: {pred}, Actual: {actual}")
        if pred == actual:
            right_counter = right_counter + 1
        else:
            wrong_counter = wrong_counter + 1

    print()
    print(header)
    print(f"Right predictions: {right_counter}")
    print(f"Wrong predictions: {wrong_counter}")
    print(f"Accuracy: {right_counter / (right_counter + wrong_counter):.2%}")


def train(diabetes_df):
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


if __name__ == "__main__":
    main()
