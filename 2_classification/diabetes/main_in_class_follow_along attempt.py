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
    
def perform_logistic_regression(diabetes_predictors_df, diabetes_response_df, balance_counter, summary_stats)

    balanced_str = 'unbalanced'
    if balance_counter == 1:
        balanced_str = 'balanced'
        
    for random_state in range(0, 3):
        if balance_counter == 1:
            random_over_sampler = ios.RandomOverSampler(random_state=random_state)
            diabetes_predictors_df, diabetes_response_df \
                = random_over_sampler.fit_resample(diabetes_predictors_df, diabetes_response_df)
                
    (diabetes_predictors_training_df, )
    summary_stats[balance_counter][ALL_NEGATIVES].insert(random_state, 
                        [true_negs, false_poss, false_negs, true_poss])
        return summary_stats
    
def predict(diabetes_df):
    diabetes_df = diabetes_df.drop(['DNR Order', 'Med Tech'], axis='columns')
    
    all_independant_vars = diabetes_df.columns.drop('Outcome').values.tolist()
    
    diabetes_predictors_df = diabetes_df[all_independant_vars]
    diabetes_response_df = diabetes_df['Outcome']
    
    for balance_counter in range(2):
        summary_stats = perform_logistic_regression(diabetes_predictors_df, diabetes_response_df, 
                                                    balance_counter, None)

def main():
    """Main function."""
    diabetes_df = pd.read_csv("pa_diabetes.csv")
    # print(diabetes_df.head())
    predict(diabetes_df)
    train(diabetes_df)


if __name__ == "__main__":
    main()
