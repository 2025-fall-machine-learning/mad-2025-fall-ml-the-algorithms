import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import seaborn as sns
import matplotlib.pyplot as plt
import imblearn.over_sampling as ios
import sklearn.metrics as metrics

# Step 4: Check the data for linearity and independence
def linearity_check(df):
    corr_matrix = df.corr()
    response_corr = corr_matrix['Survived'].drop('Survived')
    sorted_traits = response_corr.abs().sort_values(ascending=False).index.tolist()
    selected = sorted_traits + ['Survived']
    square_corr = corr_matrix.loc[selected, selected]
    mask = square_corr.abs() < 0.1
    sns.heatmap(square_corr, annot=True, mask=mask)
    plt.title(f'Correlation Matrix for Survived + Traits')
    plt.tight_layout()
    # plt.show()

# Step 6: Compare response to an array of ones and prediction
def print_comparison_results(comparison, prediction, response_df):
    np_actual = np.array(response_df)
    # print(f"Response: {np_actual}")

    if comparison == 'Logistic':
        np_prediction = np.array(prediction)
        result = np_actual == np_prediction
    else:
        prediction_length = len(prediction)
        ones_array = np.ones(prediction_length)
        int_array = ones_array.astype(int)
        result = np_actual == int_array
        # print(f"Ones Array: {int_array}")

    num_correct_predictions = np.count_nonzero(result)
    num_incorrect_predictions = np.count_nonzero(result == False)
    precision = num_correct_predictions / (num_correct_predictions + num_incorrect_predictions) if (num_correct_predictions + num_incorrect_predictions) > 0 else 0

    print(f"{comparison} Comparison: {num_correct_predictions} correct predictions, {num_incorrect_predictions} incorrect predictions, precision: {precision}")

def cross_validated_accuracy(model, predictors, response):
    scores = ms.cross_val_score(model, predictors, response, cv=5, scoring='accuracy')
    print(f"Mean CV accuracy: {scores.mean():.3f}")

# Step 11: Create confusion matrix to get sensitivity and specificity
def create_confusion_matrix(actual_data_df, prediction):
    (confusion_tuple, command_line_display_as_accuracy_top_confusion_matrix, true_negs, false_poss, false_negs, true_poss, sensitivity, specificity) \
        = compute_confusion_matrix_numbers(actual_data_df, prediction)
    
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

# Step 3: Perform Logistic Regression
def perform_logistic_regression(dataframe, balance_counter):
    titanic_df = dataframe.copy()
    titanic_df = titanic_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], axis='columns')

    # Step 7: Convert categorical variables using one-hot encoding
    titanic_df = pd.get_dummies(titanic_df, columns=['Sex', 'Embarked'], prefix=['Sex', 'Embarked'], drop_first=False)

    independent_vars = titanic_df.columns.drop('Survived').values.tolist()

    # print(f"DataFrame after encoding categorical variables:\n{titanic_df.head()}")

    titanic_predictors_df = titanic_df[independent_vars]
    titanic_response_df = titanic_df['Survived']

    # Step 10: Handle class imbalance on the second iteration
    if balance_counter == 1:
        random_over_sampler = ios.RandomOverSampler(random_state=1)
        titanic_predictors_df, titanic_response_df = random_over_sampler.fit_resample(titanic_predictors_df, titanic_response_df)

    (titanic_predictors_training_df, titanic_predictors_testing_df, titanic_response_training_df, titanic_response_testing_df) \
        = ms.train_test_split(titanic_predictors_df, titanic_response_df, test_size=0.2, random_state=1)

    # Step 9: Change max_iter to 100000 to ensure convergence
    logistic_regression_algorithm = lm.LogisticRegression(max_iter=100000)
    model = logistic_regression_algorithm.fit(titanic_predictors_training_df, titanic_response_training_df)
    prediction = model.predict(titanic_predictors_testing_df)

    linearity_check(titanic_df)

    # Step 5: Check that all values are zeros or ones
    # print(f"Prediction: {prediction}")

    print_comparison_results('Logistic', prediction, titanic_response_testing_df)
    print_comparison_results('Ones', prediction, titanic_response_testing_df)

    cross_validated_accuracy(model, titanic_predictors_df, titanic_response_df)

    (confusion_tuple, command_line_display_as_accuracy_top_confusion_matrix, true_negs,
                false_poss, false_negs, true_poss, sensitivity, specificity) \
            = compute_confusion_matrix_numbers(titanic_response_testing_df, prediction)
    
    if balance_counter == 0:
        balance_status = "Without Balancing"
    else:
        balance_status = "With Balancing"

    print(f"Confusion Matrix {balance_status}: sensitivity = {sensitivity:.3f}, specificity = {specificity:.3f}")

def main():
    # Step 1: Load the dataset
    titanic_dataset = pd.read_csv("Titanic-Dataset.csv")

    # Step 2: Drop rows with missing values
    titanic_dataset['Cabin'] = titanic_dataset['Cabin'].replace('', np.nan)
    titanic_dataset.dropna(inplace=True)

    # print(f"Dataset shape after dropping missing values: {titanic_dataset.shape}")

    balance_counter = 0

    for balance_counter in range(2):
        perform_logistic_regression(titanic_dataset, balance_counter)

if __name__ == "__main__":
    main()