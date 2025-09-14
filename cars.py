# Helper to robustly find column names
import re
def _canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

def find_col(df, candidates):
    canon_map = {_canon(c): c for c in df.columns}
    for name in candidates:
        key = _canon(name)
        if key in canon_map:
            return canon_map[key]
    raise KeyError(f"None of the columns {candidates} were found in {list(df.columns)}")
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import sklearn.model_selection as ms

# Utility to print summary of 1D data
def print_1d_data_summary(data_1d):
    numpified_data = np.array(data_1d)
    if numpified_data.ndim == 2 and numpified_data.shape[1] == 1:
        flattened_numpified_data = numpified_data.flatten()
    else:
        flattened_numpified_data = numpified_data
    first_five = ", ".join(f"{x:7.3f}" for x in flattened_numpified_data[:5])
    last_five = ", ".join(f"{x:7.3f}" for x in flattened_numpified_data[-5:])
    print(f"[{first_five}, ..., {last_five}]")

# Helper to print inspection data
def print_inspection_data(predictors=None, prediction=None, response=None, prefix=None):
    if prefix:
        print(prefix)
    if predictors is not None:
        print_1d_data_summary(predictors)
    if prediction is not None:
        print_1d_data_summary(prediction)
    if response is not None:
        print_1d_data_summary(response)

# Helper to plot regression results
def plot_regression(x, y, y_pred, xlabel, ylabel, title, color_data, color_line, label_data, label_line):
    plt.scatter(x, y, color=color_data, label=label_data)
    plt.plot(x, y_pred, color=color_line, label=label_line)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

# Simple Linear Regression: engine size vs price
def simple_linear_regression_cars(cars_df, create_testing_set):
    eng_col = find_col(cars_df, ['engine-size', 'enginesize', 'engine_size', 'Engine Size'])
    price_col = find_col(cars_df, ['price', 'msrp', 'saleprice', 'Price'])
    predictors = cars_df[[eng_col]].values
    response = cars_df[price_col].values

    training_predictors = predictors
    training_response = response

    if create_testing_set:
        training_predictors, testing_predictors, training_response, testing_response = ms.train_test_split(
            predictors, response, test_size=0.25, random_state=42)

    model = lm.LinearRegression()
    model.fit(training_predictors, training_response)
    prediction = model.predict(training_predictors)

    print("Simple Linear Regression: engine-size vs price (Training Data)")
    print_inspection_data(training_predictors, prediction, training_response)
    plot_regression(training_predictors, training_response, prediction,
                   xlabel='engine-size', ylabel='price',
                   title='Linear Regression: engine-size vs price (Training Data)',
                   color_data='blue', color_line='red',
                   label_data='Training Data', label_line='Best Fit Line')

    if create_testing_set:
        prediction = model.predict(testing_predictors)
        print("Simple Linear Regression: engine-size vs price (Testing Data)")
        print_inspection_data(testing_predictors, prediction, testing_response)
        plot_regression(testing_predictors, testing_response, prediction,
                       xlabel='engine-size', ylabel='price',
                       title='Linear Regression: engine-size vs price (Testing Data)',
                       color_data='green', color_line='red',
                       label_data='Testing Data', label_line='Best Fit Line')

# Multiple Linear Regression: use all numeric features + one-hot encoding for categoricals
def multiple_linear_regression_cars(cars_df, create_testing_set, one_hot_encode=True):
    price_col = find_col(cars_df, ['price', 'msrp', 'saleprice', 'Price'])
    predictors = cars_df.drop(columns=[price_col])
    response = cars_df[price_col].values

    # One-hot encode categorical columns if requested
    if one_hot_encode:
        predictors = pd.get_dummies(predictors, drop_first=True)
    else:
        predictors = predictors.select_dtypes(include=[np.number])

    predictors = predictors.values

    training_predictors = predictors
    training_response = response

    if create_testing_set:
        training_predictors, testing_predictors, training_response, testing_response = ms.train_test_split(
            predictors, response, test_size=0.25, random_state=42)

    model = lm.LinearRegression()
    model.fit(training_predictors, training_response)
    prediction = model.predict(training_predictors)

    print("Multiple Linear Regression: all features (Training Data)")
    print_inspection_data(prediction=prediction, response=training_response)
    # For multiple regression, plot prediction vs actual
    plt.scatter(training_response, prediction, color='blue', label='Training Data')
    plt.plot([min(training_response), max(training_response)], [min(training_response), max(training_response)], color='red', label='Ideal Fit')
    plt.xlabel('Actual price')
    plt.ylabel('Predicted price')
    plt.title('Multiple Regression: Predicted vs Actual (Training Data)')
    plt.legend()
    plt.show()

    if create_testing_set:
        prediction = model.predict(testing_predictors)
        print("Multiple Linear Regression: all features (Testing Data)")
        print_inspection_data(prediction=prediction, response=testing_response)
        plt.scatter(testing_response, prediction, color='green', label='Testing Data')
        plt.plot([min(testing_response), max(testing_response)], [min(testing_response), max(testing_response)], color='red', label='Ideal Fit')
        plt.xlabel('Actual price')
        plt.ylabel('Predicted price')
        plt.title('Multiple Regression: Predicted vs Actual (Testing Data)')
        plt.legend()
        plt.show()

# Main function
def main():
    cars_df = pd.read_csv('cars.csv')
    simple_linear_regression_cars(cars_df, create_testing_set=True)
    multiple_linear_regression_cars(cars_df, create_testing_set=True, one_hot_encode=True)

if __name__ == "__main__":
    main()
