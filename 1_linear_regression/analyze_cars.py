import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import seaborn as sns

# Function to print summaries of the data
def print_data_summary(data):
    numpified_data = np.array(data)
    # Flatten if the array is 2D
    if numpified_data.ndim == 2 and numpified_data.shape[1] == 1:
        flattened_numpified_data = numpified_data.flatten()
    else:
        flattened_numpified_data = numpified_data
    # Format first 5 and last 5 values
    first_five = ", ".join(f"{x:7.3f}" for x in flattened_numpified_data[:5])
    last_five = ", ".join(f"{x:7.3f}" for x in flattened_numpified_data[-5:])
    print(f"[{first_five}, ..., {last_five}]")

# Function to create the line of best fit
def create_linear_regression_model(predictors, response):
    model = lm.LinearRegression()
    model.fit(predictors, response)

    return model

# Function to perform prediction, creating the plotted data points
def perform_linear_regression_prediction(model, predictors):
    prediction = model.predict(predictors)

    return prediction

# Function to create modified dataframe for linearity check and regression
def create_modified_df(simple_or_multiple, df, one_hot_encode, reference_columns=None):
    # Create modified dataframe based on user inputs
    if simple_or_multiple == 'multiple' and one_hot_encode:
        carname_dummies = pd.get_dummies(df['fuelsystem'], prefix='fuelsystem') # Changed from 'carname' to 'fuelsystem'
        # Ensure the dummy columns match the reference columns if provided
        if reference_columns is not None:
            carname_dummies = carname_dummies.reindex(columns=reference_columns, fill_value=0)
        # Combine the one-hot encoded columns with the original dataframe
        predictors = pd.concat([df[['carwidth', 'enginesize']], carname_dummies], axis=1).values # Removed 'carlength' and replaced 'carheight' with 'enginesize'
        modified_cars_df = pd.DataFrame(predictors, columns=['carwidth', 'enginesize'] + list(carname_dummies.columns)) # Removed 'carlength'
        response = df['horsepower'].values
        response_name = 'horsepower'
    # Multiple linear regression without one-hot encoding
    elif simple_or_multiple == 'multiple':
        predictors = df[['carwidth', 'enginesize']].values # Removed 'carlength'
        modified_cars_df = pd.DataFrame(predictors, columns=['carwidth', 'enginesize']) # Removed 'carlength' due to high correlation with 'carwidth'
        response = df['horsepower'].values
        response_name = 'horsepower'
    # Simple linear regression
    else:
        predictors = df[['enginesize']].values
        modified_cars_df = pd.DataFrame(predictors, columns=['enginesize'])
        response = df['price'].values
        response_name = 'price'
    # Return statements based on one-hot encoding and regression type
    if reference_columns is None and one_hot_encode and simple_or_multiple == 'multiple':
        return modified_cars_df, response_name, predictors, response, list(carname_dummies.columns)
    else:
        return modified_cars_df, response_name, predictors, response, None

# Main function to perform linear regression, simple or multiple
def linear_regression(simple_or_multiple, cars_df, create_testing_set, one_hot_encode):
    # Set training and testing variable values
    if create_testing_set:
        training_df, testing_df = ms.train_test_split(cars_df, test_size=0.25)
    else:
        training_df = cars_df
        testing_df = None
    # Create modified dataframes for training and testing sets
    if create_testing_set: 
        training_modified_cars_df, response_name, training_predictors, training_response, dummy_columns \
        = create_modified_df(simple_or_multiple, training_df, one_hot_encode)
        testing_modified_cars_df, response_name, testing_predictors, testing_response, dummy_columns \
        = create_modified_df(simple_or_multiple, testing_df, one_hot_encode, reference_columns=dummy_columns)
    else:
        training_modified_cars_df, response_name, training_predictors, training_response, dummy_columns \
        = create_modified_df(simple_or_multiple, training_df, one_hot_encode)
        testing_modified_cars_df = None
        testing_predictors = None
        testing_response = None
        testing_prediction = None
    # Create model and perform prediction
    model = create_linear_regression_model(training_predictors, training_response)
    training_prediction = perform_linear_regression_prediction(model, training_predictors)
    # Perform prediction on testing set if created
    if create_testing_set:
        testing_prediction = perform_linear_regression_prediction(model, testing_predictors)

    return training_modified_cars_df, testing_modified_cars_df, response_name, training_prediction, training_response, training_predictors, \
        testing_prediction, testing_response, testing_predictors, model

# Function to sort predictors, response, and prediction values based on the first column of the predictor values
def sort_values(prediction, response, predictors):
    sorted_index = np.argsort(predictors[:, 0])
    sorted_prediction = np.array(prediction)[sorted_index]
    sorted_response = np.array(response)[sorted_index]
    sorted_predictors = np.array(predictors)[sorted_index, :]

    return sorted_prediction, sorted_response, sorted_predictors

# Function to calculate and print the r-squared value
def r_squared_value(model, predictors, response):
    r_squared = model.score(predictors, response)
    print(f'r-squared value: {r_squared:.4f}')

# Function to create a heatmap for linearity check
def linearity_check(modified_cars_df, simple_or_multiple, response_name, response_values, top_n=10):
    # Create correlation matrix
    modified_cars_df[response_name] = response_values
    correlation_matrix = modified_cars_df.corr()
    # Plot heatmap based on regression type
    if simple_or_multiple == 'simple':
        response_corr = correlation_matrix[response_name].drop(response_name)
        sns.heatmap(response_corr.to_frame(), annot=True)
        plt.show()
    else:
        # Get top N predictors most correlated with the response for one-hot encoded multiple regression
        response_correlation_matrix = correlation_matrix[[response_name]].drop(response_name)
        top_predictors = response_correlation_matrix[response_name].abs().nlargest(top_n).index
        top_plus_response = list(top_predictors) + [response_name]
        square_corr_matrix = correlation_matrix.loc[top_plus_response, top_plus_response]
        # Use a mask to display only correlations above 0.3
        mask = square_corr_matrix < 0.3
        sns.heatmap(square_corr_matrix, annot=True, mask=mask)
        plt.show()

# Function to print the values of the predictors, prediction, and response
def printing_values(simple_or_multiple, test_set_created, prediction, response, predictors):
    testing_or_training = 'testing' if test_set_created else 'training'
    if simple_or_multiple == 'simple':
        print(f'"The {testing_or_training} data predictors, prediction and response values:"')
        print_data_summary(predictors)
    else:
        print(f'"The {testing_or_training} data prediction and response values:"')
    print_data_summary(prediction)
    print_data_summary(response)

# Function to plot the values of the predictors, prediction, and response
def plotting_values(simple_or_multiple, test_set_created, prediction, response, predictors, model):
    # Calculate and print r-squared value
    r_squared_value(model, predictors, response)
    # Create scatter plot with line of best fit
    color = 'green' if test_set_created else 'blue'
    label = 'Testing Data' if test_set_created else 'Training Data'
    if simple_or_multiple == 'multiple':
        predictors = predictors[:, 0]
    else:
        predictors = predictors.flatten()
    plt.scatter(predictors, response, color=color, label=label)
    plt.plot(predictors, prediction, color = 'red', label = 'Best Fit Line')
    xlabel = 'Engine Size' if simple_or_multiple == 'simple' else 'Car Dimensions'
    plt.xlabel(xlabel)
    ylabel = 'Price' if simple_or_multiple == 'simple' else 'Horsepower'
    plt.ylabel(ylabel)
    title = f'Linear Regression: {xlabel} vs {ylabel} ({label})'
    plt.title(title)
    plt.legend()
    plt.show()

# Function to handle user input prompts
def input_prompts():
    print('Welcome to the most restricted linear regression program ever!')
    print('Please input the type of linear regression you would like to perform: simple or multiple?')

    simple_or_multiple = input().strip().lower()
    if simple_or_multiple not in ['simple', 'multiple']:
        print('Invalid input. Defaulting to simple linear regression.')
        simple_or_multiple = 'simple'

    print(f'You have selected {simple_or_multiple} linear regression.')
    print('Next, would you like to create a testing set? (yes or no)')

    testing_input = input().strip().lower()
    if testing_input not in ['yes', 'no']:
        print('Invalid input. Defaulting to no testing set.')
        use_testing_set = False
    else:
        use_testing_set = testing_input == 'yes'

    print(f'You have selected to {"create" if use_testing_set else "not create"} a testing set.')
    print('Finally, would you like to one-hot encode the car names? (yes or no)')

    if simple_or_multiple == 'simple':
        print('Note: One-hot encoding is only applicable for multiple linear regression. Defaulting to no one-hot encoding.')
        one_hot_encode = False
    else:
        encoding_input = input().strip().lower()
        if encoding_input not in ['yes', 'no']:
            print('Invalid input. Defaulting to no one-hot encoding.')
            one_hot_encode = False
        else:
            one_hot_encode = encoding_input == 'yes'

    print(f'You have selected to {"one-hot encode" if one_hot_encode else "not one-hot encode"} the car names.')
    print('Performing linear regression...')

    return simple_or_multiple, use_testing_set, one_hot_encode

def main():

    # Load the dataset
    cars_df = pd.read_csv('cars.csv')

    # Get user inputs for the type of regression and options
    simple_or_multiple, use_testing_set, one_hot_encode = input_prompts()

    # Perform linear regression
    training_modified_cars_df, testing_modified_cars_df, response_name, training_prediction, training_response, training_predictors, \
        testing_prediction, testing_response, testing_predictors, model \
            = linear_regression(simple_or_multiple, cars_df, use_testing_set, one_hot_encode)
    
    # Sort values for better plotting
    sorted_training_prediction, sorted_training_response, sorted_training_predictors = \
        sort_values(training_prediction, training_response, training_predictors)
    if use_testing_set:
        sorted_testing_prediction, sorted_testing_response, sorted_testing_predictors = \
            sort_values(testing_prediction, testing_response, testing_predictors)

    # Print and plot results
    if use_testing_set:
        printing_values(simple_or_multiple, False, sorted_training_prediction, sorted_training_response, sorted_training_predictors)
        printing_values(simple_or_multiple, use_testing_set, sorted_testing_prediction, sorted_testing_response, sorted_testing_predictors)
        # r_squared_value(model, sorted_training_predictors, sorted_training_response)
        linearity_check(training_modified_cars_df, simple_or_multiple, response_name, training_response)
        plotting_values(simple_or_multiple, False, sorted_training_prediction, sorted_training_response, sorted_training_predictors, model)
        # r_squared_value(model, sorted_testing_predictors, sorted_testing_response)
        linearity_check(testing_modified_cars_df, simple_or_multiple, response_name, testing_response)
        plotting_values(simple_or_multiple, use_testing_set, sorted_testing_prediction, sorted_testing_response, sorted_testing_predictors, model)
    else:
        printing_values(simple_or_multiple, use_testing_set, sorted_training_prediction, sorted_training_response, sorted_training_predictors)
        # r_squared_value(model, sorted_training_predictors, sorted_training_response)
        linearity_check(training_modified_cars_df, simple_or_multiple, response_name, training_response)
        plotting_values(simple_or_multiple, use_testing_set, sorted_training_prediction, sorted_training_response, sorted_training_predictors, model)

if __name__ == "__main__":
    main()