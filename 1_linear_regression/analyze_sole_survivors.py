import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import seaborn as sns
import sklearn.metrics as metrics
from scipy.stats import zscore
import statsmodels.api as sm

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
def create_modified_df(df, model, simple_or_multiple, make_predictions):
    # Multiple linear regression without one-hot encoding
    if make_predictions:
        predictors = df.drop(columns=['Name','Leadership', 'RiskTaking', 'Resourcefulness', 'Teamwork']).values
        prediction = perform_linear_regression_prediction(model, predictors)
        modified_survivors_df = df.copy()
        modified_survivors_df['PredictedSurvivalScore'] = prediction
        # Only sort and round the prediction column, keep all predictors
        modified_survivors_df['PredictedSurvivalScore'] = modified_survivors_df['PredictedSurvivalScore'].round(2)
        modified_survivors_df = modified_survivors_df.sort_values(by='PredictedSurvivalScore', ascending=False)
        response = modified_survivors_df['PredictedSurvivalScore'].values
        response_name = 'PredictedSurvivalScore'
    else:
        if simple_or_multiple == 'multiple':
            predictors = df.drop(columns=['Name', 'SurvivalScore']).values # 'Name', 'SurvivalScore'
            modified_survivors_df = df.drop(columns=['Name', 'SurvivalScore']) # 'Name', 'SurvivalScore'
            # 'Name', 'SurvivalScore', 'Leadership', 'RiskTaking', 'Resourcefulness', 'Teamwork'
            # predictors = df[['Leadership', 'MentalToughness', 'SurvivalSkills', 'Risktaking', 'Resourcefulness', 'Adaptability', 'Physicalfitness', 'Teamwork', 'Stubbornness']].values
            # modified_survivors_df = pd.DataFrame(predictors, columns=['Leadership', 'MentalToughness', 'SurvivalSkills', 'Risktaking', 'Resourcefulness', 'Adaptability', 'Physicalfitness', 'Teamwork', 'Stubbornness'])
            response = df['SurvivalScore'].values
            response_name = 'SurvivalScore'
        # Simple linear regression
        else:
            predictors = df[['RiskTaking']].values
            modified_survivors_df = pd.DataFrame(predictors, columns=['RiskTaking'])
            response = df['SurvivalSkills'].values
            response_name = 'SurvivalSkills'

    # Return statements
    return modified_survivors_df, response_name, predictors, response, prediction if make_predictions else None

# Main function to perform linear regression, simple or multiple
def linear_regression(simple_or_multiple, sole_df, create_testing_set):
    # Set training and testing variable values
    if create_testing_set:
        training_df, testing_df = ms.train_test_split(sole_df, test_size=0.25)
    else:
        training_df = sole_df
        testing_df = None
    # Create modified dataframes for training and testing sets
    if create_testing_set: 
        training_modified_survivors_df, response_name, training_predictors, training_response, training_prediction \
        = create_modified_df(training_df, None, simple_or_multiple, make_predictions=False)
        testing_modified_survivors_df, response_name, testing_predictors, testing_response, training_prediction \
        = create_modified_df(testing_df, None, simple_or_multiple, make_predictions=False)
    else:
        training_modified_survivors_df, response_name, training_predictors, training_response, training_prediction \
        = create_modified_df(training_df, None, simple_or_multiple, make_predictions=False)
        testing_modified_survivors_df = None
        testing_predictors = None
        testing_response = None
        testing_prediction = None
    # Create model and perform prediction
    model = create_linear_regression_model(training_predictors, training_response)
    training_prediction = perform_linear_regression_prediction(model, training_predictors)
    # Perform prediction on testing set if created
    if create_testing_set:
        testing_prediction = perform_linear_regression_prediction(model, testing_predictors)

    return training_modified_survivors_df, testing_modified_survivors_df, response_name, training_prediction, training_response, training_predictors, \
        testing_prediction, testing_response, testing_predictors, model

# Function to sort predictors, response, and prediction values based on the first column of the predictor values
def sort_values(prediction, response, predictors):
    sorted_index = np.argsort(predictors[:, 0])
    sorted_prediction = np.array(prediction)[sorted_index]
    sorted_response = np.array(response)[sorted_index]
    sorted_predictors = np.array(predictors)[sorted_index, :]

    return sorted_prediction, sorted_response, sorted_predictors

# Function to print the statistical summary using statsmodels
def statistical_summary(df, predictors):
    # Set up the model
    X = sm.add_constant(predictors)  # Adds a constant term to the predictor
    y = df['SurvivalScore']
    # Fit the model
    model = sm.OLS(y, X).fit()
    print(model.summary())

# Function to calculate and print the r-squared value
def r_squared_value(model, predictors, response):
    r_squared = model.score(predictors, response)
    print(f'r-squared value: {r_squared:.4f}')

def cross_validated_r_squared(model, predictors, response):
    scores = ms.cross_val_score(model, predictors, response, scoring='r2')
    print("CV r-squared scores:", scores)

    # repeated k-fold cross-validation

# Function to calculate and print the root mean squared error
def root_mean_squared_error(prediction, response):
    mse = metrics.mean_squared_error(response, prediction)
    rmse = np.sqrt(mse)
    print(f'The RMSE: {rmse}')

# Revised function to create a heatmap for linearity check
def linearity_check(df, regression_type, response_name, response_values):
    # Create correlation matrix
    df = df.copy()
    if response_name not in df.columns:
        df[response_name] = response_values
    corr_matrix = df.corr()
    # Set up the matplotlib figure
    if regression_type == 'simple':
        # One-by-one trait–response correlation
        response_corr = corr_matrix[response_name].drop(response_name)
        sns.heatmap(response_corr.to_frame(), annot=True)
        plt.title(f"Trait–Response Correlation: {response_name}")
    else:
        # Square matrix for multiple regression
        response_corr = corr_matrix[response_name].drop(response_name)
        sorted_traits = response_corr.abs().sort_values(ascending=False).index.tolist()
        selected = sorted_traits + [response_name] if response_name not in sorted_traits else sorted_traits
        square_corr = corr_matrix.loc[selected, selected]
        mask = square_corr.abs() < 0.3
        sns.heatmap(square_corr, annot=True, mask=mask)
        plt.title(f"Correlation Matrix: Traits + {response_name}")
    # Display the heatmap
    plt.tight_layout()
    # plt.savefig('linearity_check_heatmap.png')
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
    # Calculate and print root mean squared error
    root_mean_squared_error(prediction, response)
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
    xlabel = 'Survival Skills' if simple_or_multiple == 'simple' else 'All Predictors'
    plt.xlabel(xlabel)
    ylabel = 'Predicted Survival Score'
    plt.ylabel(ylabel)
    title = f'Linear Regression: {xlabel} vs {ylabel}' # ({label})
    plt.title(title)
    plt.legend()
    # plt.savefig('multiple_linear_regression_plot.png')
    plt.show()

# Function to plot the z-score distribution of a specified column in the dataframe
def plot_zscore_distribution(df, column_name):
    # Calculate z-scores for all columns
    modified_survivors_df = df.drop(columns=['Name'])
    df_z = modified_survivors_df.apply(zscore)
    # Add z-score columns to the original dataframe
    for col in df_z.columns:
        df[f'{col}_zscore'] = df_z[col]
    # Plot the z-score distribution for the specified column
    z_col = f'{column_name}_zscore'
    if z_col in df.columns:
        sns.histplot(data=df, x=z_col, kde=True)
        plt.title(f'Z-score Distribution of {column_name}')
        plt.xlabel('Z-score')
        plt.ylabel('Frequency')
        # plt.savefig('zscore_distribution.png')
        plt.show()
    else:
        print(f'Column {z_col} not found in the dataframe.')

def plot_survival_score_distribution(past_scores, predicted_scores):
    # Calculate means
    past_mean = past_scores.mean()
    predicted_mean = predicted_scores.mean()
    # Plot histograms
    plt.figure(figsize=(10, 6))
    plt.hist(past_scores, bins=15, alpha=0.6, label='Past Survival Scores', color='skyblue', edgecolor='black')
    plt.hist(predicted_scores, bins=15, alpha=0.6, label='Predicted Survival Scores', color='salmon', edgecolor='black')
    # Add mean lines
    plt.axvline(past_mean, color='blue', linestyle='dashed', linewidth=1, label=f'Past Mean: {past_mean:.2f}')
    plt.axvline(predicted_mean, color='red', linestyle='dashed', linewidth=1, label=f'Predicted Mean: {predicted_mean:.2f}')
    # Add labels and title
    plt.xlabel('Survival Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Past and Predicted Survival Scores')
    plt.legend()
    plt.tight_layout()
    # plt.savefig('survival_score_distribution.png')
    plt.show()

# Function to handle user input prompts
def input_prompts():
    print('Welcome to the most restricted linear regression program ever!')

    print('Would you like to make predictions on the next set of survivors? (yes or no)')

    predicting_input = input().strip().lower()
    if predicting_input not in ['yes', 'no']:
        print('Invalid input. Defaulting to no predictions.')
        make_predictions = False
    else:
        make_predictions = predicting_input == 'yes'

    print(f'You have selected to {"make" if make_predictions else "not make"} predictions.')

    if not make_predictions:

        print('Please input the type of linear regression you would like to perform: simple or multiple?')

        simple_or_multiple = input().strip().lower()
        if simple_or_multiple not in ['simple', 'multiple']:
            print('Invalid input. Defaulting to simple linear regression.')
            simple_or_multiple = 'simple'

        print(f'You have selected {simple_or_multiple} linear regression.')
        print('Finally, would you like to create a testing set? (yes or no)')

        testing_input = input().strip().lower()
        if testing_input not in ['yes', 'no']:
            print('Invalid input. Defaulting to no testing set.')
            use_testing_set = False
        else:
            use_testing_set = testing_input == 'yes'

        print(f'You have selected to {"create" if use_testing_set else "not create"} a testing set.')

    if not make_predictions:
        return simple_or_multiple, use_testing_set, False
    else:
        return 'multiple', False, True

def main():

    # Load the dataset
    sole_past_df = pd.read_csv('sole_survivor_past.csv')
    sole_next_df = pd.read_csv('sole_survivor_next.csv')

    # Get user inputs for the type of regression and options
    simple_or_multiple, use_testing_set, make_predictions = input_prompts()

    # Perform linear regression
    training_modified_survivor_df, testing_modified_survivor_df, response_name, training_prediction, training_response, training_predictors, \
        testing_prediction, testing_response, testing_predictors, model \
            = linear_regression(simple_or_multiple, sole_past_df, use_testing_set)
    
    # Sort values for better plotting
    sorted_training_prediction, sorted_training_response, sorted_training_predictors = \
        sort_values(training_prediction, training_response, training_predictors)
    if use_testing_set:
        sorted_testing_prediction, sorted_testing_response, sorted_testing_predictors = \
            sort_values(testing_prediction, testing_response, testing_predictors)

    # Print and plot results
    if not make_predictions:
        if use_testing_set:
            printing_values(simple_or_multiple, False, sorted_training_prediction, sorted_training_response, sorted_training_predictors)
            printing_values(simple_or_multiple, use_testing_set, sorted_testing_prediction, sorted_testing_response, sorted_testing_predictors)
            # r_squared_value(model, sorted_training_predictors, sorted_training_response)
            linearity_check(training_modified_survivor_df, simple_or_multiple, response_name, training_response)
            plotting_values(simple_or_multiple, False, sorted_training_prediction, sorted_training_response, sorted_training_predictors, model)
            # r_squared_value(model, sorted_testing_predictors, sorted_testing_response)
            linearity_check(testing_modified_survivor_df, simple_or_multiple, response_name, testing_response)
            plotting_values(simple_or_multiple, use_testing_set, sorted_testing_prediction, sorted_testing_response, sorted_testing_predictors, model)
        else:
            # Run for both simple and multiple linear regression without a testing set
            printing_values(simple_or_multiple, use_testing_set, sorted_training_prediction, sorted_training_response, sorted_training_predictors)
            # r_squared_value(model, sorted_training_predictors, sorted_training_response)
            linearity_check(training_modified_survivor_df, simple_or_multiple, response_name, training_response)
            plotting_values(simple_or_multiple, use_testing_set, sorted_training_prediction, sorted_training_response, sorted_training_predictors, model)
            # Statistical summary and z-score distribution only for multiple linear regression
            if simple_or_multiple == 'multiple':
                statistical_summary(sole_past_df, sorted_training_predictors)
                plot_zscore_distribution(sole_past_df, 'SurvivalScore')
    else:
        # Make predictions on the next set of survivors
        # Drop unnecessary columns and create model
        # reduced_sorted_past_predictors = np.delete(sorted_training_predictors, [0, 3, 4, 7], axis=1)
        selected_columns = ['MentalToughness', 'SurvivalSkills', 'Adaptability', 'PhysicalFitness', 'Stubbornness']
        reduced_past_predictors_df = sole_past_df[selected_columns]
        reduced_past_predictors = reduced_past_predictors_df.values
        reduced_model = create_linear_regression_model(reduced_past_predictors, sole_past_df['SurvivalScore'].values)
        # Print predicted top three survivors
        print('The predicted survival scores for the next set of survivors are:')
        modified_sole_next_df, response_name, next_predictors, predicted_score, next_prediction \
            = create_modified_df(sole_next_df, reduced_model, simple_or_multiple, make_predictions=True)
        print(modified_sole_next_df[['Name', 'PredictedSurvivalScore']].head(3).to_string(index=False))
        # Create heatmap for linearity check using unsorted scores
        correlation_df = modified_sole_next_df.drop(columns=['Name', 'PredictedSurvivalScore', 'Leadership', 'RiskTaking', 'Resourcefulness', 'Teamwork'])
        linearity_check(correlation_df, 'multiple', response_name, predicted_score)
        # Sort values for better plotting
        sorted_next_prediction, sorted_next_response, sorted_next_predictors = sort_values(next_prediction, predicted_score, next_predictors)
    # R-square value dropped
        # Create reduced dataframe for plotting
        next_predictors_df = pd.DataFrame(sorted_next_predictors, columns=['MentalToughness', 'SurvivalSkills', 'Adaptability', 'PhysicalFitness', 'Stubbornness'])
    # R-squared value dropped again
        # Print and plot results

        cross_validated_r_squared(reduced_model, next_predictors_df.values, sorted_next_response)

        plotting_values(simple_or_multiple, False, sorted_next_prediction, sorted_next_response, next_predictors_df.values, reduced_model)
        # Plot distribution of past and predicted survival scores
        past_score = sole_past_df['SurvivalScore'].round(2)
        plot_survival_score_distribution(past_score, predicted_score)

if __name__ == "__main__":
    main()