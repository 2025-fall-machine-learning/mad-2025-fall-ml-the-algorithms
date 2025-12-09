import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import sklearn.metrics as metrics

def linearity_check(df, predictor):
    corr_matrix = df.corr()
    response_corr = corr_matrix[predictor].drop(predictor)
    strong_corr = response_corr[response_corr.abs() > 0.1]
    strong_corr_sorted = strong_corr.reindex(strong_corr.abs().sort_values(ascending=False).index)

    print(f"Variables strongly correlated with {predictor}:\n{strong_corr_sorted}\n")

def calculate_rmse(algorithm_name, actual, predicted):
    mse = metrics.mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    print(f"{algorithm_name} RMSE: {rmse}")

def perform_linear_regression(df, predictor, test_size, random_state):
    independent_vars = df.columns.drop(predictor).values.tolist()

    # Part 1 Step 3: Define predictors and response variables
    predictors_df = df[independent_vars]
    response_df = df[predictor]

    (predictors_training_df, predictors_testing_df, response_training_df, response_testing_df) \
        = ms.train_test_split(predictors_df, response_df, test_size=test_size, random_state=random_state) # Part 1 Step 7: Remove , random_state=1)
    
    model = lm.LinearRegression()
    algorithm = model.fit(predictors_training_df, response_training_df)
    prediction = algorithm.predict(predictors_testing_df)

    # Part 1 Step 4: Calculate and print RMSE
    calculate_rmse("Linear Regression", response_testing_df, prediction)
    # print(response_testing_df.describe())

    # Part 1 Step 5: Perform lasso and ridge regression
    lr_algorithm = lm.LassoCV(cv=5)
    lr_model = lr_algorithm.fit(predictors_training_df, response_training_df.values.ravel())
    lr_prediction = lr_model.predict(predictors_testing_df)

    ridge_algorithm = lm.RidgeCV(cv=5)
    ridge_model = ridge_algorithm.fit(predictors_training_df, response_training_df)
    ridge_prediction = ridge_model.predict(predictors_testing_df)

    # Part 1 & Part 2 Step 6: Calculate and print RMSE for lasso and ridge
    calculate_rmse("Lasso Regression", response_testing_df, lr_prediction)
    calculate_rmse("Ridge Regression", response_testing_df, ridge_prediction)

def main():
    # Part 1 Step 1: Read the CSV file into a DataFrame
    marketing_csv = pd.read_csv("marketing_buckets.csv")

    # Part 2 Step 1: Read the CSV file into a DataFrame
    gene_csv = pd.read_csv("gene_expressions.csv")

    dataframe = gene_csv.copy()

    # Part 1 Step 2: Display the first 15 rows
    # print(dataframe.head(15))

    # Part 2 Step 2: Display the first few and last few rows
    # print(dataframe.head(3))
    # print(dataframe.tail(3))

    # Part 2 Step 4: Check Pearson's R correlation
    # linearity_check(dataframe, 'y')

    # Part 1 Step 3: Train and Split the Data
    # perform_linear_regression(dataframe, predictor='LVT', test_size=0.3, random_state=None)

    # Part 2 Step 5: Perform Linear Regression
    perform_linear_regression(dataframe, predictor='y', test_size=0.2, random_state=0)   

if __name__ == "__main__":
    main()