import logging
import io
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.stats.diagnostic as sms
import sklearn.metrics as metrics


def predict(lasso_ridge_sample_df):
    # Perform a cursory inspection of the data.
    print(lasso_ridge_sample_df.head())
    print(lasso_ridge_sample_df.describe())
    print(lasso_ridge_sample_df.columns)

    # correlation_matrix = lasso_ridge_sample_df.corr()
    # price_predictors_matrix = correlation_matrix[['col6']].sort_values(by='col6', ascending=False)
    # print(price_predictors_matrix)
    # print(correlation_matrix)

    predictors_df = lasso_ridge_sample_df.drop(['col6'], axis='columns')
    response_df = lasso_ridge_sample_df[['col6']]

    # # Split the data into training and testing sets.
    predictors_training_df, predictors_testing_df, \
        response_training_df, response_testing_df \
            = ms.train_test_split(predictors_df, response_df,
                train_size=5, test_size=5, random_state=4)

    # Train and predict.
    lr_algorithm = lm.LinearRegression()
    lr_model = lr_algorithm.fit(predictors_training_df, response_training_df)
    lr_prediction = lr_model.predict(predictors_testing_df)

    lasso_algorithm_1 = lm.LassoCV(cv=5)
    lasso_model_1 = lasso_algorithm_1.fit(predictors_training_df, response_training_df.values.ravel())
    lasso_prediction_1 = lasso_model_1.predict(predictors_testing_df)

    ridge_algorithm_1 = lm.RidgeCV(scoring='neg_mean_squared_error')
    ridge_model_1 = ridge_algorithm_1.fit(predictors_training_df, response_training_df)
    ridge_prediction_1 = ridge_model_1.predict(predictors_testing_df)

    # Calculate r-squared.
    lr_r_squared = lr_algorithm.score(predictors_training_df, response_training_df)
    print(f"R-squared: {lr_r_squared}")

    lasso_r_squared = lasso_algorithm_1.score(predictors_training_df, response_training_df)
    print(f"R-squared: {lasso_r_squared}")

    ridge_r_squared = ridge_algorithm_1.score(predictors_training_df, response_training_df)
    print(f"R-squared: {ridge_r_squared}")

    print(response_df.describe())
    
    lr_mse = metrics.mean_squared_error(response_testing_df, lr_prediction)
    lr_rmse = np.sqrt(lr_mse)
    print(f'Linear RMSE: {lr_rmse}.')

    lasso_mse = metrics.mean_squared_error(response_testing_df, lasso_prediction_1)
    lasso_rmse = np.sqrt(lasso_mse)
    print(f' Lasso RMSE: {lasso_rmse}.')

    ridge_mse = metrics.mean_squared_error(response_testing_df, ridge_prediction_1)
    ridge_rmse = np.sqrt(ridge_mse)
    print(f' Ridge RMSE: {ridge_rmse}.')


def main():
    print('Python HTTP trigger function processed a request.')

    filename = 'lasso_ridge_sample.csv'

    lasso_ridge_sample_df = pd.read_csv(filename)
    
    predict(lasso_ridge_sample_df)


if __name__ == "__main__":
    main()