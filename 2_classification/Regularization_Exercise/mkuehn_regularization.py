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

def marketbucket_predict(market_bucket_df):
    print(f"Head: {market_bucket_df.head()}")
    print(f"\nDescription: {market_bucket_df.describe()}")
    print(f"\nColumns: {market_bucket_df.columns}")
    
    # Set Predictors and Response
    predictors_df = market_bucket_df.drop(['LVT'], axis='columns')
    response_df = market_bucket_df['LVT']

    # Split Data
    predictors_training_df, predictors_testing_df, \
        response_training_df, response_testing_df \
            = ms.train_test_split(predictors_df, response_df, 
                test_size=0.3)
    
    # Train and Predict
    mb_algorithm = lm.LinearRegression()
    mb_model = mb_algorithm.fit(predictors_training_df, response_training_df)
    mb_prediction = mb_model.predict(predictors_testing_df)
    
    lasso_algorithm = lm.LassoCV(cv=5)
    lasso_model = lasso_algorithm.fit(predictors_training_df, response_training_df.values.ravel())
    lasso_prediction = lasso_model.predict(predictors_testing_df)
    
    ridge_algorithm = lm.RidgeCV(cv=5)
    ridge_model = ridge_algorithm.fit(predictors_training_df, response_training_df)
    ridge_prediction = ridge_model.predict(predictors_testing_df)
    
    # Calculate R-Squared
    # mb_r_squared = mb_algorithm.score(predictors_training_df, response_training_df)
    # print(f"\nR-Squared: {mb_r_squared}")
    
    # lasso_r_squared  = lasso_algorithm.score(predictors_training_df, response_training_df)
    # print(f"Lasso R-Squared: {lasso_r_squared}")
    
    # ridge_r_squared = ridge_algorithm.score(predictors_training_df, response_training_df)
    # print(f"Ridge R-Sqaured: {ridge_r_squared}")
    
    print(f"\n{response_df.describe()}")
    
    # MSE and RMSE
    mb_mse = metrics.mean_squared_error(response_testing_df, mb_prediction)
    mb_rmse = np.sqrt(mb_mse)
    print(f"\nLinear RMSE: {mb_rmse}")
    
    lasso_mse = metrics.mean_squared_error(response_testing_df, lasso_prediction)
    lasso_rmse = np.sqrt(lasso_mse)
    print(f"Lasso RMSE: {lasso_rmse}.")
    
    ridge_mse = metrics.mean_squared_error(response_testing_df, ridge_prediction)
    ridge_rmse = np.sqrt(ridge_mse)
    print(f"Ridge RMSE: {ridge_rmse}.")    
    
    
def main():
    """Main Function"""
    market_bucket_df = pd.read_csv("E:/Madison College/Machine Learning/mad-2025-fall-ml-the-algorithms/2_classification/Regularization_Exercise/marketing_buckets.csv")
    # print(market_bucket_df)
    marketbucket_predict(market_bucket_df)
    # gene_express_df = pd.read_csv("E:/Madison College/Machine Learning/mad-2025-fall-ml-the-algorithms/2_classification/Regularization_Exercise/gene_expressions.csv")
    # print(gene_express_df)
    
if __name__ == "__main__":
    main()