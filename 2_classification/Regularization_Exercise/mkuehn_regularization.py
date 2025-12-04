import logging
import io
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
import seaborn as sns
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
    
    print(f"\n{response_df.describe()}")
    
    # MSE and RMSE
    mb_mse = metrics.mean_squared_error(response_testing_df, mb_prediction)
    mb_rmse = np.sqrt(mb_mse)
    print(f"\nLinear RMSE: {mb_rmse}.")
    
    lasso_mse = metrics.mean_squared_error(response_testing_df, lasso_prediction)
    lasso_rmse = np.sqrt(lasso_mse)
    print(f"Lasso RMSE: {lasso_rmse}.")
    
    ridge_mse = metrics.mean_squared_error(response_testing_df, ridge_prediction)
    ridge_rmse = np.sqrt(ridge_mse)
    print(f"Ridge RMSE: {ridge_rmse}.")    
    
def correlation_matrix(df):
    rs = np.random.RandomState(0)
    df = pd.DataFrame(rs.rand(100, 20), columns=[f'col_{i}' for i in range(20)])
    correlation_matrix = df.corr(method='pearson')
    print(correlation_matrix)
    
def make_heatmap(corr_df):
    corr_df = corr_df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_df, annot=False, fmt=".2f", cmap='coolwarm', cbar=True, ax=ax, vmin=-1, vmax=1)
    plt.title('Pearson Correlation Matrix Heatmap')
    plt.show()
    
def genexpress_predict(gene_express_df):
    print(f"Head: {gene_express_df.head()}")
    print(f"\nTail: {gene_express_df.tail()}")
    
    # Set Predictors and Response
    predictors_df = gene_express_df.drop(['y'], axis='columns')
    response_df = gene_express_df['y']
    
    # Calculate Pearson Correlation Matrix
    # correlation_matrix = gene_express_df.corr(method='pearson')
    # print(correlation_matrix.sort_values(ascending=False))
    
    # Correlation Matrix Heatmap
    # correlation_matrix = response_df.corr('y')
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1, center=0, square=True)
    # plt.title('Gene Correlation Heatmap')
    # plt.show()
    
    # Split Data
    predictors_training_df, predictors_testing_df, \
        response_training_df, response_testing_df \
            = ms.train_test_split(predictors_df, response_df,
                test_size=0.2, random_state=0)
            
            
    # Train and Predict
    ge_algorithm = lm.LinearRegression()
    ge_model = ge_algorithm.fit(predictors_training_df, response_training_df)
    ge_prediction = ge_model.predict(predictors_testing_df)
    
    lasso_algorithm = lm.LassoCV(cv=5)
    lasso_model = lasso_algorithm.fit(predictors_training_df, response_training_df.values.ravel())
    lasso_prediction = lasso_model.predict(predictors_testing_df)
    
    ridge_algorithm = lm.RidgeCV(cv=5)
    ridge_model = ridge_algorithm.fit(predictors_training_df, response_training_df)
    ridge_prediction = ridge_model.predict(predictors_testing_df)
    
    # MSE and RMSE
    ge_mse = metrics.mean_squared_error(response_testing_df, ge_prediction)
    ge_rmse = np.sqrt(ge_mse)
    print(f"\nLinear RMSE: {ge_rmse}.")
    
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
    # marketbucket_predict(market_bucket_df)
    gene_express_df = pd.read_csv("E:/Madison College/Machine Learning/mad-2025-fall-ml-the-algorithms/2_classification/Regularization_Exercise/gene_expressions.csv")
    # print(gene_express_df)
    correlation_matrix(gene_express_df)
    # make_heatmap(gene_express_df)
    genexpress_predict(gene_express_df)
    
if __name__ == "__main__":
    main()