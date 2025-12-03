import logging
import io
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import matplotlib as plt
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
    print(f"\nLinear RMSE: {mb_rmse}.")
    
    lasso_mse = metrics.mean_squared_error(response_testing_df, lasso_prediction)
    lasso_rmse = np.sqrt(lasso_mse)
    print(f"Lasso RMSE: {lasso_rmse}.")
    
    ridge_mse = metrics.mean_squared_error(response_testing_df, ridge_prediction)
    ridge_rmse = np.sqrt(ridge_mse)
    print(f"Ridge RMSE: {ridge_rmse}.")    
    
def genexpress_predict(gene_express_df):
    print(f"Head: {gene_express_df.head()}")
    print(f"\nTail: {gene_express_df.tail()}")
    
    # Set Predictors and Response
    predictors_df = gene_express_df.drop(['y'], axis='columns')
    response_df = gene_express_df['y']
    
    # Correlation Matrix Heatmap
    correlation_matrix = response_df.corr('y')
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1, center=0, square=True)
    plt.title('Gene Correlation Heatmap')
    plt.show()
    
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
    
    # Compute Pearson r and p-value for each predictor vs response
    # results = []
    # for col in predictors_df.columns:
    #     x = predictors_df[col].values
    #     y = response_df.values
    #     # skip constant columns which cause pearsonr to fail
    #     if np.std(x) == 0 or np.std(y) == 0:
    #         results.append((col, np.nan, np.nan))
    #         continue
    #     r, p = stats.pearsonr(x, y)
    #     results.append((col, r, p))
        
    # pearson_df = pd.DataFrame(results, columns=['predictor', 'pearson_r', 'p_value'])
    # pearson_df['abs_r'] = pearson_df['pearson_r'].abs()
    # pearson_df = pearson_df.sort_values('abs_r', ascending=False).reset_index(drop=True)
    
    # Print top correlate predictors (adjust n as needed)
    # print("\nTop predictors by absolute Pearson R:")
    # print(pearson_df.head(20).to_string(index=False))
    
    # If you want the full correlation matrix (predictors x predictors + response)
    # full_corr = gene_express_df.corr(method='pearson')
    # print("\nPearson Correlation Matrix (First 10 Rows):")
    # print(full_corr.head(10).to_string())
    
    # Return the results for downstream use if needed
    # return pearson_df #, full_corr
    
    
def main():
    """Main Function"""
    market_bucket_df = pd.read_csv("E:/Madison College/Machine Learning/mad-2025-fall-ml-the-algorithms/2_classification/Regularization_Exercise/marketing_buckets.csv")
    # print(market_bucket_df)
    # marketbucket_predict(market_bucket_df)
    gene_express_df = pd.read_csv("E:/Madison College/Machine Learning/mad-2025-fall-ml-the-algorithms/2_classification/Regularization_Exercise/gene_expressions.csv")
    # print(gene_express_df)
    genexpress_predict(gene_express_df)
    
if __name__ == "__main__":
    main()