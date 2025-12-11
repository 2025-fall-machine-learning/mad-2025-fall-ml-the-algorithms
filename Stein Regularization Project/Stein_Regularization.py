import pandas as pd
import numpy as np
import sklearn.model_selection as ms
from sklearn.metrics import r2_score
import sklearn.linear_model as lm
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV

def create_linear_regression_model(predictors, response):
	model = lm.LinearRegression()
	model.fit(predictors, response)

	return model
def create_lasso_regression_model(predictors, response):
    lasso = LassoCV(cv=5)
    lasso.fit(predictors, response)
    return lasso

def create_ridge_regression_model(predictors, response):
    ridge = RidgeCV(cv=5)
    ridge.fit(predictors, response)
    return ridge

def perform_linear_regression_prediction(model, predictors):
	prediction = model.predict(predictors)

	return prediction

def regression_regularization(regularization_df, create_testing_set):
    response = regularization_df['LVT'].values
    predictors = regularization_df.drop('LVT', axis=1).values

    training_predictors = predictors
    training_response = response
    if create_testing_set:
        training_predictors, testing_predictors, training_response, testing_response = ms.train_test_split(
            predictors, response, test_size=0.3)
        model = create_linear_regression_model(training_predictors, training_response)
        prediction = perform_linear_regression_prediction(model, testing_predictors)
        print(f"Testing data R-Squared: {r2_score(testing_response, prediction)}")
        testing_rsme = np.sqrt(np.mean((testing_response - prediction) ** 2))
        print(f"Testing data RSME: {np.sqrt(np.mean((testing_response - prediction) ** 2))}")
        # print(f"Describe on  testing response:{pd.Series(testing_response).describe()}")
        lasso_model = create_lasso_regression_model(training_predictors, training_response)
        lasso_prediction = perform_linear_regression_prediction(lasso_model, testing_predictors)
        lasso_testing_rsme = np.sqrt(np.mean((testing_response - lasso_prediction) ** 2))  
        print(f"Testing Lasso RMSE: {np.sqrt(np.mean((testing_response - lasso_prediction) ** 2))}")
        ridge_model = create_ridge_regression_model(training_predictors, training_response)
        ridge_prediction = perform_linear_regression_prediction(ridge_model, testing_predictors)
        ridge_testing_rsme = np.sqrt(np.mean((testing_response - ridge_prediction) ** 2))
        print(f"Testing Ridge RMSE: {np.sqrt(np.mean((testing_response - ridge_prediction) ** 2))}")
        
        try:
            with open('rmse_results.txt', 'w') as f:
                f.write(f"RMSE_test_linear: {testing_rsme}\n")
                f.write(f"RMSE_test_lasso: {lasso_testing_rsme}\n")
                f.write(f"RMSE_test_ridge: {ridge_testing_rsme}\n")
        except Exception as e:
            print(f"Failed to write rmse_results.txt: {e}")

    
    # model = create_linear_regression_model(training_predictors, training_response)
    # prediction = perform_linear_regression_prediction(model, training_predictors)

    # print(f"\n\nTraining data R-Squared: {r2_score(training_response, prediction)}")
    # print(f"Training data RSME: {np.sqrt(np.mean((training_response - prediction) ** 2))}")
    # # print(f"Describe on training response:{pd.Series(training_response).describe()}")
    # lasso_model = create_lasso_regression_model(training_predictors, training_response)
    # lasso_prediction = perform_linear_regression_prediction(lasso_model, training_predictors)
    # print(f"Training Lasso RMSE: {np.sqrt(np.mean((training_response - lasso_prediction) ** 2))}")
    # ridge_model = create_ridge_regression_model(training_predictors, training_response)
    # ridge_prediction = perform_linear_regression_prediction(ridge_model, training_predictors)
    # print(f"Training Ridge RMSE: {np.sqrt(np.mean((training_response - ridge_prediction) ** 2))}")




def main():
    regularization_df = pd.read_csv("marketing_buckets.csv")
    # print(regularization_df)
    regression_regularization(regularization_df, True)

if __name__ == "__main__":
    main()