import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
from sklearn.metrics import r2_score
import seaborn as sns

def show_r_squared(model, predictors, response):
	r_squared = model.score(predictors, response)
	return r_squared

def print_1d_data_summary(data_1d):
	numpified_data = np.array(data_1d)
	# Flatten if 2D with one column
	if numpified_data.ndim == 2 and numpified_data.shape[1] == 1:
		flattened_numpified_data = numpified_data.flatten()
	else:
		flattened_numpified_data = numpified_data
	# Format first 5 and last 5 values
	first_five = ", ".join(f"{x:7.3f}" for x in flattened_numpified_data[:5])
	last_five = ", ".join(f"{x:7.3f}" for x in flattened_numpified_data[-5:])
	print(f"[{first_five}, ..., {last_five}]")

def create_linear_regression_model(predictors, response):
	model = lm.LinearRegression()
	model.fit(predictors, response)

	return model

def perform_linear_regression_prediction(model, predictors):
	prediction = model.predict(predictors)

	return prediction



def linear_regression(past_data, create_testing_set):

	predictors = past_data[['SurvivalSkills','MentalToughness','Stubbornness','Adaptability','PhysicalFitness']].values
	response = past_data['SurvivalScore'].values
	training_predictors = predictors
	training_response = response



	if create_testing_set:
		training_predictors, testing_predictors, training_response, testing_response \
			= ms.train_test_split(
				predictors, response, test_size=0.25, random_state=42)
		model = create_linear_regression_model(training_predictors, training_response)
		prediction = perform_linear_regression_prediction(model, testing_predictors)
		print(f"Testing data R-Squared: {show_r_squared(model, testing_predictors, testing_response)}")
		print("The testing data prediction and response values:")
		print_1d_data_summary(prediction)
		print_1d_data_summary(testing_response)	
		graph(testing_predictors[:,0], testing_response, "blue", prediction ,"Testing Data", x_label="SurvivalScore (testing)" ) 
		
	model = create_linear_regression_model(training_predictors, training_response)
	prediction = perform_linear_regression_prediction(model, training_predictors)

	print(f"Training data R-Squared: {show_r_squared(model, training_predictors, training_response)}")
	print("The training data prediction and response values:")
	print_1d_data_summary(prediction)
	print_1d_data_summary(training_response)
	graph(training_predictors[:,0], training_response, "green", prediction ,"Training Data", x_label="SurvivalScore (Training)", )

	return model

def predict_next_survivor(trained_model, next_data):
	predictors = next_data[['SurvivalSkills','MentalToughness','Stubbornness','Adaptability','PhysicalFitness']].values
	prediction = perform_linear_regression_prediction(trained_model, predictors)
	print("The next data prediction values:")
	print_1d_data_summary(prediction)
	graph(predictors[:,0], prediction, "orange", prediction ,"Next Data", x_label="SurvivalScore (Next)", )
	prediction_results = next_data.copy()
	prediction_results['PredictedSurvivalScore'] = prediction
	top_three= prediction_results.nlargest(3, 'PredictedSurvivalScore')
	print("The top three predicted survivors and their score are:")
	print(top_three[['Name','PredictedSurvivalScore']])
	#return prediction

def make_heatmap(corr_df):
	corr_df = corr_df.select_dtypes(include=[np.number]).corr()
	plt.figure(figsize=(9, 7))
	sns.heatmap(corr_df, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
	plt.title('Correlation Matrix Heatmap')
	plt.show()


def graph(predictor, resp,whatcolor,prediction,whatlabel,x_label=None, 
		y_label=None):
	plt.scatter(predictor, resp, color=whatcolor, label=whatlabel)
	plt.plot(predictor, prediction, color='red', label='Best Fit Line')
	plt.xlabel(x_label if x_label else 'Predictor')
	plt.ylabel(y_label if y_label else 'Response')
	plt.title(f'Linear Regression: {x_label} vs {y_label} ')
	plt.legend()
	plt.show()






def main():
	s_s_past = pd.read_csv("C:/Users/student/OneDrive - Madison College/Machine Learning/SoleSurvivor/sole_survivor_past.csv")
	# print(f'{s_s_past}')
	s_s_future = pd.read_csv("C:/Users/student/OneDrive - Madison College/Machine Learning/SoleSurvivor/sole_survivor_next.csv")
	# print(f'{s_s_future}')
	# make_heatmap(s_s_past)
	# make_heatmap(s_s_future)
	# linear_regression(s_s_past, False)
	# linear_regression(s_s_past, True)
	trained_model = linear_regression(s_s_past, False)
	predict_next_survivor(trained_model, s_s_future)






if __name__ == "__main__":
	main()