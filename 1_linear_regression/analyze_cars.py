import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import sklearn.model_selection as ms

def main():
	# Read dataset (relative path)
	cars_df = pd.read_csv('C:/Users/student/Documents/GitHub/mad-2025-fall-ml-the-algorithms/1_linear_regression/cars.csv')

	# Sometimes it's nice to see the raw data.
	# print(cars_df.head())

	# Options similar to analyze_cherry_trees.py
	simple_linear_regression(cars_df, False)
	# simple_linear_regression(cars_df, True)
	# multiple_linear_regression(cars_df, False, False)
	# multiple_linear_regression(cars_df, False, True)
	# multiple_linear_regression(cars_df, True, False)

# Reuse: small utilities to avoid redundancy
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


def simple_linear_regression(cars_df, create_testing_set):
	''' One predictor (enginesize) predicting one response (price). '''
	# Choose a single predictor column and response
	predictors = cars_df[['enginesize']].values
	response = cars_df['price'].values

	training_predictors = predictors
	training_response = response

	if create_testing_set:
		training_predictors, testing_predictors, training_response, testing_response = ms.train_test_split(
			predictors, response, test_size=0.25, random_state=42)

	# Train
	model = create_linear_regression_model(training_predictors, training_response)
	prediction = perform_linear_regression_prediction(model, training_predictors)

	print("The training data predictors, prediction and response values:")
	print_1d_data_summary(training_predictors)
	print_1d_data_summary(prediction)
	print_1d_data_summary(training_response)

	# Plot training: sort x and corresponding predictions so line is continuous
	x_train = training_predictors.flatten()
	order = np.argsort(x_train)
	x_train_sorted = x_train[order]
	pred_train_sorted = prediction.flatten()[order]
	plt.scatter(x_train, training_response, color='blue', label='Training Data')
	plt.plot(x_train_sorted, pred_train_sorted, color='red', label='Best Fit Line')
	plt.xlabel('Engine Size')
	plt.ylabel('Price')
	plt.title('Linear Regression: Engine Size vs Price (Training Data)')
	plt.legend()
	plt.show()

	if create_testing_set:
		prediction = perform_linear_regression_prediction(model, testing_predictors)
		print("The testing data predictors, prediction and response values:")
		print_1d_data_summary(testing_predictors)
		print_1d_data_summary(prediction)
		print_1d_data_summary(testing_response)

		# Plot testing: sort x before plotting
		x_test = testing_predictors.flatten()
		order = np.argsort(x_test)
		x_test_sorted = x_test[order]
		pred_test_sorted = prediction.flatten()[order]
		plt.scatter(x_test, testing_response, color='green', label='Testing Data')
		plt.plot(x_test_sorted, pred_test_sorted, color='red', label='Best Fit Line')
		plt.xlabel('Engine Size')
		plt.ylabel('Price')
		plt.title('Linear Regression: Engine Size vs Price (Testing Data)')
		plt.legend()
		plt.show()


def multiple_linear_regression(cars_df, create_testing_set, one_hot_encode):
	''' Multiple predictors (enginesize, horsepower optionally with carbody one-hot) predicting price. '''
	if not one_hot_encode:
		predictors = cars_df[['enginesize', 'horsepower']].values
	else:
		# One-hot encode the carbody column
		carbody_dummies = pd.get_dummies(cars_df['carbody'], prefix='carbody')
		predictors = pd.concat([cars_df[['enginesize', 'horsepower']], carbody_dummies], axis=1).values

	response = cars_df['price'].values

	training_predictors = predictors
	training_response = response

	if create_testing_set:
		training_predictors, testing_predictors, training_response, testing_response = ms.train_test_split(
			predictors, response, test_size=0.25, random_state=42)

	# Train
	model = create_linear_regression_model(training_predictors, training_response)
	prediction = perform_linear_regression_prediction(model, training_predictors)

	print("The training data prediction and response values:")
	print_1d_data_summary(prediction)
	print_1d_data_summary(training_response)

	# Plot using the first predictor (enginesize) as x; sort for continuous line
	x_train = training_predictors[:, 0]
	order = np.argsort(x_train)
	x_train_sorted = x_train[order]
	pred_train_sorted = prediction.flatten()[order]
	plt.scatter(x_train, training_response, color='blue', label='Training Data')
	plt.plot(x_train_sorted, pred_train_sorted, color='red', label='Best Fit Line')
	plt.xlabel('Engine Size')
	plt.ylabel('Price')
	plt.title('Linear Regression: Engine Size vs Price (Training Data)')
	plt.legend()
	plt.show()

	if create_testing_set:
		prediction = perform_linear_regression_prediction(model, testing_predictors)
		print("The testing data prediction and response values:")
		print_1d_data_summary(prediction)
		print_1d_data_summary(testing_response)

		# Plot testing: sort by first predictor
		x_test = testing_predictors[:, 0]
		order = np.argsort(x_test)
		x_test_sorted = x_test[order]
		pred_test_sorted = prediction.flatten()[order]
		plt.scatter(x_test, testing_response, color='green', label='Testing Data')
		plt.plot(x_test_sorted, pred_test_sorted, color='red', label='Best Fit Line')
		plt.xlabel('Engine Size')
		plt.ylabel('Price')
		plt.title('Linear Regression: Engine Size vs Price (Testing Data)')
		plt.legend()
		plt.show()

if __name__ == "__main__":
	main()
