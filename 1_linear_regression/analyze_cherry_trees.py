import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import sklearn.model_selection as ms


# Hey! Some nice pretty functions to gain reuse and avoid redundancy!
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


# Egads! The simple_linear_regression and multiple_linear_regression functions have a lot of
# common code! We are violating our most sacred principle: Reuse! (or DRY (Don't Repeat Yourself).)
def simple_linear_regression(cherry_tree_df, create_testing_set):
	''' Performs simple linear regression on the cherry tree data. That is, one predictor
	predicting one response. '''
	predictors = cherry_tree_df[['Diam']].values
	response = cherry_tree_df['Height'].values

	# If we are not creating a testing set, then we're training on 100% of the data. The name still
	# applies; it's just that the training set is the entire dataset.
	training_predictors = predictors
	training_response = response

	# The caller may want testing data, though for now we're just inspecting it. We're not
	# performing any testing other than just visual inspection of the results.
	if create_testing_set:
		# Split the data into 75% training and 25% testing.
		training_predictors, testing_predictors, training_response, testing_response \
			= ms.train_test_split(
				predictors, response, test_size=0.25, random_state=42)

	# Perform linear regression.
	model = create_linear_regression_model(training_predictors, training_response)
	prediction = perform_linear_regression_prediction(model, training_predictors)

	# # y = mx + b. Or in scikit-learn terms: y = model.coef_ * x + model.intercept_. It would take
	# # me a while, but I could plot this data by hand and calculate the predictions myself. Find a
	# # tree diameter, follow it up to the line, then over to the height. I don't have to, but this
	# # is very good to know. I need to **understand** the algorithm.
	# print(f'Slope (m): {model.coef_}, y-intercept (b): {model.intercept_}')

	print("The training data predictors, prediction and response values:")
	print_1d_data_summary(training_predictors)
	print_1d_data_summary(prediction)
	print_1d_data_summary(training_response)

	# Plot the data and the best fit line.
	plt.scatter(training_predictors, training_response, color='blue', label='Training Data')
	plt.plot(training_predictors, prediction, color='red', label='Best Fit Line')
	plt.xlabel('Diam')
	plt.ylabel('Height')
	plt.title('Linear Regression: Diam vs Height (Training Data)')
	plt.legend()
	plt.show()
	
	if create_testing_set:
		prediction = perform_linear_regression_prediction(model, testing_predictors)
		print("The testing data predictors, prediction and response values:")
		print_1d_data_summary(testing_predictors)
		print_1d_data_summary(prediction)
		print_1d_data_summary(testing_response)

        # Plot the data and the best fit line.
		plt.scatter(testing_predictors, testing_response, color='green', label='Testing Data')
		plt.plot(testing_predictors, prediction, color='red', label='Best Fit Line')
		plt.xlabel('Diam')
		plt.ylabel('Height')
		plt.title('Linear Regression: Diam vs Height (Testing Data)')
		plt.legend()
		plt.show()


def multiple_linear_regression(cherry_tree_df, create_testing_set, one_hot_encode):
	''' Performs multiple linear regression on the cherry tree data. That is, multiple predictors
	predicting one response. '''

	if not one_hot_encode:
		predictors = cherry_tree_df[['Diam', 'Height']].values
	else:
		# One-hot encode the Season column (values: Summer, Fall, Winter, Spring).
		season_dummies = pd.get_dummies(cherry_tree_df['Season'], prefix='Season')
		predictors = pd.concat([cherry_tree_df[['Diam', 'Height']], season_dummies], axis=1).values

	response = cherry_tree_df['Volume'].values

	# If we are not creating a testing set, then we're training on 100% of the data. The name still
	# applies; it's just that the training set is the entire dataset.
	training_predictors = predictors
	training_response = response

	# The caller may want testing data, though for now we're just inspecting it. We're not
	# performing any testing other than just visual inspection of the results.
	if create_testing_set:
		# Split the data into 75% training and 25% testing.
		training_predictors, testing_predictors, training_response, testing_response \
			= ms.train_test_split(
				predictors, response, test_size=0.25, random_state=42)

	# Perform linear regression.
	model = create_linear_regression_model(training_predictors, training_response)
	prediction = perform_linear_regression_prediction(model, training_predictors)

	# # y = mx + b. Or in scikit-learn terms: y = model.coef_ * x + model.intercept_. It would take
	# # me a while, but I could plot this data by hand and calculate the predictions myself. Find a
	# # tree diameter, follow it up to the line, then over to the height. I don't have to, but this
	# # is very good to know. I need to **understand** the algorithm.
	# print(f'Slope (m): {model.coef_}, y-intercept (b): {model.intercept_}')

	print("The training data prediction and response values:")
	print_1d_data_summary(prediction)
	print_1d_data_summary(training_response)

	# Plot the data and the best fit line.
	plt.scatter(training_predictors, training_response, color='blue', label='Training Data')
	plt.plot(training_predictors, prediction, color='red', label='Best Fit Line')
	plt.xlabel('Diam')
	plt.ylabel('Volume')
	plt.title('Linear Regression: Diam vs Volume (Training Data)')
	plt.legend()
	plt.show()

	if create_testing_set:
		prediction = perform_linear_regression_prediction(model, testing_predictors)
		print("The testing data prediction and response values:")
		print_1d_data_summary(prediction)
		print_1d_data_summary(testing_response)

        # Plot the data and the best fit line.
		plt.scatter(testing_predictors, testing_response, color='green', label='Testing Data')
		plt.plot(testing_predictors, prediction, color='red', label='Best Fit Line')
		plt.xlabel('Diam')
		plt.ylabel('Volume')
		plt.title('Linear Regression: Diam vs Volume (Testing Data)')
		plt.legend()
		plt.show()


def main():
	# Cherry tree diameters are easy. Heights are hard.
	cherry_tree_df = pd.read_csv('1_linear_regression\CherryTree.csv')

	# Sometimes it's nice to see the raw data.
	# print(cherry_tree_df.head())

	simple_linear_regression(cherry_tree_df, False)
	# simple_linear_regression(cherry_tree_df, True)
	# multiple_linear_regression(cherry_tree_df, False, False)
	# multiple_linear_regression(cherry_tree_df, False, True)
	# multiple_linear_regression(cherry_tree_df, True, False)


if __name__ == "__main__":
	main()
