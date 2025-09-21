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


def linear_regression(simple_or_multiple, cherry_tree_df, create_testing_set, one_hot_encode):
	if simple_or_multiple == 'multiple' and one_hot_encode:
		season_dummies = pd.get_dummies(cherry_tree_df['Season'], prefix='Season')
		predictors = pd.concat([cherry_tree_df[['Diam', 'Height']], season_dummies], axis=1).values
		response = cherry_tree_df['Volume'].values
	elif simple_or_multiple == 'multiple':
		predictors = cherry_tree_df[['Diam', 'Height']].values
		response = cherry_tree_df['Volume'].values
	else:
		predictors = cherry_tree_df[['Diam']].values
		response = cherry_tree_df['Height'].values

	if create_testing_set:
		training_predictors, testing_predictors, training_response, testing_response \
			= ms.train_test_split(
				predictors, response, test_size=0.25, random_state=42)
	else:
		training_predictors = predictors
		training_response = response
		testing_predictors = None
		testing_response = None
		testing_prediction = None
	
	model = create_linear_regression_model(training_predictors, training_response)
	training_prediction = perform_linear_regression_prediction(model, training_predictors)

	if create_testing_set:
		testing_prediction = perform_linear_regression_prediction(model, testing_predictors)

	return training_prediction, training_response, training_predictors, testing_prediction, testing_response, testing_predictors


'''
# Egads! The simple_linear_regression and multiple_linear_regression functions have a lot of
# common code! We are violating our most sacred principle: Reuse! (or DRY (Don't Repeat Yourself).)
def simple_linear_regression(cherry_tree_df, create_testing_set):
	# Performs simple linear regression on the cherry tree data. That is, one predictor
	# predicting one response.
	predictors = cherry_tree_df[['Diam']].values
	response = cherry_tree_df['Height'].values

	# If we are not creating a testing set, then we're training on 100% of the data. The name still
	# applies; it's just that the training set is the entire dataset.
	training_predictors = predictors
	training_response = response
	testing_predictors = None
	testing_response = None
	testing_prediction = None

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

	# Moved code after return statement to here.
	if create_testing_set:
		testing_prediction = perform_linear_regression_prediction(model, testing_predictors)

	# Gets the values for printing_testing_values function.
	return prediction, training_response, training_predictors, testing_prediction, testing_response, testing_predictors

	# # y = mx + b. Or in scikit-learn terms: y = model.coef_ * x + model.intercept_. It would take
	# # me a while, but I could plot this data by hand and calculate the predictions myself. Find a
	# # tree diameter, follow it up to the line, then over to the height. I don't have to, but this
	# # is very good to know. I need to **understand** the algorithm.
	# print(f'Slope (m): {model.coef_}, y-intercept (b): {model.intercept_}')
	'''
	
'''
	# Move code to new function: printing_testing_values to reduce redudancy.
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
		testing_prediction = perform_linear_regression_prediction(model, testing_predictors)
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
		'''

'''
def multiple_linear_regression(cherry_tree_df, create_testing_set, one_hot_encode):
	# Performs multiple linear regression on the cherry tree data. That is, multiple predictors
	# predicting one response.

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
	testing_predictors = None
	testing_response = None
	testing_prediction = None

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

	# Moved code after return statement to here.
	if create_testing_set:
		testing_prediction = perform_linear_regression_prediction(model, testing_predictors)

	# Gets the values for printing_testing_values function.
	return prediction, training_response, training_predictors, testing_prediction, testing_response, testing_predictors

	# # y = mx + b. Or in scikit-learn terms: y = model.coef_ * x + model.intercept_. It would take
	# # me a while, but I could plot this data by hand and calculate the predictions myself. Find a
	# # tree diameter, follow it up to the line, then over to the height. I don't have to, but this
	# # is very good to know. I need to **understand** the algorithm.
	# print(f'Slope (m): {model.coef_}, y-intercept (b): {model.intercept_}')
	'''
	
'''
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
'''


def printing_values(simple_or_multiple, test_set_created, prediction, response, predictors):
	testing_or_training = 'testing' if test_set_created else 'training'
	if simple_or_multiple == 'simple':
		print(f'"The {testing_or_training} data predictors, prediction and response values:"')
		print_1d_data_summary(predictors)
	else:
		print(f'"The {testing_or_training} data prediction and response values:"')
	print_1d_data_summary(prediction)
	print_1d_data_summary(response)


'''
def printing_values(simple_or_mulitple, test_set_created, 
							prediction, training_response, training_predictors, 
							testing_prediction=None, testing_response=None, testing_predictors=None):
	if simple_or_mulitple == 'simple':
		print("The training data predictors, prediction and response values:")
		print_1d_data_summary(training_predictors)
		print_1d_data_summary(prediction)
		print_1d_data_summary(training_response)
		if test_set_created and testing_predictors is not None:
			print("The testing data predictors, prediction and response values:")
			print_1d_data_summary(testing_predictors)
			print_1d_data_summary(testing_prediction)
			print_1d_data_summary(testing_response)
	else:
		print("The training data prediction and response values:")
		print_1d_data_summary(prediction)
		print_1d_data_summary(training_response)
		if test_set_created and testing_predictors is not None:
			print("The testing data prediction and response values:")
			print_1d_data_summary(testing_prediction)
			print_1d_data_summary(testing_response)
'''


def plotting_values(simple_or_multiple, test_set_created, prediction, response, predictors):
	color = 'green' if test_set_created else 'blue'
	label = 'Testing Data' if test_set_created else 'Training Data'
	if simple_or_multiple == 'multiple':
		predictors = predictors[:, 0]
	else:
		predictors = predictors.flatten()
	plt.scatter(predictors, response, color=color, label=label)
	plt.plot(predictors, prediction, color = 'red', label = 'Best Fit Line')
	plt.xlabel('Diam')
	ylabel = 'Volume' if simple_or_multiple == 'multiple' else 'Height'
	plt.ylabel(ylabel)
	title = f'Linear Regression: Diam vs {ylabel} ({label})'
	plt.title(title)
	plt.legend()
	plt.show()


'''
def plotting_values(simple_or_multiple, test_set_created, 
							prediction, training_response, training_predictors,
							testing_prediction=None, testing_response=None, testing_predictors=None):
	if simple_or_multiple == 'simple':
		# Plot the data and the best fit line.
		plt.scatter(training_predictors, training_response, color='blue', label='Training Data')
		plt.plot(training_predictors, prediction, color='red', label='Best Fit Line')
		plt.xlabel('Diam')
		plt.ylabel('Height')
		plt.title('Linear Regression: Diam vs Height (Training Data)')
		plt.legend()
		plt.show()
		if test_set_created and testing_predictors is not None:
			plt.scatter(testing_predictors, testing_response, color='green', label='Testing Data')
			plt.plot(testing_predictors, testing_prediction, color='red', label='Best Fit Line')
			plt.xlabel('Diam')
			plt.ylabel('Height')
			plt.title('Linear Regression: Diam vs Height (Testing Data)')
			plt.legend()
			plt.show()
	else:
		# Plot the data and the best fit line.
		plt.scatter(training_predictors, training_response, color='blue', label='Training Data')
		plt.plot(training_predictors, prediction, color='red', label='Best Fit Line')
		plt.xlabel('Diam')
		plt.ylabel('Volume')
		plt.title('Linear Regression: Diam vs Volume (Training Data)')
		plt.legend()
		plt.show()
		if test_set_created and testing_predictors is not None:
			plt.scatter(testing_predictors, testing_response, color='green', label='Testing Data')
			plt.plot(testing_predictors, testing_prediction, color='red', label='Best Fit Line')
			plt.xlabel('Diam')
			plt.ylabel('Volume')
			plt.title('Linear Regression: Diam vs Volume (Testing Data)')
			plt.legend()
			plt.show()
'''
			

def main():
	# Cherry tree diameters are easy. Heights are hard.
	cherry_tree_df = pd.read_csv('CherryTree.csv')

	# Only code that should need changing.
	simple_or_multiple = 'multiple' # 'simple' or 'multiple'
	use_testing_set = False
	one_hot_encode = True # Only used for multiple linear regression.

	# Sometimes it's nice to see the raw data.
	# print(cherry_tree_df.head())

	# simple_linear_regression(cherry_tree_df, False)
	# simple_linear_regression(cherry_tree_df, True)

	# Gets values from simple_linear_regression function for printing_testing_values function.
	training_prediction, training_response, training_predictors, testing_prediction, testing_response, testing_predictors \
		= linear_regression(simple_or_multiple, cherry_tree_df, use_testing_set, one_hot_encode)

	# multiple_linear_regression(cherry_tree_df, False, False)
	# multiple_linear_regression(cherry_tree_df, False, True)
	# multiple_linear_regression(cherry_tree_df, True, False)

	# Uses values from linear_regression functions for printing and plotting data.
	if use_testing_set:
		printing_values(simple_or_multiple, False, training_prediction, training_response, training_predictors)
		printing_values(simple_or_multiple, use_testing_set, testing_prediction, testing_response, testing_predictors)
		plotting_values(simple_or_multiple, False, training_prediction, training_response, training_predictors)
		plotting_values(simple_or_multiple, use_testing_set, testing_prediction, testing_response, testing_predictors)
	else:
		printing_values(simple_or_multiple, use_testing_set, training_prediction, training_response, training_predictors)
		plotting_values(simple_or_multiple, use_testing_set, training_prediction, training_response, training_predictors)


if __name__ == "__main__":
	main()