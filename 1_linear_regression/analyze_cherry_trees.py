import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
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


def show_r_squared(model, predictors, response):
	r_squared = model.score(predictors, response)
	print(f"R-squared: {r_squared:.4f}")


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
				predictors, response, test_size=0.25) #, random_state=42)

	# Perform linear regression.
	model = create_linear_regression_model(training_predictors, training_response)
	show_r_squared(model, training_predictors, training_response)
	show_r_squared(model, testing_predictors, testing_response)
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

	# It turns out we don't need nor use one-hot encoding because the one-hot encoded values don't
	# contribute to the response. We found this using Pearson's R.

	# if not one_hot_encode:
	predictors = cherry_tree_df[['Diam', 'Height']].values
	# For correlation matrix - simple case with just Diam and Height
	modified_cherry_tree_df = pd.DataFrame(predictors, columns=['Diam', 'Height'])
	# else:
	# 	# One-hot encode the Season column (values: Summer, Fall, Winter, Spring).
	# 	season_dummies = pd.get_dummies(cherry_tree_df['Season'], prefix='Season')
	# 	predictors_df_combined = pd.concat([cherry_tree_df[['Diam', 'Height']], season_dummies], axis=1)
	# 	predictors = predictors_df_combined.values
	# 	# For correlation matrix - reuse the combined DataFrame structure
	# 	modified_cherry_tree_df = predictors_df_combined.copy()

	response = cherry_tree_df['Volume'].values

	# Create correlation matrix by combining predictors and response
	
	# Add the response variable
	modified_cherry_tree_df['Volume'] = response

	correlation_matrix = modified_cherry_tree_df.corr()
	volume_correlation_matrix = correlation_matrix[['Volume']].sort_values(by='Volume', ascending=False)
	sns.heatmap(volume_correlation_matrix, annot=True)
	plt.show()

	mask = abs(correlation_matrix) < 0.3
	sns.heatmap(correlation_matrix, annot=True, mask=mask)
	plt.show()

	# # Calculate correlation matrix
	# correlation_matrix = modified_cherry_tree_df.corr()
	
	# # Create heatmap with annotations, highlighting correlations < 0.3
	# plt.figure(figsize=(10, 8))
	
	# # Create a mask for values where correlation < 0.3 (but not the diagonal)
	# mask = np.abs(correlation_matrix) < 0.3
	# np.fill_diagonal(mask.values, False)  # Don't mask the diagonal (self-correlation = 1)
	
	# # Create the heatmap
	# sns.heatmap(correlation_matrix, 
	#            annot=True, 
	#            cmap='coolwarm', 
	#            center=0,
	#            fmt='.3f',
	#            square=True,
	#            linewidths=0.5,
	#            mask=None)  # Show all values
	
	# # Highlight low correlations by adding a border or different color
	# # We'll use a custom approach to highlight correlations < 0.3
	# ax = plt.gca()
	# for i in range(len(correlation_matrix.columns)):
	# 	for j in range(len(correlation_matrix.columns)):
	# 		if i != j and abs(correlation_matrix.iloc[i, j]) < 0.3:
	# 			# Add a red border around cells with correlation < 0.3
	# 			rect = plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', linewidth=3)
	# 			ax.add_patch(rect)
	
	# plt.title('Correlation Matrix (Red borders indicate |correlation| < 0.3)')
	# plt.tight_layout()
	# plt.show()

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
				predictors, response, test_size=0.25) #, random_state=42)

	# Perform linear regression.
	model = create_linear_regression_model(training_predictors, training_response)
	show_r_squared(model, training_predictors, training_response)
	if create_testing_set:
		show_r_squared(model, testing_predictors, testing_response)
	prediction = perform_linear_regression_prediction(model, training_predictors)

	# # y = mx + b. Or in scikit-learn terms: y = model.coef_ * x + model.intercept_. It would take
	# # me a while, but I could plot this data by hand and calculate the predictions myself. Find a
	# # tree diameter, follow it up to the line, then over to the height. I don't have to, but this
	# # is very good to know. I need to **understand** the algorithm.
	# print(f'Slope (m): {model.coef_}, y-intercept (b): {model.intercept_}')

	print("The training data prediction and response values:")
	print_1d_data_summary(prediction)
	print_1d_data_summary(training_response)

	# # Create 3D plot with Diam and Height as independent variables, Volume as dependent variable, and best fit plane
	# fig = plt.figure(figsize=(12, 8))
	# ax = fig.add_subplot(111, projection='3d')
	
	# # Extract Diam and Height from training_predictors
	# diam_vals = training_predictors[:, 0]
	# height_vals = training_predictors[:, 1]
	
	# # Scatter plot of actual data points
	# ax.scatter(diam_vals, height_vals, training_response, color='blue', alpha=0.6, label='Training Data')
	
	# # Create meshgrid for the best fit plane
	# diam_range = np.linspace(diam_vals.min(), diam_vals.max(), 20)
	# height_range = np.linspace(height_vals.min(), height_vals.max(), 20)
	# diam_mesh, height_mesh = np.meshgrid(diam_range, height_range)
	
	# # Create predictor array for the plane
	# plane_predictors = np.column_stack([diam_mesh.ravel(), height_mesh.ravel()])
	
	# # Handle one-hot encoded features if present
	# if training_predictors.shape[1] > 2:
	# 	# Add mean values for the one-hot encoded features
	# 	mean_encoded_features = np.mean(training_predictors[:, 2:], axis=0)
	# 	plane_predictors_full = np.column_stack([
	# 		plane_predictors,
	# 		np.tile(mean_encoded_features, (plane_predictors.shape[0], 1))
	# 	])
	# else:
	# 	plane_predictors_full = plane_predictors
	
	# # Predict volume values for the plane
	# plane_predictions = model.predict(plane_predictors_full)
	# plane_predictions = plane_predictions.reshape(diam_mesh.shape)
	
	# # Plot the best fit plane
	# ax.plot_surface(diam_mesh, height_mesh, plane_predictions, alpha=0.3, color='red')
	
	# ax.set_xlabel('Diameter')
	# ax.set_ylabel('Height')
	# ax.set_zlabel('Volume')
	# ax.set_title('3D Linear Regression: Diam & Height vs Volume (Training Data)')
	# plt.show()

	if create_testing_set:
		prediction = perform_linear_regression_prediction(model, testing_predictors)
		print("The testing data prediction and response values:")
		print_1d_data_summary(prediction)
		print_1d_data_summary(testing_response)

		# # Create 3D plot for testing data with best fit plane
		# fig = plt.figure(figsize=(12, 8))
		# ax = fig.add_subplot(111, projection='3d')
		
		# # Extract Diam and Height from testing_predictors
		# test_diam_vals = testing_predictors[:, 0]
		# test_height_vals = testing_predictors[:, 1]
		
		# # Scatter plot of actual testing data points
		# ax.scatter(test_diam_vals, test_height_vals, testing_response, color='green', alpha=0.6, label='Testing Data')
		
		# # Create meshgrid for the best fit plane (same as training)
		# test_diam_range = np.linspace(test_diam_vals.min(), test_diam_vals.max(), 20)
		# test_height_range = np.linspace(test_height_vals.min(), test_height_vals.max(), 20)
		# test_diam_mesh, test_height_mesh = np.meshgrid(test_diam_range, test_height_range)
		
		# # Create predictor array for the plane
		# test_plane_predictors = np.column_stack([test_diam_mesh.ravel(), test_height_mesh.ravel()])
		
		# # Handle one-hot encoded features if present
		# if testing_predictors.shape[1] > 2:
		# 	# Add mean values for the one-hot encoded features
		# 	test_mean_encoded_features = np.mean(testing_predictors[:, 2:], axis=0)
		# 	test_plane_predictors_full = np.column_stack([
		# 		test_plane_predictors,
		# 		np.tile(test_mean_encoded_features, (test_plane_predictors.shape[0], 1))
		# 	])
		# else:
		# 	test_plane_predictors_full = test_plane_predictors
		
		# # Predict volume values for the plane
		# test_plane_predictions = model.predict(test_plane_predictors_full)
		# test_plane_predictions = test_plane_predictions.reshape(test_diam_mesh.shape)
		
		# # Plot the best fit plane
		# ax.plot_surface(test_diam_mesh, test_height_mesh, test_plane_predictions, alpha=0.3, color='red')
		
		# ax.set_xlabel('Diameter')
		# ax.set_ylabel('Height')
		# ax.set_zlabel('Volume')
		# ax.set_title('3D Linear Regression: Diam & Height vs Volume (Testing Data)')
		# plt.show()


def main():
	# Cherry tree diameters are easy. Heights are hard.
	cherry_tree_df = pd.read_csv('CherryTree.csv')

	# Sometimes it's nice to see the raw data.
	# print(cherry_tree_df.head())

	# simple_linear_regression(cherry_tree_df, False)
	# simple_linear_regression(cherry_tree_df, True)
	# multiple_linear_regression(cherry_tree_df, False, False)
	# multiple_linear_regression(cherry_tree_df, True, False)
	multiple_linear_regression(cherry_tree_df, False, True)
	# multiple_linear_regression(cherry_tree_df, True, True)


if __name__ == "__main__":
	main()
