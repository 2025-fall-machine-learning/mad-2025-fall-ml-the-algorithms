import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
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



def show_r_squared (model, predictors, response):
    r_squared = model.score(predictors, response) # Added r_squared
    print(f"R-Squared value: {r_squared:.4}") # Print R-Squared value


def create_linear_regression_model(predictors, response):
	model = lm.LinearRegression()
	model.fit(predictors, response)

	return model


def perform_linear_regression_prediction(model, predictors):
	prediction = model.predict(predictors)

	return prediction


# Egads! The simple_linear_regression and multiple_linear_regression functions have a lot of
# common code! We are violating our most sacred principle: Reuse! (or DRY (Don't Repeat Yourself).)
def simple_linear_regression(cars_df, create_testing_set):
	''' Performs simple linear regression on the cars data. That is, one predictor
	predicting one response. '''
	predictors = cars_df[['enginesize']].values
	response = cars_df['price'].values

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
	# r_squared = model.score(training_predictors, training_response) # Added r_squared
	# print(f"R-Squared value: {r_squared}") # Print R-Squared value
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
	graph(training_predictors, training_response, "blue", prediction, "Training Data")

	
	if create_testing_set:
		prediction = perform_linear_regression_prediction(model, testing_predictors)
		print("The testing data predictors, prediction and response values:")
		print_1d_data_summary(testing_predictors)
		print_1d_data_summary(prediction)
		print_1d_data_summary(testing_response)
		graph(testing_predictors, testing_response, "green", prediction, "Testing Data")

def graph(pred, resp, whatcolor, prediction, whatlabel): 
	# Plot the data and the best fit line.
	plt.scatter(pred, resp, color=whatcolor, label=whatlabel)
	plt.plot(pred, prediction, color='red', label='Best Fit Line')
	plt.xlabel('Engine Size')
	plt.ylabel('Price')
	plt.title('Linear Regression: Engine Size vs Price (Training Data)')
	plt.legend()
	plt.show()


def multiple_linear_regression(cars_df, create_testing_set, one_hot_encode):
	''' Performs multiple linear regression on the cars data. That is, multiple predictors
	predicting one response. '''

	# if not one_hot_encode:
	# 	predictors = cars_df[['enginesize', 'horsepower']].values
	# else:
	# 	# One-hot encode the carbody column (values: convertable, hardtop, hatchback, sedan, wagon).
	# 	carbody_dummies = pd.get_dummies(cars_df['carbody'], prefix='carbody')
	# 	predictors = pd.concat([cars_df[['enginesize', 'horsepower']], carbody_dummies], axis=1).values
	# 	# predictors = carbody_dummies

	# response = cars_df['price'].values

	# Create a correlation matrix by combining predictors and response

	# vvv One-hot encoding isn't necessary for this dataset because it's values don't contrtibute
	# to the repsonse. vvv

	# if not one_hot_encode:
	predictors = cars_df[['enginesize', 'horsepower']].values # Tab this line to use one-hot encoding
		# For correlation matrix, create - simple case with enginesize and horsepower
	modified_cars_df = pd.DataFrame(predictors, columns=['enginesize', 'horsepower']) # Tab this line to use one-hot encoding
	# else:
	# 	# One-hot encode the carbody column (values: convertable, hardtop, hatchback, sedan, wagon).
	# 	carbody_dummies = pd.get_dummies(cars_df['carbody'], prefix='carbody')
	# 	predictors_df_combined = pd.concat([cars_df[['enginesize', 'horsepower']], carbody_dummies], axis=1)
	# 	predictors = predictors_df_combined.values
	# 	# For correlation matrix, resuse the combined Dataframe structure
	# 	modified_cars_df = predictors_df_combined.copy()
		
	response = cars_df['price'].values
 
	# Add the response variable
	modified_cars_df['price'] = response
	correlation_matrix = modified_cars_df.corr()
	price_correlation_matrix = correlation_matrix[['price']].sort_values(by='price') # Exclude self-correlation
	sns.heatmap(price_correlation_matrix, annot=True)
	plt.show()

	mask = abs(correlation_matrix) < 0.3
	sns.heatmap(correlation_matrix, annot=True, mask=mask)
	plt.show()

	# # Calculate correlation matrix
	# correlation_matrix = modified_cars_df.corr()

	# # Create heatmap with annotations, higlighting correlations < 0.3
	# plt.figure(figsize=(10, 8))
	
	# # Create a mask for correlations < 0.3 (but not the diagonal)
	# mask = np.abs(correlation_matrix) < 0.3
	# np.fill_diagonal(mask.values, False)  # Keep diagonal unmasked

	# #Create the heatmap
	# sns.heatmap(correlation_matrix, 
	# 			annot=True,
	# 			cmap='coolwarm', 
	# 			center=0, 
	# 			fmt='.3f',
	# 			square=True,
	# 			linewidths=.5,
	# 			mask=None) # Show all values
 
	# # Highlight low correlations by adding a border or different color
	# # We'll use a custom approach to highlight correlations < 0.3
	# ax = plt.gca()
	# for i in range(len(correlation_matrix.columns)):
	# 	for j in range(len(correlation_matrix.columns)):
	# 		if i != j and abs(correlation_matrix.iloc[i, j]) < 0.3:
	# 			# Add a red border around cells with correlations 
	# 			rect = plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', linewidth=2)
	# 			ax.add_patch(rect)

	# plt.title('Corretlation Matrix (Red bordses indicate |corr| < 0.3)')
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

	# # Create 3D plot with enginesize and price as independant variables, horsepower as dependant variable, and 
 	# # best fit plane.
	# fig = plt.figure(figsize=(10, 7))
	# ax = fig.add_subplot(111, projection='3d')
 
	# # Extract enginesize and price from training_predictors
	# enginesize_vals = training_predictors[:, 0]
	# price_vals = training_predictors[:, 1]

	# # Create scatter plot
	# ax.scatter(enginesize_vals, price_vals, training_response, color='blue', alpha=0.6, label='Training Data')

	# # Create meshgrid for best fit plane
	# enginesize_range = np.linspace(enginesize_vals.min(), enginesize_vals.max(), 20)
	# price_range = np.linspace(price_vals.min(), price_vals.max(), 20)
	# enginesize_mesh, price_mesh = np.meshgrid(enginesize_range, price_range)
 
	# # Create predictor array for the plane
	# plane_predictors = np.column_stack((enginesize_mesh.ravel(), price_mesh.ravel()))
 
	# # Handle one=hot encoding if present
	# if training_predictors.shape[1] > 2:
	# 	# Add mean values for one-hot encoded features
	# 	mean_vals = np.mean(training_predictors[:, 2:], axis=0)
	# 	plane_predictors_full = np.column_stack([
    #   		plane_predictors, 
    #    		np.tile(mean_vals, (plane_predictors.shape[0], 1))
    #  	])
	# else:
	# 	plane_predictors_full = plane_predictors
  
	# # Predict horsepower values for the plane
	# horsepower_predictions = model.predict(plane_predictors_full)
	# horsepower_predictions = horsepower_predictions.reshape(enginesize_mesh.shape)
  
	# # Plot the best fit plane
	# ax.plot_surface(enginesize_mesh, price_mesh, horsepower_predictions, alpha=0.3, color='red')
	# ax.set_xlabel('Engine Size')
	# ax.set_ylabel('Price')
	# ax.set_zlabel('Horsepower')
	# ax.set_title('3D Linear Regression: Engine Size & Horsepower vs Price (Training Data)')
	# plt.show()
  
	# Plot the data and the best fit line.
	# graph(training_predictors[:,0], training_response, "blue", prediction, "Training Data")

	if create_testing_set:
		prediction = perform_linear_regression_prediction(model, testing_predictors)
		print("The testing data prediction and response values:")
		print_1d_data_summary(prediction)
		print_1d_data_summary(testing_response)
  
		# Create 3D plot for testing data with best fit plane.
		# fig = plt.figure(figsize=(10, 7))
		# ax = fig.add_subplot(111, projection='3d')
  
		# # Extract enginesize and price from testing_predictors
		# test_enginesize_vals = testing_predictors[:, 0]
		# test_price_vals = testing_predictors[:, 1]
  
		# # Scatter plot of actual testing data points
		# ax.scatter(test_enginesize_vals, test_price_vals, testing_response, color='green', alpha=0.6, label='Testing Data')
  
		# # Create a meshgrid for the best fit plane (same as training)
		# test_enginesize_range = np.linspace(test_enginesize_vals.min(), test_enginesize_vals.max(), 20)
		# test_price_range = np.linspace(test_price_vals.min(), test_price_vals.max(), 20)
		# test_enginesize_mesh, test_price_mesh = np.meshgrid(test_enginesize_range, test_price_range)
  
		# # Create predictor array for the plane
		# test_plane_predictors = np.column_stack((test_enginesize_mesh.ravel(), test_price_mesh.ravel()))
  
		# # Handle one-hot encoding if features are present
		# if testing_predictors.shape[1] > 2:
		# 	# Add mean values for one-hot encoded features
		# 	test_mean_vals = np.mean(testing_predictors[:, 2:], axis=0)
		# 	test_plane_predictors_full = np.column_stack([
		#  		test_plane_predictors, 
		#  		np.tile(test_mean_vals, (test_plane_predictors.shape[0], 1))
	  	# 	])
		# else:
		# 	test_plane_predictors_full = test_plane_predictors
   
		# # Predict horsepower values for the plane
		# test_horsepower_mesh = model.predict(test_plane_predictors_full)
		# test_horsepower_mesh = test_horsepower_mesh.reshape(test_enginesize_mesh.shape)

		# # Plot the best fit plane
		# ax.plot_surface(test_enginesize_mesh, test_price_mesh, test_horsepower_mesh, color='red', alpha=0.5, label='Best Fit Plane')
		# ax.set_xlabel('Engine Size')
		# ax.set_ylabel('Horsepower')
		# ax.set_zlabel('Price')
		# ax.set_title('Multiple Linear Regression: Engine Size & Horsepower vs Price (Testing Data)')
		# plt.show

def main():

	cars_df = pd.read_csv('E:/Madison College/Machine Learning/mad-2025-fall-ml-the-algorithms/1_linear_regression/cars.csv')

	# Sometimes it's nice to see the raw data.
	# print(cars_df.head())

	# simple_linear_regression(cars_df, False)
	# simple_linear_regression(cars_df, True)
	# multiple_linear_regression(cars_df, False, False)
	# multiple_linear_regression(cars_df, False, True)
	# multiple_linear_regression(cars_df, True, True)
	multiple_linear_regression(cars_df, True, False)


if __name__ == "__main__":
	main()
