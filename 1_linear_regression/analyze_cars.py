# Simple cars analysis script using linear regression.	

import os
import sys
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
from sklearn.metrics import mean_squared_error, r2_score


def load_cars_df():
	# Load the cars.csv file that ships with this project.
	script_dir = os.path.dirname(os.path.abspath(__file__))
	csv_path = os.path.join(script_dir, 'cars.csv')
	df = pd.read_csv(csv_path)
	return df


def inspect_df(df, n=5):
	# Print quick inspection information about the DataFrame.
	print('Columns:', df.columns.tolist())
	print('\nDtypes:\n', df.dtypes)
	print('\nHead:\n', df.head(n).to_string())
	print('\nNumeric summary:\n', df.describe())


def top_correlations(df, target='price', k=10):
	# Compute and print correlations between numeric features and a target column.
	num = df.select_dtypes(include=[np.number])
	if target not in num.columns:
		print(f"Target '{target}' not numeric or not found")
		return None
	corrs = num.corr()[target].sort_values(ascending=False)
	print(f"\nTop correlations with '{target}':\n", corrs.head(k))
	return corrs



def simple_linear_regression_plot(df, feature='enginesize', target='price', plot='both'):
	# Run a simple linear regression and plot results.
	sub = df[[feature, target]].dropna()
	X = sub[[feature]].values
	y = sub[target].values

	# Split into training and test (25% test)
	X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.25, random_state=42)

	# Fit the model on training data
	model = lm.LinearRegression()
	model.fit(X_train, y_train)

	# Predict on test set and compute metrics
	y_pred = model.predict(X_test)

	mse = mean_squared_error(y_test, y_pred)
	rmse = np.sqrt(mse)
	r2 = r2_score(y_test, y_pred)

	print(f"\nSimple linear regression: {feature} -> {target}")
	print(f"Coefficient: {model.coef_[0]:.4f}, Intercept: {model.intercept_:.4f}")
	print(f"Test RMSE: {rmse:.3f}, R^2: {r2:.3f}")

	def plot_dataset(x_vals, y_vals, label, out_name_suffix, color):
		# Plot a single dataset (train or test) and the model fit line.
		#
		# The fit line is computed using a dense sorted grid so the line appears
		# smooth even when only a few points are plotted.
		plt.figure(figsize=(8,6))
		plt.scatter(x_vals, y_vals, color=color, alpha=0.7, label=label)
		# plot the same model fit line (dense grid)
		x_grid = np.linspace(X.min(), X.max(), 200).reshape(-1,1)
		y_grid = model.predict(x_grid)
		plt.plot(x_grid, y_grid, 'r-', linewidth=2, label='Fit')
		plt.xlabel(feature)
		plt.ylabel(target)
		plt.title(f'{feature} vs {target} ({label})')
		plt.legend()
		out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{feature}_vs_{target}_{out_name_suffix}.png')
		# Show plot interactively; do not save to file when running locally.
		plt.show()

	# Plot according to the 'plot' parameter
	if plot in ('train', 'both'):
		plot_dataset(X_train, y_train, label='Train', out_name_suffix='train', color='blue')
	if plot in ('test', 'both'):
		plot_dataset(X_test, y_test, label='Test', out_name_suffix='test', color='green')


def multiple_linear_regression_with_ohe(df, target='price'):
	# Choose predictors: numeric ones and one categorical column to encode
	numeric_features = ['enginesize', 'horsepower', 'curbweight', 'citympg']
	cat_feature = 'carbody'

	# Build a working DataFrame with the chosen columns and drop rows with NA
	cols = numeric_features + [cat_feature, target]
	sub = df[cols].dropna().copy()
	if sub.shape[0] == 0:
		print('No rows available for multiple regression after dropping NA')
		return

	# One-hot encode the categorical column. pandas.get_dummies will create new
	# columns for each category (e.g., 'carbody_sedan', 'carbody_hatchback').
	cat_ohe = pd.get_dummies(sub[cat_feature], prefix=cat_feature, drop_first=True)

	# Combine numeric features and the new OHE columns to form X
	X = pd.concat([sub[numeric_features].reset_index(drop=True), cat_ohe.reset_index(drop=True)], axis=1)
	y = sub[target].values

	# Train/test split
	X_train, X_test, y_train, y_test = ms.train_test_split(X.values, y, test_size=0.25, random_state=42)

	# Fit LinearRegression
	model = lm.LinearRegression()
	model.fit(X_train, y_train)

	# Predict on test set and compute metrics
	y_pred = model.predict(X_test)
	mse = mean_squared_error(y_test, y_pred)
	rmse = np.sqrt(mse)
	r2 = r2_score(y_test, y_pred)

	print('\nMultiple linear regression (numeric + one-hot encoded categorical)')
	print(f'Predictors: {X.shape[1]} features (including OHE columns)')
	print(f'Test RMSE: {rmse:.3f}, R^2: {r2:.3f}')

	# Save a simple predicted vs actual scatter plot
	plt.figure(figsize=(7,6))
	plt.scatter(y_test, y_pred, alpha=0.7)
	maxv = max(max(y_test), max(y_pred))
	minv = min(min(y_test), min(y_pred))
	plt.plot([minv, maxv], [minv, maxv], 'r--', linewidth=1)
	plt.xlabel('Actual ' + target)
	plt.ylabel('Predicted ' + target)
	plt.title('Multiple regression: Actual vs Predicted')
	# Show plot interactively; do not save to file when running locally.
	plt.show()


def main():
	# Load dataset
	df = load_cars_df()

	# Run only the requested analyses and keep console output concise:
	# 1) Simple linear regression: enginesize -> price
	simple_linear_regression_plot(df, feature='enginesize', target='price', plot='train')

	# 2) Multiple linear regression demonstrating one-hot encoding
	multiple_linear_regression_with_ohe(df, target='price')


if __name__ == '__main__':
	main()