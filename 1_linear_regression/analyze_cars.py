import os
import sys
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
from sklearn.metrics import mean_squared_error, r2_score
import argparse
from sklearn.feature_selection import mutual_info_regression


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


def independence_checks(df, target='price'):
	"""Print compact independence / multicollinearity diagnostics.

	- Builds the same X used by the multiple-regression path (numeric + OHE carbody).
	- Prints high-correlation pairs (|r| >= 0.80), VIFs >= 5, and top mutual-info scores.
	"""
	numeric_features = ['enginesize', 'horsepower', 'curbweight', 'citympg']
	cat_feature = 'carbody'

	cols = numeric_features + [cat_feature, target]
	sub = df[cols].dropna().copy()
	if sub.shape[0] == 0:
		print('Independence checks: no rows available after dropping NA')
		return

	cat_ohe = pd.get_dummies(sub[cat_feature], prefix=cat_feature, drop_first=True)
	X = pd.concat([sub[numeric_features].reset_index(drop=True), cat_ohe.reset_index(drop=True)], axis=1)
	y = sub[target].values

	# Pairwise Pearson correlations
	corr = X.corr()
	high_pairs = []
	cols_list = list(corr.columns)
	for i in range(len(cols_list)):
		for j in range(i+1, len(cols_list)):
			a = cols_list[i]; b = cols_list[j]
			v = corr.at[a, b]
			if abs(v) >= 0.80:
				high_pairs.append((a, b, v))

	# Compute VIF per predictor via 1 / (1 - R_j^2)
	from sklearn.linear_model import LinearRegression
	vifs = {}
	X_values = X.values
	for ix, col in enumerate(X.columns):
		other_idx = [k for k in range(X_values.shape[1]) if k != ix]
		X_other = X_values[:, other_idx]
		y_col = X_values[:, ix]
		# regress y_col on X_other
		try:
			reg = LinearRegression().fit(X_other, y_col)
			r2_j = r2_score(y_col, reg.predict(X_other))
			vif = np.inf if (1 - r2_j) <= 1e-12 else 1.0 / (1.0 - r2_j)
		except Exception:
			vif = np.inf
		vifs[col] = vif

	# Mutual information ranking
	try:
		mi = mutual_info_regression(X.values, y, random_state=0)
		mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
	except Exception:
		mi_series = pd.Series(dtype=float)

	# Print concise diagnostics
	print('\nIndependence checks:')
	if len(high_pairs) == 0:
		print('  No high pairwise correlations (|r| >= 0.80).')
	else:
		pairs_str = '; '.join([f"{a} & {b} (r={v:.2f})" for a,b,v in high_pairs])
		print('  High-correlation pairs (|r|>=0.80):', pairs_str)

	high_vif = [(k, v) for k, v in vifs.items() if v >= 5]
	if len(high_vif) == 0:
		print('  No predictors with VIF >= 5')
	else:
		vif_str = '; '.join([f"{k} (VIF={v:.1f})" for k, v in high_vif])
		print('  High VIFs (>=5):', vif_str)

	if not mi_series.empty:
		top_mi = ', '.join([f"{name} ({val:.2f})" for name, val in mi_series.head(5).items()])
		print('  Top mutual-info vs price:', top_mi)
	else:
		print('  Mutual information unavailable.')




def simple_linear_regression_plot(df, feature='enginesize', target='price', plot='both', show='both'):
	# Run a simple linear regression and plot results.
	sub = df[[feature, target]].dropna()
	X = sub[[feature]].values
	y = sub[target].values

	# Split into training and test (25% test). No fixed random_state so the
	# split will be randomized on each run (useful for exploratory experiments).
	X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.25)

	# Fit the model on training data
	model = lm.LinearRegression()
	model.fit(X_train, y_train)

	# Predict on both training and test sets and compute metrics for each so
	# the console output clearly states whether a metric is computed on the
	# training or testing data.
	y_pred_train = model.predict(X_train)
	y_pred_test = model.predict(X_test)

	mse_train = mean_squared_error(y_train, y_pred_train)
	rmse_train = np.sqrt(mse_train)
	r2_train = r2_score(y_train, y_pred_train)

	mse_test = mean_squared_error(y_test, y_pred_test)
	rmse_test = np.sqrt(mse_test)
	r2_test = r2_score(y_test, y_pred_test)

	# Only show R^2 per the user's request; coefficient/intercept and
	# other metrics removed for concise output. Title still printed.
	print(f"\nSimple linear regression: {feature} -> {target}")
	if show in ('train', 'both'):
		print(f"Train R^2: {r2_train:.3f}")
		# compact quadratic check (single predictor)
		Xtr_df = pd.DataFrame(X_train, columns=[feature])
		Xtr_q = Xtr_df.copy(); Xtr_q[feature + '_sq'] = Xtr_q[feature] ** 2
		quad = lm.LinearRegression(); quad.fit(Xtr_q, y_train)
		r2_lin = r2_train
		r2_quad = r2_score(y_train, quad.predict(Xtr_q))
		delta = r2_quad - r2_lin
		verdict = 'Linear' if delta < 0.01 else ('Mild nonlinearity' if delta < 0.02 else 'Nonlinear')
		print(f"Linearity (delta R^2): {delta:.3f} -> {verdict}")
	if show in ('test', 'both'):
		print(f"Test  R^2: {r2_test:.3f}")
		# compact quadratic check on test set
		Xte_df = pd.DataFrame(X_test, columns=[feature])
		Xte_q = Xte_df.copy(); Xte_q[feature + '_sq'] = Xte_q[feature] ** 2
		r2_lin_t = r2_test
		r2_quad_t = r2_score(y_test, lm.LinearRegression().fit(Xte_q, y_test).predict(Xte_q))
		delta_t = r2_quad_t - r2_lin_t
		verdict_t = 'Linear' if delta_t < 0.01 else ('Mild nonlinearity' if delta_t < 0.02 else 'Nonlinear')
		print(f"Linearity (delta R^2): {delta_t:.3f} -> {verdict_t}")

	def plot_dataset(x_vals, y_vals, label, out_name_suffix, color):
		# Plot a single dataset (train or test) and the model fit line.
		#
		# The fit line is computed using a dense sorted grid so the line appears
		# smooth even when only a few points are plotted.
		plt.figure(figsize=(8,6))
		plt.scatter(x_vals, y_vals, color=color, alpha=0.7, label=label)
		# Make the title clearly state which dataset is being plotted
		plt.title(f'Simple regression ({label} set): {feature} vs {target}')
		# plot the same model fit line (dense grid)
		x_grid = np.linspace(X.min(), X.max(), 200).reshape(-1,1)
		y_grid = model.predict(x_grid)
		plt.plot(x_grid, y_grid, 'r-', linewidth=2, label='Fit')
		plt.xlabel(feature)
		plt.ylabel(target)
		plt.legend()
		# Show plot interactively; do not save to file when running locally.
		plt.show()

	# Plot according to the 'plot' parameter
	if plot in ('train', 'both'):
		plot_dataset(X_train, y_train, label='Train', out_name_suffix='train', color='blue')
	if plot in ('test', 'both'):
		plot_dataset(X_test, y_test, label='Test', out_name_suffix='test', color='green')



"""
Multiple regression feature selection:
- Dropped 'enginesize' due to high collinearity with 'curbweight' and 'horsepower' (see independence diagnostics).
- Kept 'curbweight', 'horsepower', 'citympg' (citympg is less correlated, but can be dropped for a stricter model).
- Kept one-hot encoded 'carbody' columns.
"""
def multiple_linear_regression_with_ohe(df, target='price', show='both'):
	# Choose predictors: drop enginesize (collinear), keep curbweight, horsepower, citympg, and OHE carbody
	numeric_features = ['curbweight', 'horsepower', 'citympg']
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

	# Train/test split (randomized each run)
	X_train, X_test, y_train, y_test = ms.train_test_split(X.values, y, test_size=0.25)

	# Fit LinearRegression
	model = lm.LinearRegression()
	model.fit(X_train, y_train)

	# Predict on both training and test sets and compute metrics for each so
	# it's clear how the model performs on held-out data vs training data.
	y_pred_train = model.predict(X_train)
	y_pred_test = model.predict(X_test)

	mse_train = mean_squared_error(y_train, y_pred_train)
	rmse_train = np.sqrt(mse_train)
	r2_train = r2_score(y_train, y_pred_train)

	mse_test = mean_squared_error(y_test, y_pred_test)
	rmse_test = np.sqrt(mse_test)
	r2_test = r2_score(y_test, y_pred_test)

	# Minimal multiple-regression output: R^2 and compact linearity check
	print('\nMultiple linear regression')
	feature_names = list(X.columns) if hasattr(X, 'columns') else [f'col{i}' for i in range(X.shape[1])]
	if show in ('train', 'both'):
		print(f'Train R^2: {r2_train:.3f}')
		# compact quadratic check (global: add squared columns)
		Xtrain_df = pd.DataFrame(X_train, columns=feature_names)
		Xtrain_q = Xtrain_df.copy()
		for col in Xtrain_df.columns:
			Xtrain_q[col + '_sq'] = Xtrain_q[col] ** 2
		quad = lm.LinearRegression(); quad.fit(Xtrain_q, y_train)
		r2_lin = r2_train
		r2_quad = r2_score(y_train, quad.predict(Xtrain_q))
		delta = r2_quad - r2_lin
		verdict = 'Linear' if delta < 0.01 else ('Mild nonlinearity' if delta < 0.02 else 'Nonlinear')
		print(f"Linearity (delta R^2): {delta:.3f} -> {verdict}")
	if show in ('test', 'both'):
		print(f'Test  R^2: {r2_test:.3f}')
		Xtest_df = pd.DataFrame(X_test, columns=feature_names)
		Xtest_q = Xtest_df.copy()
		for col in Xtest_df.columns:
			Xtest_q[col + '_sq'] = Xtest_q[col] ** 2
		r2_lin_t = r2_test
		r2_quad_t = r2_score(y_test, lm.LinearRegression().fit(Xtest_q, y_test).predict(Xtest_q))
		delta_t = r2_quad_t - r2_lin_t
		verdict_t = 'Linear' if delta_t < 0.01 else ('Mild nonlinearity' if delta_t < 0.02 else 'Nonlinear')
		print(f"Linearity (delta R^2): {delta_t:.3f} -> {verdict_t}")

	# Predicted vs actual scatter plot. Respect the 'show' parameter so the
	# user sees the chart for the same dataset whose metrics were printed.
	if show == 'train':
		xx_actual = y_train
		xx_pred = y_pred_train
		title_suffix = 'Train set'
	else:
		# default to test plot for show in ('test','both') to keep plots simple
		xx_actual = y_test
		xx_pred = y_pred_test
		title_suffix = 'Test set'

	plt.figure(figsize=(7,6))
	plt.scatter(xx_actual, xx_pred, alpha=0.7)
	maxv = max(max(xx_actual), max(xx_pred))
	minv = min(min(xx_actual), min(xx_pred))
	plt.plot([minv, maxv], [minv, maxv], 'r--', linewidth=1)
	plt.xlabel('Actual ' + target)
	plt.ylabel('Predicted ' + target)
	plt.title(f'Multiple regression ({title_suffix}): Actual vs Predicted')
	plt.show()


def main():
	# Parse a small CLI so you can choose which dataset to plot for the simple regression
	parser = argparse.ArgumentParser(description='Analyze cars: simple and multiple regression')
	# Which dataset to show for the simple regression
	group = parser.add_mutually_exclusive_group()
	group.add_argument('--train', action='store_true', help='Show only the training set plot for the simple regression')
	group.add_argument('--test', action='store_true', help='Show only the test set plot for the simple regression')

	# Allow the user to run only the simple analysis OR only the multiple analysis
	only_group = parser.add_mutually_exclusive_group()
	# Accept both a single-dash and double-dash form in case you typed -simple
	only_group.add_argument('-s', '-simple', '--simple', dest='simple', action='store_true',
							help='Run only the simple enginesize->price regression and exit')
	only_group.add_argument('-m', '-multiple', '--multiple', dest='multiple', action='store_true',
							help='Run only the multiple regression (with OHE) and exit')

	args = parser.parse_args()

	# Load dataset
	df = load_cars_df()

	# Run independence checks on every invocation (concise diagnostics)
	independence_checks(df, target='price')

	# Decide which dataset to plot: default to training if neither flag is provided
	if args.test:
		plot_choice = 'test'
	else:
		# default behavior and when --train is supplied
		plot_choice = 'train'

	# Branch according to the user's 'only' flags. If neither is provided,
	# run both analyses (backward-compatible behavior).
	if args.simple:
		simple_linear_regression_plot(df, feature='enginesize', target='price', plot=plot_choice, show=plot_choice)
		return
		return

	if args.multiple:
		multiple_linear_regression_with_ohe(df, target='price', show=plot_choice)
		return

	# Default: run both analyses
	simple_linear_regression_plot(df, feature='enginesize', target='price', plot=plot_choice, show=plot_choice)
	multiple_linear_regression_with_ohe(df, target='price', show=plot_choice)


if __name__ == '__main__':
	main()