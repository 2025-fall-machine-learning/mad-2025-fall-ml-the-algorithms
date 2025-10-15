from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
	import seaborn as sns
except Exception:
	sns = None
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline


CLEANED_DIR = Path(__file__).parent / 'cleanedFiles'


def load_cleaned():
	past_path = CLEANED_DIR / 'past_cleaned.csv'
	next_path = CLEANED_DIR / 'next_cleaned.csv'
	if not past_path.exists() or not next_path.exists():
		raise FileNotFoundError(f'Missing cleaned files in {CLEANED_DIR}; run participantAnalysis.py first')
	df_past = pd.read_csv(past_path)
	df_next = pd.read_csv(next_path)
	return df_past, df_next


def analyze_past(random_state: int = 42, print_output: bool = True):
	df_past, _ = load_cleaned()

	if 'SurvivalScore' not in df_past.columns:
		raise RuntimeError('SurvivalScore not found in past data')

	# Drop non-predictor columns if present
	X = df_past.drop(columns=['Name', 'SurvivalScore'], errors='ignore')
	y = df_past['SurvivalScore']

	# numeric predictors only
	num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
	if len(num_cols) == 0:
		raise RuntimeError('No numeric predictor columns found in past data')
	X_num = X[num_cols]

	# Correlation heatmap data
	corr_matrix = X_num.corr()

	# Train/test split
	X_train, X_test, y_train, y_test = train_test_split(X_num, y, test_size=0.2, random_state=random_state)

	# Preprocessing: median impute then standardize
	imputer = SimpleImputer(strategy='median')
	scaler = StandardScaler()
	X_train_imp = imputer.fit_transform(X_train)
	X_test_imp = imputer.transform(X_test)
	X_train_scaled = scaler.fit_transform(X_train_imp)
	X_test_scaled = scaler.transform(X_test_imp)

	# Multiple linear regression (multivariate)
	mlr = LinearRegression()
	mlr.fit(X_train_scaled, y_train)
	preds_train = mlr.predict(X_train_scaled)
	preds_test = mlr.predict(X_test_scaled)

	train_r2 = r2_score(y_train, preds_train)
	test_r2 = r2_score(y_test, preds_test)
	train_rmse = float(np.sqrt(mean_squared_error(y_train, preds_train)))
	test_rmse = float(np.sqrt(mean_squared_error(y_test, preds_test)))

	coefs = pd.Series(mlr.coef_, index=num_cols).sort_values(key=lambda v: v.abs(), ascending=False)
	residuals = y_test - preds_test

	# Single-variable linear regressions (one at a time)
	single_results = {}
	for c in num_cols:
		Xi = X_num[[c]]
		Xi_train, Xi_test, yi_train, yi_test = train_test_split(Xi, y, test_size=0.2, random_state=random_state)
		# simple pipeline: impute + scale + fit
		Xi_train_imp = imputer.fit_transform(Xi_train)
		Xi_test_imp = imputer.transform(Xi_test)
		Xi_train_scaled = scaler.fit_transform(Xi_train_imp)
		Xi_test_scaled = scaler.transform(Xi_test_imp)
		lr = LinearRegression()
		lr.fit(Xi_train_scaled, yi_train)
		p_test = lr.predict(Xi_test_scaled)
		r2 = r2_score(yi_test, p_test)
		rmse = float(np.sqrt(mean_squared_error(yi_test, p_test)))
		single_results[c] = {'r2': r2, 'rmse': rmse, 'coef': float(lr.coef_[0])}

	results = {
		'corr_matrix': corr_matrix,
		'multivar': {
			'train_r2': train_r2,
			'test_r2': test_r2,
			'train_rmse': train_rmse,
			'test_rmse': test_rmse,
			'coefs': coefs,
		},
		'single_var': pd.DataFrame(single_results).T,
		'residuals': residuals,
	}

	if print_output:
		print(f'Loaded past data: {df_past.shape}')
		print('\nMultiple linear regression (train/test holdout):')
		print(f"Train R^2: {train_r2:.3f}, Test R^2: {test_r2:.3f}")
		print(f"Train RMSE: {train_rmse:.3f}, Test RMSE: {test_rmse:.3f}")
		print('\nCoefficients (multivariate, sorted by abs):')
		print(coefs)
		print('\nSingle-variable linear regression results (test set):')
		print(pd.DataFrame(single_results).T.sort_values('r2', ascending=False))
		print('\nResiduals summary (test set):')
		print('mean {:.3f}, std {:.3f}'.format(residuals.mean(), residuals.std()))

		# Show heatmap (user can screenshot). If seaborn isn't installed, print matrix.
		if sns is not None and plt is not None:
			fig, ax = plt.subplots(figsize=(8, 6))
			sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='vlag', ax=ax)
			ax.set_title('Feature correlation heatmap')
			plt.show()
		else:
			print('\nCorrelation matrix:')
			print(corr_matrix)
	return results


def predict_next(return_df: bool = False):
	#Train on all past data and predict SurvivalScore for next season participants.
	#Returns a DataFrame sorted by predicted SurvivalScore (descending).
	df_past, df_next = load_cleaned()

	# Prepare training data
	y = df_past['SurvivalScore']
	X = df_past.drop(columns=['Name', 'SurvivalScore'], errors='ignore')

	# Prepare next-season features
	X_next = df_next.drop(columns=['Name'], errors='ignore')

	# Use central pipeline factory for consistent preprocessing + model
	def make_regression_pipeline():
		return make_pipeline(
			SimpleImputer(strategy='median'),
			StandardScaler(),
			LinearRegression()
		)

	pipe = make_regression_pipeline()
	pipe.fit(X, y)

	# Ensure feature columns align and are in the same order as training data
	missing = [c for c in X.columns if c not in X_next.columns]
	if len(missing) > 0:
		raise RuntimeError(f"Next-season data is missing predictor columns required for prediction: {missing}")
	# Reorder columns of X_next to match X
	X_next = X_next[X.columns]

	preds = pipe.predict(X_next)

	df_pred = df_next.copy()
	df_pred['PredictedSurvivalScore'] = preds

	df_pred = df_pred[['Name', 'PredictedSurvivalScore']].sort_values('PredictedSurvivalScore', ascending=False).reset_index(drop=True)

	# Clear header for next-data section and print all predictions then top-3
	print('\n' + '='*10 + ' Next-data analysis: survival score prediction ' + '='*10 + '\n')

	# Verification: ensure predictions align to rows and print verification after past-data summary
	if len(preds) != len(df_next):
		raise RuntimeError('Prediction length does not match next-season rows')
	print(f"Verified: predictions produced for {len(preds)} participants; feature columns aligned.")

	# Print every predicted contestant (sorted by predicted score descending)
	df_sorted = df_pred.sort_values('PredictedSurvivalScore', ascending=False).reset_index(drop=True)
	print('All predicted participants (next season) — sorted by predicted score:')
	for _, r in df_sorted.iterrows():
		print(f"{r['Name']}: {r['PredictedSurvivalScore']:.3f}")

	# Then print top-3
	top3 = df_sorted.head(3)
	print('\nTop 3 predicted participants (next season):')
	for i, r in top3.iterrows():
		print(f"{i+1}. {r['Name']} — {r['PredictedSurvivalScore']:.3f}")

	if return_df:
		return df_pred


if __name__ == '__main__':
	# Default behavior: always run past-data analysis then next-data predictions.
	print('\n' + '='*10 + ' Past-data specialist scoring analysis ' + '='*10 + '\n')
	res = analyze_past(print_output=True)
	m = res['multivar']
	print('\n--- SHORT SUMMARY ---')
	print(f"Train R2: {m['train_r2']:.3f}, Test R2: {m['test_r2']:.3f}")
	print(f"Train RMSE: {m['train_rmse']:.3f}, Test RMSE: {m['test_rmse']:.3f}")

	# Then predictions for next data
	predict_next()