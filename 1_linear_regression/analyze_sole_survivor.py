import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score

"""analyze_sole_survivor.py

Reads `sole_survivor_past.csv`, fits a standardized linear regression to predict
the expert-provided `SurvivalScore` from the sub-scores, and writes a short
briefing for the data science team lead. Saves plots and a text briefing in the
same folder as this script.

Usage: run this file with the repository root as working directory or run
from its folder. It expects `sole_survivor_past.csv` to live in the same folder.
"""


def main(show_plots: bool = False):
	base_dir = os.path.dirname(__file__)
	csv_path = os.path.join(base_dir, "sole_survivor_past.csv")

	# Read CSV (skip initial spaces that appear in the provided file)
	df = pd.read_csv(csv_path, skipinitialspace=True)
	df.columns = [c.strip() for c in df.columns]

	# Clean names and ensure numeric columns
	df['Name'] = df['Name'].astype(str).str.strip()
	# Convert features and target to numeric safely; coerce errors to NaN
	feature_cols = [c for c in df.columns if c not in ('Name', 'SurvivalScore')]
	X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
	y = pd.to_numeric(df['SurvivalScore'], errors='coerce')

	# Drop rows with missing target or any non-numeric feature values
	valid_mask = y.notna() & X.notna().all(axis=1)
	if not valid_mask.all():
		dropped = (~valid_mask).sum()
		print(f"Warning: dropping {dropped} rows with non-numeric/missing feature or target values")
	df = df.loc[valid_mask].copy()
	X = X.loc[valid_mask]
	y = y.loc[valid_mask]

	# Convert target to 1d numpy array for sklearn compatibility
	y = np.asarray(y).ravel()

	# Short dataset summary
	print(f"Rows: {len(df)}; Features: {feature_cols}")

	# Correlation heatmap (features + target) - matplotlib only
	corr = df[feature_cols + ['SurvivalScore']].corr()
	plt.figure(figsize=(10, 8))
	im = plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
	plt.colorbar(im)
	ticks = np.arange(len(corr.columns))
	plt.xticks(ticks, corr.columns, rotation=45, ha='right')
	plt.yticks(ticks, corr.columns)
	# annotate cells
	for i in range(len(corr.columns)):
		for j in range(len(corr.columns)):
			plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', color='black', fontsize=8)
	plt.title('Correlation matrix (features and SurvivalScore)')
	plt.tight_layout()
	plt.show()

	# Train/test split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Ensure y_train / y_test are 1d numpy arrays
	y_train = np.asarray(y_train).ravel()
	y_test = np.asarray(y_test).ravel()

	# Standardize features so coefficients are comparable
	scaler = StandardScaler().fit(X_train)
	X_train_s = scaler.transform(X_train)
	X_test_s = scaler.transform(X_test)

	# Fit linear model
	model = LinearRegression()
	model.fit(X_train_s, y_train)

	# Cross-validated metrics (5-fold) on full standardized data
	# Use a fresh, unfitted estimator for cross-validation to avoid side-effects
	from sklearn.linear_model import LinearRegression as _LR
	X_s_full = scaler.transform(X)
	cv_r2 = cross_val_score(_LR(), X_s_full, y, cv=5, scoring='r2')
	cv_mae = -cross_val_score(_LR(), X_s_full, y, cv=5, scoring='neg_mean_absolute_error')

	# Test set evaluation
	y_pred = model.predict(X_test_s)
	y_pred = np.asarray(y_pred).ravel()
	test_r2 = r2_score(y_test, y_pred)
	test_mae = mean_absolute_error(y_test, y_pred)
	# compute RMSE without using the `squared` kwarg for compatibility with older sklearn
	test_mse = mean_squared_error(y_test, y_pred)
	test_rmse = float(np.sqrt(test_mse))

	# Coefficients (standardized)
	# Coefficients (standardized) - make robust to shape issues
	coefs = np.ravel(model.coef_)
	# Align length with feature list if necessary
	if coefs.shape[0] != len(feature_cols):
		# If mismatch, warn and align to shortest length
		print(f"Warning: coefficient length {coefs.shape[0]} != features {len(feature_cols)}; aligning to minimum length")
		n = min(coefs.shape[0], len(feature_cols))
		coefs = coefs[:n]
		features_for_coefs = feature_cols[:n]
	else:
		features_for_coefs = feature_cols

	coef_df = pd.DataFrame({'feature': features_for_coefs, 'coef': coefs}).copy()
	coef_df['coef'] = coef_df['coef'].astype(float)
	coef_df['abs_coef'] = coef_df['coef'].abs()
	coef_df = coef_df.sort_values('abs_coef', ascending=False).drop(columns=['abs_coef'])

	plt.figure(figsize=(9, 5))
	plt.barh(coef_df['feature'], coef_df['coef'], color='C0')
	plt.axvline(0, color='k', linewidth=0.5)
	plt.xlabel('Standardized coefficient')
	plt.title('Standardized linear regression coefficients')
	plt.tight_layout()
	plt.show()

	# Predicted vs actual
	plt.figure(figsize=(6, 6))
	plt.scatter(y_test, y_pred, alpha=0.8)
	mn, mx = min(float(np.min(y_test)), float(np.min(y_pred))), max(float(np.max(y_test)), float(np.max(y_pred)))
	plt.plot([mn, mx], [mn, mx], 'r--', linewidth=1)
	plt.xlabel('Actual SurvivalScore (test)')
	plt.ylabel('Predicted SurvivalScore')
	plt.title(f'Predicted vs Actual (R²={test_r2:.3f}, MAE={test_mae:.2f})')
	plt.tight_layout()
	plt.show()

	# Residuals
	residuals = y_test - y_pred
	plt.figure(figsize=(6, 4))
	plt.hist(residuals, bins=15, alpha=0.7, color='C1')
	plt.xlabel('Residual (actual - predicted)')
	plt.title('Residual distribution (test set)')
	plt.tight_layout()
	plt.show()

	# Compose briefing
	top_pos = coef_df[coef_df['coef'] > 0].head(3)
	top_neg = coef_df[coef_df['coef'] < 0].head(3)

	briefing_lines = [
		'Briefing: Evaluation of expert SurvivalScore ratings',
		'--------------------------------------------------',
		f'Rows analyzed: {len(df)}',
		'',
		'Modeling approach: standardized linear regression using expert sub-scores.',
		f'Cross-validated R² (5-fold): mean={cv_r2.mean():.3f}, std={cv_r2.std():.3f}',
		f'Cross-validated MAE (5-fold): mean={cv_mae.mean():.2f}',
		'',
		f'Test set (20%) performance: R²={test_r2:.3f}, MAE={test_mae:.2f}, RMSE={test_rmse:.2f}',
		'',
		'Top features pushing SurvivalScore higher (standardized coef):',
	]
	for _, r in top_pos.iterrows():
		briefing_lines.append(f" - {r['feature']}: coef={r['coef']:.3f}")
	briefing_lines.append('')
	briefing_lines.append('Top features pushing SurvivalScore lower (standardized coef):')
	for _, r in top_neg.iterrows():
		briefing_lines.append(f" - {r['feature']}: coef={r['coef']:.3f}")

	briefing_lines.extend([
		'',
		'Interpretation:',
		f'- The linear model explains about {cv_r2.mean():.2f} (mean CV R²) of the variance in the expert SurvivalScore. ',
		'- This indicates experts are generally consistent: their numeric sub-scores contain meaningful signal about the final SurvivalScore, but a large portion of variance remains unexplained (subjectivity, omitted features, or nonlinearity).',
		f'- Typical prediction error (CV MAE) is {cv_mae.mean():.2f} points on the SurvivalScore scale; test MAE={test_mae:.2f}.',
	'- Visual artifacts (correlation heatmap, coefficient importance, predicted vs actual, and residuals) are displayed inline when the script runs.',
		'',
		'Validity and caveats:',
		'- Sample size is small (n=%d), so variance in CV estimates may be large.' % len(df),
		'- Linear model assumes additive effects and no strong interactions; if experts weigh features nonlinearly, the linear fit will miss that.',
		'- There may be collinearity between sub-scores (see heatmap) which affects coefficient stability.',
		'',
		'Recommendations:',
		'- For a winner predictor, test regularized linear models (Ridge/Lasso) and tree-based models (RandomForest/GradientBoosting) and validate on future-season data.',
		'- Consider collecting multiple expert ratings per contestant to measure inter-rater reliability and reduce noise.',
		'',
	'Note: plots are displayed inline; no image files are written by default.',
	])

	briefing_text = "\n".join(briefing_lines)

	briefing_file = os.path.join(base_dir, 'briefing_for_team_lead.txt')
	with open(briefing_file, 'w', encoding='utf-8') as f:
		f.write(briefing_text)

	print('\nBriefing saved to', briefing_file)
	print('\nPlots were displayed inline during the run; no image files were written.')


if __name__ == '__main__':
	main()
