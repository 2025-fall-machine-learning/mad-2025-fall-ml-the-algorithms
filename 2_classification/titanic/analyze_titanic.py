import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import os

try:
	from imblearn.over_sampling import RandomOverSampler
	IMBLEARN_AVAILABLE = True
except Exception:
	IMBLEARN_AVAILABLE = False


def main():
	base_dir = os.path.dirname(__file__)
	csv_path = os.path.join(base_dir, "Titanic-Dataset.csv")
	df = pd.read_csv(csv_path, skipinitialspace=True)
	run_titanic_analysis(df)


def run_titanic_analysis(df):
	print(f'Loaded rows: {len(df)}')

	df_clean = drop_invalid_rows(df)
	print(f'After dropping invalid rows: {len(df_clean)}')

	features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

	print('\nFeature correlations with Survived:')
	try:
		corr = df_clean[features + ['Survived']].corr()['Survived'].sort_values(ascending=False)
		print(corr.round(3))
	except Exception:
		print('Could not compute correlations (data may be non-numeric).')

	# Train baseline logistic regression (random_state=1 per assignment)
	print('\nTraining baseline logistic regression (random_state=1)')
	model, metrics_before = train_logistic(df_clean, features, random_state=1, max_iter=100000)
	print('Baseline accuracy:', metrics_before['accuracy'])
	print('Baseline sensitivity:', metrics_before['sensitivity'])
	print('Baseline specificity:', metrics_before['specificity'])
	print('Confusion matrix:\n', metrics_before['confusion_matrix'])

	# Baseline comparison: all-ones predictor
	y_test = metrics_before['y_test']
	all_ones = np.ones_like(y_test)
	metrics_allones = evaluate_classification(y_test, all_ones)
	print('\nAll-ones predictor accuracy:', metrics_allones['accuracy'])
	print('All-ones sensitivity:', metrics_allones['sensitivity'])
	print('All-ones specificity:', metrics_allones['specificity'])

	# Now perform balancing using RandomOverSampler and compare metrics
	if IMBLEARN_AVAILABLE:
		print('\nBalancing training data with RandomOverSampler (random_state=1) and re-training')
		model_bal, metrics_after = train_logistic(df_clean, features, random_state=1, max_iter=100000, balance=True)
		print('After balancing accuracy:', metrics_after['accuracy'])
		print('After balancing sensitivity:', metrics_after['sensitivity'])
		print('After balancing specificity:', metrics_after['specificity'])
	else:
		print('\nimblearn not installed; skipping balancing step')
		metrics_after = None

	print('\nEncoding Sex and Embarked, then re-training with extended feature set')
	df_ext = df_clean.copy()
	# Sex: male=1, female=0
	df_ext['Sex'] = df_ext['Sex'].map({'male': 1, 'female': 0})
	emp_dummies = pd.get_dummies(df_ext['Embarked'], prefix='Embarked', drop_first=True)
	df_ext = pd.concat([df_ext, emp_dummies], axis=1)

	features_ext = features + ['Sex'] + list(emp_dummies.columns)

	model_ext, metrics_ext_before = train_logistic(df_ext, features_ext, random_state=1, max_iter=100000, balance=False)
	print('Extended (before balancing) accuracy:', metrics_ext_before['accuracy'])
	print('Extended (before balancing) sensitivity:', metrics_ext_before['sensitivity'])
	print('Extended (before balancing) specificity:', metrics_ext_before['specificity'])

	if IMBLEARN_AVAILABLE:
		model_ext_bal, metrics_ext_after = train_logistic(df_ext, features_ext, random_state=1, max_iter=100000, balance=True)
		print('Extended (after balancing) accuracy:', metrics_ext_after['accuracy'])
		print('Extended (after balancing) sensitivity:', metrics_ext_after['sensitivity'])
		print('Extended (after balancing) specificity:', metrics_ext_after['specificity'])
	else:
		metrics_ext_after = None

	# Save sensitivity/specificity before and after balancing, for both original and extended features
	out_file = 'titanic_sensitivity_specificity.txt'
	write_sensitivity_specificity(out_file, metrics_before, metrics_after, extended_before=metrics_ext_before, extended_after=metrics_ext_after)
	print('\nWrote sensitivity/specificity to', out_file)


def drop_invalid_rows(df: pd.DataFrame):
	return df.dropna()


def evaluate_classification(y_true, y_pred):
	acc = metrics.accuracy_score(y_true, y_pred)
	cm = metrics.confusion_matrix(y_true, y_pred)
	# cm = [[tn, fp],[fn, tp]]
	tn, fp, fn, tp = cm.ravel()
	sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
	specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
	return {
		'accuracy': float(acc), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
		'sensitivity': float(sensitivity), 'specificity': float(specificity), 'confusion_matrix': cm
	}


def train_logistic(df: pd.DataFrame, feature_cols, random_state=1, max_iter=100000, balance=False):
	X = df[feature_cols].copy()
	y = df['Survived'].astype(int).copy()

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

	if balance:
		if not IMBLEARN_AVAILABLE:
			raise RuntimeError('imblearn not installed; install imbalanced-learn to enable balancing')
		ros = RandomOverSampler(random_state=1)
		X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
	else:
		X_train_res, y_train_res = X_train, y_train

	model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter, solver='lbfgs'))
	model.fit(X_train_res, y_train_res)

	# cross-validated accuracy on the training set (5-fold)
	try:
		cv_acc = float(np.mean(cross_val_score(model, X_train_res, y_train_res, cv=5, scoring='accuracy')))
	except Exception:
		cv_acc = None

	y_pred = model.predict(X_test)
	metrics_dict = evaluate_classification(y_test.values, y_pred)
	metrics_dict['y_test'] = y_test.values
	metrics_dict['y_pred'] = y_pred
	metrics_dict['cv_accuracy'] = cv_acc
	return model, metrics_dict


def write_sensitivity_specificity(path, original_before: dict, original_after: dict = None,
								  extended_before: dict = None, extended_after: dict = None):
	lines = []
	lines.append('Original features (Pclass, Age, SibSp, Parch, Fare)')
	lines.append(f" - before balancing: sensitivity={original_before['sensitivity']:.3f}, specificity={original_before['specificity']:.3f}")
	if original_after is not None:
		lines.append(f" - after balancing:  sensitivity={original_after['sensitivity']:.3f}, specificity={original_after['specificity']:.3f}")
	else:
		lines.append(' - after balancing:  (not performed)')
	lines.append('')
	if extended_before is not None:
		lines.append('Extended features (included Sex and Embarked)')
		lines.append(f" - before balancing: sensitivity={extended_before['sensitivity']:.3f}, specificity={extended_before['specificity']:.3f}")
		if extended_after is not None:
			lines.append(f" - after balancing:  sensitivity={extended_after['sensitivity']:.3f}, specificity={extended_after['specificity']:.3f}")
		else:
			lines.append(' - after balancing:  (not performed)')

	with open(path, 'w', encoding='utf-8') as f:
		f.write('\n'.join(lines))


if __name__ == '__main__':
	main()
