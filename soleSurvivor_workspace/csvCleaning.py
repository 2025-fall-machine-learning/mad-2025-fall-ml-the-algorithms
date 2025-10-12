import os
import pandas as pd
import numpy as np


def load_data(base_path='.'):
	past_path = os.path.join(base_path, 'sole_survivor_past.csv')
	next_path = os.path.join(base_path, 'sole_survivor_next.csv')
	df_past = pd.read_csv(past_path)
	df_next = pd.read_csv(next_path)
	return df_past, df_next


def inspect_df(name, df):
	lines = []
	lines.append(f'=== {name} ===')
	lines.append(f'Shape: {df.shape}')
	lines.append('\nDtypes:')
	lines.append(str(df.dtypes))
	lines.append('\nHead:')
	lines.append(str(df.head().to_string()))
	lines.append('\nMissing values per column:')
	lines.append(str(df.isnull().sum()))
	lines.append('\nNumeric summary:')
	lines.append(str(df.describe().T))
	return '\n'.join(lines)


def main():
	base = os.path.dirname(__file__)
	dpast, dnext = load_data(base)

	# Convert negative numeric values to absolute value and report counts
	def convert_negatives_to_abs(df):
		num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
		neg_counts = (df[num_cols] < 0).sum()
		total_neg = int(neg_counts.sum())
		if total_neg > 0:
			print(f'Found {total_neg} negative numeric entries; converting to absolute value (per-column counts below):')
			for c, v in neg_counts.items():
				if v > 0:
					print(f'  {c}: {int(v)}')
			# take absolute value for numeric columns (preserves magnitude)
			df[num_cols] = df[num_cols].abs()
		else:
			print('No negative numeric entries found.')
		return df

	dpast = convert_negatives_to_abs(dpast)
	dnext = convert_negatives_to_abs(dnext)

	# Save cleaned copies (do NOT overwrite originals)
	cleaned_dir = os.path.join(base, 'cleanedFiles')
	os.makedirs(cleaned_dir, exist_ok=True)
	past_cleaned_path = os.path.join(cleaned_dir, 'past_cleaned.csv')
	next_cleaned_path = os.path.join(cleaned_dir, 'next_cleaned.csv')
	dpast.to_csv(past_cleaned_path, index=False)
	dnext.to_csv(next_cleaned_path, index=False)
	print(f'Wrote cleaned copies (negatives -> NaN) to:')
	print(f'  {past_cleaned_path}')
	print(f'  {next_cleaned_path}')
	report_past = inspect_df('past', dpast)
	report_next = inspect_df('next', dnext)

	report = report_past + '\n\n' + report_next
	out_path = os.path.join(base, 'inspection_report.txt')
	with open(out_path, 'w', encoding='utf8') as f:
		f.write(report)

	print(report_past[:1000])
	print('\n--- next dataset ---\n')
	print(report_next[:1000])
	print(f'\nFull inspection report written to: {out_path}')


if __name__ == '__main__':
	main()

