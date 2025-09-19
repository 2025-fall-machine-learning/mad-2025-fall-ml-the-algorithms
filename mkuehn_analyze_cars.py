import os
import numpy as np
import pandas as pd
import csv
from collections import Counter
import matplotlib.pyplot as plt

# Path to CSV
path = r'E:/Madison College/Machine Learning/mad-2025-fall-ml-the-algorithms/1_linear_regression/cars.csv'


def detect_column(df, keywords):
	# prefer exact or normalized matches first (avoid matching 'enginelocation')
	keys = [k.lower() for k in keywords]
	cols = list(df.columns)

	# 1) exact match (case-insensitive)
	for col in cols:
		if col.lower() in keys:
			return col

	# 2) normalized exact match (remove spaces and dashes)
	norm_keys = [k.replace(' ', '').replace('-', '') for k in keys]
	for col in cols:
		low_norm = col.lower().replace(' ', '').replace('-', '')
		if low_norm in norm_keys:
			return col

	# 3) prefer columns that mention engine AND size/displacement/cc/cyl
	for col in cols:
		low = col.lower()
		if 'engine' in low and ('size' in low or 'displacement' in low or 'cc' in low or 'cyl' in low):
			return col

	# 4) fallback: any column containing a keyword as substring
	for col in cols:
		low = col.lower()
		for k in keys:
			if k in low:
				return col

	return None


def summarize_unique_engine_sizes(path):
	# keep original Counter behavior for engine-size-like values (strings)
	with open(path, newline='', encoding='utf-8') as f:
		reader = csv.DictReader(f)
		vals = []
		if 'enginesize' in reader.fieldnames:
			for row in reader:
				v = row.get('enginesize', '').strip()
				if v != '':
					vals.append(v)
		else:
			# fallback: try to collect any column that looks like engine size
			for row in reader:
				for k in ('enginesize', 'engine size', 'engine', 'displacement'):
					if k in row:
						v = row.get(k, '').strip()
						if v != '':
							vals.append(v)
						break
	cnt = Counter(vals)
	print('Unique engine sizes (count):')
	for size, c in cnt.most_common():
		print(size, c)


def run_regression_and_plot(path):
	df = pd.read_csv(path)

	# detect engine-size and price columns
	eng_col = None
	price_col = None

	# common engine-size candidates
	eng_candidates = ['enginesize', 'engine size', 'engine', 'displacement', 'cc', 'cyl']
	price_candidates = ['price', 'msrp', 'cost', 'price($)', 'price_usd']

	eng_col = detect_column(df, eng_candidates)
	price_col = detect_column(df, price_candidates)

	if eng_col is None:
		# try heuristic: any column containing 'engine' substring
		for col in df.columns:
			if 'engine' in col.lower() or 'displacement' in col.lower() or 'cc' in col.lower():
				eng_col = col
				break

	if price_col is None:
		for col in df.columns:
			if 'price' in col.lower() or 'msrp' in col.lower() or 'cost' in col.lower():
				price_col = col
				break

	if eng_col is None or price_col is None:
		print('Could not automatically detect engine-size and/or price columns.')
		print('Available columns:')
		print(list(df.columns))
		return

	print(f'Using engine column: "{eng_col}" and price column: "{price_col}"')

	# normalize and convert to numeric
	df[eng_col] = df[eng_col].astype(str).str.replace(',', '').str.strip()
	df[price_col] = df[price_col].astype(str).str.replace(',', '').str.strip()

	df[eng_col] = pd.to_numeric(df[eng_col], errors='coerce')
	df[price_col] = pd.to_numeric(df[price_col], errors='coerce')

	sub = df[[eng_col, price_col]].dropna()
	if sub.empty:
		print('No numeric data available after conversion/cleaning.')
		return

	x = sub[eng_col].values.astype(float)
	y = sub[price_col].values.astype(float)

	# compute linear regression with numpy
	slope, intercept = np.polyfit(x, y, 1)
	y_pred = slope * x + intercept
	ss_res = np.sum((y - y_pred) ** 2)
	ss_tot = np.sum((y - np.mean(y)) ** 2)
	r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float('nan')

	print('\nLinear regression result:')
	print(f'  n = {len(x)}')
	print(f'  slope = {slope:.6f}')
	print(f'  intercept = {intercept:.6f}')
	print(f'  R^2 = {r2:.6f}')

	# plot
	try:
		plt.figure(figsize=(8, 6))
		plt.scatter(x, y, alpha=0.6, label='data')
		xs = np.linspace(np.min(x), np.max(x), 200)
		ys = slope * xs + intercept
		plt.plot(xs, ys, color='red', linewidth=2, label=f'fit: y={slope:.3f}x+{intercept:.1f}')
		plt.xlabel(eng_col)
		plt.ylabel(price_col)
		plt.title(f'Linear regression ({eng_col} vs {price_col})\nR^2={r2:.3f}')
		plt.legend()

		out_dir = os.path.dirname(path) or '.'
		out_path = os.path.join(out_dir, 'enginesize_price_regression.png')
		plt.tight_layout()
		plt.savefig(out_path)
		print(f'Plot saved to: {out_path}')
		# also show the plot (will open a window when running locally)
		plt.show()
	except Exception as e:
		print('Plotting failed:', e)


if __name__ == '__main__':
	summarize_unique_engine_sizes(path)
	run_regression_and_plot(path)
