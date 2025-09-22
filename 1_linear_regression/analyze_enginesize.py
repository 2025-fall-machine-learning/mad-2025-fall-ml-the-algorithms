import csv
import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_engine_price(csv_path):
    engines = []
    prices = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            e = row.get('enginesize', '').strip()
            p = row.get('price', '').strip()
            if e == '' or p == '':
                continue
            try:
                ev = float(e)
                pv = float(p)
            except ValueError:
                continue
            if math.isfinite(ev) and math.isfinite(pv):
                engines.append(ev)
                prices.append(pv)
    return np.array(engines), np.array(prices)


def fit_and_plot(x, y, show=True):
    # Fit linear regression using numpy.polyfit
    coeffs = np.polyfit(x, y, 1)
    m, b = coeffs[0], coeffs[1]
    y_pred = m * x + b
    # R^2
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float('nan')

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.6, label='data', edgecolor='k', s=40)
    xs = np.linspace(np.min(x), np.max(x), 200)
    plt.plot(xs, m * xs + b, color='red', lw=2, label=f'fit: y={m:.2f}x+{b:.0f}')
    plt.xlabel('Engine Size (cc)')
    plt.ylabel('Price (USD)')
    plt.title('Engine Size vs Price with Linear Fit')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.annotate(f'slope={m:.2f}\nintercept={b:.0f}\n$R^2$={r2:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=10, ha='left', va='top', bbox=dict(boxstyle='round', fc='w'))
    plt.tight_layout()
    if show:
        # Try to display the plot; do not save to disk per user request.
        try:
            plt.show()
        finally:
            plt.close()
        return m, b, r2
    else:
        # If not showing, just close the figure and return fit values.
        plt.close()
        return m, b, r2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze engine size vs price')
    # The script will display the plot only and will not save it to disk.
    parser.add_argument('--no-show', action='store_true', help="Do not display the plot (keeps script silent)")
    args = parser.parse_args()

    csv_path = os.path.join(os.path.dirname(__file__), 'cars.csv')
    x, y = load_engine_price(csv_path)
    if x.size == 0:
        print('No engine size / price data found in', csv_path)
    else:
        m, b, r2 = fit_and_plot(x, y, show=not args.no_show)
        print(f'Points: {x.size}')
        print(f'slope: {m:.6f}')
        print(f'intercept: {b:.6f}')
        print(f'R^2: {r2:.6f}')
        if not args.no_show:
            print('Displayed plot (no image was saved).')
        else:
            print('Plot display skipped by --no-show. No file was saved.')
