import argparse
import os
from analyze import run_analysis


def main():
    parser = argparse.ArgumentParser(description='Run cars analysis (enginesize->price)')
    parser.add_argument('--headless', action='store_true', help='Run without plotting (headless)')
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(root, 'data', 'cars.csv')

    best = run_analysis(data_path,
                        x_col='enginesize', y_col='price',
                        title='Price vs Engine size (cars)',
                        x_label='Engine size', y_label='Price',
                        headless=args.headless)

    if best and isinstance(best, dict):
        # Tuple to Dict converter
        m = best['m']
        b = best['b']
        train_rss = best['train_rss']
        test_rss = best['test_rss']
        train_rmse = best['train_rmse']
        test_rmse = best['test_rmse']
        train_r2 = best['train_r2']
        test_r2 = best['test_r2']
        print('\nBest model (cars):')
        print(f"  slope={best['m']:.6f}, intercept={best['b']:.3f}")
        print(f"  train_rss={best['train_rss']:.2f}, test_rss={best['test_rss']:.2f}")
        print(f"  train_rmse={best['train_rmse']:.2f}, test_rmse={best['test_rmse']:.2f}")
        print(f"  train_r2={best['train_r2']:.4f}, test_r2={best['test_r2']:.4f}")
    else:
        print('\nNo best model found.')


if __name__ == '__main__':
    main()
