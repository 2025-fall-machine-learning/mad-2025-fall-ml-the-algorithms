import os

if __package__ is None:
    # when run directly (python scripts/manual_linear_regression.py) ensure the
    # project root is on sys.path so package imports work
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Prefer normal package imports; but be tolerant in case the script is launched
# from an environment where the package path is not available (some IDE/debuggers).
try:
    from scripts.RegressionPlot import RegressionPlot
    from scripts.LinearRegression import LinearRegression
    from scripts.Split import Split
except Exception:
    # Fallback: import by file path so users can run the script in any way.
    import importlib.util
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    rp_path = os.path.join(ROOT, 'scripts', 'RegressionPlot.py')
    lr_path = os.path.join(ROOT, 'scripts', 'LinearRegression.py')

    def _load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    _rp_mod = _load_module('scripts.RegressionPlot', rp_path)
    _lr_mod = _load_module('scripts.LinearRegression', lr_path)
    _split_mod = _load_module('scripts.Split', os.path.join(ROOT, 'scripts', 'Split.py'))
    RegressionPlot = _rp_mod.RegressionPlot
    LinearRegression = _lr_mod.LinearRegression
    Split = _split_mod.Split

import threading
import sys
import time

# Hard-coded headless flag for testing or CI. Change to True to disable plotting.
HEADLESS = False

_ASSISTANT_HEADLESS_FLAG = '--_assistant-headless'
assistant_headless = False
if _ASSISTANT_HEADLESS_FLAG in sys.argv:
    assistant_headless = True
    try:
        sys.argv.remove(_ASSISTANT_HEADLESS_FLAG)
    except ValueError:
        pass

HEADLESS_MODE = HEADLESS or assistant_headless

# Change the TRIAL_DELAY value to slow or speed up the animated iterations. Try TRIAL_DELAY = 0.03.
TRIAL_DELAY = 0.000003 if not HEADLESS_MODE else 0.0

def run_analysis(data_path, x_col, y_col, seed=42, train_frac=0.8,
                 title=None, x_label=None, y_label=None, headless=None):
    
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    datasets = Split(data_path, x_col=x_col, y_col=y_col, seed=seed, train_frac=train_frac)
    x_all, y_all, x_train, y_train, x_test, y_test = datasets.as_arrays()

    lr = LinearRegression()

    effective_headless = HEADLESS_MODE if headless is None else bool(headless)

    if effective_headless:
        print('Running in HEADLESS mode: no plotting, running fit to completion...')
        for m, b, r, epoch_idx in lr.fit(x_train, y_train):
            pass
        best = lr.best
        if best and best[0] is not None:
            m_best, b_best, train_rss = best
            result = {
                'm': m_best,
                'b': b_best,
                'train_rss': train_rss,
                'test_rss': lr.rss(x_test, y_test, m_best, b_best),
                'train_rmse': lr.rmse(x_train, y_train, m_best, b_best),
                'test_rmse': lr.rmse(x_test, y_test, m_best, b_best),
                'train_r2': lr.r2(x_train, y_train, m_best, b_best),
                'test_r2': lr.r2(x_test, y_test, m_best, b_best),
            }
            return result
        return None

    rp = RegressionPlot(x_all, y_all, x_train, y_train, x_test, y_test, lr.epochs,
                        cur_rss=None, trial_iter=None, external_producer=True,
                        title=title, x_label=x_label, y_label=y_label)

    def _producer():
        print('producer thread starting')

        def _on_trial(m, b, r, epoch_idx):
            rp.add_trial(m, b, r, epoch_idx)
            if TRIAL_DELAY > 0.0:
                time.sleep(TRIAL_DELAY)

        def _on_done(best):
                try:
                    m_best, b_best, train_rss = best
                    rp.test_rss = lr.rss(x_test, y_test, m_best, b_best)
                    rp.train_rmse = lr.rmse(x_train, y_train, m_best, b_best)
                    rp.test_rmse = lr.rmse(x_test, y_test, m_best, b_best)
                    rp.train_r2 = lr.r2(x_train, y_train, m_best, b_best)
                    rp.test_r2 = lr.r2(x_test, y_test, m_best, b_best)
                    # store final metrics onto lr for the caller
                    lr._final_metrics = {
                        'm': m_best,
                        'b': b_best,
                        'train_rss': train_rss,
                        'test_rss': rp.test_rss,
                        'train_rmse': rp.train_rmse,
                        'test_rmse': rp.test_rmse,
                        'train_r2': rp.train_r2,
                        'test_r2': rp.test_r2,
                    }
                except Exception:
                    rp.test_rss = None
                    rp.train_rmse = None
                    rp.test_rmse = None
                    rp.train_r2 = None
                    rp.test_r2 = None
                    lr._final_metrics = None
                print('producer done, best=', best)
                rp.producer_done()

        try:
            print('producer calling lr.fit')
            lr.fit(x_train, y_train, on_trial=_on_trial, on_done=_on_done)
            print('producer returned from lr.fit')
        except Exception as e:
            print('Producer thread raised exception:', e)

    prod_thread = threading.Thread(target=_producer)
    prod_thread.start()
    print('producer thread started')

    rp.run()
    prod_thread.join()
    # return rich metrics if available
    if hasattr(lr, '_final_metrics') and lr._final_metrics:
        return lr._final_metrics
    best = lr.best
    if best and best[0] is not None:
        m_best, b_best, train_rss = best
        return {
            'm': m_best,
            'b': b_best,
            'train_rss': train_rss,
            'test_rss': lr.rss(x_test, y_test, m_best, b_best),
            'train_rmse': lr.rmse(x_train, y_train, m_best, b_best),
            'test_rmse': lr.rmse(x_test, y_test, m_best, b_best),
            'train_r2': lr.r2(x_train, y_train, m_best, b_best),
            'test_r2': lr.r2(x_test, y_test, m_best, b_best),
        }
    return None


if __name__ == '__main__':
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(root, 'data', 'cars.csv')
    best = run_analysis(data_path)
    if best and best[0] is not None:
        m, b, train_rss = best
        # reload arrays to compute test metrics for reporting
        datasets = Split(data_path, x_col='enginesize', y_col='price', seed=42, train_frac=0.8)
        x_all, y_all, x_train, y_train, x_test, y_test = datasets.as_arrays()
        lr = LinearRegression()
        test_rss = lr.rss(x_test, y_test, m, b)
        train_rmse = lr.rmse(x_train, y_train, m, b)
        test_rmse = lr.rmse(x_test, y_test, m, b)
        train_r2 = lr.r2(x_train, y_train, m, b)
        test_r2 = lr.r2(x_test, y_test, m, b)
        print('\nComparison:')
        print(f'  Final rss={train_rss:.2f} (train), test_rss={test_rss:.2f}')
        print(f'  train_rmse={train_rmse:.2f}, test_rmse={test_rmse:.2f}')
        print(f'  train_r2={train_r2:.4f}, test_r2={test_r2:.4f}')
    else:
        print('\nNo best model found during streaming run.')
