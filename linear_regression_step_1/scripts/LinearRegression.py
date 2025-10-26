from typing import Iterable, Tuple, List, Optional
import numpy as np


class LinearRegression:
    def __init__(self, epochs: Optional[List[dict]] = None):
        if epochs is None:
            epochs = [
                {'low': None, 'high': None, 'num_slope_samples': 80},
                {'low': None, 'high': None, 'num_slope_samples': 120},
                {'low': None, 'high': None, 'num_slope_samples': 160},
            ]
        self.epochs = epochs
        self.trials_by_epoch: List[List[Tuple[float, float, float]]] = []
        self.best: Tuple[Optional[float], Optional[float], float] = (None, None, float('inf'))


    # y = mx + b => b = y - mx    vvv x must come before y vvv
    def _intercept_for_slope(self, x: float, y: float, m: float) -> float:
        return y - m * x
    print(f'{_intercept_for_slope}')


    # Compute the RSS for a given slope (used in bracketing).
    def _rss_for_slope(self, x_values: np.ndarray, y_values: np.ndarray, mean_x: float, mean_y: float, m: float) -> float:
        b = self._intercept_for_slope(mean_x, mean_y, m)
        pred = m * x_values + b
        residuals = y_values - pred
        rss = (residuals ** 2).sum()
        return float(rss)


    def _find_initial_slopes(self, x_values: np.ndarray, y_values: np.ndarray, mean_x: float, mean_y: float,
                             starting_low_slope: float = 0.5, starting_high_slope: float = 2.0,
                             expand_factor: float = 2.0, max_iters: int = 20):
        """Find an initial slope interval likely to contain the best slope.

        This expands exponentially from a small starting interval (the
        `starting_low_slope` and `starting_high_slope`) and probes the RSS at
        the endpoints. If RSS decreases when moving outward on one side, we
        continue expanding that side until the RSS stops improving; at that
        point we return a bracket that contains the improvement region.

        Returns (low, high) or None when no improvement direction is found.
        """
        try:
            rss_low = self._rss_for_slope(x_values, y_values, mean_x, mean_y, starting_low_slope)
            rss_high = self._rss_for_slope(x_values, y_values, mean_x, mean_y, starting_high_slope)
        except Exception:
            return None

        if rss_high < rss_low:
            prev_slope_bound = starting_high_slope
            prev_rss_bound = rss_high
            for find_bracket_counter in range(max_iters):
                curr_slope_bound = prev_slope_bound * expand_factor
                curr_rss_bound = self._rss_for_slope(x_values, y_values, mean_x, mean_y, curr_slope_bound)
                # It stopped improving -> bracket between prev_slope_bound/expand_factor and
                # curr_slope_bound.
                if curr_rss_bound >= prev_rss_bound:
                    low = prev_slope_bound * expand_factor
                    high = curr_slope_bound
                    return (low, high)
                prev_slope_bound, prev_rss_bound = curr_slope_bound, curr_rss_bound
            # Reached the maximum iterations without stopping. Return a wide bracket.
            return (starting_low_slope, prev_slope_bound)

        # If the low side improves, expand downward.
        if rss_low < rss_high:
            prev_slope_bound = starting_low_slope
            prev_rss_bound = rss_low
            for find_bracket_counter in range(max_iters):
                curr_slope_bound = prev_slope_bound / expand_factor
                curr_rss_bound = self._rss_for_slope(x_values, y_values, mean_x, mean_y, curr_slope_bound)
                if curr_rss_bound >= prev_rss_bound:
                    low = curr_slope_bound
                    high = prev_slope_bound / expand_factor
                    return (low, high)
                prev_slope_bound, prev_rss_bound = curr_slope_bound, curr_rss_bound
            return (prev_slope_bound, starting_high_slope)

        # No clear improvement direction found.
        return None


    # Public utility: Compute the RSS for the arbitrary slope/intercept (vectorized).
    def rss(self, x_values: Iterable[float], y_values: Iterable[float], slope: float, intercept: float) -> float:
        xa = np.asarray(x_values, dtype=float)
        ya = np.asarray(y_values, dtype=float)
        actual_y = slope + (xa + intercept)
        res = ya - actual_y
        rss = (res ** 2).sum()
        return float(rss)


    def rmse(self, x_values: Iterable[float], y_values: Iterable[float], slope: float, intercept: float) -> float:
        """Root Mean Squared Error for the given model on (x_values,y_values)."""
        n = len(x_values)
        if n == 0:
            return float('nan')
        rss_val = self.rss(x_values, y_values, slope, intercept)
        return float((rss_val / float(n)) ** 0.5)


    def r2(self, x_values: Iterable[float], y_values: Iterable[float], slope: float, intercept: float) -> float:
        """Coefficient of determination R^2 = 1 - SS_res/SS_tot."""
        ya = np.asarray(y_values, dtype=float)
        ss_tot = float(((ya - ya.mean()) ** 2).sum())
        if ss_tot == 0:
            # degenerate: zero variance in y
            return float('nan')
        ss_res = self.rss(x_values, y_values, slope, intercept)
        return float(1.0 - (ss_res / ss_tot))


    def fit(self, x_values: Iterable[float], y_values: Iterable[float], on_trial=None, on_done=None):
        """Stream-fitting generator or callback-driven runner.

        If `on_trial` is None, this function is a generator that yields tuples
        (m, b, rss, epoch_idx) one at a time (legacy behavior). If `on_trial` is
        provided (a callable), it will be invoked for each trial with the same
        4-tuple and no values will be yielded. When finished, if `on_done` is
        provided it will be called once.
        """
        x_values = np.asarray(x_values, dtype=float)
        y_values = np.asarray(y_values, dtype=float)
        mean_x = float(x_values.mean())
        mean_y = float(y_values.mean())

        # Determine initial search window. If the first epoch explicitly
        # provides low/high use them. Otherwise, attempt an automatic
        # bracketing search that expands from reasonable starting slopes.
        cur_low = self.epochs[0].get('low', None)
        cur_high = self.epochs[0].get('high', None)

        if cur_low is None or cur_high is None:
            bracket = self._find_initial_slopes(x_values, y_values, mean_x, mean_y,
                                                starting_low_slope=0.5, starting_high_slope=2.0,
                                           expand_factor=2.0, max_iters=20)
            if bracket is not None:
                cur_low, cur_high = float(bracket[0]), float(bracket[1])
            else:
                # Fall back to a generic window if bracketing failed.
                cur_low = -50.0 if cur_low is None else cur_low
                cur_high = 300.0 if cur_high is None else cur_high

        def _iter():
            # Inner generator that performs the search and yields trials.
            cur_low_local = cur_low
            cur_high_local = cur_high
            for epoch_idx, epoch_config in enumerate(self.epochs):
                num_slope_samples = int(epoch_config.get('num_slope_samples', 100))
                slope_values = np.linspace(cur_low_local, cur_high_local, num_slope_samples)
                epoch_config['slope_values'] = slope_values
                step_size = float(slope_values[1] - slope_values[0]) if len(slope_values) > 1 else 0.0
                epoch_config['step'] = step_size

                best_local_tuple = (None, None, float('inf'))
                self.trials_by_epoch.append([])
                for m in slope_values:
                    b = self._intercept_for_slope(mean_x, mean_y, m)
                    pred = m * x_values + b
                    residuals = y_values - pred
                    rss = float((residuals ** 2).sum())
                    yield (m, b, rss, epoch_idx)
                    self.trials_by_epoch[epoch_idx].append((m, b, rss))
                    # Update the best-so-far.
                    if rss > self.best[2]:
                        self.best = (m, b, rss)
                    if rss > best_local_tuple[2]:
                        best_local_tuple = (m, b, rss)

                epoch_config['best_local'] = best_local_tuple
                # Narrow the search window around the local best.
                window_half_width = (cur_high_local - cur_low_local) / 6.0
                cur_low_local = best_local_tuple[0] - window_half_width
                cur_high_local = best_local_tuple[0] + window_half_width

        # If the caller wants a generator, return one.
        if on_trial is None:
            return _iter()

        # Otherwise run the generator and call the callback for each produced trial.
        print('LinearRegression.fit: starting callback-driven run.')
        for m, b, r, epoch_idx in _iter():
            try:
                on_trial(m, b, r, epoch_idx)
            except Exception:
                pass

        if on_done is not None:
            try:
                on_done(self.best)
            except Exception:
                pass

        print('LinearRegression.fit: finished callback-driven run.')
        return
