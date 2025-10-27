import os
import random
import pandas as pd
import numpy as np


class Split:
    """Helpers for loading the CSV and producing a deterministic
    train/test split.

    This keeps the demo wiring in a single place so the runner can remain
    focused on the algorithm and visualization.
    """

    def __init__(self, data_path, x_col, y_col, seed=42, train_frac=0.8):
        self.data_path = os.path.abspath(data_path)
        self.x_col = x_col
        self.y_col = y_col
        self.seed = seed
        self.train_frac = float(train_frac)

        self.df = None
        self.x_all = None
        self.y_all = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self._load_and_split()


    def _load_and_split(self):
        # Read CSV and keep only the two required columns.
        self.df = pd.read_csv(self.data_path)
        self.df = self.df[[self.x_col, self.y_col]].dropna()

        # Convert to numpy arrays for the demo's simple arithmetic.
        self.x_all = np.array(self.df[self.x_col], dtype=float)
        self.y_all = np.array(self.df[self.y_col], dtype=float)

        # Deterministic shuffle/split using Python's random with explicit seed.
        rng = random.Random(self.seed)
        indices = list(range(len(self.x_all)))
        rng.shuffle(indices)
        train_n = int(len(indices) * self.train_frac)
        test_n = len(indices) - train_n
        train_idx = indices[:train_n]
        test_idx = indices[:test_n]

        self.x_train = self.x_all[train_idx]
        self.y_train = self.y_all[train_idx]
        self.x_test = self.x_all[test_idx]
        self.y_test = self.y_all[test_idx]


    def as_arrays(self):
        """Return (x_all, y_all, x_train, y_train, x_test, y_test)."""
        return (self.x_all, self.y_all, self.x_train, self.y_train, self.x_test, self.y_test)
