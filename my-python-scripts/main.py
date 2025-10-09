#!/usr/bin/env python3
"""Simple sample script for the workspace.

Usage:
  python my-python-scripts/main.py [args...]

This prints a greeting and echoes any provided arguments.
"""
import sys
from typing import Optional


def load_wine() -> Optional[object]:
    """Try to load a wine dataset from common packages and return a pandas DataFrame or similar.

    Attempts (in order): scikit-learn, seaborn, vega_datasets, sklearn.datasets, and sklearn's load_wine.
    Returns None if none available.
    """
    try:
        # scikit-learn utility (returns Bunch)
        from sklearn.datasets import load_wine as _load_wine
        import pandas as pd

        data = _load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        if hasattr(data, "target"):
            df["target"] = data.target
        print("Loaded wine dataset via scikit-learn")
        return df
    except Exception:
        pass

    try:
        # seaborn may have an example dataset called 'wine' in some environments
        import seaborn as sns
        df = sns.load_dataset("wine")
        print("Loaded wine dataset via seaborn")
        return df
    except Exception:
        pass

    try:
        # vega_datasets
        from vega_datasets import data as vdata
        df = vdata.wine()
        print("Loaded wine dataset via vega_datasets")
        return df
    except Exception:
        pass

    print("No supported wine dataset found in installed packages.")
    return None


def main() -> None:
    print("my-python-scripts: hello! This is a sample script.")
    if "--show-wine" in sys.argv:
        df = load_wine()
        if df is None:
            print("Could not load wine dataset. Install scikit-learn, seaborn, or vega_datasets.")
        else:
            # Print a short preview
            try:
                print(df.head().to_string(index=False))
            except Exception:
                print(df)
        return

    if len(sys.argv) > 1:
        print("Arguments:", " ".join(sys.argv[1:]))
    else:
        print("No arguments provided.")


if __name__ == "__main__":
    main()
