# analyze_cars.py Guide

This script performs simple and multiple linear regression on the cars dataset, with built-in diagnostics for R², linearity, and independence.

## How to Run
Open a terminal in the project root (where this README is located).

### 1. Simple Linear Regression (enginesize → price)
    Command to run test set:
    py 1_linear_regression\analyze_cars.py -s --test
    ```
    Command to run training set:
    py 1_linear_regression\analyze_cars.py -s --train

### 2. Multiple Linear Regression (curbweight, horsepower, citympg, carbody)
    Command to run test set:
    py 1_linear_regression\analyze_cars.py -m --test
    ```
    Command to run training set:
    py 1_linear_regression\analyze_cars.py -m --train

### 3. Run Both Analyses (default)
    py 1_linear_regression\analyze_cars.py

## Output
- Independence diagnostics print first.
- R² and linearity outputs.
- Plots open interactively.

## Notes
- The script always prints independence checks (collinearity, VIF, MI) before running regressions.
- Multiple regression uses: curbweight, horsepower, citympg, and one-hot carbody.
