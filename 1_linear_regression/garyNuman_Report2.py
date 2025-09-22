"""
cars.csv — Price Prediction Analysis

This script demonstrates:
  1) Simple Linear Regression (engine size → price)
  2) Multiple Linear Regression (numeric + one-hot encoded categorical features → price)

Outputs:
  - R² (proportion of variance in the dependent variable)
  - Root Mean Squared Error metrics for both models 
    - (average magnitude of errors between predicted and actual values)
  - Scatter plot with regression line for the simple model
"""

# =======================
# 1. Import libraries
# =======================
# We use pandas/numpy for data handling, matplotlib/seaborn for plotting,
# and scikit-learn for modeling and evaluation.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# =======================
# 2. Load and clean data
# =======================
# - Load cars.csv
# - Normalize column names
# - Convert numeric columns to numeric types
# - Drop rows missing enginesize or price (needed for regression)
df = pd.read_csv("cars.csv")

# Standardize column names to lowercase and remove - and " " chars for consistency
df.columns = df.columns.str.lower().str.replace("-", "").str.replace(" ", "")

# Ensure core numeric columns are numeric ('coerce' bad entries to NaN)
for col in ['price', 'enginesize', 'horsepower', 'curbweight', 'highwaympg']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove rows missing price or enginesize
df = df.dropna(subset=['price', 'enginesize'])

# ===================================================
# 3. SIMPLE LINEAR REGRESSION (engine size → price)
# ===================================================
# - Train/test split (75/25)
# - Fit linear model
# - Evaluate on test set
# - Plot regression line over scatter of all data
X = df[['enginesize']].values
y = df['price'].values

# Train/test split - this goofed me for a bit with the order of X, y. Random_state is funny
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Fit linear regression model
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)

# Calculate metrics (Coeffiecent of Determination and Root Mean Squared Error)
r2_simple = r2_score(y_test, y_pred)
rmse_simple = mean_squared_error(y_test, y_pred) ** 0.5

# Print results
print("\n=== Simple Linear Regression (Engine Size → Price) ===")
print(f"Coeffiecent of Determination: {r2_simple:.3f}")
print(f"Root Mean Squared Error: ${rmse_simple:,.2f}")

# Plot results
plt.figure(figsize=(6, 4))
sns.scatterplot(x=df['enginesize'], y=df['price'], alpha=0.6, edgecolor=None)
sns.lineplot(x=df['enginesize'], y=linreg.predict(df[['enginesize']]), color='red')
plt.title('Engine Size vs Price (Simple Linear Regression)')
plt.xlabel('Engine Size')
plt.ylabel('Price')
plt.tight_layout()
plt.show()

# ============================================================
# 4. MULTIPLE LINEAR REGRESSION (numeric + categorical → price)
# ============================================================
# - Use multiple numeric predictors
# - One-hot encode categorical columns
# - Train/test split, fit, evaluate

# Define numeric and categorical features and filter to those present in df
numeric_feats = ['enginesize', 'horsepower', 'curbweight', 'highwaympg']
numeric_feats = [c for c in numeric_feats if c in df.columns]

# Define categorical features (if present) and filter to those present in df
categorical_feats = ['fueltype', 'aspiration', 'doornumber', 'drivewheels', 'bodystyle']
categorical_feats = [c for c in categorical_feats if c in df.columns]

# One-hot encode categoricals into dummy columns
df_encoded = pd.get_dummies(
    df[numeric_feats + categorical_feats + ['price']],
    columns=categorical_feats,
    drop_first=True,     # avoid dummy variable trap lolrofl
    dtype=int
)

# Prepare data for modeling
X_multi = df_encoded.drop('price', axis=1).values
y_multi = df_encoded['price'].values

# Split, fit, predict
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.25, random_state=42
)

multi_reg = LinearRegression()
multi_reg.fit(X_train_m, y_train_m)
y_pred_m = multi_reg.predict(X_test_m)

# Calculate metrics
r2_multi = r2_score(y_test_m, y_pred_m)
rmse_multi = mean_squared_error(y_test_m, y_pred_m) ** 0.5

# Print results
print("\n=== Multiple Linear Regression (Numeric + One-Hot Encoded Categorical) ===")
print(f"Coeffiecent of Determination:   {r2_multi:.3f}")
print(f"Root Mean Squared Error: ${rmse_multi:,.2f}")
