import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def print_1d_data_summary(data_1d, label=None):
    arr = np.array(data_1d)
    # Only flatten if it's a 2D array with one column
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.flatten()
    first_five = ", ".join(f"{x:7.3f}" for x in arr[:5])
    last_five = ", ".join(f"{x:7.3f}" for x in arr[-5:])
    print(f"{label + ': ' if label else ''}[{first_five}, ..., {last_five}]")


def plot_regression(x, y_true, y_pred, x_label, y_label, title, color):
    
    #Creates a scatter plot of true values and an optional regression line.
    
    plt.scatter(x, y_true, color=color, label=f"{x_label} Data")
    plt.plot(x, y_pred, color='red', label='Best Fit Line')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()


def process_dataset(X, y, model, set_name, labels, color):
    
    #Handles prediction, printing summaries, and plotting for one dataset split.
    # Predict using the model
    y_pred = model.predict(X)
    # Print summaries for predictors, predictions, and response
    x_data = X[:, 0] if X.ndim > 1 else X
    print_1d_data_summary(x_data, f"{labels['predictors']} predictors ({set_name})")
    print_1d_data_summary(y_pred, f"Prediction ({set_name})")
    print_1d_data_summary(y, f"{labels['response']} response ({set_name})")
    # Plot regression
    plot_regression(
        x_data,
        y,
        y_pred,
        labels['predictors'],
        labels['response'],
        f"Regression Methods: {labels['predictors']} vs {labels['response']} ({set_name})",
        color
    )


def regression_workflow(
    df,
    predictors_cols,
    response_col,
    create_testing_set=False,
    one_hot_encode=False,
    categorical_col=None,
    labels=None,
    model_type='linear'  # Could be: 'linear' or 'multiple' 

):
    
    #Main workflow to:
    # select predictors/response
    # (optionally) one-hot encode categorical variables
    # split into train/test sets (if requested)
    # fit a linear regression model
    # process each dataset (train/test)
     
    # 1. Prepare predictors (X)
    X = df[predictors_cols].values
    # Add dummy variables if specified
    if one_hot_encode and categorical_col:
        dummies = pd.get_dummies(df[categorical_col], prefix=categorical_col)
        X = np.hstack([X, dummies.values])

    # 2. Prepare response (y)
    y = df[response_col].values

    # 3. Split or use all data
    if create_testing_set:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42)
        datasets = [
            ('Training', X_train, y_train, 'purple'),
            ('Testing', X_test, y_test, 'orange')
        ]
    else:
        datasets = [('All Data', X, y, 'violet')]

    # 4. Fit model on training data (or all data)
    if model_type == 'multiple' and X.shape[1] > 1:
        print("Using Multiple Linear Regression (multi-feature)")
        model = LinearRegression().fit(datasets[0][1], datasets[0][2])
    else:
        print("Using Simple Linear Regression (single feature)")
        model = LinearRegression().fit(datasets[0][1], datasets[0][2])

    # 5. Process each dataset (print + plot)
    for set_name, X_set, y_set, color in datasets:
        process_dataset(X_set, y_set, model, set_name, labels, color)


def main():
    
    # Main entry point:
    # Reads CSV
    # Runs regression_workflow with desired settings
    
    df = pd.read_csv('CherryTree.csv')

    # Example linear regression: single predictor, no test split
    regression_workflow(
       df,
       predictors_cols=['Diam'],
       response_col='Height',
       create_testing_set=False,
       labels={'predictors': 'Diam', 'response': 'Height'},
       model_type='linear'
    )

    # To try other runs, just call regression_workflow again with new parameters:
    # regression_workflow(df, ['Diam'], 'Height', create_testing_set=True, labels={'predictors':'Diam','response':'Height'})
    # regression_workflow(df, ['Diam','Height'], 'Volume', create_testing_set=False, labels={'predictors':'Diam','response':'Volume'})

    # Example: Multiple linear regression (Diam and Height as predictors for Volume)
    # regression_workflow(
    #     df,
    #     predictors_cols=['Diam', 'Height'],
    #     response_col='Volume',
    #     create_testing_set=True,
    #     labels={'predictors': 'Diam & Height', 'response': 'Volume'},
    #     model_type='multiple'
    # )


if __name__ == "__main__":
    main()