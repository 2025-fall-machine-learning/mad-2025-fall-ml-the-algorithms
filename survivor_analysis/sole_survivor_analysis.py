import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_data(base_path):
    past_path = os.path.join(base_path, "sole_survivor_past.csv")
    next_path = os.path.join(base_path, "sole_survivor_next.csv")
    past = pd.read_csv(past_path)
    nxt = pd.read_csv(next_path)
    # Clean whitespace in column names and names
    past.columns = [c.strip() for c in past.columns]
    nxt.columns = [c.strip() for c in nxt.columns]
    past['Name'] = past['Name'].str.strip()
    nxt['Name'] = nxt['Name'].str.strip()
    return past, nxt


def prepare_features(past, nxt):
    # Features are all columns except Name and SurvivalScore
    feature_cols = [c for c in past.columns if c not in ('Name', 'SurvivalScore')]
    X = past[feature_cols].copy()
    y = past['SurvivalScore'].copy()
    X_next = nxt[feature_cols].copy()

    # Select features by absolute correlation with SurvivalScore
    # Keep both strong positive and strong negative correlations
    corr_with_target = past[feature_cols + ['SurvivalScore']].corr()['SurvivalScore'].drop('SurvivalScore')
    # threshold for 'strong' correlation (absolute value)
    thresh = 0.20
    selected = corr_with_target[ corr_with_target.abs() >= thresh ].sort_values(key=lambda s: s.abs(), ascending=False)
    if len(selected) == 0:
        # fallback: keep all features
        selected_features = feature_cols
    else:
        selected_features = list(selected.index)

    return X[selected_features].copy(), y, X_next[selected_features].copy(), selected_features, corr_with_target


def train_and_evaluate(X, y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    return model, X_test, y_test, y_pred, metrics


def save_plots(base_path, past, X_test, y_test, y_pred, feature_cols, model, X_next, predictions_next):
    plots_dir = os.path.join(base_path, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Predicted vs Actual (test)
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--')
    plt.xlabel('Actual SurvivalScore')
    plt.ylabel('Predicted SurvivalScore')
    plt.title('Predicted vs Actual (test set)')
    plt.tight_layout()
    pvap = os.path.join(plots_dir, 'predicted_vs_actual.png')
    plt.savefig(pvap)
    plt.close()

    # Residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, kde=True)
    plt.title('Residuals distribution (test set)')
    plt.xlabel('Actual - Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'residuals_hist.png'))
    plt.close()

    # Correlation heatmap (include SurvivalScore)
    corr_cols = feature_cols + ['SurvivalScore']
    corr = past[corr_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap='coolwarm', center=0)
    plt.title('Feature correlation matrix (including SurvivalScore)')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'))
    plt.close()

    # Coefficients (top positive and negative)
    coefs = pd.Series(model.coef_, index=feature_cols).sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    coefs.plot(kind='bar')
    plt.title('Linear regression coefficients')
    plt.ylabel('Coefficient value')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'coefficients_bar.png'))
    plt.close()

    # Predictions for next season: top 10 bar chart
    pred_df = pd.DataFrame({'Name': X_next.index.map(lambda i: X_next.index[i])})
    # Instead create from provided predictions_next and names already in next CSV
    # This function will not rely on pred_df above; plots will be built by caller instead





def write_briefing(base_path, metrics, coef_series, next_preds_df, selected_features, corr_with_target, plots_relative='plots'):
    # next_preds_df: DataFrame with columns ['Name', 'Predicted'] sorted desc
    briefing_path = os.path.join(base_path, 'briefing.md')
    with open(briefing_path, 'w', encoding='utf-8') as f:
        f.write('# Sole Survivor - Data Analysis Briefing\n\n')
        f.write('This briefing summarizes the exploratory analysis, model performance, and predictions for the next season.\n\n')

        f.write('## Model performance (on held-out test set)\n')
        f.write(f"- RMSE: {metrics['rmse']:.3f}\n")
        f.write(f"- MAE: {metrics['mae']:.3f}\n")
        f.write(f"- R^2: {metrics['r2']:.3f}\n\n")

        f.write('## Are survival specialists scoring contestants well?\n')
        f.write('We train a linear regression on available features to predict the official SurvivalScore. If the model explains the scores well (high R^2, low errors), then scores correlate strongly with measurable features.\n\n')

        if metrics['r2'] > 0.6:
            f.write('- The model explains a large portion of the variance (R^2 > 0.6). This suggests the specialists are applying consistent, measurable criteria.\n')
        elif metrics['r2'] > 0.3:
            f.write('- The model explains a moderate portion of the variance (0.3 < R^2 <= 0.6). The specialists are partially consistent, but some subjective/unmeasured factors likely affect scores.\n')
        else:
            f.write('- The model explains little variance (R^2 <= 0.3). The specialists appear to rely on factors not captured in the numeric features or apply inconsistent scoring.\n')

        f.write('\n## Feature selection by (absolute) correlation with SurvivalScore\n')
        f.write('Selected features (|corr| >= 0.20), showing correlation and sign:\n')
        for feat in selected_features:
            f.write(f"- {feat}: {corr_with_target.loc[feat]:.3f}\n")
        f.write('\n')

        f.write('## Important features (top coefficients)\n')
        for feat, val in coef_series.head(5).items():
            f.write(f'- {feat}: {val:.3f}\n')
        f.write('\n')

        f.write('## Top predicted contestants for next season\n')
        top3 = next_preds_df.head(3)
        for i, row in top3.reset_index(drop=True).iterrows():
            f.write(f"{i+1}. {row['Name']} — predicted SurvivalScore: {row['Predicted']:.3f}\n")
        f.write('\n')

        f.write('## Charts\n')
        f.write(f'- Predicted vs Actual: {plots_relative}/predicted_vs_actual.png\n')
        f.write(f'- Residuals distribution: {plots_relative}/residuals_hist.png\n')
        f.write(f'- Correlation heatmap: {plots_relative}/correlation_heatmap.png\n')
        f.write(f'- Coefficients bar chart: {plots_relative}/coefficients_bar.png\n')


def main():
    base_path = os.path.dirname(__file__)
    past, nxt = load_data(base_path)
    X, y, X_next, feature_cols, corr_with_target = prepare_features(past, nxt)

    # Keep Name list for next dataset
    names_next = nxt['Name'].values

    model, X_test, y_test, y_pred, metrics = train_and_evaluate(X, y)

    # Create and save plots
    plots_dir = os.path.join(base_path, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Predicted vs Actual
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--')
    plt.xlabel('Actual SurvivalScore')
    plt.ylabel('Predicted SurvivalScore')
    plt.title('Predicted vs Actual (test set)')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'predicted_vs_actual.png'))
    plt.close()

    # Residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, kde=True)
    plt.title('Residuals distribution (test set)')
    plt.xlabel('Actual - Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'residuals_hist.png'))
    plt.close()

    # Correlation heatmap (include SurvivalScore)
    corr_cols = feature_cols + ['SurvivalScore']
    corr = past[corr_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap='coolwarm', center=0)
    plt.title('Feature correlation matrix (including SurvivalScore)')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'))
    plt.close()

    # Coefficients bar (index matches selected features)
    coefs = pd.Series(model.coef_, index=feature_cols).sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    coefs.plot(kind='bar')
    plt.title('Linear regression coefficients')
    plt.ylabel('Coefficient value')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'coefficients_bar.png'))
    plt.close()

    # Predict next season
    preds_next = model.predict(X_next)
    next_preds_df = pd.DataFrame({'Name': names_next, 'Predicted': preds_next})
    next_preds_df = next_preds_df.sort_values('Predicted', ascending=False).reset_index(drop=True)

    # Top 10 bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Predicted', y='Name', data=next_preds_df.head(10), palette='viridis')
    plt.title('Top 10 predicted SurvivalScore (next season)')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'next_top10.png'))
    plt.close()

    # Write briefing markdown (include selected features and their correlations)
    write_briefing(base_path, metrics, coefs, next_preds_df, feature_cols, corr_with_target)

    # Also write a CSV with predictions
    next_preds_df.to_csv(os.path.join(base_path, 'next_predictions.csv'), index=False)

    # Print summary to stdout
    print('Model metrics:')
    for k, v in metrics.items():
        print(f'- {k}: {v:.4f}')
    print('\nTop 3 predicted for next season:')
    for i, row in next_preds_df.head(3).iterrows():
        print(f"{i+1}. {row['Name']}: {row['Predicted']:.4f}")


if __name__ == '__main__':
    main()
