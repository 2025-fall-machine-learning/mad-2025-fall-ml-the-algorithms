import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt


def print_1d_summary(arr, name=None):
    a = np.asarray(arr).ravel()
    if name:
        print(name)
    if a.size == 0:
        print('  <empty>')
        return
    preview = ', '.join(f"{x:.3f}" for x in np.concatenate([a[:5], a[-5:]])[:10])
    print(f"  [{preview}]")


def train_and_predict(past_csv, next_csv, test_size=0.2, random_state=42):
    # Read past data
    df_past = pd.read_csv(past_csv, skipinitialspace=True)
    df_past.columns = [c.strip() for c in df_past.columns]

    # Features are all columns except Name and SurvivalScore
    feature_cols = [c for c in df_past.columns if c not in ('Name', 'SurvivalScore')]
    X = df_past[feature_cols].apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(df_past['SurvivalScore'], errors='coerce')

    valid = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=test_size, random_state=random_state)

    # Fit simple linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_test = model.predict(X_test)
    test_r2 = float(r2_score(y_test, y_pred_test))
    test_mae = float(mean_absolute_error(y_test, y_pred_test))
    print('Test R2:', test_r2)
    print('Test MAE:', test_mae)

    # Read next-season data and predict
    df_next = pd.read_csv(next_csv, skipinitialspace=True)
    df_next.columns = [c.strip() for c in df_next.columns]
    X_next = df_next[feature_cols].apply(pd.to_numeric, errors='coerce')
    # Drop rows with missing features
    valid_next = X_next.notna().all(axis=1)
    if not valid_next.all():
        print('Dropping rows with missing features in next data:', (~valid_next).sum())
    df_next = df_next.loc[valid_next].copy()
    X_next = X_next.loc[valid_next].values

    preds_next = model.predict(X_next)
    df_next['PredictedSurvivalScore'] = preds_next

    # Show top 3 predicted contestants
    top3 = df_next.sort_values('PredictedSurvivalScore', ascending=False).head(3)
    print('\nTop 3 predicted contestants for next season:')
    for i, row in top3.iterrows():
        print(f"{row['Name'].strip()}: predicted score={row['PredictedSurvivalScore']:.3f}")

    # Optional: quick scatter of predicted distribution
    plt.hist(preds_next, bins=10, color='C2', alpha=0.8)
    plt.title('Predicted SurvivalScore distribution (next season)')
    plt.xlabel('Predicted SurvivalScore')
    plt.show()

    return model, df_past, df_next, test_r2, test_mae, top3


def main():
    base = 'c:/Users/student/Documents/GitHub/mad-2025-fall-ml-the-algorithms/1_linear_regression'
    past = base + '/sole_survivor_past.csv'
    nxt = base + '/sole_survivor_next.csv'
    model, df_past, df_next, test_r2, test_mae, top3 = train_and_predict(past, nxt)

    # Hard-coded briefing text (#text#)
    briefing = "#text#\n" + (
        "Briefing: Expert SurvivalScore evaluation\n"
        "Summary: We trained a simple linear model on past expert sub-scores to predict the final SurvivalScore.\n"
        f"Model test R² = {test_r2:.2f}, test MAE = {test_mae:.2f}.\n"
        "Lay summary: The experts' sub-scores do contain useful information — the model explains some, but not all, of the final score. "
        "A moderate R² and non-zero MAE mean the experts are partly consistent, but human judgment and missing factors still affect the final rating.\n"
        "Top 3 predicted contestants (next season):\n"
    )
    for i, row in top3.iterrows():
        briefing += f" - {row['Name'].strip()}: predicted score={row['PredictedSurvivalScore']:.1f}\n"

    briefing += (
        "Recommendation: use regularized models and larger datasets before trusting a single model for predicting winners. "
        "Collecting multiple expert ratings per contestant would reduce noise and improve accuracy.\n"
    )

    # Write briefing to file
    briefing_file = base + '/briefing_for_team_lead.txt'
    with open(briefing_file, 'w', encoding='utf-8') as f:
        f.write(briefing)
    print('\nBriefing written to', briefing_file)
    print('\n' + briefing)


if __name__ == '__main__':
    main()
