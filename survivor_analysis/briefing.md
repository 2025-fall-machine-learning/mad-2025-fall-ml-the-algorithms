# Sole Survivor - Data Analysis Briefing

This briefing summarizes the exploratory analysis, model performance, and predictions for the next season.

## Model performance (on held-out test set)
- RMSE: 8.019
- MAE: 5.822
- R^2: 0.782

## Are survival specialists scoring contestants well?
We train a linear regression on available features to predict the official SurvivalScore. If the model explains the scores well (high R^2, low errors), then scores correlate strongly with measurable features.

- The model explains a large portion of the variance (R^2 > 0.6). This suggests the specialists are applying consistent, measurable criteria.

## Feature selection by (absolute) correlation with SurvivalScore
Selected features (|corr| >= 0.20), showing correlation and sign:
- RiskTaking: -0.434
- SurvivalSkills: 0.430
- PhysicalFitness: 0.350
- Stubbornness: -0.326
- Adaptability: 0.321
- MentalToughness: 0.310
- Resourcefulness: 0.307

## Important features (top coefficients)
- Adaptability: 0.824
- SurvivalSkills: 0.639
- PhysicalFitness: 0.500
- MentalToughness: 0.434
- RiskTaking: 0.330

## Top predicted contestants for next season
1. Nico — predicted SurvivalScore: 82.250
2. Byron — predicted SurvivalScore: 69.714
3. Jonah — predicted SurvivalScore: 62.230

## Charts
- Predicted vs Actual: plots/predicted_vs_actual.png
- Residuals distribution: plots/residuals_hist.png
- Correlation heatmap: plots/correlation_heatmap.png
- Coefficients bar chart: plots/coefficients_bar.png
