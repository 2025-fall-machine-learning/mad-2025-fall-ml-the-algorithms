# Sole Survivor - Data Analysis Briefing

This briefing summarizes the exploratory analysis, model performance, and predictions for the next season.

## Model performance (on held-out test set)
- RMSE: 7.848
- MAE: 5.693
- R^2: 0.791

## Are survival specialists scoring contestants well?
We train a linear regression on available features to predict the official SurvivalScore. If the model explains the scores well (high R^2, low errors), then scores correlate strongly with measurable features.

- The model explains a large portion of the variance (R^2 > 0.6). This suggests the specialists are applying consistent, measurable criteria.

## Important features (top coefficients)
- Adaptability: 0.850
- Teamwork: 0.720
- SurvivalSkills: 0.599
- PhysicalFitness: 0.510
- MentalToughness: 0.428

## Top predicted contestants for next season
1. Nico — predicted SurvivalScore: 84.390
2. Byron — predicted SurvivalScore: 70.765
3. Jonah — predicted SurvivalScore: 66.002

## Charts
- Predicted vs Actual: plots/predicted_vs_actual.png
- Residuals distribution: plots/residuals_hist.png
- Correlation heatmap: plots/correlation_heatmap.png
- Coefficients bar chart: plots/coefficients_bar.png
