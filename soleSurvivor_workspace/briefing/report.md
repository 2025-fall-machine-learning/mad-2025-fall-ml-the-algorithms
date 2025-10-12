#1 Are survival specialists scoring the contestants well and why.
    Yes, the specialists score well. The multiple linear regression using all their initial ratings explains about 82% of the final score variation (see MODEL1.png — Test R² ≈ 0.816) with a small typical error (Test RMSE ≈ 7.374), while any single rating is much weaker (e.g., Adaptability alone: R² ≈ 0.23, RMSE ≈ 15 — see MODEL1.png → “Single-variable…” row), and the heatmap (MODEL2.png) shows the ratings carry different complementary information rather than being redundant, so combining them yields a strong reliable predictions.

    • Shown in data seen in MODEL1.png the combined ratings explain roughly 82% of the final score variation which has been determined through multiple linear regression "Test R^2: 0.816"
    • Error is small, "Test RMSE: 7.374" shows predictions are generally within 7 points of actual values.
    • Single ratings are much weaker (see MODEL1.png → ‘Single-variable linear regression results (test set)’): best single predictor (Adaptability) has R² ≈ 0.23 and RMSE ≈ 15, so no single metric explains the outcome nearly as well.
    • The heatmap (MODEL2.png) shows how each rating relates to the final score and to the other ratings. For example, see MODEL1.png → “Single-variable…” row “Adaptability” (R² ≈ 0.23, RMSE ≈ 15) versus the full model header (Test R² ≈ 0.82, Test RMSE ≈ 7.37). The contrast shows the ratings add complementary information, so combining them gives much stronger, more reliable predictions.

    Notes on validity:
    • Findings rely on a linear model trained on historical data and validated with an 80/20 train/test split (Test R² ≈ 0.816). This gives reasonable confidence, but it assumes relationships are ~linear.
    • Sample size and unmeasured factors may limit generalizability; the model may miss interactions or rare events that affect final scores.

#2 Survival Score Predictions & Predicted Top 3 Participants
    See MODEL3.png for the full score predictions
    Top 3 predicted participants (next season):
    1.     Nico — 84.817
    2.     Byron — 71.524
    3.     Jonah — 66.604

    How the survival scores were predicted:
        • After confirming model performance on an 80/20 training/testing split, I retrained the same model on the full cleaned past dataset so it used all available historical data.
        • The same input handling was applied to the next participants' ratings so the new ratings were treated exactly like the training data.
        • The next‑season rating columns are aligned to match the training feature names and order, then produced a predicted final SurvivalScore for each contestant.
        • Results were sorted from highest to lowest predicted SurvivalScore to create a ranked list; the top three were identified and printed.
        • Simple checks confirmed the columns matched and that the number of predictions equaled the number of contestants.