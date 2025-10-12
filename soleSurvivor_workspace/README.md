Code explanation and data analysis (along with supporting images) are written in briefing.md in the briefing folder.

csvCleaning.py - reads the raw CSVs from csvFiles, cleans and normalizes the data (types, missing values, etc.), and writes the processed outputs into cleanedFiles (e.g., past_cleaned.csv and next_cleaned.csv).
    raw csv files are in folder: csvFiles
    cleaned csv files are in folder: cleanedFiles

modeling.py - loads the cleaned CSVs from cleanedFiles, runs an 80/20 train/test multiple linear regression analysis (metrics, coefficients, residuals, heatmap), then refits the model on all past data and predicts next‑season SurvivalScore for each participant—printing the full ranked list and the top‑3.