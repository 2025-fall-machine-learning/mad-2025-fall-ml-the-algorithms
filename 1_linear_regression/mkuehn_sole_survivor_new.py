import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection as ms

def print_1d_data_summary(data_1d):
    numpified_data = np.array(data_1d)
# Flatten if 2D with one column
    if numpified_data.ndim == 2 and numpified_data.shape[1] == 1:
        arr = numpified_data.flatten()
    else:
        arr = numpified_data

    def format_row(r):
        # numeric row (1D numpy array)
        if isinstance(r, np.ndarray) and np.issubdtype(r.dtype, np.number):
            return "[" + ", ".join(f"{v:7.3f}" for v in r) + "]"
        # scalar numeric
        if np.isscalar(r) and np.issubdtype(type(r), np.number):
            return f"{r:7.3f}"
        # fallback to safe string representation
        return repr(r)

    # Prepare first/last pieces depending on shape
    if arr.ndim == 1:
        first_five = ", ".join(format_row(x) for x in arr[:5])
        last_five = ", ".join(format_row(x) for x in arr[-5:])
    elif arr.ndim == 2:
        first_five = ", ".join(format_row(row) for row in arr[:5])
        last_five = ", ".join(format_row(row) for row in arr[-5:])
    else:
        # unexpected shape: just show reprs
        flat = arr.flatten()
        first_five = ", ".join(repr(x) for x in flat[:5])
        last_five = ", ".join(repr(x) for x in flat[-5:])

    print(f"[{first_five}, ..., {last_five}]")
    
def show_r_squared(model, predictors, response):
    r_squared = model.score(predictors, response)
    print(f"R-Squared: {r_squared:.3f}")
    
def create_linear_regression_model(predictors, response):
    model = lm.LinearRegression()
    model.fit(predictors, response)
    
    return model

def perform_linear_regression_prediction(model, predictors):
    prediction = model.predict(predictors)
    
    return prediction

def make_heatmap(corr_df):
    corr_df = corr_df.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Correlation Matrix Heatmap')
    plt.show()

def create_linear_regression_model(predictors, response):
    model = lm.LinearRegression()
    model.fit(predictors, response)
    
    return model

def perform_linear_regression_prediction(model, predictors):
    prediction = model.predict(predictors)
    
    return prediction

def top3survivors(sole_survivor_past_df):
        # Show top 3 names with highest SurvivalScore
    try:
        top3 = sole_survivor_past_df.nlargest(3, 'SurvivalScore')[['Name', 'SurvivalScore']]
        print("Top 3 Sole Survivors by Survival Score:")
        for _, r in top3.iterrows():
            print(f"Name: {r['Name']}, Survival Score: {r['SurvivalScore']}")
    except Exception:
        # fail silently if column missing or invalid
        print("Unable to determine top 3 Sole Survivors by Survival Score (check data).")

def linear_regression_ss(sole_survivor_past_df, create_testing_set):
    
    predictors = sole_survivor_past_df[["MentalToughness", "SurvivalSkills", "Adaptability", "PhysicalFitness", "Stubbornness"]].values
    response = sole_survivor_past_df["SurvivalScore"].values
    
    training_predictors = predictors
    training_response = response
    
    if create_testing_set:
        # Split the data into 75% training and 25% testing.
        training_predictors, testing_predictors, training_response, testing_response \
            = ms.train_test_split(
                predictors, response, test_size=0.25, random_state=42)
                
        model = create_linear_regression_model(training_predictors, training_response)
        prediction = perform_linear_regression_prediction(model, testing_predictors)
        show_r_squared(model, training_predictors, training_response)
        graph(testing_predictors[:,0], testing_response, 'blue', prediction, "Testing Data")
        
    model = create_linear_regression_model(training_predictors, training_response)
    prediction = perform_linear_regression_prediction(model, training_predictors)
    
    print(f"Training data R-Squared: {show_r_squared(model, training_predictors, training_response)}")
    print("Training Predictors Summary:")
    print_1d_data_summary(prediction)
    print(training_response)
    # graph(training_predictors[:,0], training_response, 'green', prediction, 'Training Data')
    
    return model

def predict_new_survivor(future_data, trained_model):
    predictors = future_data[["MentalToughness", "SurvivalSkills", "Adaptability", "PhysicalFitness", "Stubbornness"]].values
    prediction = perform_linear_regression_prediction(trained_model, predictors)
    print("Next Sole Survivor Predictions Summary:")
    print_1d_data_summary(prediction)
    print(prediction)
    prediction_results = future_data.copy()
    prediction_results["PredictedSurvivalScore"] = prediction
    top_three = prediction_results.nlargest(3, 'PredictedSurvivalScore')
    print("Top 3 Predicted Sole Survivors by Survival Score:")
    print(top_three[['Name', 'PredictedSurvivalScore']])
    graph(predictors[:,0], prediction_results["PredictedSurvivalScore"], 'purple', prediction, "Next Sole Survivor Predictions", 
          xlabel="Sole Survivor", ylabel="Predicted Survival Score")
   
def graph(pred, resp, whatcolor, prediction, whatlabel, xlabel=None, ylabel=None): 
	# Plot the data and the best fit line.
	plt.scatter(pred, resp, color=whatcolor, label=whatlabel)
	plt.plot(pred, prediction, color='red', label='Best Fit Line')
	plt.xlabel("Sole Survivor")
	plt.ylabel("Sole Survivor Survival Score")
	plt.title("Sole Survivor Data: Survival Score vs. Name")
	plt.legend()
	plt.show()
   
def main():
    
    sole_survivor_past_df = pd.read_csv("E:/Madison College/Machine Learning/mad-2025-fall-ml-the-algorithms/1_linear_regression/sole_survivor_past.csv", skipinitialspace=True)
    sole_survivor_past_df["Name"] = sole_survivor_past_df["Name"].astype(str).str.strip()
    
    sole_survivor_next_df = pd.read_csv("E:/Madison College/Machine Learning/mad-2025-fall-ml-the-algorithms/1_linear_regression/sole_survivor_next.csv", skipinitialspace=True)
    sole_survivor_next_df["Name"] = sole_survivor_next_df["Name"].astype(str).str.strip()
    
    # vvv In-class follow-along debug section vvv
    # print("Start of debug section:")
    # verification_model = lm.LinearRegression()
    # verification_model.fit(sole_survivor_past_df.iloc[:, 1:-1], sole_survivor_past_df.iloc[:, -1])
    # verification_prediction = verification_model.predict(sole_survivor_past_df.iloc[:, 1:])
    # r_past = verification_model.score(sole_survivor_past_df.iloc[:, 1:-1], sole_survivor_past_df.iloc[:, -1])
    # print(f'R-squared value for past verification model: {r_past}')
    
    # r_next = verification_model.score(sole_survivor_next_df.iloc[:, 1:-1], verification_prediction)
    # print(f'R-squared value for next verification model: {r_next}')
    
    # (verification_training, verification_testing) = ms.train_test_split(sole_survivor_past_df, test_size=0.25, random_state=42)
    # verification_model_2 = lm.LinearRegression()
    # verification_model_2.fit(verification_training.iloc[:, 1:-1], verification_training.iloc[:, -1])
    # verification_prediction_training = verification_model_2.predict(verification_training.iloc[:, 1:-1])
    # verification_prediction_testing = verification_model_2.predict(verification_testing.iloc[:, 1:-1])
    # r_past_training = verification_model_2.score(verification_training.iloc[:, 1:-1], verification_training.iloc[:, -1])
    # r_past_training_with_predictions = verification_model_2.score(verification_training.iloc[:, 1:-1], verification_prediction_training)
    
    # plotting_values(True simple_or_mulitple, False)
    # ^^^ End of debug section ^^^
    
    # top3survivors(sole_survivor_past_df)
    
    # make_heatmap(sole_survivor_past_df)
    
    # linear_regression_ss(sole_survivor_past_df, create_testing_set=False)
    # linear_regression_ss(sole_survivor_past_df, create_testing_set=True)
    
    fully_trained_ss_model = linear_regression_ss(sole_survivor_past_df, create_testing_set=False)
    # predict_new_survivor(sole_survivor_past_df, fully_trained_ss_model)
    predict_new_survivor(sole_survivor_next_df, fully_trained_ss_model)
    
if __name__ == "__main__":
    main()