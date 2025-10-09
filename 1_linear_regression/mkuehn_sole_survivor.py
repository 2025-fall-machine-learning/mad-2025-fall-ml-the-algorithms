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

def simple_linear_regression_sspast(sole_survivor_past_df, create_testing_set=True):
    # Ensure names are strings without leading/trailing whitespace, then one-hot encode
    sole_survivor_past_df["Name"] = sole_survivor_past_df["Name"].astype(str).str.strip().fillna("")
    predictors = pd.get_dummies(sole_survivor_past_df[["Name"]], prefix='Name')
    response = sole_survivor_past_df["SurvivalScore"].values
    
    # Show top 3 names with highest SurvivalScore
    try:
        top3 = sole_survivor_past_df.nlargest(3, 'SurvivalScore')[['Name', 'SurvivalScore']]
        print("Top 3 Sole Survivors by Survival Score:")
        for _, r in top3.iterrows():
            print(f"Name: {r['Name']}, Survival Score: {r['SurvivalScore']}")
    except Exception:
        # fail silently if column missing or invalid
        print("Unable to determine top 3 Sole Survivors by Survival Score (check data).")
    
    training_predictors = predictors
    training_response = response
    
    if create_testing_set:
        training_predictors, testing_predictors, training_response, testing_response \
            = ms.train_test_split(
                predictors, response, test_size=0.2) #, random_state=42)
            
    model = create_linear_regression_model(training_predictors, training_response)
    show_r_squared(model, training_predictors, training_response)

    if create_testing_set:
        show_r_squared(model, testing_predictors, testing_response)
    prediction = perform_linear_regression_prediction(model, training_predictors)
    
    print("Training Data Predictors, Predictions and Response Values:")
    print_1d_data_summary(training_predictors)
    print_1d_data_summary(prediction)
    print_1d_data_summary(training_response)
    graph(training_predictors, training_response, "blue", prediction, "Training Data")
    
    if create_testing_set:
        testing_prediction = perform_linear_regression_prediction(model, testing_predictors)
        print("Testing Data Predictors, Predictions and Response Values:")
        print_1d_data_summary(testing_predictors)
        print_1d_data_summary(testing_prediction)
        print_1d_data_summary(testing_response)
        graph(testing_predictors, testing_response, "green", testing_prediction, "Testing Data")
    
def multiple_linear_regression_sspast(sole_survivor_past_df, create_testing_set, one_hot_encode):
    
    new_past_df = sole_survivor_past_df.copy()
    new_past_df["Name"] = new_past_df["Name"].astype(str).str.strip().fillna("")
    
    # numeric columns that may exist in your dataset - adjust as needed
    numeric_cols = ["Leadership", "MentalToughness", "SurvivalSkills", "RiskTaking", 
                    "Resourcefulness", "Adaptability", "PhysicalFitness", "Teamwork", "Stubbornness"]
    
    # ensure numeric columns exist and are numeric
    for col in numeric_cols:
        if col not in new_past_df.columns:
            new_past_df[col] = 0.0
        new_past_df[col] = pd.to_numeric(new_past_df[col], errors='coerce').fillna(0.0)
    
    # one-hot encode Name
    name_dummies = pd.get_dummies(new_past_df["Name"], prefix='Name')
    
    # choose predictors
    if one_hot_encode:
        predictors = pd.concat([new_past_df[numeric_cols].reset_index(drop=True), name_dummies.reset_index(drop=True)], axis=1)
    else:
        predictors = name_dummies
    
    # response (ensure numeric)
    if "NewSurvivalScore" in new_past_df.columns:
        new_past_df["NewSurvivalScore"] = pd.to_numeric(new_past_df["SurvivalScore"], errors='coerce').fillna(0.0)
    else:
        new_past_df["NewSurvivalScore"] = 0.0
        
    # If SurvivalScore is missing or all zeros, compute it as the sum of the numeric predictors
    if new_past_df["NewSurvivalScore"].abs().sum() == 0:
        # sum only the numeric predictors except Stubbornness, then subtract Stubbornness (exclude one-hot name columns). 
        # If you want to include the one-hot name columns, use: df[[*numeric_cols, *name_dummies.columns]].sum(axis=1)
        cols_except_stubborn = [c for c in numeric_cols if c != "Stubbornness"]
        new_past_df["NewSurvivalScore"] = new_past_df[cols_except_stubborn].sum(axis=1) - new_past_df["Stubbornness"]
    
    response = new_past_df["NewSurvivalScore"].values
    
    # Show top 3 names with highest SurvivalScore
    try:
        top3 = new_past_df.nlargest(3, 'NewSurvivalScore')[['Name', 'NewSurvivalScore']]
        print("Top 3 Sole Survivors by Survival Score:")
        for _, r in top3.iterrows():
            print(f"Name: {r['Name']}, Survival Score: {r['NewSurvivalScore']}")
    except Exception:
        # fail silently if column missing or invalid
        print("Unable to determine top 3 Sole Survivors by Survival Score (check data).")
        
    # optional train/test split
    if create_testing_set:
        X_train, X_test, y_train, y_test = ms.train_test_split(predictors, response, test_size=0.25, random_state=42)
    else:
        X_train, y_train = predictors, response
        X_test, y_test = None, None
    
    # train model
    model = create_linear_regression_model(X_train.values, y_train)
    
    # show coefficients with feature names
    feat_names = list(predictors.columns)
    coef = np.asarray(model.coef_).ravel()
    coef_df = pd.DataFrame({"feature": feat_names, "coefficient": coef})
    coef_df["abs_coef"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False).drop(columns=["abs_coef"]).reset_index(drop=True)
    print("Feature coefficients (sorted by absolute value):")
    print(coef_df.to_string(index=False))
    print(f"Intercept: {model.intercept_:.4f}")
    
    # predictions for full dataset (aligned with predictors)
    preds_all = perform_linear_regression_prediction(model, predictors.values)
    
    # attach predictions to a copy of the original dataframe
    df_out = new_past_df.reset_index(drop=True).copy()
    df_out["PredictedSurvivalScore"] = preds_all
    
    # show a combined table of predictors + actual + predicted (first 10 rows)
    combined = pd.concat([predictors.reset_index(drop=True), df_out[["NewSurvivalScore", "PredictedSurvivalScore"]]], axis=1)
    print("\nSample of predictors with actual and predicted SurvivalScore:")
    print(combined.head(10).to_string(index=False))
    
    # show R^2 for train (and test if available)
    try:
        r2_train = model.score(X_train.values if hasattr(X_train, "values") else X_train, y_train)
        print(f"\nTraining R^2: {r2_train:.3f}")
    except Exception:
        pass
    if X_test is not None:
        try:
            r2_test = model.score(X_test.values if hasattr(X_test, "values") else X_test, y_test)
            print(f"Testing R^2:  {r2_test:.3f}")
            test_preds = perform_linear_regression_prediction(model, X_test.values)
            test_combined = pd.concat([X_test.reset_index(drop=True), pd.DataFrame({"Actual": y_test, "Predicted": test_preds})], axis=1)
            print("\nTest set sample (first 10 rows):")
            print(test_combined.head(10).to_string(index=False))
        except Exception as e:
            print(f"Could not evaluate test set: {e}")
    
    # optional: plot using existing graph() (pass name labels so x-axis shows actual names)
    
    # pass original names aligned to training rows for x-axis labels
    graph(predictors, response, "blue", preds_all, "Training Data")
    
    if create_testing_set:
        prediction = perform_linear_regression_prediction(model, predictors)
        print("The testing data and response values:")
        print_1d_data_summary(prediction)
        print_1d_data_summary(response)
        graph(predictors, response, "green", prediction, "Testing Data")

def multiple_linear_regression_ssnext(sole_survivor_next_df, create_testing_set, one_hot_encode):
    
    # defensive copy and cleanup
    new_next_df = sole_survivor_next_df.copy()
    new_next_df["Name"] = new_next_df["Name"].astype(str).str.strip().fillna("")
    
    # numeric columns that may exist in your dataset - adjust as needed
    numeric_cols = ["Leadership", "MentalToughness", "SurvivalSkills", "RiskTaking", 
                    "Resourcefulness", "Adaptability", "PhysicalFitness", "Teamwork", "Stubbornness"]
    
    # ensure numeric columns exist and are numeric
    for col in numeric_cols:
        if col not in new_next_df.columns:
            new_next_df[col] = 0.0
        new_next_df[col] = pd.to_numeric(new_next_df[col], errors='coerce').fillna(0.0)
            
    # one-hot encode Name
    name_dummies = pd.get_dummies(new_next_df["Name"], prefix='Name')
    
    # choose predictors
    if one_hot_encode:
        predictors = pd.concat([new_next_df[numeric_cols].reset_index(drop=True), name_dummies.reset_index(drop=True)], axis=1)
    else:
        predictors = name_dummies
    
    # response (ensure numeric)
    if "SurvivalScore" in new_next_df.columns:
        new_next_df["SurvivalScore"] = pd.to_numeric(new_next_df["SurvivalScore"], errors='coerce').fillna(0.0)
    else:
        new_next_df["SurvivalScore"] = 0.0

    # If SurvivalScore is missing or all zeros, compute it as the sum of the numeric predictors
    if new_next_df["SurvivalScore"].abs().sum() == 0:
        # sum only the numeric predictors except Stubbornness, then subtract Stubbornness (exclude one-hot name columns). 
        # If you want to include the one-hot name columns, use: df[[*numeric_cols, *name_dummies.columns]].sum(axis=1)
        cols_except_stubborn = [c for c in numeric_cols if c != "Stubbornness"]
        new_next_df["SurvivalScore"] = new_next_df[cols_except_stubborn].sum(axis=1) - new_next_df["Stubbornness"]

    response = new_next_df["SurvivalScore"].values
    
    # Show top 3 names with highest SurvivalScore
    try:
        top3 = new_next_df.nlargest(3, 'SurvivalScore')[['Name', 'SurvivalScore']]
        print("Top 3 Sole Survivors by Survival Score:")
        for _, r in top3.iterrows():
            print(f"Name: {r['Name']}, Survival Score: {r['SurvivalScore']}")
            
    except Exception:
        # fail silently if column missing or invalid
        print("Unable to determine top 3 Sole Survivors by Survival Score (check data).")
    
    # optional train/test split
    if create_testing_set:
        X_train, X_test, y_train, y_test = ms.train_test_split(predictors, response, test_size=0.25, random_state=42)
    else:
        X_train, y_train = predictors, response
        X_test, y_test = None, None

    # train model
    model = create_linear_regression_model(X_train.values, y_train)

    # show coefficients with feature names
    feat_names = list(predictors.columns)
    coef = np.asarray(model.coef_).ravel()
    coef_df = pd.DataFrame({"feature": feat_names, "coefficient": coef})
    coef_df["abs_coef"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False).drop(columns=["abs_coef"]).reset_index(drop=True)
    print("Feature coefficients (sorted by absolute value):")
    print(coef_df.to_string(index=False))
    print(f"Intercept: {model.intercept_:.4f}")
    
    # predictions for full dataset (aligned with predictors)
    preds_all = perform_linear_regression_prediction(model, predictors.values)

    # attach predictions to a copy of the original dataframe
    df_out = new_next_df.reset_index(drop=True).copy()
    df_out["PredictedSurvivalScore"] = preds_all

    # show a combined table of predictors + actual + predicted (first 10 rows)
    combined = pd.concat([predictors.reset_index(drop=True), df_out[["SurvivalScore", "PredictedSurvivalScore"]]], axis=1)
    print("\nSample of predictors with actual and predicted SurvivalScore:")
    print(combined.head(10).to_string(index=False))
    
    # show R^2 for train (and test if available)
    try:
        r2_train = model.score(X_train.values if hasattr(X_train, "values") else X_train, y_train)
        print(f"\nTraining R^2: {r2_train:.3f}")
    except Exception:
        pass
    if X_test is not None:
        try:
            r2_test = model.score(X_test.values if hasattr(X_test, "values") else X_test, y_test)
            print(f"Testing R^2:  {r2_test:.3f}")
            test_preds = perform_linear_regression_prediction(model, X_test.values)
            test_combined = pd.concat([X_test.reset_index(drop=True), pd.DataFrame({"Actual": y_test, "Predicted": test_preds})], axis=1)
            print("\nTest set sample (first 10 rows):")
            print(test_combined.head(10).to_string(index=False))
        except Exception as e:
            print(f"Could not evaluate test set: {e}")

    # optional: plot using existing graph() (pass name labels so x-axis shows actual names)
    
    # pass original names aligned to training rows for x-axis labels
    graph(predictors, response, "blue", preds_all, "Training Data")
    
    if create_testing_set:
        prediction = perform_linear_regression_prediction(model, predictors)
        print("The testing data and response values:")
        print_1d_data_summary(prediction)
        print_1d_data_summary(response)
        graph(predictors, response, "green", prediction, "Testing Data")
    
    # try:
    #     graph(predictors, response, "blue", preds_all, "All Data", name_labels=df_out["Name"].reset_index(drop=True))
    # except Exception:
    #     # non-fatal; plotting not required for textual results
    #     pass

    # return augmented dataframe and model for downstream use
    return df_out, model
        
def graph(pred, resp, whatcolor, prediction, whatlabel, name_labels=None):

    original_pred = pred
    
    # Ensure resp and prediction are 1D numeric arrays
    resp = np.array(resp).ravel()
    prediction = np.array(prediction).ravel()
    
    pred_arr = np.array(pred)
    if pred_arr.ndim ==1:
        pred = pred_arr
        name_labels = None
    elif pred_arr.ndim == 2:
        # If pred is one-hot-like, use argmax to map to a single label index;
        # otherwise fall back to sample indices so sizes match resp/prediction.
        try:
            row_sums = pred_arr.sum(axis=1)
            if np.all(np.isclose(row_sums, 1)) or np.all(row_sums == 0) | np.all(row_sums == 1):
                pred = pred_arr.argmax(axis=1)
                # If original_pred is a DataFrame, get column names for labels
                if isinstance(original_pred, pd.DataFrame):
                    cols = list(original_pred.columns)
                    name_labels = [cols[i] for i in pred]
                else:
                    name_labels = None
            else:
                pred = np.arange(pred_arr.shape[0])
                name_labels = None
        except Exception:
            pred = np.arange(pred_arr.shape[0])
            name_labels = None
    else:
        pred = np.arange(resp.shape[0])
        name_labels = None
        
    # Ensure all arrays have the same length (trim to the minimum)
    min_len = min(len(pred), len(resp), len(prediction))
    if min_len == 0:
        raise ValueError("Empty array(s) provided for graphing.")
    if len(pred) != min_len or len(resp) != min_len or len(prediction) != min_len:
        print(f"Warning: Arrays have different lengths. Trimming to minimum length {min_len}.")
        pred = pred[:min_len]
        resp = resp[:min_len]
        prediction = prediction[:min_len]
        if name_labels is not None:
            name_labels = name_labels[:min_len]
                
                
    #Plot and set x-axis labels to names if available
    # plt.bar(pred, prediction, color=whatcolor, label="Best Fit Line")
    plt.scatter(pred, resp, color=whatcolor, label=whatlabel)
    plt.plot(pred, prediction, color='red', label='Best Fit Line')
    if name_labels is not None:
        plt.xticks(pred, name_labels, rotation=90, fontsize=6)
    plt.xlabel("Sole Survivor")
    plt.ylabel("Sole Survivor Survival Score")
    plt.legend()
    plt.title("Sole Survivor Past Data: Survival Score vs. Name")
    plt.show()
        
def main():
    
    sole_survivor_past_df = pd.read_csv("E:/Madison College/Machine Learning/mad-2025-fall-ml-the-algorithms/1_linear_regression/sole_survivor_past.csv", skipinitialspace=True)
    sole_survivor_past_df["Name"] = sole_survivor_past_df["Name"].astype(str).str.strip()
    
    sole_survior_next_df = pd.read_csv("E:/Madison College/Machine Learning/mad-2025-fall-ml-the-algorithms/1_linear_regression/sole_survivor_next.csv", skipinitialspace=True)
    sole_survior_next_df["Name"] = sole_survior_next_df["Name"].astype(str).str.strip()
    
    # simple_linear_regression_sspast(sole_survivor_past_df, create_testing_set=False)
    # simple_linear_regression_sspast(sole_survivor_past_df, create_testing_set=True)
    # multiple_linear_regression_sspast(sole_survivor_past_df, create_testing_set=False, one_hot_encode=False)
    # multiple_linear_regression_sspast(sole_survivor_past_df, create_testing_set=False, one_hot_encode=True)
    # multiple_linear_regression_sspast(sole_survivor_past_df, create_testing_set=True, one_hot_encode=False)
    # multiple_linear_regression_sspast(sole_survivor_past_df, create_testing_set=True, one_hot_encode=True)
    
    # multiple_linear_regression_ssnext(sole_survior_next_df, create_testing_set=False, one_hot_encode=False)
    # multiple_linear_regression_ssnext(sole_survior_next_df, create_testing_set=False, one_hot_encode=True)
    # multiple_linear_regression_ssnext(sole_survior_next_df, create_testing_set=True, one_hot_encode=False)
    multiple_linear_regression_ssnext(sole_survior_next_df, create_testing_set=True, one_hot_encode=True)
    
if __name__ == "__main__":
    main()