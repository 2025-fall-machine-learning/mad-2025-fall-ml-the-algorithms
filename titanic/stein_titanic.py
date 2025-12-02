import numpy as np
import pandas as pd
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import sklearn.metrics as metrics
import imblearn.over_sampling as ios
import matplotlib.pyplot as plt
import seaborn as sns

#  https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/

# Cleaning the data! 
def clean(df):
    # print(f"un clean data :\n{df}")
    cleaned_data=df.dropna()
    # print(f"cleaned data :\n{cleaned_data}")
    
    return cleaned_data


# Create a heatmap for correlation analysis
def create_heatmap(df, title='Correlation Heatmap'):
    numerical_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numerical_df.corr()
    
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def create_regression_model(predictors, response):
	model = lm.LogisticRegression(max_iter=100000)
	model.fit(predictors, response)
	return model

def perform_regression_prediction(model, predictors):
	prediction = model.predict(predictors)
	return prediction


def compute_confusion_matrix_numbers(actual_data_df, prediction):
    confusion_tuple = metrics.confusion_matrix(actual_data_df, prediction)
    # print(f"confusion_tuple: {confusion_tuple}")
    command_line_display_as_accuracy_top_confusion_matrix = confusion_tuple.T
    command_line_display_as_accuracy_top_confusion_matrix = np.flip(command_line_display_as_accuracy_top_confusion_matrix, axis=0)
    command_line_display_as_accuracy_top_confusion_matrix = np.flip(command_line_display_as_accuracy_top_confusion_matrix, axis=1)
    true_negs = 0
    false_poss = 0
    false_negs = 0
    true_poss = 0
    sensitivity = 0
    specificity = 0
    if len(confusion_tuple.ravel()) == 4:
        (true_negs, false_poss, false_negs, true_poss) = confusion_tuple.ravel()
        if ((true_poss + false_negs) > 0) and ((true_negs + false_poss) > 0):
            sensitivity = true_poss / (true_poss + false_negs)
            specificity = true_negs / (true_negs + false_poss)
    # print(f'confusion_tuple:{confusion_tuple}')
    # print(f'command_line_display_as_accuracy_top_confusion_matrix:{command_line_display_as_accuracy_top_confusion_matrix}')
    # print(f'true_negs:{true_negs}, false_poss:{false_poss}, false_negs:{false_negs}, true_poss:{true_poss}')
    return (confusion_tuple, command_line_display_as_accuracy_top_confusion_matrix, true_negs, false_poss, false_negs, true_poss, sensitivity, specificity)


def create_confusion_matrix(actual_data_df, prediction):
    (confusion_tuple, command_line_display_as_accuracy_top_confusion_matrix, true_negs, false_poss, false_negs, true_poss, sensitivity, specificity) \
        = compute_confusion_matrix_numbers(actual_data_df, prediction)

 
def linear_regression(cleaned_data_df, create_testing_set):
    
    # Add male as numeric column
    cleaned_data_df['Sex_male'] = (cleaned_data_df['Sex'] == 'male').astype(int)
    # Add embarked as numeric column
    embarked_dummies = pd.get_dummies(cleaned_data_df['Embarked'], prefix='Embarked', drop_first=True)
    cleaned_data_df = pd.concat([cleaned_data_df, embarked_dummies], axis=1) 

    # predictors = pd.concat([cleaned_data_df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]], axis=1).values
    pred_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male'] + list(embarked_dummies.columns)
    predictors = cleaned_data_df[pred_cols].values


    # predictors = cleaned_data_df[['Age', 'SibSp', 'Fare']].values
    response = cleaned_data_df['Survived'].values

    training_predictors = predictors
    training_response = response

    if create_testing_set:
    # Split the data into training and testing.
        training_predictors, testing_predictors, training_response, testing_response \
        = ms.train_test_split(
            predictors, response, test_size=0.25, random_state=1)

    
    model = create_regression_model(training_predictors, training_response)
    prediction = perform_regression_prediction(model, training_predictors)
    # graph(training_predictors[:,0], training_response, "green", prediction ,"Training Data")

    if create_testing_set:
            prediction = perform_regression_prediction(model, testing_predictors)
    # print(f"prediction: {prediction}")
    # print(f"testing_response: {testing_response}")
    r_squared = model.score(predictors, response)
    print(f"R-Squared: {r_squared}")
    # --- Cross-validation accuracy ---
    cv_scores = ms.cross_val_score(model, predictors, response, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean()}")


    unbalanced_results = compute_confusion_matrix_numbers(testing_response, prediction)
    unbalanced_sensitivity = unbalanced_results[6]
    unbalanced_specificity = unbalanced_results[7]
    print(f"Unbalanced sensitivity: {unbalanced_sensitivity}, specificity: {unbalanced_specificity}")

    #balance the  data
    ros = ios.RandomOverSampler(random_state=1)
    X_train_bal, y_train_bal = ros.fit_resample(training_predictors, training_response)

    model_bal = create_regression_model(X_train_bal, y_train_bal)
    y_pred_bal = perform_regression_prediction(model_bal, testing_predictors)

    balanced_results = compute_confusion_matrix_numbers(testing_response, y_pred_bal)
    balanced_sensitivity = balanced_results[6]
    balanced_specificity = balanced_results[7]

    print(f"Balanced sensitivity: {balanced_sensitivity}, specificity: {balanced_specificity}")


    all_ones_prediction = np.ones_like(testing_response)
    # print(f"all_ones_prediction: {all_ones_prediction}")
    all_ones_metrics = compute_confusion_matrix_numbers(testing_response, all_ones_prediction)
    # print(f"all_ones_metrics: {all_ones_metrics}")

    with open("titanic_sensitivity_specificity.txt", "w") as file:
        file.write("Unbalanced model:\n")
        file.write(f"Sensitivity: {unbalanced_sensitivity}\n")
        file.write(f"Specificity: {unbalanced_specificity}\n\n")

        file.write("Balanced model:\n")
        file.write(f"Sensitivity: {balanced_sensitivity}\n")
        file.write(f"Specificity: {balanced_specificity}\n")






def graph(predictor, resp,whatcolor,prediction,whatlabel):
	plt.scatter(predictor, resp, color=whatcolor, label=whatlabel)
	plt.plot(predictor, prediction, color='red', label='Best Fit Line')
	plt.xlabel('Predictor')
	plt.ylabel('Survival')
	plt.title('Who knows maybe Titanic Survival Prediction')
	plt.legend()
	plt.show()


def main():
    titanic_df = pd.read_csv('Titanic-Dataset.csv')
    cleaned_data_df=clean(titanic_df)
    # create_heatmap(cleaned_data_df, title='Titanic Dataset Correlation Heatmap')
    linear_regression(cleaned_data_df, True)



if __name__ == '__main__':
    main()
