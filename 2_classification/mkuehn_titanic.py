import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import sklearn.metrics as metrics
import imblearn.over_sampling as ios
import seaborn as sns
from sklearn.metrics import accuracy_score

N_FOLD_CROSS_VALIDATION = 10

def show_prediction_results(header, prediction, actual_data):
    np_prediction = np.round(prediction).astype(int)
    np_actual = np.array(actual_data).astype(int)
    result = np_actual == np_prediction
    
    num_correct_predictions = np.count_nonzero(result)
    num_incorrect_predictions = np.count_nonzero(result == False)
    print('{0}: {1} correct, {2} incorrect, precision: {3}.'
          .format(header, num_correct_predictions, num_incorrect_predictions,
                  (num_correct_predictions/(num_correct_predictions+num_incorrect_predictions))))
    return (num_correct_predictions/(num_correct_predictions+num_incorrect_predictions))

def make_heatmap(corr_df):
    corr_df = corr_df.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Correlation Matrix Heatmap')
    plt.show()

def compute_confusion_matrix_numbers(actual_data_df, prediction):
    confusion_tuple = metrics.confusion_matrix(actual_data_df, prediction)
    command_line_display_as_accuracy_top_confusion_matrix = confusion_tuple.T
    command_line_display_as_accuracy_top_confusion_matrix = np.flip(command_line_display_as_accuracy_top_confusion_matrix, axis=0)
    command_line_display_as_accuracy_top_confusion_matrix = np.flip(command_line_display_as_accuracy_top_confusion_matrix, axis=1)
    true_negs = 0
    false_pos = 0
    false_negs = 0
    true_pos = 0
    sensitivity = 0
    specificity = 0
    if len(confusion_tuple.ravel()) == 4:
        (true_negs, false_pos, false_negs, true_pos) = confusion_tuple.ravel()
        if ((true_pos + false_negs) > 0) and ((true_negs + false_pos) > 0):
            sensitivity = true_pos / (true_pos + false_negs)
            specificity = true_negs / (true_negs + false_pos)
    return (confusion_tuple, command_line_display_as_accuracy_top_confusion_matrix, true_negs, false_pos, false_negs, true_pos, sensitivity, specificity)

# def create_confusion_matrix(actual_data_df, prediction):
#     (confusion_tuple, command_line_display_as_accuracy_top_confusion_matrix, true_negs, false_pos, false_negs, true_pos, sensitivity, specificity) \
#         = compute_confusion_matrix_numbers(actual_data_df, prediction)
#     if (sensitivity > 0) or (specificity > 0):
#         print(f'tp: {true_pos}, fn: {false_negs}, tn: {true_negs}, fp: {false_pos}, sensitivity: {sensitivity}, specificity: {specificity}.')
#     print(command_line_display_as_accuracy_top_confusion_matrix)
#     sns.heatmap(confusion_tuple, annot=True)
#     plt.show()

UNBALANCED_POS = 0
BALANCED_POS = 1
ACTUAL_DATA = 0
ALL_NEGATIVES = 1

def perform_logistic_regression(titanic_predictors_df, titanic_response_df, balance_counter, summary_stats):
    balanced_str = 'unbalanced'
    # titanic_predictors_copy = titanic_predictors_df.copy()
    # titanic_response_copy = titanic_response_df.copy()
    if balance_counter == 1:
        balanced_str = 'balanced'
        
    accuracy_values = []    
    sensitivity_values = []
    specificity_values = []
        
    for random_state in range(0, 3):
        if balance_counter == 1:
            random_over_sampler = ios.RandomOverSampler(random_state=0) #<-- changed random state here
            titanic_predictors_df, titanic_response_df \
                = random_over_sampler.fit_resample(titanic_predictors_df, titanic_response_df)
                
        (titanic_predictors_training_df, titanic_predictors_testing_df,
            titanic_response_training_df, titanic_response_testing_df) = ms.train_test_split(
            titanic_predictors_df, titanic_response_df,
            test_size = 0.2, random_state=0) #<-- changed random state here
            
        algorithm = lm.LogisticRegression(max_iter=100000)
        model = algorithm.fit(titanic_predictors_training_df, titanic_response_training_df)
        prediction = model.predict(titanic_predictors_testing_df)

        acc = accuracy_score(titanic_response_testing_df, prediction)
        accuracy_values.append(acc)
        print(f"{balanced_str} fold {random_state} accuracy: {acc:.4f}")
        
        (confusion_tuple, command_line_display_as_accuracy_top_confusion_matrix, true_negs,
                false_pos, false_negs, true_pos, sensitivity, specificity) \
            = compute_confusion_matrix_numbers(titanic_response_testing_df, prediction)
            
        sensitivity_values.append(sensitivity)
        specificity_values.append(specificity)
            
        summary_stats[balance_counter][ACTUAL_DATA].insert(random_state,
                            [true_negs, false_pos, false_negs, true_pos])
        
        all_negatives_prediction = [0]*len(titanic_response_testing_df)
        
        (confusion_tuple, command_line_display_as_accuracy_top_confusion_matrix, true_negs,
                false_pos, false_negs, true_pos, sensitivity, specificity) \
            = compute_confusion_matrix_numbers(titanic_response_testing_df, all_negatives_prediction)
            
        summary_stats[balance_counter][ALL_NEGATIVES].insert(random_state,
                            [true_negs, false_pos, false_negs, true_pos])
    
    # Compute average accuracy
    avg_accuracy = sum(accuracy_values) / len(accuracy_values) if accuracy_values else 0
    print(f"{balanced_str.capitalize()} - Average Accuracy: {avg_accuracy:.4f}")
        
    # Compute averages
    avg_sensitivity = sum(sensitivity_values) / len(sensitivity_values) if sensitivity_values else 0
    avg_specificity = sum(specificity_values) / len(specificity_values) if specificity_values else 0
    
    print(f"\n{balanced_str.capitalize()} - Average Sensitivity: {avg_sensitivity:.4f}, Average Specificity: {avg_specificity:.4f}")
    
    return summary_stats
     
   
def predict(titanic_fixed_df, one_hot_encode):
    
    if not one_hot_encode:
        titanic_predictors_df = titanic_fixed_df[['Age', 'SibSp', 'Fare']] #<--'Pclass' and 'Parch' were removed, they were below .1 when compared to 'Survived'
    else:
        # inside predict(titanic_fixed_df, one_hot_encode):
        tfcopy_df = titanic_fixed_df.copy()
        
        # Drop Age, SibSp, Fare
        tfcopy_df = tfcopy_df.drop(columns=['Age', 'SibSp', 'Fare'])
        
        # Map Sex -> 0/1
        tfcopy_df['Sex'] = tfcopy_df['Sex'].map({'male': 0, 'female': 1})
        
        # Map Embarked -> 0/1/2
        # tfcopy_df['Embarked'] = tfcopy_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}) 
        
        # One-hot encode Embarked (keeps columns for S, C, Q)
        titanic_embarked_dummies = pd.get_dummies(tfcopy_df['Embarked'], prefix='Embarked')
        
        # Choose numeric base predictors plus encoded columns
        titanic_predictors_df = pd.concat([tfcopy_df[['Sex']], titanic_embarked_dummies], axis=1)   
        
        # titanic_sex_dummies = pd.get_dummies(titanic_fixed_df['Sex'], prefix='Sex')
        # titanic_embarked_dummies = pd.get_dummies(titanic_fixed_df['Embarked'], prefix='Embarked')
        # titanic_predictors_df = pd.concat([titanic_fixed_df[['Age', 'SibSp', 'Fare']], titanic_sex_dummies, titanic_embarked_dummies], axis=1)   
    
    titanic_response_df = titanic_fixed_df['Survived']
    
    summary_stats = [[[],[]],[[],[]]]
    for balance_counter in range(2):
        summary_stats = perform_logistic_regression(titanic_predictors_df, titanic_response_df, 
                                                    balance_counter, summary_stats)
        
    summary_array = np.array(summary_stats, dtype=int)
    
    print("Training Predictors Summary:")
    print(summary_array)
    print(titanic_predictors_df)
    print(titanic_response_df)
    
def main():
    """Main function"""
    titanic_df = pd.read_csv("E:/Madison College/Machine Learning/mad-2025-fall-ml-the-algorithms/2_classification/Titanic-Dataset.csv")
    # print(titanic_df)
    titanic_fixed_df = titanic_df.dropna() ## <-- Dropped the NaN rows
    # print(titanic_fixed_df, false)
    # make_heatmap(titanic_fixed_df)
    # create_confusion_matrix(titanic_fixed_df)
    # predict(titanic_fixed_df, False)
    predict(titanic_fixed_df, True)
    
if __name__ == "__main__":
    main()