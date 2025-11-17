import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import imblearn.over_sampling as ios
import seaborn as sns

def predict(titanic_fixed_df):
    titanic_predictors_df = titanic_fixed_df['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    titanic_response_df = titanic_fixed_df['Survived']
    
    training_predictors = titanic_predictors_df
    training_response = titanic_response_df

def main():
    """Main function"""
    titanic_df = pd.read_csv("E:/Madison College/Machine Learning/mad-2025-fall-ml-the-algorithms/2_classification/Titanic-Dataset.csv")
    # print(titanic_df.head())
    titanic_fixed_df = titanic_df.dropna() ## <-- Dropped the NaN rows
    # print(titanic_fixed_df.head())
    
    
if __name__ == "__main__":
    main()