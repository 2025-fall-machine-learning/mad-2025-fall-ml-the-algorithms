import  numpy as np
import pandas as pd
import madplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def main():
    """Main function"""
    titanic_df = pd.read_csv("Titanic-Dataset.csv")
    print(titanic_df.head())
    
if __name__ == "__main__":
    main()