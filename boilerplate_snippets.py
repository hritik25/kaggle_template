import pandas as pd
from typing import List, Tuple

# function to separate continuous and categorical features
def separate_cont_cat(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    continuous_features = df.select_dtypes(include='number').columns
    categorical_features = df.select_dtypes(exclude='number').columns
    return continuous_features, categorical_features

# function to return X, y (training features, labels)
def split_df_x_y(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = df.columns
    feature_cols = [col for col in cols if col != target_col]
    return df[feature_cols], df[[target_col]]

"""
1. walk through the hw2_task1 in order and see what code you wrote 
in the notebook, and write a snippet for generic enough usage
2. test the utility of the snippets to solve hw2_task2 and add missing functionality
"""