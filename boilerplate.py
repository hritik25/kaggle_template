import pandas as pd
from typing import List, Tuple

# **general**
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# **missing values imputation**
from sklearn.impute import MissingIndicator
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

# **feature engineering**
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder

# **cross-validation and model tuning**
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

# **classifiction models**
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from lightgbm.sklearn import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

# **class imbalance**
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline as make_imb_pipeline

# **target transformation** 
from sklearn.compose import TransformedTargetRegressor

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