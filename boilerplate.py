import pandas as pd
from typing import List, Tuple, Optional

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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import TargetEncoder
# from sentence_transformers import SentenceTransformer

# **cross-validation and model tuning**
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import ndcg_score

# **classifiction models**
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
# import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

# **class imbalance**
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.pipeline import make_pipeline as make_imb_pipeline

# **target transformation** 
from sklearn.compose import TransformedTargetRegressor

# function to separate continuous and categorical features
def separate_cont_cat(df: pd.DataFrame, target_col: Optional[str]=None) -> Tuple[List[str], List[str]]:
    continuous_features = df.select_dtypes(include='number').columns
    categorical_features = df.select_dtypes(exclude='number').columns
    if target_col:
        continuous_features = [x for x in continuous_features if x!=target_col]
        categorical_features = [x for x in categorical_features if x!=target_col]
    return continuous_features, categorical_features

# function to return X, y (training features, labels)
def split_df_x_y(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = df.columns
    feature_cols = [col for col in cols if col != target_col]
    return df[feature_cols], df[[target_col]]

# function to summarize the dataset body at a high level (assuming well structured data)
def summarize_data(df: pd.DataFrame) -> None:
    print("No. of samples = ", len(df))
    print("No. of features = ", len(df.columns))
    cont_x, cat_x = separate_cont_cat(df)
    print("No. of numerical features = ", len(cont_x))
    print("No. of non-numerical features = ", len(cat_x))
    df.info()

# function to generate quick numbers on the cardinality of categorical variables
def categorical_variables_stats(df: pd.DataFrame, cat: List[str]) -> pd.DataFrame:
    n_unique =   []
    for var in cat:
        n_unique.append(df[var].nunique())
    return pd.DataFrame({"name": cat, "num_unique": n_unique})

# function to filter a continuous variable by interquartile range
def filter_cont_target_by_iqr(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    q3 = np.quantile(df[target_col], 0.75)
    q1 = np.quantile(df[target_col], 0.25)
    iqr = q3 - q1
 
    lower_range = q1 - 1.5 * iqr
    upper_range = q3 + 1.5 * iqr

    df = df[df[target_col] > lower_range]
    df = df[df[target_col] < upper_range]
    return df

# function to quickly plot feature distributions
def continuous_feature_histograms(df: pd.DataFrame, cont_features: List[str]) -> None:
    # visualize continuous features
    for col in cont_features:
        fig = plt.figure(figsize=(5, 5))
        fig.suptitle(col)
        plt.hist(df[col])

def plot_classification_distributions(df: pd.DataFrame, 
                                    target_col: str,
                                    cont_features: List[str],
                                    cat_features: List[str],
                                    figsize: Tuple[int, int]=(20, 4)) -> None:
    """
    Plot feature distributions for classification problems in a single figure.
    For categorical features: nested bar plots
    For continuous features: histograms colored by class
    """
    # Calculate number of rows needed
    n_cat = len(cat_features)
    n_cont = len(cont_features)
    n_cols = 4  # Number of plots per row
    n_rows = ((n_cat + n_cont - 1) // n_cols) + 1
    
    fig = plt.figure(figsize=(figsize[0], figsize[1] * n_rows))
    
    # Plot categorical features
    for idx, feature in enumerate(cat_features, 1):
        ax = plt.subplot(n_rows, n_cols, idx)
        df_grouped = df.groupby(feature)[target_col].value_counts(normalize=True).unstack()
        df_grouped.plot(kind='bar', stacked=False, ax=ax)
        ax.set_title(f'{feature} vs {target_col}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Proportion')
        ax.legend(title=target_col)
        ax.tick_params(axis='x', rotation=45)
    
    # Plot continuous features
    for idx, feature in enumerate(cont_features, len(cat_features) + 1):
        ax = plt.subplot(n_rows, n_cols, idx)
        for target_value in df[target_col].unique():
            mask = df[target_col] == target_value
            ax.hist(df[mask][feature], alpha=0.5, label=str(target_value), 
                   density=True, bins=30)
        ax.set_title(f'{feature} Distribution')
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.legend(title=target_col)
    
    plt.tight_layout()
    plt.show()

def plot_regression_distributions(df: pd.DataFrame,
                                target_col: str,
                                cont_features: List[str],
                                cat_features: List[str],
                                figsize: Tuple[int, int]=(20, 4)) -> None:
    """
    Plot feature distributions for regression problems in a single figure.
    For categorical features: box plots
    For continuous features: scatter plots
    """
    n_cat = len(cat_features)
    n_cont = len(cont_features)
    n_cols = 4
    n_rows = ((n_cat + n_cont - 1) // n_cols) + 1
    
    fig = plt.figure(figsize=(figsize[0], figsize[1] * n_rows))
    
    # Plot categorical features
    for idx, feature in enumerate(cat_features, 1):
        ax = plt.subplot(n_rows, n_cols, idx)
        df.boxplot(column=target_col, by=feature, ax=ax)
        ax.set_title(f'{target_col} by {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel(target_col)
        ax.tick_params(axis='x', rotation=45)
    
    # Plot continuous features
    for idx, feature in enumerate(cont_features, len(cat_features) + 1):
        ax = plt.subplot(n_rows, n_cols, idx)
        ax.scatter(df[feature], df[target_col], alpha=0.5)
        ax.set_title(f'{target_col} vs {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel(target_col)
    
    plt.tight_layout()
    plt.show()