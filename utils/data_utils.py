import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from typing import Tuple, List

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file and clean the data by dropping unnecessary columns 
    and converting specified binary columns to numeric categorical codes.
    """
    # Load the dataset from the CSV file
    df = pd.read_csv(file_path)
    
    # Define columns to drop as they are not needed for further analysis
    cols_to_drop = ['cool', 'business_id', 'zip_code', 'user_id', 'useful', 'ambience_count']
    df.drop(columns=cols_to_drop, inplace=True)
    
    # Convert binary columns to categorical codes.
    # Codes of -1 (representing missing categories) are replaced with NaN.
    for col in ['price_binary', 'attire_binary']:
        df[col] = df[col].astype('category').cat.codes.replace(-1, np.nan)
    
    return df


def transform_numeric_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    skewed_features: List[str],
    numeric_features: List[str],
    stars_feature: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transform numeric features in the training and testing datasets.
    
    Skewed features undergo a log transformation (using log1p) followed by standard scaling,
    while other numeric features (including star rating features) are scaled using standard scaling.
    """
    # Pipeline for processing skewed features: log transformation then standard scaling
    skewed_pipeline = Pipeline([
        ("log_transform", FunctionTransformer(np.log1p, validate=True)),
        ("scaler", StandardScaler())
    ])
    # Fit and transform skewed features on training data, then transform test data
    X_train.loc[:, skewed_features] = skewed_pipeline.fit_transform(X_train[skewed_features])
    X_test.loc[:, skewed_features] = skewed_pipeline.transform(X_test[skewed_features])
    
    # Combine other numeric features with star rating features
    other_numeric = numeric_features + stars_feature
    
    # Pipeline for scaling other numeric features
    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])
    # Fit and transform other numeric features on training data, then transform test data
    X_train.loc[:, other_numeric] = numeric_pipeline.fit_transform(X_train[other_numeric])
    X_test.loc[:, other_numeric] = numeric_pipeline.transform(X_test[other_numeric])
    
    return X_train, X_test