"""
Utility functions for Model Explainability Dashboard
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import joblib
import os

# Paths
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'artifacts')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


def load_data():
    """
    Load the Breast Cancer dataset from sklearn.
    
    Returns:
        X (pd.DataFrame): Feature dataframe
        y (pd.Series): Target series
        feature_names (list): List of feature names
        target_names (list): List of target class names
    """
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    
    return X, y, list(data.feature_names), list(data.target_names)


def validate_data(X, y):
    """
    Validate the dataset for common issues.
    
    Args:
        X (pd.DataFrame): Feature dataframe
        y (pd.Series): Target series
        
    Returns:
        dict: Validation report
    """
    report = {
        'n_samples': len(X),
        'n_features': X.shape[1],
        'null_counts': X.isnull().sum().to_dict(),
        'has_nulls': X.isnull().any().any(),
        'duplicate_rows': X.duplicated().sum(),
        'data_types': X.dtypes.astype(str).to_dict(),
        'target_distribution': y.value_counts().to_dict()
    }
    
    return report


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        X (pd.DataFrame): Feature dataframe
        y (pd.Series): Target series
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test: Split datasets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def save_artifact(obj, filename):
    """
    Save an artifact to the artifacts directory.
    
    Args:
        obj: Object to save
        filename (str): Name of the file
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    filepath = os.path.join(ARTIFACTS_DIR, filename)
    joblib.dump(obj, filepath)
    print(f"Saved artifact: {filepath}")
    return filepath


def load_artifact(filename):
    """
    Load an artifact from the artifacts directory.
    
    Args:
        filename (str): Name of the file
        
    Returns:
        Loaded object
    """
    filepath = os.path.join(ARTIFACTS_DIR, filename)
    if os.path.exists(filepath):
        return joblib.load(filepath)
    else:
        raise FileNotFoundError(f"Artifact not found: {filepath}")


def get_feature_statistics(X):
    """
    Get statistical summary of features.
    
    Args:
        X (pd.DataFrame): Feature dataframe
        
    Returns:
        pd.DataFrame: Statistics summary
    """
    stats = X.describe().T
    stats['missing'] = X.isnull().sum()
    stats['dtype'] = X.dtypes
    return stats


if __name__ == "__main__":
    # Test the utility functions
    print("Loading data...")
    X, y, feature_names, target_names = load_data()
    
    print("\nValidating data...")
    report = validate_data(X, y)
    print(f"Samples: {report['n_samples']}")
    print(f"Features: {report['n_features']}")
    print(f"Has nulls: {report['has_nulls']}")
    print(f"Duplicates: {report['duplicate_rows']}")
    print(f"Target distribution: {report['target_distribution']}")
    
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    
    print("\nData layer setup complete!")
