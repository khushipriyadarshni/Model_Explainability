"""
Model Training Pipeline for Model Explainability Dashboard
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.utils import load_data, split_data, save_artifact, validate_data


def create_preprocessing_pipeline():
    """
    Create a preprocessing pipeline with scaling.
    
    Returns:
        Pipeline: Sklearn preprocessing pipeline
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    return pipeline


def create_model(random_state=42):
    """
    Create an XGBoost classifier with default parameters.
    
    Args:
        random_state (int): Random seed
        
    Returns:
        XGBClassifier: XGBoost model
    """
    model = XGBClassifier(
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )
    return model


def get_hyperparameter_grid():
    """
    Get hyperparameter grid for GridSearchCV.
    
    Returns:
        dict: Parameter grid
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    return param_grid


def train_with_grid_search(X_train, y_train, cv=5, verbose=1):
    """
    Train XGBoost model with GridSearchCV for hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training targets
        cv (int): Number of cross-validation folds
        verbose (int): Verbosity level
        
    Returns:
        Best model, best parameters, CV results
    """
    print("Starting hyperparameter tuning with GridSearchCV...")
    
    # Use a smaller grid for faster training
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.1, 0.2],
        'min_child_weight': [1, 3]
    }
    
    model = create_model()
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=verbose
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV ROC-AUC score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_


def train_model(X_train, y_train, params=None, random_state=42):
    """
    Train XGBoost model with specified parameters.
    
    Args:
        X_train: Training features
        y_train: Training targets
        params (dict): Model parameters
        random_state (int): Random seed
        
    Returns:
        Trained model
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
    
    model = XGBClassifier(
        **params,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model


def evaluate_cv(model, X, y, cv=5):
    """
    Evaluate model using cross-validation.
    
    Args:
        model: Trained model
        X: Features
        y: Targets
        cv (int): Number of folds
        
    Returns:
        dict: CV scores for multiple metrics
    """
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_results = {}
    
    for metric in scoring_metrics:
        scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
        cv_results[metric] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores.tolist()
        }
    
    return cv_results


def run_training_pipeline(use_grid_search=True):
    """
    Run the complete training pipeline.
    
    Args:
        use_grid_search (bool): Whether to use GridSearchCV
        
    Returns:
        dict: Training results and artifacts
    """
    print("=" * 60)
    print("MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Load and validate data
    print("\n1. Loading data...")
    X, y, feature_names, target_names = load_data()
    
    print("\n2. Validating data...")
    validation_report = validate_data(X, y)
    print(f"   Samples: {validation_report['n_samples']}")
    print(f"   Features: {validation_report['n_features']}")
    
    # Split data
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"   Train size: {len(X_train)}")
    print(f"   Test size: {len(X_test)}")
    
    # Preprocess data
    print("\n4. Preprocessing data...")
    preprocessor = create_preprocessing_pipeline()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    # Convert back to DataFrame for better handling
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)
    
    # Train model
    print("\n5. Training model...")
    if use_grid_search:
        best_model, best_params, cv_results = train_with_grid_search(
            X_train_scaled, y_train, cv=5, verbose=1
        )
    else:
        best_model = train_model(X_train_scaled, y_train)
        best_params = best_model.get_params()
    
    # Cross-validation evaluation
    print("\n6. Cross-validation evaluation...")
    cv_scores = evaluate_cv(best_model, X_train_scaled, y_train, cv=5)
    for metric, scores in cv_scores.items():
        print(f"   {metric}: {scores['mean']:.4f} (+/- {scores['std']:.4f})")
    
    # Save artifacts
    print("\n7. Saving artifacts...")
    
    # Save model
    save_artifact(best_model, 'trained_model.pkl')
    
    # Save preprocessor
    save_artifact(preprocessor, 'preprocessor.pkl')
    
    # Save training data for SHAP
    training_data = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'target_names': target_names,
        'X_train_original': X_train,
        'X_test_original': X_test
    }
    save_artifact(training_data, 'training_data.pkl')
    
    # Save best parameters
    save_artifact(best_params, 'best_params.pkl')
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    return {
        'model': best_model,
        'preprocessor': preprocessor,
        'best_params': best_params,
        'cv_scores': cv_scores,
        'training_data': training_data
    }


if __name__ == "__main__":
    results = run_training_pipeline(use_grid_search=True)
    print("\nTraining pipeline completed successfully!")
