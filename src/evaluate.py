"""
Model Evaluation and Metrics for Model Explainability Dashboard
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    classification_report, average_precision_score
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.utils import load_artifact, save_artifact


def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)
        
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1_score': f1_score(y_true, y_pred, average='binary'),
    }
    
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        metrics['average_precision'] = average_precision_score(y_true, y_prob)
    
    return metrics


def get_classification_report(y_true, y_pred, target_names=None):
    """
    Get detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names of target classes
        
    Returns:
        str: Classification report
    """
    return classification_report(y_true, y_pred, target_names=target_names)


def plot_confusion_matrix(y_true, y_pred, target_names=None, figsize=(8, 6)):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names of target classes
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=target_names if target_names else ['0', '1'],
                yticklabels=target_names if target_names else ['0', '1'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    
    return fig


def plot_roc_curve(y_true, y_prob, figsize=(8, 6)):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


def plot_precision_recall_curve(y_true, y_prob, figsize=(8, 6)):
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(recall, precision, color='darkorange', lw=2,
            label=f'PR curve (AP = {avg_precision:.3f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


def plot_calibration_curve(y_true, y_prob, n_bins=10, figsize=(8, 6)):
    """
    Plot calibration curve.
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        n_bins: Number of bins
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Curve (Reliability Diagram)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


def plot_feature_correlation(X, figsize=(12, 10)):
    """
    Plot feature correlation heatmap.
    
    Args:
        X: Feature dataframe
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    corr_matrix = X.corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r',
                center=0, ax=ax, square=True, linewidths=0.5)
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    
    return fig


def plot_feature_distributions(X, n_cols=5, figsize=(15, 20)):
    """
    Plot feature distributions (histograms).
    
    Args:
        X: Feature dataframe
        n_cols: Number of columns in subplot grid
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    n_features = X.shape[1]
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, col in enumerate(X.columns):
        axes[i].hist(X[col], bins=30, edgecolor='black', alpha=0.7)
        axes[i].set_title(col[:20] + '...' if len(col) > 20 else col, fontsize=8)
        axes[i].tick_params(labelsize=6)
    
    # Hide empty subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Feature Distributions', fontsize=14, y=1.02)
    plt.tight_layout()
    
    return fig


def run_evaluation(model=None, training_data=None):
    """
    Run complete evaluation pipeline.
    
    Args:
        model: Trained model (loads from artifact if None)
        training_data: Training data dict (loads from artifact if None)
        
    Returns:
        dict: Evaluation results
    """
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Load artifacts if not provided
    if model is None:
        print("\n1. Loading model...")
        model = load_artifact('trained_model.pkl')
    
    if training_data is None:
        print("2. Loading training data...")
        training_data = load_artifact('training_data.pkl')
    
    X_test = training_data['X_test']
    y_test = training_data['y_test']
    target_names = training_data['target_names']
    
    # Make predictions
    print("\n3. Making predictions...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    print("\n4. Computing metrics...")
    metrics = compute_metrics(y_test, y_pred, y_prob)
    
    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    # Classification report
    print("\n5. Classification Report:")
    print(get_classification_report(y_test, y_pred, target_names))
    
    # Save evaluation results
    evaluation_results = {
        'metrics': metrics,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'y_test': y_test,
        'target_names': target_names
    }
    save_artifact(evaluation_results, 'evaluation_results.pkl')
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
    
    return evaluation_results


if __name__ == "__main__":
    results = run_evaluation()
    print("\nEvaluation completed successfully!")
