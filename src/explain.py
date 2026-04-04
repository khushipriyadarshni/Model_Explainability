"""
SHAP Explainability Module for Model Explainability Dashboard
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.utils import load_artifact, save_artifact


def create_explainer(model, X_background=None):
    """
    Create SHAP TreeExplainer for XGBoost model.
    
    Args:
        model: Trained XGBoost model
        X_background: Background data for explainer
        
    Returns:
        SHAP TreeExplainer
    """
    explainer = shap.TreeExplainer(model)
    return explainer


def compute_shap_values(explainer, X):
    """
    Compute SHAP values for given data.
    
    Args:
        explainer: SHAP explainer
        X: Data to explain
        
    Returns:
        SHAP values array
    """
    shap_values = explainer.shap_values(X)
    return shap_values


def get_shap_explanation(explainer, X):
    """
    Get SHAP Explanation object for given data.
    
    Args:
        explainer: SHAP explainer
        X: Data to explain
        
    Returns:
        SHAP Explanation object
    """
    return explainer(X)


def plot_feature_importance(shap_values, feature_names, max_display=20, figsize=(10, 8)):
    """
    Plot global feature importance (bar plot).
    
    Args:
        shap_values: SHAP values
        feature_names: List of feature names
        max_display: Maximum features to display
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    # Calculate mean absolute SHAP values
    if isinstance(shap_values, list):
        # For binary classification, use positive class
        shap_vals = np.abs(shap_values[1]).mean(axis=0)
    else:
        shap_vals = np.abs(shap_values).mean(axis=0)
    
    # Create dataframe for plotting
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': shap_vals
    }).sort_values('importance', ascending=True).tail(max_display)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(importance_df['feature'], importance_df['importance'], color='#1f77b4')
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title('Global Feature Importance (SHAP)')
    plt.tight_layout()
    
    return fig


def plot_summary(shap_values, X, feature_names=None, max_display=20, figsize=(10, 8)):
    """
    Plot SHAP summary plot (beeswarm).
    
    Args:
        shap_values: SHAP values or Explanation object
        X: Feature data
        feature_names: List of feature names
        max_display: Maximum features to display
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    plt.sca(ax)
    
    if isinstance(shap_values, list):
        # For binary classification, use positive class
        shap.summary_plot(shap_values[1], X, feature_names=feature_names,
                         max_display=max_display, show=False)
    else:
        shap.summary_plot(shap_values, X, feature_names=feature_names,
                         max_display=max_display, show=False)
    
    plt.tight_layout()
    return fig


def plot_waterfall(shap_explanation, instance_idx=0, max_display=15):
    """
    Plot waterfall plot for individual prediction explanation.
    
    Args:
        shap_explanation: SHAP Explanation object
        instance_idx: Index of instance to explain
        max_display: Maximum features to display
        
    Returns:
        matplotlib figure
    """
    fig = plt.figure(figsize=(10, 8))
    shap.plots.waterfall(shap_explanation[instance_idx], max_display=max_display, show=False)
    plt.tight_layout()
    return fig


def plot_force(explainer, shap_values, X, instance_idx=0, feature_names=None):
    """
    Plot force plot for individual prediction.
    
    Args:
        explainer: SHAP explainer
        shap_values: SHAP values
        X: Feature data
        instance_idx: Index of instance
        feature_names: List of feature names
        
    Returns:
        SHAP force plot visualization
    """
    if isinstance(shap_values, list):
        sv = shap_values[1][instance_idx]
    else:
        sv = shap_values[instance_idx]
    
    if isinstance(X, pd.DataFrame):
        x_instance = X.iloc[instance_idx]
    else:
        x_instance = X[instance_idx]
    
    return shap.force_plot(
        explainer.expected_value[1] if isinstance(explainer.expected_value, list) 
        else explainer.expected_value,
        sv,
        x_instance,
        feature_names=feature_names
    )


def plot_dependence(shap_values, X, feature, interaction_feature=None, figsize=(10, 6)):
    """
    Plot SHAP dependence plot for feature interactions.
    
    Args:
        shap_values: SHAP values
        X: Feature data
        feature: Feature to plot
        interaction_feature: Feature for interaction coloring
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    plt.sca(ax)
    
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values
    
    shap.dependence_plot(feature, sv, X, interaction_index=interaction_feature, 
                         ax=ax, show=False)
    plt.tight_layout()
    return fig


def get_instance_explanation(model, explainer, X, instance_idx, feature_names):
    """
    Get detailed explanation for a single instance.
    
    Args:
        model: Trained model
        explainer: SHAP explainer
        X: Feature data
        instance_idx: Index of instance
        feature_names: List of feature names
        
    Returns:
        dict: Instance explanation details
    """
    if isinstance(X, pd.DataFrame):
        instance = X.iloc[[instance_idx]]
        instance_values = X.iloc[instance_idx].values
    else:
        instance = X[instance_idx:instance_idx+1]
        instance_values = X[instance_idx]
    
    # Get prediction
    prediction = model.predict(instance)[0]
    prediction_proba = model.predict_proba(instance)[0]
    
    # Get SHAP values for this instance
    shap_values = explainer.shap_values(instance)
    if isinstance(shap_values, list):
        instance_shap = shap_values[1][0]  # Positive class
    else:
        instance_shap = shap_values[0]
    
    # Get expected value
    if isinstance(explainer.expected_value, list):
        expected_value = explainer.expected_value[1]
    else:
        expected_value = explainer.expected_value
    
    # Create feature contribution dataframe
    contributions = pd.DataFrame({
        'feature': feature_names,
        'value': instance_values,
        'shap_value': instance_shap
    }).sort_values('shap_value', key=abs, ascending=False)
    
    return {
        'prediction': prediction,
        'prediction_proba': prediction_proba,
        'expected_value': expected_value,
        'contributions': contributions,
        'shap_values': instance_shap
    }


def run_shap_analysis(model=None, training_data=None):
    """
    Run complete SHAP analysis pipeline.
    
    Args:
        model: Trained model (loads from artifact if None)
        training_data: Training data dict (loads from artifact if None)
        
    Returns:
        dict: SHAP analysis results
    """
    print("=" * 60)
    print("SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 60)
    
    # Load artifacts if not provided
    if model is None:
        print("\n1. Loading model...")
        model = load_artifact('trained_model.pkl')
    
    if training_data is None:
        print("2. Loading training data...")
        training_data = load_artifact('training_data.pkl')
    
    X_test = training_data['X_test']
    feature_names = training_data['feature_names']
    
    # Create explainer
    print("\n3. Creating SHAP explainer...")
    explainer = create_explainer(model)
    
    # Compute SHAP values
    print("4. Computing SHAP values (this may take a moment)...")
    shap_values = compute_shap_values(explainer, X_test)
    
    # Get Explanation object
    print("5. Creating SHAP explanation object...")
    shap_explanation = get_shap_explanation(explainer, X_test)
    
    # Save SHAP artifacts
    print("\n6. Saving SHAP artifacts...")
    shap_artifacts = {
        'explainer': explainer,
        'shap_values': shap_values,
        'shap_explanation': shap_explanation,
        'feature_names': feature_names
    }
    save_artifact(shap_artifacts, 'shap_artifacts.pkl')
    
    print("\n" + "=" * 60)
    print("SHAP ANALYSIS COMPLETE!")
    print("=" * 60)
    
    return shap_artifacts


if __name__ == "__main__":
    results = run_shap_analysis()
    print("\nSHAP analysis completed successfully!")
