"""
Model Explainability Dashboard - Streamlit App
A production-quality ML explainability system with SHAP explanations
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.utils import load_data, load_artifact, validate_data, split_data, save_artifact
from src.evaluate import (
    compute_metrics, plot_confusion_matrix, plot_roc_curve,
    plot_precision_recall_curve, plot_calibration_curve,
    plot_feature_correlation
)
from src.explain import (
    create_explainer, compute_shap_values, get_shap_explanation,
    plot_feature_importance, plot_summary, plot_waterfall,
    plot_dependence, get_instance_explanation
)
from src.train import run_training_pipeline

# Page configuration
st.set_page_config(
    page_title="ML Explainability Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model_and_data():
    """Load or train model and data with caching."""
    artifacts_dir = os.path.join(os.path.dirname(__file__), 'artifacts')
    model_path = os.path.join(artifacts_dir, 'trained_model.pkl')
    
    # Check if model exists
    if not os.path.exists(model_path):
        with st.spinner(" Training model for the first time... This may take a few minutes."):
            results = run_training_pipeline(use_grid_search=True)
        st.success(" Model trained successfully!")
    
    # Load artifacts
    model = load_artifact('trained_model.pkl')
    training_data = load_artifact('training_data.pkl')
    
    return model, training_data


@st.cache_resource
def load_shap_artifacts(_model, _training_data):
    """Load or compute SHAP artifacts with caching."""
    artifacts_dir = os.path.join(os.path.dirname(__file__), 'artifacts')
    shap_path = os.path.join(artifacts_dir, 'shap_artifacts.pkl')
    
    if os.path.exists(shap_path):
        return load_artifact('shap_artifacts.pkl')
    
    # Compute SHAP values
    with st.spinner(" Computing SHAP values... This may take a moment."):
        explainer = create_explainer(_model)
        X_test = _training_data['X_test']
        shap_values = compute_shap_values(explainer, X_test)
        shap_explanation = get_shap_explanation(explainer, X_test)
        
        shap_artifacts = {
            'explainer': explainer,
            'shap_values': shap_values,
            'shap_explanation': shap_explanation,
            'feature_names': _training_data['feature_names']
        }
        save_artifact(shap_artifacts, 'shap_artifacts.pkl')
    
    return shap_artifacts


def main():
    """Main application entry point."""
    # Header
    st.title(" Model Explainability Dashboard")
    st.markdown("### Powered by SHAP & XGBoost")
    st.markdown("An interactive dashboard for understanding machine learning model predictions using SHAP explanations.")
    st.markdown("---")
    
    # Load data and model
    try:
        model, training_data = load_model_and_data()
        shap_artifacts = load_shap_artifacts(model, training_data)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Click the button below to train a new model.")
        if st.button(" Train New Model"):
            with st.spinner("Training model..."):
                run_training_pipeline(use_grid_search=True)
            st.success("Model trained! Please refresh the page.")
        return
    
    # Get raw data for overview
    X, y, feature_names, target_names = load_data()
    validation_report = validate_data(X, y)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("##  Dashboard Controls")
        page = st.radio(
            "Select Section",
            [" Overview", " Model Performance", " Feature Importance", 
             " Instance Explanation", " Feature Analysis"]
        )
        st.markdown("---")
        st.markdown("### ℹ About")
        st.markdown("""
        This dashboard provides:
        - Model performance metrics
        - Global feature importance
        - Individual prediction explanations
        - Feature interaction analysis
        """)
    
    # Page content
    if page == " Overview":
        render_overview(X, y, feature_names, target_names, validation_report)
    elif page == " Model Performance":
        render_model_performance(model, training_data)
    elif page == " Feature Importance":
        render_feature_importance(model, training_data, shap_artifacts)
    elif page == " Instance Explanation":
        render_instance_explanation(model, training_data, shap_artifacts)
    elif page == " Feature Analysis":
        render_feature_analysis(training_data)
    
    # Footer
    st.markdown("---")
    st.markdown(" Model Explainability Dashboard | Built with Streamlit & SHAP")


def render_overview(X, y, feature_names, target_names, validation_report):
    """Render the dataset overview section."""
    st.header(" Dataset Overview")
    
    # Dataset stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples", validation_report['n_samples'])
    col2.metric("Features", validation_report['n_features'])
    col3.metric("Malignant", validation_report['target_distribution'].get(0, 0))
    col4.metric("Benign", validation_report['target_distribution'].get(1, 0))
    
    st.markdown("---")
    
    # Data preview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(" Data Preview")
        display_df = X.copy()
        display_df['Target'] = y.map({0: target_names[0], 1: target_names[1]})
        st.dataframe(display_df.head(10), use_container_width=True)
    
    with col2:
        st.subheader(" Target Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#667eea', '#764ba2']
        target_counts = y.value_counts()
        ax.pie(target_counts, labels=target_names, autopct='%1.1f%%', colors=colors,
               explode=(0.05, 0), shadow=True, startangle=90)
        ax.set_title('Class Distribution')
        st.pyplot(fig)
        plt.close()


def render_model_performance(model, training_data):
    """Render model performance metrics section."""
    st.header(" Model Performance")
    
    X_test = training_data['X_test']
    y_test = training_data['y_test']
    target_names = training_data['target_names']
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    metrics = compute_metrics(y_test, y_pred, y_prob)
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    col2.metric("Precision", f"{metrics['precision']:.4f}")
    col3.metric("Recall", f"{metrics['recall']:.4f}")
    col4.metric("F1 Score", f"{metrics['f1_score']:.4f}")
    col5.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    
    st.markdown("---")
    
    # Visualization tabs
    tab1, tab2, tab3 = st.tabs([" Confusion Matrix", " ROC Curve", " PR Curve"])
    
    with tab1:
        fig = plot_confusion_matrix(y_test, y_pred, target_names)
        st.pyplot(fig)
        plt.close()
    
    with tab2:
        fig = plot_roc_curve(y_test, y_prob)
        st.pyplot(fig)
        plt.close()
    
    with tab3:
        fig = plot_precision_recall_curve(y_test, y_prob)
        st.pyplot(fig)
        plt.close()


def render_feature_importance(model, training_data, shap_artifacts):
    """Render global feature importance section."""
    st.header(" Feature Importance (SHAP)")
    
    X_test = training_data['X_test']
    feature_names = training_data['feature_names']
    shap_values = shap_artifacts['shap_values']
    
    # Tabs for different views
    tab1, tab2 = st.tabs([" Bar Plot", " Summary Plot"])
    
    with tab1:
        st.subheader("Global Feature Importance")
        st.markdown("Shows the average impact of each feature on model predictions.")
        fig = plot_feature_importance(shap_values, feature_names)
        st.pyplot(fig)
        plt.close()
    
    with tab2:
        st.subheader("SHAP Summary Plot")
        st.markdown("Shows how feature values affect predictions.")
        fig = plot_summary(shap_values, X_test, feature_names)
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # Feature dependence plot
    st.subheader(" Feature Dependence")
    selected_feature = st.selectbox("Select Feature", feature_names, index=0)
    fig = plot_dependence(shap_values, X_test, selected_feature)
    st.pyplot(fig)
    plt.close()


def render_instance_explanation(model, training_data, shap_artifacts):
    """Render individual instance explanation section."""
    st.header(" Individual Prediction Explanation")
    
    X_test = training_data['X_test']
    y_test = training_data['y_test']
    feature_names = training_data['feature_names']
    target_names = training_data['target_names']
    explainer = shap_artifacts['explainer']
    shap_explanation = shap_artifacts['shap_explanation']
    
    # Instance selector
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader(" Select Instance")
        instance_idx = st.number_input(
            "Instance Index",
            min_value=0,
            max_value=len(X_test) - 1,
            value=0,
            step=1
        )
        
        if st.button(" Random Instance"):
            instance_idx = np.random.randint(0, len(X_test))
            st.rerun()
    
    with col2:
        # Get prediction details
        instance = X_test.iloc[[instance_idx]]
        prediction = model.predict(instance)[0]
        prediction_proba = model.predict_proba(instance)[0]
        actual = y_test.iloc[instance_idx]
        
        st.subheader(" Prediction Summary")
        
        pred_col1, pred_col2, pred_col3 = st.columns(3)
        
        with pred_col1:
            predicted_class = target_names[prediction]
            color = "" if prediction == actual else ""
            st.metric("Predicted Class", f"{color} {predicted_class}")
        
        with pred_col2:
            confidence = max(prediction_proba) * 100
            st.metric("Confidence", f"{confidence:.1f}%")
        
        with pred_col3:
            actual_class = target_names[actual]
            st.metric("Actual Class", actual_class)
    
    st.markdown("---")
    
    # Probability display
    st.subheader(" Prediction Probability")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{target_names[0]} (Malignant)**")
        st.progress(float(prediction_proba[0]))
        st.caption(f"{prediction_proba[0]*100:.1f}%")
    with col2:
        st.markdown(f"**{target_names[1]} (Benign)**")
        st.progress(float(prediction_proba[1]))
        st.caption(f"{prediction_proba[1]*100:.1f}%")
    
    st.markdown("---")
    
    # SHAP waterfall
    st.subheader(" SHAP Waterfall Explanation")
    fig = plot_waterfall(shap_explanation, instance_idx)
    st.pyplot(fig)
    plt.close()
    
    # Feature contributions table
    st.subheader(" Feature Contributions")
    explanation = get_instance_explanation(model, explainer, X_test, instance_idx, feature_names)
    contributions_df = explanation['contributions'].copy()
    contributions_df['impact'] = contributions_df['shap_value'].apply(
        lambda x: ' Increases Risk' if x > 0 else ' Decreases Risk'
    )
    contributions_df['shap_value'] = contributions_df['shap_value'].round(4)
    contributions_df['value'] = contributions_df['value'].round(4)
    st.dataframe(contributions_df[['feature', 'value', 'shap_value', 'impact']].head(15), use_container_width=True)


def render_feature_analysis(training_data):
    """Render feature analysis section."""
    st.header(" Feature Analysis")
    
    X = training_data['X_train_original']
    
    st.subheader(" Feature Correlation Heatmap")
    st.markdown("Understanding correlations helps interpret SHAP values.")
    fig = plot_feature_correlation(X)
    st.pyplot(fig)
    plt.close()


if __name__ == "__main__":
    main()
