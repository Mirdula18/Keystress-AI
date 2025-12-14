"""
Model Training Module for Keystress-AI

This module trains a machine learning model to classify burnout levels
based on typing behavior features. It uses a Random Forest Classifier
as the primary model with comprehensive evaluation metrics.

Model Output:
    0 = Low Burnout
    1 = Medium Burnout
    2 = High Burnout
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# Feature columns used for training
FEATURE_COLUMNS = [
    'avg_typing_speed',
    'avg_inter_key_delay',
    'max_pause_duration',
    'backspace_ratio',
    'typing_consistency'
]


def load_training_data(filepath: str = 'data/synthetic_typing_data.csv') -> pd.DataFrame:
    """
    Load the training dataset from CSV file.
    
    Parameters:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Training data not found at {filepath}")
    
    return pd.read_csv(filepath)


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Prepare data for model training.
    
    Parameters:
        df (pd.DataFrame): Full dataset
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler
    """
    X = df[FEATURE_COLUMNS].values
    y = df['burnout_level'].values
    
    # Scale features for better model performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, scaler


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, 
                        n_estimators: int = 100, random_state: int = 42):
    """
    Train a Random Forest Classifier.
    
    Parameters:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees in the forest
        random_state: Random seed
        
    Returns:
        RandomForestClassifier: Trained model
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray,
                              random_state: int = 42):
    """
    Train a Logistic Regression model (alternative).
    
    Parameters:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed
        
    Returns:
        LogisticRegression: Trained model
    """
    model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate model performance with multiple metrics.
    
    Parameters:
        model: Trained classifier
        X_test: Test features
        y_test: True labels
        
    Returns:
        dict: Evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, 
                                                        target_names=['Low', 'Medium', 'High'])
    }
    
    return metrics


def get_feature_importance(model, feature_names: list = None) -> dict:
    """
    Get feature importance from the trained model.
    
    Parameters:
        model: Trained Random Forest model
        feature_names: List of feature names
        
    Returns:
        dict: Feature importance scores
    """
    if feature_names is None:
        feature_names = FEATURE_COLUMNS
    
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        return dict(zip(feature_names, importance.tolist()))
    
    return {}


def save_model(model, scaler, model_path: str = 'models/burnout_model.pkl',
               scaler_path: str = 'models/scaler.pkl'):
    """
    Save the trained model and scaler to disk.
    
    Parameters:
        model: Trained model
        scaler: Fitted scaler
        model_path: Path to save model
        scaler_path: Path to save scaler
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")


def load_model(model_path: str = 'models/burnout_model.pkl',
               scaler_path: str = 'models/scaler.pkl'):
    """
    Load trained model and scaler from disk.
    
    Parameters:
        model_path: Path to model file
        scaler_path: Path to scaler file
        
    Returns:
        tuple: (model, scaler)
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler


def train_and_evaluate(data_path: str = 'data/synthetic_typing_data.csv',
                       save_models: bool = True) -> dict:
    """
    Complete training pipeline: load data, train model, evaluate, and save.
    
    Parameters:
        data_path: Path to training data
        save_models: Whether to save trained models
        
    Returns:
        dict: Training results including metrics and feature importance
    """
    print("Loading training data...")
    df = load_training_data(data_path)
    print(f"Loaded {len(df)} samples")
    
    print("\nPreparing data...")
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    print("\nTraining Random Forest Classifier...")
    model = train_random_forest(X_train, y_train)
    
    print("\nEvaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print("\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"           Predicted")
    print(f"           Low  Med  High")
    print(f"Actual Low  {cm[0][0]:3d}  {cm[0][1]:3d}  {cm[0][2]:3d}")
    print(f"      Med  {cm[1][0]:3d}  {cm[1][1]:3d}  {cm[1][2]:3d}")
    print(f"     High  {cm[2][0]:3d}  {cm[2][1]:3d}  {cm[2][2]:3d}")
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    # Feature importance
    importance = get_feature_importance(model)
    print("Feature Importance:")
    for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {score:.4f}")
    
    if save_models:
        print("\nSaving models...")
        save_model(model, scaler)
    
    return {
        'model': model,
        'scaler': scaler,
        'metrics': metrics,
        'feature_importance': importance
    }


if __name__ == "__main__":
    # Run the training pipeline
    results = train_and_evaluate()
