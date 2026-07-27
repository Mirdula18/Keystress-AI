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

import json
import os
from datetime import datetime, timezone

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

from .disclosure import (
    FEATURE_SET_VERSION,
    FEATURES_V1,
    SHIPPED_DATA_SOURCE,
    SYNTHETIC_MODEL_NOTICE,
    format_metric,
)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


# Feature columns used for training. Sourced from the versioned feature set so that
# training, inference, and the model metadata can never drift apart silently.
FEATURE_COLUMNS = list(FEATURES_V1)

#: Sidecar describing what a saved model is and what its metrics actually mean.
DEFAULT_METADATA_PATH = 'models/model_metadata.json'


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


def build_model_version(data_source: str, n_samples: int, random_state: int) -> str:
    """
    Build a deterministic model version identifier.

    The identifier is derived only from inputs that determine the model, so an identical
    training run produces an identical version string. It deliberately contains no
    timestamp — a clean checkout must be able to reproduce the same version (F13).

    Parameters:
        data_source: ``"synthetic"`` or ``"real"``.
        n_samples: Number of training samples.
        random_state: Seed used for generation, splitting, and fitting.

    Returns:
        str: e.g. ``"rf-v1-synthetic-s42-n1500"``.
    """
    return f"rf-{FEATURE_SET_VERSION}-{data_source}-s{random_state}-n{n_samples}"


def build_model_metadata(metrics: dict, n_samples: int, random_state: int,
                         data_source: str = SHIPPED_DATA_SOURCE) -> dict:
    """
    Describe a trained model and, critically, what its metrics mean.

    Every metric this project stores travels with the data source it was measured on.
    For the shipped model that source is ``"synthetic"``, meaning the scores below
    describe how separable the hand-authored generator classes are — not real-world
    burnout detection. See :data:`src.disclosure.SYNTHETIC_MODEL_NOTICE`.

    Parameters:
        metrics: Evaluation metrics from :func:`evaluate_model`.
        n_samples: Number of samples in the training dataset.
        random_state: Seed used throughout the pipeline.
        data_source: Source of the training data.

    Returns:
        dict: Registry-shaped metadata (see ``ARCHITECTURE.md`` §4.4).
    """
    return {
        'model_version': build_model_version(data_source, n_samples, random_state),
        'model_type': 'RandomForestClassifier',
        'trained_on': data_source,
        'data_source': data_source,
        'feature_set': FEATURE_SET_VERSION,
        'features': list(FEATURE_COLUMNS),
        'random_seed': random_state,
        'n_samples': n_samples,
        'metrics': {
            'accuracy': float(metrics['accuracy']),
            'precision_weighted': float(metrics['precision']),
            'recall_weighted': float(metrics['recall']),
            'f1_weighted': float(metrics['f1_score']),
        },
        # The metrics above are meaningless without this field. Never drop it.
        'metrics_data_source': data_source,
        'metrics_caveat': SYNTHETIC_MODEL_NOTICE if data_source == 'synthetic' else '',
        'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }


def save_model(model, scaler, model_path: str = 'models/burnout_model.pkl',
               scaler_path: str = 'models/scaler.pkl',
               metadata: dict = None,
               metadata_path: str = DEFAULT_METADATA_PATH):
    """
    Save the trained model, scaler, and its disclosure metadata to disk.

    Parameters:
        model: Trained model
        scaler: Fitted scaler
        model_path: Path to save model
        scaler_path: Path to save scaler
        metadata: Metadata dict from :func:`build_model_metadata`. Optional only to keep
            the signature backward compatible; omitting it leaves the served model
            without a version or data source, which the prediction layer then reports
            honestly as unknown.
        metadata_path: Path to save the metadata sidecar
    """
    import joblib
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

    if metadata is not None:
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, 'w', encoding='utf-8') as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        print(f"Model metadata saved to {metadata_path}")


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
    import joblib
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler


def train_and_evaluate(data_path: str = 'data/synthetic_typing_data.csv',
                       save_models: bool = True,
                       random_state: int = 42,
                       data_source: str = SHIPPED_DATA_SOURCE) -> dict:
    """
    Complete training pipeline: load data, train model, evaluate, and save.

    Every metric printed or returned by this function carries its data source. For the
    default synthetic dataset the labels were authored by the generator, so the scores
    measure class separability of a hand-built distribution rather than any real-world
    detection ability.

    Parameters:
        data_path: Path to training data
        save_models: Whether to save trained models
        random_state: Seed applied to splitting and model fitting
        data_source: Source of the training data, recorded in the model metadata

    Returns:
        dict: Training results including metrics, feature importance, and metadata
    """
    print("Loading training data...")
    df = load_training_data(data_path)
    n_samples = len(df)
    print(f"Loaded {n_samples} samples (data source: {data_source})")

    print("\nPreparing data...")
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, random_state=random_state)
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    print("\nTraining Random Forest Classifier...")
    model = train_random_forest(X_train, y_train, random_state=random_state)

    print("\nEvaluating model...")
    metrics = evaluate_model(model, X_test, y_test)

    metadata = build_model_metadata(metrics, n_samples, random_state, data_source)

    print("\n" + "=" * 72)
    print(f"MODEL EVALUATION RESULTS - measured on {data_source.upper()} DATA")
    print("=" * 72)
    if data_source == 'synthetic':
        print(SYNTHETIC_MODEL_NOTICE)
        print("-" * 72)
    print(format_metric("Accuracy ", metrics['accuracy'], data_source))
    print(format_metric("Precision", metrics['precision'], data_source))
    print(format_metric("Recall   ", metrics['recall'], data_source))
    print(format_metric("F1-Score ", metrics['f1_score'], data_source))
    print(f"\nConfusion Matrix ({data_source} data):")
    cm = metrics['confusion_matrix']
    print(f"           Predicted")
    print(f"           Low  Med  High")
    print(f"Actual Low  {cm[0][0]:3d}  {cm[0][1]:3d}  {cm[0][2]:3d}")
    print(f"      Med  {cm[1][0]:3d}  {cm[1][1]:3d}  {cm[1][2]:3d}")
    print(f"     High  {cm[2][0]:3d}  {cm[2][1]:3d}  {cm[2][2]:3d}")
    print(f"\nClassification Report ({data_source} data):")
    print(metrics['classification_report'])

    # Feature importance. Also synthetic-derived: it reflects the generator's chosen
    # distributions, not a measured property of human typing.
    importance = get_feature_importance(model)
    print(f"Feature Importance (derived from {data_source} data):")
    for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {score:.4f} ({data_source}-derived)")

    print(f"\nModel version: {metadata['model_version']}")

    if save_models:
        print("\nSaving models...")
        save_model(model, scaler, metadata=metadata)

    return {
        'model': model,
        'scaler': scaler,
        'metrics': metrics,
        'feature_importance': importance,
        'metadata': metadata,
        'data_source': data_source,
    }


if __name__ == "__main__":
    # Run the training pipeline
    results = train_and_evaluate()
