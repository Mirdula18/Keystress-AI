"""
Feature Engineering Module for Keystress-AI

This module transforms raw typing metadata into meaningful features
for burnout detection. These features are designed to capture
typing behavior patterns indicative of stress and fatigue.

Features computed:
    - avg_typing_speed: Average keys per second
    - avg_inter_key_delay: Average time between keystrokes
    - max_pause_duration: Maximum pause between keystrokes
    - backspace_ratio: Ratio of backspaces to total keys
    - typing_consistency: Standard deviation of inter-key delays
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union


def extract_typing_features(session_data: Dict) -> Dict[str, float]:
    """
    Extract typing behavior features from session metadata.
    
    Parameters:
        session_data (dict): Dictionary containing typing metadata with keys:
            - total_keys: Total number of keys pressed
            - backspace_count: Number of backspace keys pressed
            - duration: Total typing duration in seconds
            - inter_key_delays: List of delays between consecutive keystrokes
            
    Returns:
        dict: Dictionary containing computed features
    """
    total_keys = session_data.get('total_keys', 0)
    backspace_count = session_data.get('backspace_count', 0)
    duration = session_data.get('duration', 0)
    inter_key_delays = session_data.get('inter_key_delays', [])
    
    # Handle edge cases
    if total_keys == 0 or duration == 0:
        return {
            'avg_typing_speed': 0.0,
            'avg_inter_key_delay': 0.0,
            'max_pause_duration': 0.0,
            'backspace_ratio': 0.0,
            'typing_consistency': 0.0
        }
    
    # Calculate average typing speed (keys per second)
    avg_typing_speed = total_keys / duration if duration > 0 else 0.0
    
    # Calculate average inter-key delay
    if len(inter_key_delays) > 0:
        avg_inter_key_delay = np.mean(inter_key_delays)
        max_pause_duration = np.max(inter_key_delays)
        typing_consistency = np.std(inter_key_delays)
    else:
        avg_inter_key_delay = 0.0
        max_pause_duration = 0.0
        typing_consistency = 0.0
    
    # Calculate backspace ratio
    backspace_ratio = backspace_count / total_keys if total_keys > 0 else 0.0
    
    return {
        'avg_typing_speed': float(avg_typing_speed),
        'avg_inter_key_delay': float(avg_inter_key_delay),
        'max_pause_duration': float(max_pause_duration),
        'backspace_ratio': float(backspace_ratio),
        'typing_consistency': float(typing_consistency)
    }


def features_to_dataframe(features: Dict[str, float]) -> pd.DataFrame:
    """
    Convert features dictionary to a pandas DataFrame.
    
    Parameters:
        features (dict): Dictionary of feature values
        
    Returns:
        pd.DataFrame: Single-row DataFrame with features
    """
    return pd.DataFrame([features])


def batch_extract_features(sessions: List[Dict]) -> pd.DataFrame:
    """
    Extract features from multiple typing sessions.
    
    Parameters:
        sessions (list): List of session data dictionaries
        
    Returns:
        pd.DataFrame: DataFrame with features for all sessions
    """
    features_list = [extract_typing_features(session) for session in sessions]
    return pd.DataFrame(features_list)


def normalize_features(df: pd.DataFrame, 
                       feature_columns: List[str] = None) -> pd.DataFrame:
    """
    Normalize feature values using min-max scaling.
    
    Parameters:
        df (pd.DataFrame): DataFrame with features
        feature_columns (list): Columns to normalize (default: all numeric)
        
    Returns:
        pd.DataFrame: DataFrame with normalized features
    """
    if feature_columns is None:
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_normalized = df.copy()
    
    for col in feature_columns:
        min_val = df[col].min()
        max_val = df[col].max()
        
        if max_val - min_val > 0:
            df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df_normalized[col] = 0.0
    
    return df_normalized


def get_feature_summary(df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics for typing features.
    
    Parameters:
        df (pd.DataFrame): DataFrame with typing features
        
    Returns:
        dict: Dictionary with summary statistics
    """
    feature_cols = ['avg_typing_speed', 'avg_inter_key_delay', 
                    'max_pause_duration', 'backspace_ratio', 'typing_consistency']
    
    summary = {}
    for col in feature_cols:
        if col in df.columns:
            summary[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max())
            }
    
    return summary


if __name__ == "__main__":
    # Demo of feature engineering
    print("Feature Engineering Module Demo")
    print("=" * 50)
    
    # Sample session data
    sample_session = {
        'total_keys': 50,
        'backspace_count': 5,
        'duration': 15.0,
        'inter_key_delays': [0.2, 0.3, 0.25, 0.4, 0.2, 0.35, 0.28, 0.22]
    }
    
    features = extract_typing_features(sample_session)
    
    print("\nExtracted Features:")
    for name, value in features.items():
        print(f"  {name}: {value:.4f}")
    
    # Convert to DataFrame
    df = features_to_dataframe(features)
    print("\nFeatures DataFrame:")
    print(df)
