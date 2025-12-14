"""
Synthetic Data Generation Module for Keystress-AI

This module generates synthetic typing behavior data to train
the burnout detection model. Since real labeled burnout data is
unavailable, we create realistic synthetic data with burnout labels.

Burnout Levels:
    0 = Low Burnout (normal, healthy typing patterns)
    1 = Medium Burnout (showing some stress indicators)
    2 = High Burnout (significant stress indicators)
"""

import numpy as np
import pandas as pd
import os


def generate_synthetic_typing_data(n_samples=1000, random_state=42):
    """
    Generate synthetic typing behavior dataset with burnout labels.
    
    The data simulates realistic typing patterns associated with different
    burnout levels based on research insights:
    - Low burnout: Consistent, moderate speed typing
    - Medium burnout: Slightly slower, more variable patterns
    - High burnout: Slow, inconsistent, more corrections
    
    Parameters:
        n_samples (int): Number of samples to generate
        random_state (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: DataFrame with typing features and burnout labels
    """
    np.random.seed(random_state)
    
    # Distribute samples across burnout levels
    n_low = n_samples // 3
    n_medium = n_samples // 3
    n_high = n_samples - n_low - n_medium
    
    data = []
    
    # Generate Low Burnout samples (label = 0)
    # Characteristics: Fast, consistent typing, few corrections
    for _ in range(n_low):
        avg_typing_speed = np.random.normal(5.0, 0.8)  # ~5 keys/sec
        avg_inter_key_delay = np.random.normal(0.2, 0.05)  # ~200ms
        max_pause_duration = np.random.exponential(1.0) + 0.5  # Short pauses
        backspace_ratio = np.random.beta(2, 20)  # Low correction rate (~9%)
        typing_consistency = np.random.normal(0.05, 0.02)  # Very consistent
        
        data.append({
            'avg_typing_speed': max(0.5, avg_typing_speed),
            'avg_inter_key_delay': max(0.05, avg_inter_key_delay),
            'max_pause_duration': max(0.3, max_pause_duration),
            'backspace_ratio': np.clip(backspace_ratio, 0, 0.5),
            'typing_consistency': max(0.01, typing_consistency),
            'burnout_level': 0
        })
    
    # Generate Medium Burnout samples (label = 1)
    # Characteristics: Moderate speed, some variability, moderate corrections
    for _ in range(n_medium):
        avg_typing_speed = np.random.normal(3.5, 1.0)  # ~3.5 keys/sec
        avg_inter_key_delay = np.random.normal(0.35, 0.1)  # ~350ms
        max_pause_duration = np.random.exponential(2.0) + 1.0  # Medium pauses
        backspace_ratio = np.random.beta(3, 15)  # Moderate correction rate (~17%)
        typing_consistency = np.random.normal(0.12, 0.04)  # Some variation
        
        data.append({
            'avg_typing_speed': max(0.5, avg_typing_speed),
            'avg_inter_key_delay': max(0.05, avg_inter_key_delay),
            'max_pause_duration': max(0.5, max_pause_duration),
            'backspace_ratio': np.clip(backspace_ratio, 0, 0.5),
            'typing_consistency': max(0.01, typing_consistency),
            'burnout_level': 1
        })
    
    # Generate High Burnout samples (label = 2)
    # Characteristics: Slow, inconsistent typing, many corrections, long pauses
    for _ in range(n_high):
        avg_typing_speed = np.random.normal(2.0, 0.8)  # ~2 keys/sec
        avg_inter_key_delay = np.random.normal(0.6, 0.15)  # ~600ms
        max_pause_duration = np.random.exponential(3.5) + 2.0  # Long pauses
        backspace_ratio = np.random.beta(4, 10)  # High correction rate (~29%)
        typing_consistency = np.random.normal(0.25, 0.08)  # Very inconsistent
        
        data.append({
            'avg_typing_speed': max(0.5, avg_typing_speed),
            'avg_inter_key_delay': max(0.05, avg_inter_key_delay),
            'max_pause_duration': max(1.0, max_pause_duration),
            'backspace_ratio': np.clip(backspace_ratio, 0, 0.5),
            'typing_consistency': max(0.01, typing_consistency),
            'burnout_level': 2
        })
    
    # Create DataFrame and shuffle
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df


def save_synthetic_data(df, filepath='data/synthetic_typing_data.csv'):
    """
    Save the synthetic dataset to a CSV file.
    
    Parameters:
        df (pd.DataFrame): The synthetic dataset
        filepath (str): Path to save the CSV file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Synthetic dataset saved to {filepath}")
    print(f"Total samples: {len(df)}")
    print(f"Burnout distribution:\n{df['burnout_level'].value_counts().sort_index()}")


if __name__ == "__main__":
    # Generate and save synthetic data
    print("Generating synthetic typing behavior dataset...")
    df = generate_synthetic_typing_data(n_samples=1500)
    save_synthetic_data(df)
    
    # Display sample statistics
    print("\nDataset Statistics:")
    print(df.describe())
