"""
Prediction Module for Keystress-AI

This module provides functions to predict burnout levels
from typing behavior features using the trained model.

Prediction Output:
    0 / "Low Burnout" - Healthy typing patterns
    1 / "Medium Burnout" - Some stress indicators
    2 / "High Burnout" - Significant stress indicators
"""

import os
import numpy as np
from typing import Dict, Tuple

# Import local modules
from .train_model import FEATURE_COLUMNS


# Burnout level labels
BURNOUT_LABELS = {
    0: "Low Burnout",
    1: "Medium Burnout",
    2: "High Burnout"
}

# Detailed descriptions for each level
BURNOUT_DESCRIPTIONS = {
    0: "Your typing patterns indicate healthy, consistent behavior. "
       "Keep up the good work and maintain your current balance!",
    1: "Your typing shows some signs of stress or fatigue. "
       "Consider taking short breaks and practicing self-care.",
    2: "Your typing patterns suggest you may be experiencing significant stress. "
       "We recommend taking a break, talking to someone, or seeking support."
}


def load_trained_model(model_path: str = 'models/burnout_model.pkl',
                       scaler_path: str = 'models/scaler.pkl'):
    """
    Load the trained model and scaler.
    
    Parameters:
        model_path: Path to the saved model
        scaler_path: Path to the saved scaler
        
    Returns:
        tuple: (model, scaler)
    """
    import joblib
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(
            "Trained model not found. Please run train_model.py first."
        )
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler


def predict_burnout(typing_features: Dict[str, float], 
                    model=None, scaler=None) -> Tuple[int, str, str, float]:
    """
    Predict burnout level from typing features.
    
    Parameters:
        typing_features (dict): Dictionary containing:
            - avg_typing_speed: Average keys per second
            - avg_inter_key_delay: Average time between keystrokes
            - max_pause_duration: Maximum pause between keystrokes
            - backspace_ratio: Ratio of backspaces to total keys
            - typing_consistency: Standard deviation of inter-key delays
        model: Pre-loaded model (optional)
        scaler: Pre-loaded scaler (optional)
        
    Returns:
        tuple: (level_number, level_label, description, confidence)
    """
    # Load model if not provided
    if model is None or scaler is None:
        model, scaler = load_trained_model()
    
    # Prepare feature vector in correct order
    features = np.array([[
        typing_features.get('avg_typing_speed', 0),
        typing_features.get('avg_inter_key_delay', 0),
        typing_features.get('max_pause_duration', 0),
        typing_features.get('backspace_ratio', 0),
        typing_features.get('typing_consistency', 0)
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    confidence = float(np.max(probabilities))
    
    # Get label and description
    label = BURNOUT_LABELS.get(prediction, "Unknown")
    description = BURNOUT_DESCRIPTIONS.get(prediction, "")
    
    return int(prediction), label, description, confidence


def get_prediction_details(typing_features: Dict[str, float],
                          model=None, scaler=None) -> Dict:
    """
    Get detailed prediction results including probabilities for each class.
    
    Parameters:
        typing_features: Dictionary of typing features
        model: Pre-loaded model (optional)
        scaler: Pre-loaded scaler (optional)
        
    Returns:
        dict: Detailed prediction results
    """
    if model is None or scaler is None:
        model, scaler = load_trained_model()
    
    # Prepare feature vector
    features = np.array([[
        typing_features.get('avg_typing_speed', 0),
        typing_features.get('avg_inter_key_delay', 0),
        typing_features.get('max_pause_duration', 0),
        typing_features.get('backspace_ratio', 0),
        typing_features.get('typing_consistency', 0)
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict with probabilities
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    return {
        'prediction': int(prediction),
        'label': BURNOUT_LABELS.get(prediction, "Unknown"),
        'description': BURNOUT_DESCRIPTIONS.get(prediction, ""),
        'confidence': float(np.max(probabilities)),
        'probabilities': {
            'Low Burnout': float(probabilities[0]),
            'Medium Burnout': float(probabilities[1]),
            'High Burnout': float(probabilities[2])
        },
        'features': typing_features
    }


def format_prediction_output(prediction_result: Dict) -> str:
    """
    Format prediction results for display.
    
    Parameters:
        prediction_result: Dictionary from get_prediction_details
        
    Returns:
        str: Formatted output string
    """
    output = []
    output.append("=" * 50)
    output.append("BURNOUT DETECTION RESULT")
    output.append("=" * 50)
    output.append(f"\n📊 Risk Level: {prediction_result['label']}")
    output.append(f"🎯 Confidence: {prediction_result['confidence']:.1%}")
    output.append(f"\n💡 {prediction_result['description']}")
    output.append("\n📈 Probability Breakdown:")
    
    for level, prob in prediction_result['probabilities'].items():
        bar = "█" * int(prob * 20)
        output.append(f"  {level}: {bar} {prob:.1%}")
    
    output.append("\n📝 Your Typing Features:")
    features = prediction_result['features']
    output.append(f"  • Typing Speed: {features.get('avg_typing_speed', 0):.2f} keys/sec")
    output.append(f"  • Avg Key Delay: {features.get('avg_inter_key_delay', 0):.3f} sec")
    output.append(f"  • Max Pause: {features.get('max_pause_duration', 0):.2f} sec")
    output.append(f"  • Correction Rate: {features.get('backspace_ratio', 0):.1%}")
    output.append(f"  • Consistency Score: {features.get('typing_consistency', 0):.3f}")
    
    return "\n".join(output)


if __name__ == "__main__":
    # Demo prediction
    print("Prediction Module Demo")
    print("=" * 50)
    
    # Sample typing features (simulating medium burnout)
    sample_features = {
        'avg_typing_speed': 3.5,
        'avg_inter_key_delay': 0.35,
        'max_pause_duration': 2.5,
        'backspace_ratio': 0.15,
        'typing_consistency': 0.12
    }
    
    try:
        result = get_prediction_details(sample_features)
        print(format_prediction_output(result))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first by running: python -m src.train_model")
