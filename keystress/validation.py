"""Data validation utilities for keystroke analysis."""

from typing import Any, Dict, List, Optional, Union


def validate_keystroke_data(data: Dict[str, Any]) -> bool:
    """Validate keystroke data structure."""
    required_fields = ["timestamp", "key", "duration"]
    return all(field in data for field in required_fields)


def validate_timing_data(timings: List[float], min_value: float = 0.0, max_value: float = 10.0) -> bool:
    """Validate timing data within expected range."""
    if not timings:
        return False
    return all(min_value <= t <= max_value for t in timings)


def validate_session_data(session: Dict[str, Any]) -> Dict[str, List[str]]:
    """Validate session data and return any validation errors."""
    errors = {}
    
    if "user_id" not in session:
        errors.setdefault("user_id", []).append("User ID is required")
    
    if "start_time" not in session:
        errors.setdefault("start_time", []).append("Start time is required")
    
    if "keystrokes" in session:
        if not isinstance(session["keystrokes"], list):
            errors.setdefault("keystrokes", []).append("Keystrokes must be a list")
        elif not all(validate_keystroke_data(ks) for ks in session["keystrokes"]):
            errors.setdefault("keystrokes", []).append("Invalid keystroke data found")
    
    return errors


def sanitize_input(value: str, max_length: int = 1000) -> str:
    """Sanitize string input."""
    if not isinstance(value, str):
        return ""
    return value[:max_length].strip()


def validate_metric_range(value: Union[int, float], metric_name: str, 
                          min_val: float = 0.0, max_val: float = 1.0) -> Optional[str]:
    """Validate a metric is within expected range."""
    if not (min_val <= value <= max_val):
        return f"{metric_name} must be between {min_val} and {max_val}"
    return None
