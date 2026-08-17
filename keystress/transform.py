"""Data transformation utilities for keystroke analysis."""

from datetime import datetime
from typing import Any, Dict, List


def timestamps_to_durations(timestamps: List[float]) -> List[float]:
    """Convert timestamps to durations between events."""
    if len(timestamps) < 2:
        return []
    return [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]


def raw_to_features(raw_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Transform raw keystroke data to feature vectors."""
    if not raw_data:
        return {}
    
    hold_times = [d.get("hold_time", 0) for d in raw_data if "hold_time" in d]
    flight_times = [d.get("flight_time", 0) for d in raw_data if "flight_time" in d]
    
    return {
        "hold_times": hold_times,
        "flight_times": flight_times,
        "count": len(raw_data),
        "total_duration": sum(hold_times) + sum(flight_times)
    }


def normalize_timestamps(timestamps: List[float], reference: float = None) -> List[float]:
    """Normalize timestamps relative to the first timestamp."""
    if not timestamps:
        return []
    if reference is None:
        reference = timestamps[0]
    return [t - reference for t in timestamps]


def merge_keystroke_sessions(sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple keystroke sessions."""
    if not sessions:
        return {}
    
    merged = {
        "user_id": sessions[0].get("user_id"),
        "start_time": sessions[0].get("start_time"),
        "keystrokes": []
    }
    
    for session in sessions:
        merged["keystrokes"].extend(session.get("keystrokes", []))
    
    merged["keystroke_count"] = len(merged["keystrokes"])
    return merged


def create_time_windows(data: List[Dict[str, Any]], window_size_ms: float = 1000.0) -> List[List[Dict]]:
    """Create time windows from timestamped data."""
    if not data:
        return []
    
    windows = []
    current_window = [data[0]]
    window_start = data[0].get("timestamp", 0)
    
    for item in data[1:]:
        item_time = item.get("timestamp", 0)
        if item_time - window_start <= window_size_ms:
            current_window.append(item)
        else:
            windows.append(current_window)
            current_window = [item]
            window_start = item_time
    
    if current_window:
        windows.append(current_window)
    
    return windows
