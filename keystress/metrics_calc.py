"""Metrics calculation utilities for keystroke analysis."""

import math
from typing import List, Optional, Tuple


def calculate_mean(values: List[float]) -> float:
    """Calculate the mean of a list of values."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def calculate_std_dev(values: List[float], mean: Optional[float] = None) -> float:
    """Calculate standard deviation."""
    if len(values) < 2:
        return 0.0
    if mean is None:
        mean = calculate_mean(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def calculate_cv(values: List[float]) -> float:
    """Calculate coefficient of variation."""
    mean = calculate_mean(values)
    if mean == 0:
        return 0.0
    std_dev = calculate_std_dev(values, mean)
    return std_dev / mean


def calculate_percentile(values: List[float], percentile: float) -> float:
    """Calculate a percentile value."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = (percentile / 100) * (len(sorted_values) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return sorted_values[lower]
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def calculate_iqr(values: List[float]) -> Tuple[float, float, float]:
    """Calculate interquartile range (Q1, Q3, IQR)."""
    q1 = calculate_percentile(values, 25)
    q3 = calculate_percentile(values, 75)
    return q1, q3, q3 - q1


def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Normalize a score to 0-1 range."""
    if max_val == min_val:
        return 0.0
    return (score - min_val) / (max_val - min_val)


def calculate_stress_index(hold_times: List[float], flight_times: List[float]) -> float:
    """Calculate a stress index from timing data."""
    if not hold_times or not flight_times:
        return 0.0
    
    hold_cv = calculate_cv(hold_times)
    flight_cv = calculate_cv(flight_times)
    
    combined = (hold_cv + flight_cv) / 2
    return min(max(combined, 0.0), 1.0)
