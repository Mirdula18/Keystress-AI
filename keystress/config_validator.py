"""Configuration validation for Keystress-AI."""

from typing import Any, Dict, List, Optional


DEFAULT_CONFIG = {
    "analysis": {
        "min_samples": 10,
        "max_samples": 1000,
        "window_size_ms": 5000,
        "threshold": 0.5
    },
    "privacy": {
        "anonymize": True,
        "retention_days": 30,
        "min_aggregation": 5
    },
    "api": {
        "rate_limit": 100,
        "timeout": 30,
        "max_payload_mb": 10
    }
}


def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate configuration and return list of errors."""
    errors = []
    
    analysis = config.get("analysis", {})
    if "min_samples" in analysis and "max_samples" in analysis:
        if analysis["min_samples"] > analysis["max_samples"]:
            errors.append("min_samples cannot be greater than max_samples")
    
    privacy = config.get("privacy", {})
    if "retention_days" in privacy:
        if privacy["retention_days"] < 1:
            errors.append("retention_days must be at least 1")
    
    api = config.get("api", {})
    if "rate_limit" in api:
        if api["rate_limit"] < 1:
            errors.append("rate_limit must be positive")
    
    return errors


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return DEFAULT_CONFIG.copy()


def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge configuration with override."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Get a nested configuration value using dot notation."""
    keys = key_path.split(".")
    current = config
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current
