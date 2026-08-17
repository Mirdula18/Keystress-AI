"""Utility helper functions for Keystress-AI."""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional


def generate_session_id() -> str:
    """Generate a unique session identifier."""
    timestamp = datetime.now().isoformat()
    return hashlib.sha256(timestamp.encode()).hexdigest()[:16]


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely get a value from a dictionary."""
    return data.get(key, default)


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """Format a datetime object to ISO string."""
    if dt is None:
        dt = datetime.now()
    return dt.isoformat()


def parse_json_safely(json_string: str) -> Optional[Dict[str, Any]]:
    """Safely parse a JSON string."""
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return None
