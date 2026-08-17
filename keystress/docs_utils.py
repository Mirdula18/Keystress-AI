"""Documentation utilities for Keystress-AI."""

import inspect
from typing import Any, Callable, Dict, Optional


def get_function_signature(func: Callable) -> str:
    """Get the signature of a function."""
    sig = inspect.signature(func)
    return str(sig)


def get_docstring_summary(func: Callable) -> str:
    """Get a summary from a function's docstring."""
    docstring = inspect.getdoc(func)
    if docstring:
        first_line = docstring.strip().split("\n")[0]
        return first_line
    return "No description available"


def generate_api_docs(endpoints: Dict[str, Dict[str, Any]]) -> str:
    """Generate API documentation from endpoint definitions."""
    lines = ["# API Documentation\n"]
    
    for endpoint, info in endpoints.items():
        lines.append(f"## {endpoint}")
        lines.append(f"**Method:** {info.get('method', 'GET')}")
        lines.append(f"**Description:** {info.get('description', 'No description')}")
        
        if "parameters" in info:
            lines.append("\n**Parameters:**")
            for param in info["parameters"]:
                lines.append(f"- `{param['name']}` ({param['type']}): {param['description']}")
        
        lines.append("\n---\n")
    
    return "\n".join(lines)


def format_metric_report(metrics: Dict[str, Any]) -> str:
    """Format metrics into a readable report."""
    lines = ["# Metrics Report\n"]
    
    for name, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"- **{name}:** {value:.4f}")
        else:
            lines.append(f"- **{name}:** {value}")
    
    return "\n".join(lines)


def create_changelog_entry(version: str, changes: list) -> str:
    """Create a changelog entry."""
    lines = [f"## [{version}]\n"]
    lines.append("### Changes")
    for change in changes:
        lines.append(f"- {change}")
    return "\n".join(lines)


def validate_docstring(func: Callable) -> Dict[str, bool]:
    """Validate that a function has proper documentation."""
    docstring = inspect.getdoc(func)
    sig = inspect.signature(func)
    
    return {
        "has_docstring": docstring is not None,
        "has_parameters": len(sig.parameters) > 0,
        "has_return_annotation": sig.return_annotation != inspect.Parameter.empty
    }
