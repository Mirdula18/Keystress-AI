"""API response helpers for Keystress-AI."""

from typing import Any, Dict, Optional


def success_response(data: Any, message: str = "Success") -> Dict[str, Any]:
    """Create a success response."""
    return {
        "status": "success",
        "message": message,
        "data": data
    }


def error_response(message: str, status_code: int = 400, details: Optional[Dict] = None) -> Dict[str, Any]:
    """Create an error response."""
    response = {
        "status": "error",
        "message": message,
        "status_code": status_code
    }
    if details:
        response["details"] = details
    return response


def paginated_response(data: list, total: int, page: int, page_size: int) -> Dict[str, Any]:
    """Create a paginated response."""
    return {
        "status": "success",
        "data": data,
        "pagination": {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size
        }
    }


def validation_error_response(errors: Dict[str, list]) -> Dict[str, Any]:
    """Create a validation error response."""
    return {
        "status": "error",
        "message": "Validation failed",
        "status_code": 422,
        "errors": errors
    }


def health_check_response(status: str = "healthy", version: str = "1.0.0") -> Dict[str, Any]:
    """Create a health check response."""
    return {
        "status": status,
        "version": version,
        "service": "keystress-api"
    }
