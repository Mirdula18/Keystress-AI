"""Custom exceptions for Keystress-AI."""


class KeystressError(Exception):
    """Base exception for Keystress-AI."""
    pass


class ValidationError(KeystressError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, field: str = None):
        self.field = field
        super().__init__(message)


class DataNotFoundError(KeystressError):
    """Raised when requested data is not found."""
    pass


class AnalysisError(KeystressError):
    """Raised when analysis fails."""
    
    def __init__(self, message: str, metric: str = None):
        self.metric = metric
        super().__init__(message)


class ConfigurationError(KeystressError):
    """Raised when configuration is invalid."""
    pass


class StorageError(KeystressError):
    """Raised when storage operations fail."""
    pass


class APIError(KeystressError):
    """Raised when API operations fail."""
    
    def __init__(self, message: str, status_code: int = 500):
        self.status_code = status_code
        super().__init__(message)
