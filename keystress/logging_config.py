"""Logging configuration for Keystress-AI."""

import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger("keystress")
    
    if not logger.handlers:
        logger.setLevel(getattr(logging, level.upper()))
        
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(f"keystress.{name}")


class PerformanceLogger:
    """Logger for performance metrics."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger("performance")
    
    def log_duration(self, operation: str, duration_ms: float) -> None:
        """Log operation duration."""
        self.logger.info(f"{operation} completed in {duration_ms:.2f}ms")
    
    def log_metric(self, metric_name: str, value: float) -> None:
        """Log a metric value."""
        self.logger.debug(f"Metric {metric_name}: {value:.4f}")
