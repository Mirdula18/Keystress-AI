"""Performance monitoring utilities for Keystress-AI."""

import time
from contextlib import contextmanager
from typing import Dict, Optional


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, list] = {}
        self.timers: Dict[str, float] = {}
    
    @contextmanager
    def timer(self, name: str):
        """Context manager for timing operations."""
        start = time.time()
        yield
        duration = (time.time() - start) * 1000
        self.record_metric(f"{name}_duration_ms", duration)
    
    def record_metric(self, name: str, value: float) -> None:
        """Record a performance metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_average(self, name: str) -> float:
        """Get average value for a metric."""
        values = self.metrics.get(name, [])
        if not values:
            return 0.0
        return sum(values) / len(values)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all metrics."""
        summary = {}
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    "count": len(values),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "total": sum(values)
                }
        return summary
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.timers.clear()


class RequestTracker:
    """Track API request performance."""
    
    def __init__(self):
        self.requests: Dict[str, list] = {}
    
    def track_request(self, endpoint: str, duration_ms: float, status_code: int) -> None:
        """Track a request."""
        if endpoint not in self.requests:
            self.requests[endpoint] = []
        self.requests[endpoint].append({
            "duration_ms": duration_ms,
            "status_code": status_code,
            "timestamp": time.time()
        })
    
    def get_endpoint_stats(self, endpoint: str) -> Dict[str, float]:
        """Get statistics for an endpoint."""
        requests = self.requests.get(endpoint, [])
        if not requests:
            return {}
        
        durations = [r["duration_ms"] for r in requests]
        success_count = sum(1 for r in requests if 200 <= r["status_code"] < 300)
        
        return {
            "count": len(requests),
            "average_duration": sum(durations) / len(durations),
            "success_rate": success_count / len(requests) if requests else 0
        }
