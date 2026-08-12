"""
Flask extension singletons (F3).

The rate limiter lives here rather than in the app factory so that view functions can
reference it with a decorator at import time, while the factory still owns binding it to a
concrete app via :meth:`Limiter.init_app`. This is the documented Flask-Limiter pattern
for an application-factory layout.

Storage is in-memory. That is correct for the local-first dev server this project ships
today; a multi-process production deployment (F14) will point ``RATELIMIT_STORAGE_URI`` at
a shared backend so the limit holds across workers.
"""

from __future__ import annotations

from flask import current_app
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

#: Shared limiter. ``key_func`` buckets by client address — the right unit for a public,
#: unauthenticated endpoint. No ``default_limits``: only ``/api/predict`` is limited, so
#: health checks and static assets are never throttled.
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="memory://",
    headers_enabled=True,
)


def predict_rate_limit() -> str:
    """
    Resolve the ``/api/predict`` limit at request time.

    Returning a callable's value (rather than a literal string on the decorator) lets the
    limit come from configuration, so an operator can tune it without touching code.

    Returns:
        str: A Flask-Limiter limit expression such as ``"60/minute"``.
    """
    return current_app.config.get("KEYSTRESS_RATE_LIMIT", "60/minute")
