"""
Response security headers (F3).

Every response leaves with a small, conservative set of hardening headers. The goal is
defence-in-depth for a tool that handles sensitive keystroke *timing*: reduce the blast
radius of any injected content, forbid framing, and leak no referrer.

Content-Security-Policy note
----------------------------
The policy is now strict: every directive is ``'self'`` or ``'none'``, with no CDN and no
``'unsafe-inline'`` anywhere. That was not true when F3 shipped it — the page then carried
inline ``onclick`` handlers and ``style="width:…"`` attributes and pulled fonts and icons
from two CDNs, so the loose directives were a documented compromise (D-021) rather than an
oversight. F16 removed all four causes, and D-023 closes the compromise.

What this buys: with ``script-src 'self'``, injected markup cannot execute. An attacker who
manages to get a ``<script>alert(1)</script>`` into the page achieves nothing, because the
browser refuses to run script that did not arrive from this origin as a file. That is the
entire point of a CSP, and ``'unsafe-inline'`` gives it back wholesale.

The cost is a real constraint on the frontend: no inline handler, no inline ``<style>``, no
``style="…"`` attribute may be added to ``web/index.html`` again. Anything that needs to
run must live in ``app.js``, and anything that needs to look different must be a class.
``tests/test_offline_assets.py`` enforces both, because the failure mode is silent — a
blocked handler is a button that simply does nothing.
"""

from __future__ import annotations

from flask import Response

#: Sources permitted by the Content-Security-Policy. Every value is ``'self'`` or
#: ``'none'``; there is no host allowlist to keep in step with the page, because the page
#: loads nothing from anywhere else.
#:
#: Kept as a dict so the policy can be asserted directive-by-directive in tests rather
#: than string-matched against one long header.
CSP_DIRECTIVES: dict[str, str] = {
    "default-src": "'self'",
    "base-uri": "'self'",
    "form-action": "'self'",
    "frame-ancestors": "'none'",
    "object-src": "'none'",
    # `data:` is for the inline SVG favicon — an image the page carries, not one it fetches.
    "img-src": "'self' data:",
    "script-src": "'self'",
    "style-src": "'self'",
    "font-src": "'self'",
    "connect-src": "'self'",
}

#: Directives that must never be weakened without a recorded decision. ``'unsafe-inline'``
#: or ``'unsafe-eval'`` in either of these hands script execution back to injected markup,
#: which is the specific attack the policy exists to stop.
STRICT_DIRECTIVES: frozenset[str] = frozenset({"script-src", "style-src"})

CONTENT_SECURITY_POLICY = "; ".join(f"{k} {v}" for k, v in CSP_DIRECTIVES.items())

#: Static headers applied to every response.
_STATIC_HEADERS: dict[str, str] = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    # No geolocation/mic/camera is ever needed; deny them so a compromised page cannot ask.
    "Permissions-Policy": "geolocation=(), microphone=(), camera=(), interest-cohort=()",
}


def apply_security_headers(response: Response, *, is_secure: bool = False) -> Response:
    """
    Attach hardening headers to a response.

    Parameters:
        response: The outgoing response.
        is_secure: Whether the request arrived over HTTPS. HSTS is only meaningful — and
            only safe — on a secure origin, so it is added exclusively in that case. On a
            plain-HTTP loopback dev server it would be pointless and could pin ``https``
            for ``localhost``.

    Returns:
        Response: The same response, with headers set.
    """
    for name, value in _STATIC_HEADERS.items():
        response.headers.setdefault(name, value)

    if is_secure:
        response.headers.setdefault(
            "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
        )

    return response
