"""
Response security headers (F3).

Every response leaves with a small, conservative set of hardening headers. The goal is
defence-in-depth for a tool that handles sensitive keystroke *timing*: reduce the blast
radius of any injected content, forbid framing, and leak no referrer.

Content-Security-Policy note
----------------------------
The policy below still allows ``'unsafe-inline'`` for scripts and styles, and two CDNs
(Google Fonts, Font Awesome). That is a deliberate, documented compromise, not an
oversight: the current page carries inline ``onclick`` handlers and ``style="width:…"``
attributes, and pulls fonts/icons from a CDN. F16 vendors those assets locally and moves
the handlers into ``app.js``; when it does, this policy tightens to ``'self'`` with no
CDN and no ``'unsafe-inline'``. The prediction path itself already makes no third-party
call, which is what HARD RULE 5 and the F3 acceptance criteria require.
"""

from __future__ import annotations

from flask import Response

#: Sources permitted by the Content-Security-Policy. Kept as a dict so F16 can drop the
#: CDN and inline entries by editing one obvious place.
_CSP_DIRECTIVES: dict[str, str] = {
    "default-src": "'self'",
    "base-uri": "'self'",
    "form-action": "'self'",
    "frame-ancestors": "'none'",
    "object-src": "'none'",
    "img-src": "'self' data:",
    # 'unsafe-inline' scripts: inline onclick handlers in index.html (removed in F16).
    "script-src": "'self' 'unsafe-inline'",
    # 'unsafe-inline' styles: inline style="width:…" bars; CDNs supply fonts/icons (F16).
    "style-src": "'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com",
    "font-src": "'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com",
    "connect-src": "'self'",
}

CONTENT_SECURITY_POLICY = "; ".join(f"{k} {v}" for k, v in _CSP_DIRECTIVES.items())

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
