"""
Privacy-hardening tests (F3): security headers, the request-body cap, and rate limiting.

These own the behaviour the shared fixtures deliberately switch off. ``conftest`` disables
rate limiting for every other test so they are not coupled through the limiter's shared
counter; here we build our own apps that turn each guard on and prove it fires.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from keystress.app import create_app
from keystress.config import Settings
from keystress.core.model import ModelRegistry
from keystress.core.storage import Store
from keystress.extensions import limiter
from keystress.security import (
    CONTENT_SECURITY_POLICY,
    CSP_DIRECTIVES,
    STRICT_DIRECTIVES,
)
from tests.conftest import make_events


def _build_app(registry: ModelRegistry, store: Store, **overrides: Any):
    """Build a test app whose settings differ only in the given fields."""
    application = create_app(
        settings=Settings(**overrides), registry=registry, store=store, load_model=False
    )
    # Consent is not what these tests exercise; disable the gate so predict returns 200.
    application.config.update(TESTING=True, KEYSTRESS_REQUIRE_CONSENT=False)
    return application


class TestSecurityHeaders:
    """Every response carries the hardening headers (F3)."""

    def test_headers_present_on_the_page(self, client) -> None:
        headers = client.get("/").headers
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"
        assert headers["Referrer-Policy"] == "no-referrer"
        assert headers["Cross-Origin-Opener-Policy"] == "same-origin"
        assert headers["Content-Security-Policy"] == CONTENT_SECURITY_POLICY
        assert "camera=()" in headers["Permissions-Policy"]

    def test_headers_present_on_the_api(self, client, events) -> None:
        response = client.post("/api/predict", json={"keystroke_events": events})
        assert response.status_code == 200
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "Content-Security-Policy" in response.headers

    def test_no_hsts_on_plain_http(self, client) -> None:
        # HSTS on a plain-HTTP loopback dev server is pointless and would wrongly pin
        # https for localhost; it is added only on a secure origin.
        assert "Strict-Transport-Security" not in client.get("/").headers

    def test_csp_keeps_the_prediction_path_first_party(self) -> None:
        # The core function must need no third-party call: connect-src is 'self' only.
        assert "connect-src 'self'" in CONTENT_SECURITY_POLICY
        assert "frame-ancestors 'none'" in CONTENT_SECURITY_POLICY


class TestStrictContentSecurityPolicy:
    """
    The policy is strict since F16 (D-023), and must stay that way.

    `'unsafe-inline'` in `script-src` re-permits exactly the injected inline script a CSP
    exists to block, so a regression here would quietly undo the header rather than break
    anything visible. These tests are the alarm.
    """

    @pytest.mark.parametrize("directive", sorted(STRICT_DIRECTIVES))
    def test_no_unsafe_source_in_a_strict_directive(self, directive: str) -> None:
        value = CSP_DIRECTIVES[directive]
        assert "unsafe-inline" not in value, f"{directive} allows inline {directive[:-4]}"
        assert "unsafe-eval" not in value, f"{directive} allows eval"

    def test_no_directive_allows_a_remote_host(self) -> None:
        """
        No directive names an external origin.

        The two CDNs that used to appear here were removed by F16; a re-added font or
        icon host would break offline use and leak the visitor's IP to a third party on
        page load, before they have consented to anything.
        """
        for directive, value in CSP_DIRECTIVES.items():
            assert "//" not in value, f"{directive} allows a remote host: {value}"

    def test_every_fetch_directive_is_self_or_none(self) -> None:
        allowed_sources = {"'self'", "'none'", "data:"}
        for directive, value in CSP_DIRECTIVES.items():
            unexpected = set(value.split()) - allowed_sources
            assert not unexpected, f"{directive} allows {sorted(unexpected)}"

    def test_the_page_actually_complies_with_its_own_policy(self, client) -> None:
        """
        A strict policy the page violates would be worse than a loose one: the header
        would look right in an audit while the app silently lost its buttons.
        """
        body = client.get("/").get_data(as_text=True)
        assert 'style="' not in body
        assert not re.search(r"\son[a-z]+=\"", body)


class TestPayloadCap:
    """An oversized body is rejected with 413 before it is parsed."""

    def test_oversized_body_is_rejected_with_413(self, registry, store) -> None:
        app = _build_app(registry, store, max_content_length=200, rate_limit_enabled=False)
        client = app.test_client()

        oversized = {"keystroke_events": make_events(count=50)}
        response = client.post("/api/predict", json=oversized)

        assert response.status_code == 413
        assert "too large" in response.get_json()["error"].lower()

    def test_body_within_the_cap_is_accepted(self, registry, store, events) -> None:
        app = _build_app(registry, store, max_content_length=1_048_576, rate_limit_enabled=False)
        response = app.test_client().post("/api/predict", json={"keystroke_events": events})
        assert response.status_code == 200

    def test_cap_is_read_from_settings(self, registry, store) -> None:
        app = _build_app(registry, store, max_content_length=4096)
        assert app.config["MAX_CONTENT_LENGTH"] == 4096


class TestRateLimiting:
    """The model endpoint is throttled per client (F3)."""

    def test_exceeding_the_limit_returns_429(self, registry, store, events) -> None:
        app = _build_app(registry, store, rate_limit="2/minute", rate_limit_enabled=True)
        with app.app_context():
            limiter.reset()  # clear any counter left by an earlier enabled run
        client = app.test_client()

        payload = {"keystroke_events": events}
        first = client.post("/api/predict", json=payload)
        second = client.post("/api/predict", json=payload)
        third = client.post("/api/predict", json=payload)

        assert first.status_code == 200
        assert second.status_code == 200
        assert third.status_code == 429
        assert "too many" in third.get_json()["error"].lower()
        assert "Retry-After" in third.headers

    def test_health_endpoint_is_not_throttled(self, registry, store) -> None:
        # Only /api/predict is limited; liveness checks must never be throttled.
        app = _build_app(registry, store, rate_limit="1/minute", rate_limit_enabled=True)
        with app.app_context():
            limiter.reset()
        client = app.test_client()
        for _ in range(5):
            assert client.get("/api/health").status_code == 200

    def test_disabled_limiter_does_not_throttle(self, registry, store, events) -> None:
        app = _build_app(registry, store, rate_limit="1/minute", rate_limit_enabled=False)
        client = app.test_client()
        payload = {"keystroke_events": events}
        for _ in range(4):
            assert client.post("/api/predict", json=payload).status_code == 200
