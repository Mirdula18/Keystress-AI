"""
Shared pytest fixtures.

The key fixture here is :func:`model_bundle`, which builds a small model in memory. It
exists so that API and inference tests never depend on a trained artifact being present on
disk — the thing the inherited module-level globals made impossible (D-009).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import keystress.app
from keystress.app import create_app
from keystress.config import PROJECT_ROOT
from keystress.core.model import ModelBundle, ModelRegistry
from keystress.core.storage import Store
from keystress.ml.synthetic import generate_synthetic_typing_data
from keystress.ml.train import FEATURE_COLUMNS

#: Deterministic seed for every fixture, so failures are reproducible.
TEST_SEED = 1234


@pytest.fixture(autouse=True)
def forbid_the_real_store(monkeypatch) -> None:
    """
    Fail any test that would open the developer's real consent database.

    :func:`keystress.app.create_app` builds a :class:`Store` at ``settings.store_path``
    when none is injected, and that default resolves inside the repository. A test that
    forgets to inject one would quietly write consent rows into the working tree — the
    precise kind of unintended persistence F2 exists to prevent, and the kind that is
    invisible until someone commits it. Making the omission a loud failure is cheaper
    than noticing the stray file later.
    """
    real_data_dir = (PROJECT_ROOT / "data").resolve()
    real_store = keystress.app.Store

    def guarded(path, *args: Any, **kwargs: Any) -> Store:
        if Path(path).resolve().is_relative_to(real_data_dir):
            raise AssertionError(
                f"test opened the real consent database at {path!s}. "
                "Pass the `store` fixture to create_app(store=...)."
            )
        return real_store(path, *args, **kwargs)

    monkeypatch.setattr(keystress.app, "Store", guarded)


@pytest.fixture(scope="session")
def synthetic_frame():
    """A small synthetic dataset, generated once per session."""
    return generate_synthetic_typing_data(n_samples=300, random_state=TEST_SEED)


@pytest.fixture(scope="session")
def model_bundle(synthetic_frame) -> ModelBundle:
    """
    A trained, in-memory model bundle.

    Deliberately built in memory rather than loaded from ``models/``: tests must not
    depend on a prior training run, and must not be affected by whatever artifact happens
    to be sitting on the developer's disk.

    Returns:
        ModelBundle: A bundle whose metadata declares its synthetic origin.
    """
    X = synthetic_frame[FEATURE_COLUMNS].to_numpy()
    y = synthetic_frame["burnout_level"].to_numpy()

    scaler = StandardScaler().fit(X)
    estimator = RandomForestClassifier(
        n_estimators=25, max_depth=6, random_state=TEST_SEED
    ).fit(scaler.transform(X), y)

    return ModelBundle(
        estimator=estimator,
        scaler=scaler,
        metadata={
            "model_version": "rf-v1-synthetic-test",
            "data_source": "synthetic",
            "feature_set": "v1",
        },
    )


@pytest.fixture
def registry(model_bundle: ModelBundle) -> ModelRegistry:
    """A registry preloaded with the fixture bundle."""
    reg = ModelRegistry()
    reg.set(model_bundle)
    return reg


@pytest.fixture
def store(tmp_path) -> Store:
    """A store backed by a temp database, so tests never touch the real one."""
    return Store(tmp_path / "test.db")


@pytest.fixture
def app(registry: ModelRegistry, store: Store):
    """A Flask app wired to the fixture model and a temp store, never touching disk."""
    application = create_app(registry=registry, store=store, load_model=False)
    # Rate limiting shares an in-process counter, so leaving it on would couple otherwise
    # independent tests through a global. Consent enforcement is likewise off here so the
    # many prediction tests need not each mint a token; both are exercised deliberately in
    # their own modules (test_security.py, test_consent_api.py).
    application.config.update(
        TESTING=True, RATELIMIT_ENABLED=False, KEYSTRESS_REQUIRE_CONSENT=False
    )
    return application


@pytest.fixture
def client(app):
    """A test client for the fixture app."""
    return app.test_client()


@pytest.fixture
def empty_app(tmp_path):
    """An app with no model loaded, for testing graceful degradation."""
    application = create_app(
        registry=ModelRegistry(), store=Store(tmp_path / "empty.db"), load_model=False
    )
    application.config.update(
        TESTING=True, RATELIMIT_ENABLED=False, KEYSTRESS_REQUIRE_CONSENT=False
    )
    return application


@pytest.fixture
def empty_client(empty_app):
    """A test client for the app with no model."""
    return empty_app.test_client()


def make_events(count: int = 40, interval: float = 0.25,
                backspace_every: int = 0) -> list[dict[str, Any]]:
    """
    Build a well-formed keystroke event list.

    Parameters:
        count: Number of events.
        interval: Seconds between events.
        backspace_every: Mark every Nth event as a correction; 0 for none.

    Returns:
        list[dict]: Events carrying only ``timestamp`` and ``is_backspace``.
    """
    return [
        {
            "timestamp": round(i * interval, 6),
            "is_backspace": bool(backspace_every and i % backspace_every == 0),
        }
        for i in range(count)
    ]


def make_varied_events(count: int = 60, seed: int = TEST_SEED) -> list[dict[str, Any]]:
    """
    Build events with realistic jitter and occasional long pauses.

    Parameters:
        count: Number of events.
        seed: Random seed.

    Returns:
        list[dict]: Events with varied inter-key delays.
    """
    rng = np.random.default_rng(seed)
    timestamp = 0.0
    events: list[dict[str, Any]] = []
    for i in range(count):
        gap = float(rng.gamma(2.0, 0.12)) + (1.5 if i and i % 17 == 0 else 0.0)
        timestamp += gap
        events.append({
            "timestamp": round(timestamp, 6),
            "is_backspace": bool(i % 13 == 0),
        })
    return events


@pytest.fixture
def events() -> list[dict[str, Any]]:
    """A standard well-formed event list."""
    return make_events()


def page_bundle(client) -> str:
    """
    Fetch `/` together with every same-origin stylesheet and script it links.

    This is what a browser actually ends up with, and it is deliberately indifferent to
    *where* the CSS and JS are stored. Before the F10 extraction everything is inline, so
    the bundle is just the page; afterwards the same content arrives over separate
    requests and the bundle is identical in substance.

    Characterization tests assert against this rather than against the raw `/` body,
    because a test that fails purely because bytes moved between files would block the
    very refactor it exists to protect — while still catching content that went missing.

    Parameters:
        client: Flask test client.

    Returns:
        str: The page source concatenated with all linked local assets.
    """
    body = client.get("/").get_data(as_text=True)
    parts = [body]

    hrefs = re.findall(r'<link[^>]+href="([^"]+)"', body)
    srcs = re.findall(r'<script[^>]+src="([^"]+)"', body)

    for url in hrefs + srcs:
        # Third-party URLs and data: URIs are not servable same-origin assets: the
        # former are skipped until F16 removes them, and the latter are inline content
        # (e.g. the favicon) that a browser renders directly.
        if url.startswith(("http://", "https://", "//", "data:")):
            continue
        response = client.get(url)
        assert response.status_code == 200, f"linked asset {url} returned {response.status_code}"
        parts.append(response.get_data(as_text=True))

    return "\n".join(parts)
