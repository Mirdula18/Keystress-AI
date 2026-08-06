"""
Keystress-AI Flask application factory.

A privacy-preserving research prototype examining whether typing rhythm relates to
academic wellbeing. It produces research *indicators*, not assessments or diagnoses.

The served model is trained on synthetic data whose classes were hand-authored by the
generator, so every number it produces is labelled ``data_source: "synthetic"`` and means
only that. See ``docs/CLAUDE.md`` §1.

Structure
---------
This module is a thin entrypoint: it builds the app, wires the model registry, and
registers the API blueprints. Request handling lives in ``keystress.api``, and the
domain logic in ``keystress.core``. There is no module-level mutable model state — the
registry attached to ``app.extensions`` owns the loaded model (F11).

The frontend lives in ``keystress/web/`` as real files (F10) — ``index.html`` plus
``static/styles.css`` and ``static/app.js``. No HTML, CSS, or JavaScript appears in Python
any more. The setup is build-free: Flask serves the directory directly, so editing the
frontend needs no toolchain.
"""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

from .config import Settings, load_settings
from .core.model import ModelRegistry, ModelUnavailableError
from .core.storage import Store
from .extensions import limiter
from .security import apply_security_headers

logger = logging.getLogger(__name__)

#: Extracted frontend. ``index.html`` is rendered as a Jinja template purely so it can use
#: ``url_for('static', ...)``; it contains no template logic beyond those asset URLs.
WEB_ROOT = Path(__file__).resolve().parent / "web"
STATIC_ROOT = WEB_ROOT / "static"


def configure_logging(level: str = "INFO") -> None:
    """
    Configure application logging.

    Replaces the inherited ``print`` calls (F11). Emoji are deliberately absent: they
    raise ``UnicodeEncodeError`` on legacy Windows console codepages, which turned a
    cosmetic banner into a startup crash.

    Parameters:
        level: Log level name; an unrecognised value falls back to INFO.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_model(registry: ModelRegistry, settings: Settings) -> None:
    """
    Load the model, training one from synthetic data if none exists.

    Parameters:
        registry: The registry to populate.
        settings: Resolved settings supplying artifact paths.

    Note:
        A failure here is logged and swallowed rather than raised. The app must still
        start so that ``/api/health`` and ``/readyz`` can report the degraded state —
        HARD RULE 6 asks for a clear message, and a process that refuses to boot cannot
        deliver one.
    """
    try:
        registry.load(settings.model_path, settings.scaler_path, settings.metadata_path)
        return
    except ModelUnavailableError as exc:
        logger.warning("%s", exc)

    if not settings.auto_train:
        logger.error("No model available and auto-training is disabled")
        return

    logger.info("Training a model from synthetic data (this happens once)")
    try:
        from .ml.synthetic import generate_synthetic_typing_data, save_synthetic_data
        from .ml.train import train_and_evaluate

        if not settings.data_path.exists():
            save_synthetic_data(
                generate_synthetic_typing_data(n_samples=1500), settings.data_path
            )

        train_and_evaluate(
            data_path=settings.data_path,
            model_path=settings.model_path,
            scaler_path=settings.scaler_path,
            metadata_path=settings.metadata_path,
        )
        registry.load(settings.model_path, settings.scaler_path, settings.metadata_path)
    except (ModelUnavailableError, FileNotFoundError, ValueError, OSError) as exc:
        logger.error("Could not train a model: %s. Predictions will be unavailable.", exc)


def create_app(settings: Settings | None = None,
               registry: ModelRegistry | None = None,
               store: Store | None = None,
               load_model: bool = True) -> Flask:
    """
    Build the Flask application.

    Parameters:
        settings: Configuration; read from the environment when omitted.
        registry: Model registry to use. Injectable so tests can supply a fixture model
            without touching disk — the thing the inherited module globals made impossible.
        store: Consent/donation store (F2). Injectable so tests can point at a temporary
            database instead of the real one.
        load_model: Whether to load or train a model at startup.

    Returns:
        Flask: The configured application.
    """
    settings = settings if settings is not None else load_settings()
    registry = registry if registry is not None else ModelRegistry()
    store = store if store is not None else Store(settings.store_path)

    app = Flask(
        __name__,
        template_folder=str(WEB_ROOT),
        static_folder=str(STATIC_ROOT),
        static_url_path="/static",
    )
    app.config["KEYSTRESS_SETTINGS"] = settings
    app.extensions["keystress_registry"] = registry
    app.extensions["keystress_store"] = store
    app.config["KEYSTRESS_REQUIRE_CONSENT"] = settings.require_consent

    # F3 privacy hardening. The body cap rejects an oversized payload with 413 before it
    # is parsed; the limiter throttles abuse of the model endpoint. Both read from config
    # so they can be tuned or, for tests, switched off without code changes.
    app.config["MAX_CONTENT_LENGTH"] = settings.max_content_length
    app.config["KEYSTRESS_RATE_LIMIT"] = settings.rate_limit
    app.config["RATELIMIT_ENABLED"] = settings.rate_limit_enabled
    limiter.init_app(app)

    from .api.consent import bp as consent_bp
    from .api.health import bp as health_bp
    from .api.predict import bp as predict_bp

    app.register_blueprint(predict_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(consent_bp)

    @app.route("/")
    def index() -> str:
        """Serve the single-page application from ``web/index.html``."""
        return render_template("index.html")

    @app.after_request
    def _security_headers(response: Response) -> Response:
        """Harden every response (F3)."""
        return apply_security_headers(response, is_secure=request.is_secure)

    @app.errorhandler(413)
    def _payload_too_large(_exc: Exception) -> tuple[Response, int]:
        """Return the oversized-body rejection as JSON, matching the API error shape."""
        return jsonify({"error": "Request body too large."}), 413

    @app.errorhandler(429)
    def _rate_limited(_exc: Exception) -> tuple[Response, int]:
        """Return the rate-limit rejection as JSON. Flask-Limiter sets ``Retry-After``."""
        return jsonify({"error": "Too many requests; please slow down."}), 429

    if load_model and not registry.is_loaded:
        ensure_model(registry, settings)

    return app


def main() -> int:
    """
    Run the development server.

    Returns:
        int: Process exit code.
    """
    settings = load_settings()
    configure_logging(settings.log_level)

    logger.info("Keystress-AI: typing-dynamics research prototype")
    logger.info(
        "Research indicators only - not a diagnostic tool. The shipped model is "
        "trained on synthetic data; no real-world performance has been established."
    )

    app = create_app(settings)

    if not settings.is_loopback:
        # HARD RULE 5: local-first. A wider bind is allowed but never silent.
        logger.warning(
            "Bound to %s, which may be reachable from your network. Raw keystroke "
            "timing is sensitive; prefer 127.0.0.1.", settings.host,
        )

    logger.info("Serving on http://%s:%d", settings.host, settings.port)
    app.run(debug=settings.debug, host=settings.host, port=settings.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
