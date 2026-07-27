"""
Environment-driven configuration for Keystress-AI.

Local-first by default (``docs/CLAUDE.md`` HARD RULE 5): the server binds loopback unless
an operator explicitly opts into a wider bind. Every setting has a safe default, so the
application runs correctly with no environment configured at all.

This is the Phase 0 subset — enough to remove hardcoded values and the `0.0.0.0` default.
The full config surface with `.env.example` and health/readiness wiring arrives with F14.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

#: Repository root, used to resolve default artifact paths regardless of the working
#: directory the process was started from. This replaces the fragile assumption that the
#: app is always launched from the repository root.
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

#: Addresses considered loopback-only.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _env_str(name: str, default: str) -> str:
    """
    Read a string setting from the environment.

    Parameters:
        name: Environment variable name.
        default: Value used when unset or empty.

    Returns:
        str: The configured value.
    """
    value = os.environ.get(name, "").strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    """
    Read an integer setting from the environment.

    Parameters:
        name: Environment variable name.
        default: Value used when unset, empty, or unparseable.

    Returns:
        int: The configured value. An unparseable value falls back to the default rather
        than crashing at import time.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    """
    Read a boolean setting from the environment.

    Parameters:
        name: Environment variable name.
        default: Value used when unset.

    Returns:
        bool: ``True`` for "1", "true", "yes", "on" (case-insensitive).
    """
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path) -> Path:
    """
    Read a filesystem path from the environment.

    Parameters:
        name: Environment variable name.
        default: Value used when unset.

    Returns:
        Path: The configured path.
    """
    raw = os.environ.get(name, "").strip()
    return Path(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    """
    Resolved application settings.

    Attributes:
        host: Bind address. Defaults to loopback (HARD RULE 5).
        port: Bind port.
        debug: Flask debug mode. Off unless explicitly enabled.
        model_path: Path to the serialised classifier.
        scaler_path: Path to the serialised feature scaler.
        metadata_path: Path to the model disclosure metadata sidecar.
        data_path: Path to the training dataset.
        log_level: Root log level for the application logger.
        auto_train: Whether a missing model triggers a synthetic training run at startup.
    """

    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False
    model_path: Path = field(default_factory=lambda: PROJECT_ROOT / "models" / "burnout_model.pkl")
    scaler_path: Path = field(default_factory=lambda: PROJECT_ROOT / "models" / "scaler.pkl")
    metadata_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "models" / "model_metadata.json"
    )
    data_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "synthetic_typing_data.csv"
    )
    log_level: str = "INFO"
    auto_train: bool = True

    @property
    def is_loopback(self) -> bool:
        """Report whether the configured bind address is loopback-only."""
        return self.host in LOOPBACK_HOSTS


def load_settings() -> Settings:
    """
    Build settings from the environment.

    Returns:
        Settings: Resolved configuration with local-first defaults.
    """
    models_dir = PROJECT_ROOT / "models"
    return Settings(
        host=_env_str("KEYSTRESS_HOST", "127.0.0.1"),
        port=_env_int("KEYSTRESS_PORT", 5000),
        # FLASK_DEBUG retained for compatibility with the inherited entrypoint.
        debug=_env_bool("KEYSTRESS_DEBUG") or _env_bool("FLASK_DEBUG"),
        model_path=_env_path("KEYSTRESS_MODEL_PATH", models_dir / "burnout_model.pkl"),
        scaler_path=_env_path("KEYSTRESS_SCALER_PATH", models_dir / "scaler.pkl"),
        metadata_path=_env_path("KEYSTRESS_METADATA_PATH", models_dir / "model_metadata.json"),
        data_path=_env_path(
            "KEYSTRESS_DATA_PATH", PROJECT_ROOT / "data" / "synthetic_typing_data.csv"
        ),
        log_level=_env_str("KEYSTRESS_LOG_LEVEL", "INFO").upper(),
        auto_train=_env_bool("KEYSTRESS_AUTO_TRAIN", default=True),
    )
