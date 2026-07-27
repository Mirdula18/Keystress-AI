"""
Model loading and lifecycle (F11).

Replaces the module-level mutable ``model``/``scaler`` globals that the inherited
application mutated from ``load_models()`` and read from two request handlers. That
pattern made the app untestable (no way to inject a model), racy (mutation after import),
and silently wrong when loading failed partway.

The replacement is an immutable :class:`ModelBundle` plus a small explicit registry. The
loaded bundle is never mutated: reloading swaps in a whole new bundle atomically, so a
request either sees a fully loaded model or none at all — never a half-populated pair.

Graceful degradation (``docs/CLAUDE.md`` HARD RULE 6): a missing or unreadable model
yields :class:`ModelUnavailableError`, which callers turn into a clear message. It never
produces a fake result and never crashes the process at import time.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .disclosure import FEATURE_SET_VERSION, SHIPPED_DATA_SOURCE, SYNTHETIC_MODEL_NOTICE

logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """
    Raised when no usable model is available.

    Carries a message safe to show a user: it explains the situation without leaking
    filesystem paths or stack detail.
    """


@dataclass(frozen=True)
class ModelBundle:
    """
    An immutable, fully loaded model with the metadata describing what it is.

    Frozen on purpose. The estimator and scaler always travel together with the metadata
    that says what data trained them, so a prediction can never be served without knowing
    its data source — the failure mode F1 exists to prevent.

    Attributes:
        estimator: Fitted classifier exposing ``predict`` and ``predict_proba``.
        scaler: Fitted feature scaler exposing ``transform``.
        metadata: Disclosure metadata (model version, data source, feature set).
    """

    estimator: Any
    scaler: Any
    metadata: dict

    @property
    def model_version(self) -> str:
        """Version identifier of the loaded model."""
        return str(self.metadata.get("model_version", "unknown"))

    @property
    def data_source(self) -> str:
        """Data source the model was trained on."""
        return str(self.metadata.get("data_source", SHIPPED_DATA_SOURCE))

    @property
    def feature_set(self) -> str:
        """Feature-set version the model expects."""
        return str(self.metadata.get("feature_set", FEATURE_SET_VERSION))

    def describe(self) -> str:
        """
        Return a one-line human description including the data source.

        Returns:
            str: e.g. ``"rf-v1-synthetic-g2-s42-n1500 (trained on synthetic data)"``.
        """
        return f"{self.model_version} (trained on {self.data_source} data)"


def default_metadata() -> dict:
    """
    Build metadata for a model that arrived without a sidecar.

    An unlabelled model is reported as ``"unknown"`` rather than being given a flattering
    default: not knowing a model's provenance is exactly the condition the disclosure
    contract exists to surface.

    Returns:
        dict: Minimal metadata with an unknown version.
    """
    return {
        "model_version": "unknown",
        "data_source": SHIPPED_DATA_SOURCE,
        "feature_set": FEATURE_SET_VERSION,
        "metrics_caveat": SYNTHETIC_MODEL_NOTICE,
    }


def read_metadata(metadata_path: Path) -> dict:
    """
    Read the model metadata sidecar.

    Parameters:
        metadata_path: Path to the JSON sidecar.

    Returns:
        dict: Metadata, falling back to :func:`default_metadata` when the file is absent
        or unreadable. A corrupt sidecar degrades to "unknown provenance" rather than
        preventing the model from serving — but it is logged as a warning, because a model
        whose provenance cannot be read is a real problem.
    """
    if not metadata_path.exists():
        logger.warning(
            "No model metadata at %s; model provenance will be reported as unknown",
            metadata_path,
        )
        return default_metadata()

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read model metadata at %s: %s", metadata_path, exc)
        return default_metadata()

    if not isinstance(metadata, dict):
        logger.warning("Model metadata at %s is not a JSON object", metadata_path)
        return default_metadata()

    metadata.setdefault("model_version", "unknown")
    metadata.setdefault("data_source", SHIPPED_DATA_SOURCE)
    metadata.setdefault("feature_set", FEATURE_SET_VERSION)
    return metadata


def load_bundle(model_path: Path, scaler_path: Path, metadata_path: Path) -> ModelBundle:
    """
    Load a model bundle from disk.

    Parameters:
        model_path: Path to the serialised classifier.
        scaler_path: Path to the serialised scaler.
        metadata_path: Path to the metadata sidecar.

    Returns:
        ModelBundle: A fully loaded, immutable bundle.

    Raises:
        ModelUnavailableError: If either artifact is missing or cannot be deserialised.
    """
    import joblib

    missing = [str(p) for p in (model_path, scaler_path) if not Path(p).exists()]
    if missing:
        logger.error("Model artifacts missing: %s", ", ".join(missing))
        raise ModelUnavailableError(
            "No trained model is available. Run `keystress-train` "
            "(or `python -m keystress.ml.train`) to build one."
        )

    try:
        estimator = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    except Exception as exc:  # noqa: BLE001 - deliberate; see below
        # Deliberately broad, and not a bare `except`.
        #
        # Unpickling a corrupt or truncated file raises essentially anything: the
        # narrower `(OSError, ValueError, EOFError, AttributeError, ImportError)` tuple
        # tried first here let a real `IndexError` from `pickle.pop_mark` escape as a raw
        # traceback, which `tests/test_core.py::test_corrupt_artifact_raises_clean_error`
        # caught. Enumerating pickle's failure modes is a losing game; at a
        # deserialisation boundary the correct rule is that *no* artifact problem may
        # escape as a stack trace (HARD RULE 6). The original is chained and logged, so
        # nothing is hidden from an operator.
        logger.error("Failed to deserialise model artifacts: %s: %s",
                     type(exc).__name__, exc)
        raise ModelUnavailableError(
            "The trained model could not be loaded; it may be corrupt or built by an "
            "incompatible scikit-learn version. Retrain with `keystress-train`."
        ) from exc

    metadata = read_metadata(Path(metadata_path))
    bundle = ModelBundle(estimator=estimator, scaler=scaler, metadata=metadata)
    logger.info("Loaded model %s", bundle.describe())
    return bundle


class ModelRegistry:
    """
    Holds the process-wide model bundle.

    A registry rather than a bare global: the bundle is swapped atomically under a lock,
    reads return an immutable object, and tests can inject a bundle without patching
    module attributes. This is the "loader/singleton" F11 requires — the single instance
    is a deliberate, inspectable object rather than an implicit module global.
    """

    def __init__(self) -> None:
        """Create an empty registry."""
        self._bundle: ModelBundle | None = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """Report whether a bundle is currently available."""
        return self._bundle is not None

    def get(self) -> ModelBundle:
        """
        Return the loaded bundle.

        Returns:
            ModelBundle: The current bundle.

        Raises:
            ModelUnavailableError: If nothing has been loaded.
        """
        bundle = self._bundle
        if bundle is None:
            raise ModelUnavailableError(
                "No model is loaded. The service cannot produce a prediction right now."
            )
        return bundle

    def get_or_none(self) -> ModelBundle | None:
        """
        Return the loaded bundle, or ``None`` when unavailable.

        Useful for health reporting, which must describe an unloaded state rather than
        raise on it.

        Returns:
            Optional[ModelBundle]: The bundle, or ``None``.
        """
        return self._bundle

    def set(self, bundle: ModelBundle) -> None:
        """
        Install a bundle, replacing any existing one atomically.

        Parameters:
            bundle: The bundle to install.
        """
        with self._lock:
            self._bundle = bundle

    def load(self, model_path: Path, scaler_path: Path, metadata_path: Path) -> ModelBundle:
        """
        Load a bundle from disk and install it.

        The bundle is built completely before being installed, so a failed load leaves any
        previously working model in place rather than tearing it down.

        Parameters:
            model_path: Path to the serialised classifier.
            scaler_path: Path to the serialised scaler.
            metadata_path: Path to the metadata sidecar.

        Returns:
            ModelBundle: The newly installed bundle.

        Raises:
            ModelUnavailableError: If loading fails.
        """
        bundle = load_bundle(model_path, scaler_path, metadata_path)
        self.set(bundle)
        return bundle

    def clear(self) -> None:
        """Unload the current bundle. Primarily for tests."""
        with self._lock:
            self._bundle = None


#: Process-wide registry. Application code reaches the model through this object or, in
#: request handlers, through the registry stored on the Flask app — never through a
#: mutable module-level `model` variable.
registry = ModelRegistry()
