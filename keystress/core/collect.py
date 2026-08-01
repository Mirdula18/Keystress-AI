"""
Typing Data Collection for Keystress-AI — **the privacy boundary**.

This module is where the project's central promise is kept or broken. It converts
keystroke events into session metadata, and it must never let anything that could
reconstruct typed text through.

What may pass
-------------
Per event: a timestamp, and a boolean saying whether the key was a correction.
Nothing else. The boolean is derived in the browser (``event.key === 'Backspace'``), so
key identity is discarded before the data ever leaves the keyboard.

What must never pass (``docs/ARCHITECTURE.md`` §4.1)
----------------------------------------------------
``key``, ``char``, ``code``, ``which``, ``keyCode``, target element value, text-length
deltas that reveal content, clipboard contents, or focused-element identity.

:data:`ALLOWED_EVENT_FIELDS` is the allowlist, and :func:`process_keystroke_data` reads
*only* those fields — an unknown field is dropped, never stored, never echoed back. The
guarantee is enforced here on the server rather than assumed from client behaviour,
because a modified client can send anything.

An accepted, bounded disclosure: ``total_keys`` reveals roughly how many keys were pressed.
That is inherent to timing analysis and cannot reconstruct content, and it is documented
in ``docs/AUDIT.md`` §6 rather than left implicit.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: The complete set of fields read from an incoming keystroke event. Adding a field here
#: is a privacy-boundary change and requires re-reading HARD RULE 1.
ALLOWED_EVENT_FIELDS = frozenset({"timestamp", "is_backspace"})

#: Fields that must never appear in a session record or a response. Asserted by
#: ``tests/test_privacy.py``. Kept explicit so the prohibition is testable, not folklore.
FORBIDDEN_EVENT_FIELDS = frozenset({
    "key", "char", "chars", "character", "code", "keycode", "keyCode", "which",
    "text", "value", "content", "input", "clipboard", "selection",
    "target", "element", "window", "title", "url", "app",
})


@dataclass
class TypingSession:
    """
    Metadata about a typing session. Holds no content, only timing.

    Attributes:
        timestamps: Timestamp of each key press.
        is_backspace: Whether each key press was a correction.
        start_time: Session start timestamp.
        end_time: Session end timestamp.
    """

    timestamps: list[float] = field(default_factory=list)
    is_backspace: list[bool] = field(default_factory=list)
    start_time: float | None = None
    end_time: float | None = None

    def record_keypress(self, is_backspace: bool = False,
                        timestamp: float | None = None) -> None:
        """
        Record a single key press.

        Note the signature: it accepts *whether* the key was a correction, never *which*
        key it was. There is deliberately no parameter through which character identity
        could be passed.

        Parameters:
            is_backspace: Whether the key pressed was a correction.
            timestamp: Event time; defaults to now. Injectable so tests need not sleep.

        Raises:
            ValueError: If the timestamp is not a finite number. A ``NaN`` or ``Inf``
                value would poison every aggregate computed from the session, and those
                aggregates silently reaching a model is worse than a loud failure.
        """
        current_time = time.monotonic() if timestamp is None else float(timestamp)

        if not math.isfinite(current_time):
            raise ValueError("Keypress timestamp must be a finite number")

        if self.start_time is None:
            self.start_time = current_time

        self.timestamps.append(current_time)
        self.is_backspace.append(bool(is_backspace))
        self.end_time = current_time

    def get_inter_key_delays(self) -> list[float]:
        """
        Compute delays between consecutive key presses.

        Returns:
            List[float]: Inter-key delays in seconds; empty for fewer than two presses.
        """
        return [
            self.timestamps[i] - self.timestamps[i - 1]
            for i in range(1, len(self.timestamps))
        ]

    def get_total_keys(self) -> int:
        """Return the number of keys pressed."""
        return len(self.timestamps)

    def get_backspace_count(self) -> int:
        """Return the number of corrections."""
        return sum(self.is_backspace)

    def get_duration(self) -> float:
        """
        Return the session duration in seconds.

        Returns:
            float: Duration, or ``0.0`` when the session has not started.
        """
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

    def reset(self) -> None:
        """Clear the session for a new recording."""
        self.timestamps = []
        self.is_backspace = []
        self.start_time = None
        self.end_time = None


class TypingDataCollector:
    """
    Collects typing metadata from local input.

    Used for offline/demo capture. The live web path uses
    :func:`process_keystroke_data` instead.
    """

    def __init__(self) -> None:
        """Initialise the collector with an empty session."""
        self.session = TypingSession()

    def start_session(self) -> None:
        """Begin a new session, discarding any previous one."""
        self.session.reset()

    def record_key(self, is_backspace: bool = False) -> None:
        """
        Record a key press.

        Parameters:
            is_backspace: Whether the key was a correction.
        """
        self.session.record_keypress(is_backspace)

    def end_session(self) -> TypingSession:
        """
        End the current session.

        Returns:
            TypingSession: The completed session.
        """
        if self.session.start_time is not None:
            self.session.end_time = time.monotonic()
        return self.session

    def get_session_data(self) -> dict[str, Any]:
        """
        Return session metadata.

        Returns:
            dict: Timing metadata only — no content.
        """
        return {
            "total_keys": self.session.get_total_keys(),
            "backspace_count": self.session.get_backspace_count(),
            "duration": self.session.get_duration(),
            "inter_key_delays": self.session.get_inter_key_delays(),
            "start_time": self.session.start_time,
            "end_time": self.session.end_time,
        }


def empty_session_metadata() -> dict[str, Any]:
    """
    Return the metadata record for a session with no events.

    Returns:
        dict: Zeroed session metadata.
    """
    return {
        "total_keys": 0,
        "backspace_count": 0,
        "duration": 0.0,
        "inter_key_delays": [],
        "start_time": None,
        "end_time": None,
    }


def process_keystroke_data(keystroke_events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """
    Convert raw keystroke events into session metadata.

    **Privacy boundary.** Only :data:`ALLOWED_EVENT_FIELDS` are read. Any other field in
    an incoming event — including anything content-bearing that a modified client might
    send — is ignored and never reaches the returned record. The output shape is fixed and
    contains no per-event data beyond timing.

    Parameters:
        keystroke_events: Events, each carrying ``timestamp`` and optionally
            ``is_backspace``.

    Returns:
        dict: ``total_keys``, ``backspace_count``, ``duration``, ``inter_key_delays``,
        ``start_time``, ``end_time``.

    Raises:
        ValueError: If an event lacks a timestamp or carries a non-numeric or non-finite
            one (``NaN``/``Inf``). Malformed input fails loudly rather than being silently
            coerced to a plausible number or allowed to poison the session aggregates.
    """
    if not keystroke_events:
        return empty_session_metadata()

    timestamps: list[float] = []
    is_backspace: list[bool] = []

    for index, event in enumerate(keystroke_events):
        if not isinstance(event, Mapping):
            raise ValueError(f"Keystroke event {index} is not an object")

        if "timestamp" not in event:
            raise ValueError(f"Keystroke event {index} is missing 'timestamp'")

        raw_timestamp = event["timestamp"]
        if isinstance(raw_timestamp, bool) or not isinstance(raw_timestamp, (int, float)):
            raise ValueError(f"Keystroke event {index} has a non-numeric 'timestamp'")

        timestamp_value = float(raw_timestamp)
        if not math.isfinite(timestamp_value):
            # NaN/Inf pass the isinstance check but would turn every downstream aggregate
            # (duration, delays, features) into NaN/Inf that then silently reaches a
            # model. Rejected here, loudly, like any other malformed input.
            raise ValueError(f"Keystroke event {index} has a non-finite 'timestamp'")

        # Only these two values are read. Everything else in `event` is discarded here
        # and cannot appear anywhere downstream.
        timestamps.append(timestamp_value)
        is_backspace.append(bool(event.get("is_backspace", False)))

    inter_key_delays = [
        timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))
    ]

    return {
        "total_keys": len(timestamps),
        "backspace_count": sum(is_backspace),
        "duration": timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0,
        "inter_key_delays": inter_key_delays,
        "start_time": timestamps[0],
        "end_time": timestamps[-1],
    }
