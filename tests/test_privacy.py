"""
Privacy tests — the project's central guarantee (F12).

Keystress-AI's entire claim is that it analyses *how* you type and never *what* you type.
Until these tests existed that guarantee was a convention: the code happened to read only
two fields, and nothing checked that it stayed that way. These tests make it enforceable.

What is asserted
----------------
1. Content-bearing fields on an incoming event never survive processing, never reach the
   feature vector, and never appear in the API response — even when a hostile client
   deliberately sends them.
2. The output shape is fixed. Session metadata and prediction responses carry a known set
   of keys, so a future change that starts echoing input fails here.
3. Typed text cannot be reconstructed from what the system retains. Sessions differing
   only in *what* was typed produce identical output.
4. The collection API offers no channel for character data: no function parameter, no
   dataclass field, no allowlist entry.
5. The frontend records a boolean, not a key.

Design note
-----------
The adversary modelled here is a modified client, not a malicious server operator. A user
running the server can obviously read what they receive; the guarantee is that the
*protocol and pipeline* never carry content in the first place, so there is nothing to
read. Tests therefore push content-bearing payloads in and assert they vanish.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest

from keystress.core import collect, features, inference
from keystress.core.collect import (
    ALLOWED_EVENT_FIELDS,
    FORBIDDEN_EVENT_FIELDS,
    TypingSession,
    process_keystroke_data,
)
from keystress.core.disclosure import FEATURES_V1
from keystress.core.features import extract_typing_features
from keystress.core.inference import get_prediction_details
from tests.conftest import make_events

pytestmark = pytest.mark.privacy


#: A sentence whose words must never appear anywhere downstream.
SECRET_TEXT = "my bank password is hunter2 correct horse battery staple"

#: Distinctive tokens searched for in every output.
SECRET_TOKENS = (
    "hunter2", "password", "bank", "correct", "horse", "battery", "staple",
    "KeyA", "ShiftLeft", "secret",
)

#: The exact keys session metadata may contain.
ALLOWED_SESSION_KEYS = frozenset({
    "total_keys", "backspace_count", "duration",
    "inter_key_delays", "start_time", "end_time",
})

#: The exact keys a prediction response may contain.
ALLOWED_RESPONSE_KEYS = frozenset({
    "prediction", "label", "description", "confidence", "probabilities", "labels",
    "insufficient_data", "features", "data_source", "model_version", "feature_set",
    "disclaimer", "level_class",
})


def hostile_events(count: int = 40) -> list[dict]:
    """
    Build events carrying every content-bearing field an attacker might try.

    A well-behaved client never sends these. The point is that the server must not care.

    Parameters:
        count: Number of events.

    Returns:
        list[dict]: Events with legitimate timing plus a payload of forbidden fields.
    """
    words = SECRET_TEXT.split()
    return [
        {
            "timestamp": round(i * 0.21, 6),
            "is_backspace": i % 9 == 0,
            # Everything below must be discarded at the privacy boundary.
            "key": words[i % len(words)][0],
            "char": words[i % len(words)][0],
            "code": "KeyA",
            "keyCode": 65,
            "which": 65,
            "text": words[i % len(words)],
            "value": SECRET_TEXT,
            "content": SECRET_TEXT,
            "input": SECRET_TEXT,
            "clipboard": "secret clipboard contents",
            "selection": "secret",
            "target": "#password-field",
            "window": "Online Banking - Chrome",
            "title": "Online Banking",
            "url": "https://bank.example/login",
            "app": "chrome.exe",
        }
        for i in range(count)
    ]


def assert_no_content(blob: object, context: str) -> None:
    """
    Assert that a serialised object contains no content-bearing data.

    Parameters:
        blob: Any JSON-serialisable object.
        context: Description used in the failure message.

    Raises:
        AssertionError: If a secret token or forbidden field name is present.
    """
    text = json.dumps(blob, default=str)
    lowered = text.lower()

    leaked = [token for token in SECRET_TOKENS if token.lower() in lowered]
    assert not leaked, f"{context} leaked content tokens: {leaked}\n{text[:400]}"

    present = [name for name in FORBIDDEN_EVENT_FIELDS if f'"{name}"' in text]
    assert not present, f"{context} carries forbidden field(s): {present}\n{text[:400]}"


# --------------------------------------------------------------------------------------
# 1. The pipeline discards content
# --------------------------------------------------------------------------------------


class TestPipelineDiscardsContent:
    """Content-bearing input must not survive any pipeline stage."""

    def test_session_metadata_is_content_free(self) -> None:
        session = process_keystroke_data(hostile_events())
        assert_no_content(session, "session metadata")

    def test_session_metadata_keys_are_exactly_the_allowed_set(self) -> None:
        session = process_keystroke_data(hostile_events())
        assert set(session) == ALLOWED_SESSION_KEYS, (
            "session metadata shape changed; every new key needs a privacy review"
        )

    def test_features_are_content_free(self) -> None:
        session = process_keystroke_data(hostile_events())
        assert_no_content(extract_typing_features(session), "features")

    def test_features_keys_are_exactly_the_v1_feature_set(self) -> None:
        session = process_keystroke_data(hostile_events())
        assert set(extract_typing_features(session)) == set(FEATURES_V1)

    def test_features_are_all_numeric(self) -> None:
        """
        A numeric-only feature vector cannot carry a string of typed text.

        This is the structural reason content cannot reach the model at all.
        """
        session = process_keystroke_data(hostile_events())
        for name, value in extract_typing_features(session).items():
            assert isinstance(value, float), f"feature {name} is {type(value).__name__}, not float"

    def test_prediction_is_content_free(self, model_bundle) -> None:
        session = process_keystroke_data(hostile_events())
        result = get_prediction_details(extract_typing_features(session), model_bundle)
        assert_no_content(result, "prediction result")


# --------------------------------------------------------------------------------------
# 2. The API response discards content
# --------------------------------------------------------------------------------------


class TestApiResponseDiscardsContent:
    """The full HTTP round-trip must not echo anything content-bearing."""

    def test_response_to_hostile_payload_is_content_free(self, client) -> None:
        response = client.post("/api/predict", json={"keystroke_events": hostile_events()})
        assert response.status_code == 200
        assert_no_content(response.get_json(), "API response")

    def test_raw_response_bytes_contain_no_secret(self, client) -> None:
        """Belt and braces: scan the wire bytes, not just the parsed object."""
        response = client.post("/api/predict", json={"keystroke_events": hostile_events()})
        raw = response.get_data(as_text=True).lower()
        for token in SECRET_TOKENS:
            assert token.lower() not in raw, f"raw response body contains {token!r}"

    def test_response_keys_are_exactly_the_allowed_set(self, client) -> None:
        body = client.post(
            "/api/predict", json={"keystroke_events": hostile_events()}
        ).get_json()
        unexpected = set(body) - ALLOWED_RESPONSE_KEYS
        assert not unexpected, (
            f"response gained unreviewed key(s): {sorted(unexpected)}. "
            "Every new response field needs a privacy review."
        )

    def test_top_level_hostile_fields_are_ignored(self, client) -> None:
        """Content smuggled beside `keystroke_events` must not be echoed."""
        response = client.post("/api/predict", json={
            "keystroke_events": make_events(),
            "typed_text": SECRET_TEXT,
            "clipboard": SECRET_TEXT,
            "note": "hunter2",
        })
        assert response.status_code == 200
        assert_no_content(response.get_json(), "API response with smuggled top-level fields")

    def test_error_responses_do_not_echo_input(self, client) -> None:
        """A rejection message must not quote back what was sent."""
        response = client.post("/api/predict", json={
            "keystroke_events": [{"timestamp": SECRET_TEXT, "char": "x"}] * 6
        })
        assert response.status_code == 400
        assert_no_content(response.get_json(), "error response")

    def test_health_endpoint_is_content_free(self, client) -> None:
        assert_no_content(client.get("/api/health").get_json(), "health response")


# --------------------------------------------------------------------------------------
# 3. Typed text is not reconstructable
# --------------------------------------------------------------------------------------


class TestTextIsNotReconstructable:
    """What survives must be insufficient to recover what was typed."""

    def test_identical_timing_different_content_yields_identical_output(self) -> None:
        """
        The strongest formulation of the guarantee.

        Two sessions with the same rhythm but completely different typed text must be
        indistinguishable in everything the system retains. If they ever differ, some
        content has leaked into the pipeline.
        """
        timings = [round(i * 0.19, 6) for i in range(40)]

        first = [
            {"timestamp": t, "is_backspace": i % 7 == 0,
             "key": "a", "text": "the quick brown fox"}
            for i, t in enumerate(timings)
        ]
        second = [
            {"timestamp": t, "is_backspace": i % 7 == 0,
             "key": "z", "text": SECRET_TEXT}
            for i, t in enumerate(timings)
        ]

        assert process_keystroke_data(first) == process_keystroke_data(second)
        assert (extract_typing_features(process_keystroke_data(first))
                == extract_typing_features(process_keystroke_data(second)))

    def test_api_responses_identical_for_different_content(self, client) -> None:
        timings = [round(i * 0.19, 6) for i in range(40)]
        payloads = [
            [{"timestamp": t, "is_backspace": False, "text": "aaaa"} for t in timings],
            [{"timestamp": t, "is_backspace": False, "text": SECRET_TEXT} for t in timings],
        ]
        bodies = [
            client.post("/api/predict", json={"keystroke_events": p}).get_json()
            for p in payloads
        ]
        assert bodies[0] == bodies[1]

    def test_retained_data_is_smaller_than_the_text_it_came_from(self) -> None:
        """
        Five aggregate numbers cannot encode a 200-keystroke sequence.

        An information-theoretic sanity check: the feature vector is fixed-size regardless
        of session length, so it cannot carry per-keystroke content.
        """
        short = extract_typing_features(process_keystroke_data(make_events(count=20)))
        long = extract_typing_features(process_keystroke_data(make_events(count=2000)))
        assert len(short) == len(long) == 5


# --------------------------------------------------------------------------------------
# 4. The API offers no channel for content
# --------------------------------------------------------------------------------------


class TestNoChannelForContent:
    """It must be structurally impossible to pass content in."""

    def test_allowlist_contains_only_timing_fields(self) -> None:
        assert {"timestamp", "is_backspace"} == ALLOWED_EVENT_FIELDS

    def test_allowlist_and_forbidden_list_are_disjoint(self) -> None:
        assert not (ALLOWED_EVENT_FIELDS & FORBIDDEN_EVENT_FIELDS)

    def test_record_keypress_has_no_content_parameter(self) -> None:
        """The recording API accepts *whether*, never *which*."""
        params = set(inspect.signature(TypingSession.record_keypress).parameters)
        assert params == {"self", "is_backspace", "timestamp"}

    def test_typing_session_fields_are_timing_only(self) -> None:
        fields = set(TypingSession.__dataclass_fields__)
        assert fields == {"timestamps", "is_backspace", "start_time", "end_time"}

    def test_session_stores_no_strings(self) -> None:
        session = TypingSession()
        for i in range(10):
            session.record_keypress(is_backspace=i % 3 == 0, timestamp=i * 0.2)
        assert all(isinstance(t, float) for t in session.timestamps)
        assert all(isinstance(b, bool) for b in session.is_backspace)


# --------------------------------------------------------------------------------------
# 5. Source-level guarantees
# --------------------------------------------------------------------------------------


class TestSourceLevelGuarantees:
    """Guard against a future change quietly reintroducing content capture."""

    @pytest.mark.parametrize("module", [collect, features, inference])
    def test_core_modules_never_read_content_fields(self, module) -> None:
        """
        No core module may subscript an event with a content-bearing key.

        Catches the realistic regression: someone adds `event["key"]` to "improve" a
        feature and does not think of it as a privacy change.
        """
        source = Path(inspect.getfile(module)).read_text(encoding="utf-8")
        # Ignore the documentation and the FORBIDDEN_EVENT_FIELDS declaration itself.
        code = re.sub(r'"""[\s\S]*?"""', "", source)
        code = re.sub(r"FORBIDDEN_EVENT_FIELDS[\s\S]*?\}\)", "", code)

        for name in ("key", "char", "code", "keyCode", "which", "text", "clipboard"):
            for pattern in (f'["{name}"]', f"['{name}']", f'.get("{name}"', f".get('{name}'"):
                assert pattern not in code, (
                    f"{module.__name__} reads content-bearing field via {pattern}"
                )

    def test_frontend_records_a_boolean_not_a_key(self) -> None:
        """
        The browser must discard key identity at the source.

        `event.key` may be *compared* to 'Backspace', but its value must never be stored
        into the event objects that are sent to the server.
        """
        script = Path("keystress/web/static/app.js").read_text(encoding="utf-8")

        assert "is_backspace: event.key === 'Backspace'" in script, (
            "the backspace flag must be a comparison result, not a stored key value"
        )

        for forbidden in ("key: event.key", "char:", "code: event.code",
                          "value: typingArea.value", "clipboardData",
                          "text: ", "keyCode"):
            assert forbidden not in script, f"app.js may leak content via `{forbidden}`"

    def test_frontend_sends_only_two_fields(self) -> None:
        """The outgoing payload construction must carry exactly timestamp + flag."""
        script = Path("keystress/web/static/app.js").read_text(encoding="utf-8")
        payload = re.search(
            r"keystroke_events:\s*keystrokeData\.map\(k\s*=>\s*\(\{(.*?)\}\)\)",
            script, re.S,
        )
        assert payload, "could not locate the outgoing payload construction in app.js"

        sent_fields = set(re.findall(r"(\w+)\s*:", payload.group(1)))
        assert sent_fields == {"timestamp", "is_backspace"}, (
            f"app.js sends unexpected field(s): {sorted(sent_fields - {'timestamp', 'is_backspace'})}"
        )

    def test_no_persistence_of_raw_timing_on_the_serving_path(self) -> None:
        """
        Phase 0 stores nothing about a user. Consent-gated storage arrives in F2.

        Fails if a write call appears in the serving path without the consent flow that
        HARD RULE 4 requires alongside it.
        """
        for name in ("collect.py", "features.py", "inference.py"):
            source = Path("keystress/core") / name
            code = re.sub(r'"""[\s\S]*?"""', "", source.read_text(encoding="utf-8"))
            for pattern in ("open(", "write_text(", "to_csv(", "json.dump(", "pickle."):
                assert pattern not in code, (
                    f"{name} performs I/O ({pattern}); raw timing data must not be "
                    "persisted before the F2 consent flow exists"
                )


class TestSuiteIntegrity:
    """
    Guard the guard.

    A test suite that collects zero tests passes. Without this, the privacy guarantee
    could be silently removed by deleting its tests rather than by breaking them — which
    would look identical in a green CI run.
    """

    #: Floor on the number of privacy assertions. Raise it when tests are added; lowering
    #: it should require explaining which guarantee is being given up.
    MINIMUM_PRIVACY_TESTS = 25

    def test_suite_has_not_been_emptied(self) -> None:
        import sys

        module = sys.modules[__name__]
        count = sum(
            1
            for obj in vars(module).values()
            if inspect.isclass(obj) and obj.__name__.startswith("Test")
            for name in vars(obj)
            if name.startswith("test_")
        )
        assert count >= self.MINIMUM_PRIVACY_TESTS, (
            f"Only {count} privacy tests found, expected at least "
            f"{self.MINIMUM_PRIVACY_TESTS}. The no-content-capture guarantee must not be "
            "weakened by deleting the tests that verify it."
        )

    def test_every_privacy_test_carries_the_marker(self) -> None:
        """`pytestmark` applies the marker module-wide; CI selects on it."""
        assert pytestmark.name == "privacy"
