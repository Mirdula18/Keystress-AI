"""
Accessibility contract of the served page.

These are static assertions over the markup and `app.js`, not a substitute for testing
with an actual screen reader. They exist to catch the failure this module was written in
response to: the page *advertised* two accessibility affordances — a polite live region
and a Ctrl+Enter shortcut — and neither was wired to anything. A promise in the markup
with no implementation behind it is worse than no promise, because it is invisible to
everyone who does not need it and useless to everyone who does.

So each test below pairs a piece of markup with the code that has to honour it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

APP_JS = Path("keystress/web/static/app.js")


@pytest.fixture(scope="module")
def script() -> str:
    """The frontend script, read once."""
    return APP_JS.read_text(encoding="utf-8")


class TestLandmarksAndSkipLink:
    """A keyboard user must be able to get past the header in one keystroke."""

    def test_skip_link_targets_the_main_landmark(self, client) -> None:
        body = client.get("/").get_data(as_text=True)
        match = re.search(r'<a class="skip-link" href="#([^"]+)"', body)
        assert match, "no skip link"
        assert f'id="{match.group(1)}"' in body, "skip link points at a missing target"

    def test_page_has_the_three_landmarks(self, client) -> None:
        body = client.get("/").get_data(as_text=True)
        for tag in ("<header", "<main", "<footer"):
            assert tag in body, f"missing {tag} landmark"

    def test_cards_are_sections_with_accessible_names(self, client) -> None:
        """
        Each card is a labelled region, so a screen reader can list and jump between them.

        A `<div>` soup gives no such outline: every card would be undifferentiated text.
        """
        body = client.get("/").get_data(as_text=True)
        sections = re.findall(r"<section[^>]*>", body)
        assert sections, "cards are not sections"
        for section in sections:
            assert "aria-label" in section or "aria-labelledby" in section, (
                f"section without an accessible name: {section}"
            )


class TestLiveRegion:
    """State changes are announced, because most of them are purely visual."""

    def test_page_carries_a_polite_live_region(self, client) -> None:
        body = client.get("/").get_data(as_text=True)
        marker = body.index('id="announcer"')
        element = body[body.rindex("<", 0, marker):body.index(">", marker) + 1]
        assert 'aria-live="polite"' in element, "the announcer must be a polite live region"
        assert "sr-only" in element, "the announcer must not be visible"

    def test_announce_writes_to_that_region(self, script: str) -> None:
        assert "function announce(" in script
        assert "getElementById('announcer')" in script

    @pytest.mark.parametrize("transition", [
        # Every state change that is otherwise conveyed only by a card appearing.
        "showConsentedView", "showConsentGate", "newTest",
    ])
    def test_each_silent_transition_announces(self, script: str, transition: str) -> None:
        body = script[script.index(f"function {transition}("):]
        body = body[:body.index("\n}")]
        assert "announce(" in body, f"{transition} changes the view without announcing it"

    def test_the_result_announcement_carries_its_qualifier(self, script: str) -> None:
        """
        HARD RULE 3 applies to what is spoken as much as to what is drawn.

        A screen-reader user hearing an unqualified confidence figure has been told
        exactly the number the whole disclosure contract exists to prevent — the
        spoken channel is not exempt from HARD RULE 3.
        """
        announcement = re.search(r"announce\('Result: '(.*?)\);", script, re.S)
        assert announcement, "no result announcement found"
        assert "qualifier" in announcement.group(1)
        assert "uncalibrated" in announcement.group(1)

    def test_announcements_never_include_typed_text(self, script: str) -> None:
        """A live region is read aloud, so content must not be able to reach it."""
        for call in re.findall(r"announce\((.*?)\);", script, re.S):
            assert "typingArea" not in call, "an announcement reads the typing box"
            assert ".value" not in call, "an announcement reads an input value"


class TestKeyboardShortcut:
    """The advertised shortcut has to exist."""

    def test_page_advertises_the_shortcut(self, client) -> None:
        body = client.get("/").get_data(as_text=True)
        assert "<kbd>Ctrl</kbd>" in body and "<kbd>Enter</kbd>" in body

    def test_the_shortcut_is_bound_and_runs_the_analysis(self, script: str) -> None:
        assert "function handleShortcut(" in script
        handler = script[script.index("function handleShortcut("):]
        handler = handler[:handler.index("\n}")]

        assert "ctrlKey" in handler and "metaKey" in handler, (
            "Cmd+Enter must work too; a Mac user has no Ctrl convention here"
        )
        assert "analyzeTyping()" in handler
        assert "analyzeBtn.disabled" in handler, (
            "the shortcut must respect the same minimum-length gate as the button, "
            "or it would submit a session the button refuses"
        )
        assert "addEventListener('keydown', handleShortcut)" in script

    def test_the_shortcut_stores_no_key_identity(self, script: str) -> None:
        """
        The handler compares `event.key` and discards it, exactly as the Backspace check
        does. Comparing is fine; keeping is not (HARD RULE 1).
        """
        handler = script[script.index("function handleShortcut("):]
        handler = handler[:handler.index("\n}")]
        assert "push" not in handler
        assert "keystrokeData" not in handler


class TestControlSemantics:
    """Buttons behave like buttons and inputs have labels."""

    def test_every_button_declares_its_type(self, client) -> None:
        """
        An untyped `<button>` inside a form submits it. There is no form here today, so
        this is cheap insurance rather than a live bug — but it is the kind of default
        that bites silently the moment markup is wrapped in one.
        """
        body = client.get("/").get_data(as_text=True)
        for button in re.findall(r"<button[^>]*>", body):
            assert "type=" in button, f"button without an explicit type: {button}"

    def test_every_checkbox_sits_inside_a_label(self, client) -> None:
        body = client.get("/").get_data(as_text=True)
        for match in re.finditer(r'<input type="checkbox"[^>]*id="([^"]+)"', body):
            preceding = body[:match.start()]
            assert preceding.rstrip().endswith(">"), "unexpected markup shape"
            assert "<label" in preceding[-300:], (
                f"checkbox {match.group(1)} is not wrapped in a label"
            )

    def test_decorative_icons_are_hidden_from_assistive_technology(self, client) -> None:
        """
        Every icon is decorative — the text beside it carries the meaning. An unhidden
        inline SVG is announced as "graphic", which is noise on every single control.
        """
        body = client.get("/").get_data(as_text=True)
        for svg in re.findall(r"<svg[^>]*>", body):
            assert 'aria-hidden="true"' in svg, f"icon not hidden from readers: {svg[:80]}"
