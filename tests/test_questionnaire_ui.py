"""
Questionnaire UI tests (F4).

The page is where the instrument is actually *used*, so the obligations that attach to
using someone else's instrument attach here: show the citation, say how it was adapted,
say it is not a diagnosis, and do not put an unvalidated model's guess next to the score
(D-019).

None of that is enforceable by the server — a page can fetch `/api/instrument` and render
only the questions. So it is enforced here, against the served markup and script.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.conftest import page_bundle

APP_JS = Path("keystress/web/static/app.js")


@pytest.fixture(scope="module")
def script() -> str:
    return APP_JS.read_text(encoding="utf-8")


class TestMarkup:
    """What the page ships."""

    @pytest.mark.parametrize("element_id", [
        "questionnaire-card", "questionnaire-items", "questionnaire-submit",
        "questionnaire-progress", "questionnaire-result", "questionnaire-scores",
        "questionnaire-band", "questionnaire-caveat", "questionnaire-storage",
        # Provenance, filled from the instrument payload.
        "instrument-name", "instrument-citation", "instrument-adaptation",
        "instrument-disclaimer",
    ])
    def test_element_is_present(self, client, element_id: str) -> None:
        assert f'id="{element_id}"' in client.get("/").get_data(as_text=True)

    def test_the_card_starts_hidden(self, client) -> None:
        """
        The questionnaire is offered after a session has been analysed, so it labels
        something rather than arriving out of context.
        """
        body = client.get("/").get_data(as_text=True)
        marker = body.index('id="questionnaire-card"')
        opening = body[body.rindex("<section", 0, marker):marker]
        assert "is-hidden" in opening

    def test_the_submit_button_starts_disabled(self, client) -> None:
        body = client.get("/").get_data(as_text=True)
        marker = body.index('id="questionnaire-submit"')
        assert "disabled" in body[marker:marker + 80]


class TestNoFreeText:
    """The one control type that could carry content does not exist here."""

    def test_the_page_has_no_text_input_beyond_the_typing_box(self, client) -> None:
        body = client.get("/").get_data(as_text=True)

        text_inputs = re.findall(r'<input[^>]*type="(?:text|email|search|tel|url)"[^>]*>', body)
        assert not text_inputs, f"free-text input added to the page: {text_inputs}"

        # One textarea, and it is the typing box whose *content* is never read.
        assert body.count("<textarea") == 1
        assert 'id="typing-area"' in body

    def test_answers_are_built_from_radio_values_only(self, script: str) -> None:
        collect = script[script.index("function collectAnswers("):]
        collect = collect[:collect.index("\n}")]
        assert 'input[type="radio"]:checked' in collect
        assert "parseInt" in collect, "an answer must be an integer, not free text"

    def test_the_submitted_payload_carries_only_responses_and_a_donation_id(
        self, script: str
    ) -> None:
        submit = script[script.index("function submitQuestionnaire("):]
        submit = submit[:submit.index("\n}")]

        assigned = set(re.findall(r"body\.(\w+)\s*=", submit))
        declared = set(re.findall(r"const body = \{ (\w+):", submit))
        assert assigned | declared <= {"responses", "donation_id"}, (
            f"questionnaire request carries unexpected field(s): {assigned | declared}"
        )


class TestProvenanceIsShown:
    """Using an instrument without naming it is misuse, however tidy the page looks."""

    @pytest.mark.parametrize("field", [
        "citation", "adaptation_note", "disclaimer", "name",
    ])
    def test_the_page_renders_each_provenance_field(self, script: str, field: str) -> None:
        render = script[script.index("function renderInstrument("):]
        render = render[:render.index("\n}")]
        assert f"payload.{field}" in render, f"the page never displays {field}"

    def test_provenance_is_rendered_as_text_not_markup(self, script: str) -> None:
        """
        `textContent`, never `innerHTML`: server-supplied strings are never parsed as
        markup. Under the strict CSP an injected `<script>` would not run anyway, but
        markup that renders at all is still a page defacement, and this is free.
        """
        render = script[script.index("function renderInstrument("):]
        render = render[:render.index("\n}")]
        assert "innerHTML" not in render


class TestHonestResultPanel:
    """What the participant is told about their own score."""

    def test_the_caveat_is_displayed_with_the_band(self, script: str) -> None:
        show = script[script.index("function showQuestionnaireResult("):]
        show = show[:show.index("\n}")]
        assert "result.caveat" in show
        assert "result.band" in show

    def test_scores_are_shown_out_of_one_hundred(self, script: str) -> None:
        """
        A bare "62" invites being read as a percentage of something, or as a probability
        of being burned out. It is neither, and saying "out of 100" costs nothing.
        """
        show = script[script.index("function showQuestionnaireResult("):]
        show = show[:show.index("\n}")]
        assert "out of 100" in show

    def test_the_storage_outcome_is_stated(self, script: str) -> None:
        show = script[script.index("function showQuestionnaireResult("):]
        show = show[:show.index("\n}")]
        assert "storage_note" in show, "the page must say whether the answers were kept"

    def test_the_result_is_announced(self, script: str) -> None:
        show = script[script.index("function showQuestionnaireResult("):]
        show = show[:show.index("\n}")]
        assert "announce(" in show

    def test_no_model_output_is_shown_beside_the_questionnaire_score(
        self, script: str
    ) -> None:
        """
        D-019 again, at the last mile. The model is synthetic-trained and unvalidated;
        rendering its prediction next to a real self-report would let the page imply the
        two measured the same thing.
        """
        show = script[script.index("function showQuestionnaireResult("):]
        show = show[:show.index("\n}")]
        for forbidden in ("probabilities", "level_class", "confidence", "prediction"):
            assert forbidden not in show


class TestFlow:
    """When the card appears and disappears."""

    def test_the_questionnaire_is_offered_after_a_result(self, script: str) -> None:
        display = script[script.index("function displayResults("):]
        display = display[:display.index("\nfunction ")]
        assert "showQuestionnaire()" in display

    def test_starting_a_new_session_clears_the_pairing(self, script: str) -> None:
        """
        `donationId` names the session being labelled. Left over from a previous run, it
        would attach this questionnaire to the wrong typing session — a wrong label,
        which is the one kind of bad row nothing downstream can detect.
        """
        new_test = script[script.index("function newTest("):]
        new_test = new_test[:new_test.index("\n}")]
        assert "donationId = null" in new_test
        assert "hideCard('questionnaire-card')" in new_test

    def test_returning_to_the_consent_gate_clears_it_too(self, script: str) -> None:
        gate = script[script.index("function showConsentGate("):]
        gate = gate[:gate.index("\n}")]
        assert "donationId = null" in gate

    def test_the_instrument_endpoint_is_used(self, client) -> None:
        assert "/api/instrument" in page_bundle(client)
        assert "/api/questionnaire" in page_bundle(client)
