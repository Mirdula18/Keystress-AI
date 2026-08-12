"""
Offline capability and third-party isolation (F16).

**Acceptance:** the app is fully functional with no network, and the page makes no
third-party request. Both are properties of the *served bytes*, so they are tested here
rather than trusted to a reviewer noticing a re-added CDN link in a diff.

The tests below split into two claims:

1. **Nothing is fetched from anywhere but this origin.** Every `href`/`src` the page
   carries must be same-origin and servable, so a machine with no route to the internet
   renders exactly what a connected one does.
2. **No behaviour is expressed inline.** Inline `onclick`/`onchange` attributes are what
   forced `script-src 'unsafe-inline'` (D-021). With the handlers bound in `app.js`, the
   policy can be strict — but only for as long as no one re-adds one, which is what
   :class:`TestNoInlineHandlers` is for.
"""

from __future__ import annotations

import re

import pytest

from tests.conftest import page_bundle

#: Origins the page must never reach for. The two named CDNs were real dependencies
#: before F16 and are the regression most worth naming explicitly.
FORBIDDEN_ORIGINS = (
    "fonts.googleapis.com",
    "fonts.gstatic.com",
    "cdnjs.cloudflare.com",
    "cdn.jsdelivr.net",
    "unpkg.com",
)

#: Matches the URL of a *subresource* — something the browser fetches on its own while
#: rendering. `<a href>` is deliberately excluded: a link the reader may choose to follow
#: is navigation, not a third-party request the page makes, and the footer's GitHub link
#: is both wanted and harmless offline.
_ASSET_URL = re.compile(
    r'<(?!a[\s>])[a-z]+[^>]*?\s(?:href|src)="([^"]+)"', re.IGNORECASE
)

#: Matches an inline event-handler attribute: `onclick=`, `onchange=`, `onsubmit=`, ...
_INLINE_HANDLER = re.compile(r'\son[a-z]+\s*=\s*"', re.IGNORECASE)


class TestNoThirdPartyRequests:
    """The page must be renderable with no network access at all."""

    def test_page_references_no_forbidden_origin(self, client) -> None:
        bundle = page_bundle(client)
        for origin in FORBIDDEN_ORIGINS:
            assert origin not in bundle, f"page still reaches for {origin}"

    def test_every_asset_url_is_same_origin_or_inline(self, client) -> None:
        """
        Every linked asset is either a local path or a `data:` URI.

        `data:` is allowed because it is inline content, not a request: the favicon is a
        few hundred bytes of SVG that a browser renders without touching the network.
        Anything with a scheme and a host is a third-party fetch and fails here.
        """
        body = client.get("/").get_data(as_text=True)

        for url in _ASSET_URL.findall(body):
            if url.startswith("data:"):
                continue
            assert not url.startswith(("http://", "https://", "//")), (
                f"asset {url} is fetched from another origin"
            )

    def test_linked_assets_are_actually_served(self, client) -> None:
        """An offline-capable page whose own CSS 404s is not offline-capable."""
        body = client.get("/").get_data(as_text=True)
        local = [
            url for url in _ASSET_URL.findall(body)
            if not url.startswith(("data:", "http://", "https://", "//", "#"))
        ]
        assert local, "page links no local assets at all — did the template change?"

        for url in local:
            assert client.get(url).status_code == 200, f"{url} is linked but not served"

    def test_no_font_face_pulls_a_remote_file(self, client) -> None:
        """A vendored `@font-face` pointing at a CDN would be offline-broken too."""
        css = client.get("/static/styles.css").get_data(as_text=True)
        for origin in FORBIDDEN_ORIGINS:
            assert origin not in css


class TestNoInlineHandlers:
    """
    Behaviour lives in `app.js`, so the CSP can forbid inline script.

    This is the guard on D-023: re-adding a single `onclick` attribute would make the page
    silently non-functional under the strict policy — the handler would be blocked, the
    button would do nothing, and no test that only checks element ids would notice.
    """

    def test_page_carries_no_inline_event_handler(self, client) -> None:
        body = client.get("/").get_data(as_text=True)
        found = _INLINE_HANDLER.findall(body)
        assert not found, f"inline handler attributes found: {sorted(set(found))}"

    def test_page_carries_no_inline_script_block(self, client) -> None:
        body = client.get("/").get_data(as_text=True)
        assert not re.search(r"<script(?![^>]*\ssrc=)[^>]*>", body), (
            "inline <script> block found; move the code into app.js"
        )

    @pytest.mark.parametrize("element_id", [
        "consent-analysis", "consent-donate", "consent-btn",
        "reset-btn", "analyze-btn", "new-test-btn",
        "donate-toggle", "view-data-btn", "delete-btn",
    ])
    def test_every_bound_control_exists(self, client, element_id: str) -> None:
        """
        Each id in `CONTROL_BINDINGS` must exist in the markup.

        The binding table logs and skips a missing element rather than throwing, which
        keeps one typo from breaking every other control — so the typo needs catching
        here instead.
        """
        assert f'id="{element_id}"' in client.get("/").get_data(as_text=True)

    def test_page_carries_no_inline_style_attribute(self, client) -> None:
        """
        Styling lives in the stylesheet, so the CSP can forbid inline style too.

        Scripted width changes (`bar.style.width = …`) go through the CSSOM and are not
        restricted by CSP; a `style="…"` attribute in the markup is, and would be silently
        dropped under the strict policy — leaving, for instance, a permanently visible
        typing card in front of someone who has not consented.
        """
        body = client.get("/").get_data(as_text=True)
        assert not re.search(r'\sstyle\s*=\s*"', body), (
            "inline style attribute found; use a class instead"
        )

    def test_binding_table_covers_every_control(self, client) -> None:
        """The reverse direction: no control is left without a listener."""
        script = client.get("/static/app.js").get_data(as_text=True)
        table = script[script.index("const CONTROL_BINDINGS"):]
        table = table[:table.index("];")]

        for element_id in ("consent-btn", "analyze-btn", "delete-btn", "donate-toggle"):
            assert f"'{element_id}'" in table, f"{element_id} is not bound in app.js"
