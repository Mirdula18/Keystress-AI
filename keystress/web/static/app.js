/*
 * Keystress-AI frontend behaviour.
 *
 * Extracted from the HTML_TEMPLATE string literal in app.py (F10). Behaviour-preserving:
 * the logic below is unchanged from the inlined original apart from indentation.
 *
 * PRIVACY: this script records a timestamp and a Backspace boolean per keydown, and
 * nothing else. `event.key` is compared against 'Backspace' and then discarded - key
 * identity never enters `keystrokeData`, and never leaves the browser. Do not add any
 * field here that could carry character identity (docs/CLAUDE.md HARD RULE 1).
 *
 * Loaded with `defer`, so the DOM is parsed before this runs.
 */

// Keystroke data collection
let keystrokeData = [];
let startTime = null;
let updateInterval = null;

const typingArea = document.getElementById('typing-area');
const analyzeBtn = document.getElementById('analyze-btn');

// Event listeners
typingArea.addEventListener('keydown', recordKeyDown);
typingArea.addEventListener('input', updateStats);

// -------------------------------------------------------------------------------------
// Control bindings (F16)
//
// Every control used to carry an inline `onclick`/`onchange` attribute, which forced the
// Content-Security-Policy to allow 'unsafe-inline' scripts — the one directive that
// meaningfully weakens a CSP, since it re-permits exactly the injected inline script the
// policy exists to stop. Binding here instead lets `script-src` be plain 'self'.
//
// Table-driven so adding a control means adding a row, not remembering to also add a
// listener; a missing element is reported rather than silently unbound.
// -------------------------------------------------------------------------------------

const CONTROL_BINDINGS = [
    ['consent-analysis', 'change', updateConsentButton],
    ['consent-donate', 'change', updateConsentButton],
    ['consent-btn', 'click', grantConsent],
    ['reset-btn', 'click', resetTest],
    ['analyze-btn', 'click', analyzeTyping],
    ['new-test-btn', 'click', newTest],
    ['donate-toggle', 'change', changeDonateConsent],
    ['view-data-btn', 'click', viewMyData],
    ['delete-btn', 'click', deleteMyData],
    ['questionnaire-submit', 'click', submitQuestionnaire]
];

function bindControls() {
    CONTROL_BINDINGS.forEach(function (binding) {
        const element = document.getElementById(binding[0]);
        if (!element) {
            // A renamed id would otherwise produce a dead button with no clue why.
            console.error('Keystress: no element #' + binding[0] + ' to bind');
            return;
        }
        element.addEventListener(binding[1], binding[2]);
    });
}

bindControls();

// Show/hide by class rather than by writing `style.display`, so the markup carries no
// `style` attribute and `style-src` can stay strict (F16). Toggling a class also keeps
// the display mode in the stylesheet, where a card that is not a plain block (the
// flex-laid-out ones) does not need JS to know that.
function showCard(elementId) {
    document.getElementById(elementId).classList.remove('is-hidden');
}

function hideCard(elementId) {
    document.getElementById(elementId).classList.add('is-hidden');
}

// -------------------------------------------------------------------------------------
// Consent (F2)
//
// The participant token is an opaque UUID minted by POST /api/consent. It is the only
// credential, carries no personal content, and lives in localStorage so a returning user
// need not re-consent. Losing it costs nothing but reachability of the stored rows.
//
// The gate here is a UI courtesy, not the enforcement point: /api/predict refuses a
// request without a valid token regardless of what this page does.
// -------------------------------------------------------------------------------------

const CONSENT_KEY = 'keystress_consent_id';

// Mirrors the participant's donate opt-in, so the page can say what will happen before
// it happens. Refreshed from the server on load and after every consent change.
let donateConsent = false;

function consentId() {
    return window.localStorage.getItem(CONSENT_KEY);
}

function rememberConsent(participantId) {
    window.localStorage.setItem(CONSENT_KEY, participantId);
}

function forgetConsent() {
    window.localStorage.removeItem(CONSENT_KEY);
    donateConsent = false;
}

// Every request that needs the token goes through here, so it cannot be forgotten
// on one call site and silently 403.
function consentHeaders() {
    const headers = { 'Content-Type': 'application/json' };
    const id = consentId();
    if (id) { headers['X-Consent-Id'] = id; }
    return headers;
}

function loadConsentPolicy() {
    return fetch('/api/consent/policy')
        .then(response => response.json())
        .then(policy => {
            document.getElementById('consent-summary').textContent = policy.summary;
            document.getElementById('consent-disclaimer').textContent = policy.disclaimer;
            document.getElementById('consent-version').textContent = policy.consent_version;
        })
        .catch(() => {
            // Consent cannot be given against wording we failed to load, so the button
            // stays disabled and the page says why rather than asking for blind agreement.
            document.getElementById('consent-summary').textContent =
                'The consent policy could not be loaded, so nothing can be analysed. '
                + 'Please reload the page.';
        });
}

function updateConsentButton() {
    // Only the required box gates the button; donation is genuinely optional.
    document.getElementById('consent-btn').disabled =
        !document.getElementById('consent-analysis').checked;
}

function grantConsent() {
    const analysis = document.getElementById('consent-analysis').checked;
    const donate = document.getElementById('consent-donate').checked;
    if (!analysis) { return; }

    fetch('/api/consent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ analysis: analysis, donate: donate })
    })
    .then(response => response.json().then(body => ({ ok: response.ok, body: body })))
    .then(result => {
        if (!result.ok) {
            alert(result.body.error || 'Consent could not be recorded.');
            return;
        }
        rememberConsent(result.body.participant_id);
        donateConsent = result.body.donate;
        showConsentedView();
    })
    .catch(() => alert('Consent could not be recorded. Please try again.'));
}

// Reveal the tool. Called after consent is granted, and on load for a returning
// participant whose token the server still recognises.
function showConsentedView() {
    hideCard('consent-card');
    showCard('test-card');
    showCard('data-card');
    document.getElementById('donate-toggle').checked = donateConsent;
    document.getElementById('data-status').textContent = donateConsent
        ? 'This session’s timing features will be stored for research when you analyse.'
        : 'Nothing is being stored. Analysis runs and the result is discarded.';
    announce('Consent recorded. The typing session is now available.');
}

// Show the gate again, e.g. after deletion or a token the server no longer knows.
function showConsentGate(message) {
    showCard('consent-card');
    hideCard('test-card');
    hideCard('data-card');
    document.getElementById('results-card').classList.remove('show');
    document.getElementById('consent-analysis').checked = false;
    document.getElementById('consent-donate').checked = false;
    document.getElementById('data-output').textContent = '';
    hideCard('questionnaire-card');
    donationId = null;
    updateConsentButton();
    announce(message || 'Consent is required before anything can be analysed.');
    if (message) { alert(message); }
}

// A stored token is only trusted if the server still has the record: it may have been
// deleted from another tab, or the database reset. Verifying avoids showing someone a
// working tool that will 403 the moment they press Analyze.
function restoreConsent() {
    const id = consentId();
    if (!id) { return Promise.resolve(); }

    return fetch('/api/data/' + encodeURIComponent(id))
        .then(response => {
            if (!response.ok) {
                forgetConsent();
                return null;
            }
            return response.json();
        })
        .then(summary => {
            if (!summary) { return; }
            donateConsent = summary.donate;
            showConsentedView();
        })
        .catch(() => { /* offline: leave the gate up rather than assume consent. */ });
}

function changeDonateConsent() {
    const id = consentId();
    const donate = document.getElementById('donate-toggle').checked;
    if (!id) { return; }

    fetch('/api/consent/' + encodeURIComponent(id), {
        method: 'PATCH',
        headers: consentHeaders(),
        body: JSON.stringify({ analysis: true, donate: donate })
    })
    .then(response => response.json().then(body => ({ ok: response.ok, body: body })))
    .then(result => {
        if (!result.ok) {
            document.getElementById('donate-toggle').checked = donateConsent;
            alert(result.body.error || 'That change could not be saved.');
            return;
        }
        donateConsent = result.body.donate;
        showConsentedView();
    })
    .catch(() => {
        document.getElementById('donate-toggle').checked = donateConsent;
        alert('That change could not be saved. Please try again.');
    });
}

function viewMyData() {
    const id = consentId();
    if (!id) { return; }

    fetch('/api/data/' + encodeURIComponent(id))
        .then(response => response.json().then(body => ({ ok: response.ok, body: body })))
        .then(result => {
            if (!result.ok) {
                forgetConsent();
                showConsentGate('That record no longer exists. Please consent again to continue.');
                return;
            }
            // Printed verbatim: transparency means showing exactly what is held, not a
            // summary of it. It is timing features and consent flags - nothing else exists.
            document.getElementById('data-output').textContent =
                JSON.stringify(result.body, null, 2);
            announce('Everything stored about you is now shown below.');
        })
        .catch(() => alert('Your data could not be loaded. Please try again.'));
}

function deleteMyData() {
    const id = consentId();
    if (!id) { return; }
    if (!window.confirm(
        'Permanently delete your consent record and every stored session? '
        + 'This cannot be undone.')) {
        return;
    }

    fetch('/api/data/' + encodeURIComponent(id), { method: 'DELETE' })
        .then(response => {
            forgetConsent();
            showConsentGate(response.ok
                ? 'Everything stored about you has been deleted.'
                : 'There was nothing left to delete.');
        })
        .catch(() => alert('Deletion failed. Please try again.'));
}

// Donation is never silent: it happens only with an active opt-in, and the results card
// says so either way.
function donateSession() {
    const note = document.getElementById('donation-note');
    if (!donateConsent) {
        note.textContent = 'This session was not stored. Analysis ran and the timing was '
            + 'discarded.';
        return;
    }

    fetch('/api/donate', {
        method: 'POST',
        headers: consentHeaders(),
        body: JSON.stringify({
            keystroke_events: keystrokeData.map(k => ({
                timestamp: k.timestamp,
                is_backspace: k.is_backspace
            }))
        })
    })
    .then(response => response.json().then(body => ({ ok: response.ok, body: body })))
    .then(result => {
        if (result.ok) {
            // Remember which session was stored, so the questionnaire can label *this*
            // one rather than the participant in general.
            donationId = result.body.donation_id;
        }
        note.textContent = result.ok
            ? 'You opted in, so this session’s five timing features were stored for '
              + 'research. Use "Your data" below to see or delete them.'
            : 'This session could not be stored, so nothing was saved for it.';
    })
    .catch(() => {
        note.textContent = 'This session could not be stored, so nothing was saved for it.';
    });
}

// -------------------------------------------------------------------------------------
// The research questionnaire (F4)
//
// This is the label side of the dataset. It is offered to everyone who has consented,
// not only to donors: someone who wants their own score without contributing it is a
// legitimate visitor, and the server simply does not store the answers in that case.
//
// PRIVACY: the payload is item id → integer scale value. There is no free-text field
// anywhere in this flow, deliberately — it is the one control type that could carry
// content, and it is where content would eventually arrive.
// -------------------------------------------------------------------------------------

// The donation this questionnaire labels, when there is one. Null means the answers are
// scored (and possibly stored) without being tied to a typing session.
let donationId = null;

// The instrument as served, cached so re-answering does not re-fetch it.
let instrument = null;

function loadInstrument() {
    if (instrument) { return Promise.resolve(instrument); }

    return fetch('/api/instrument')
        .then(response => response.json())
        .then(payload => {
            instrument = payload;
            renderInstrument(payload);
            return payload;
        })
        .catch(() => {
            document.getElementById('questionnaire-progress').textContent =
                'The questionnaire could not be loaded. Your typing session is unaffected.';
            return null;
        });
}

// Provenance first, then the items. Both are built with createElement and textContent
// rather than innerHTML: server-supplied strings are never parsed as markup, which keeps
// this honest under the strict CSP and immune to a malformed item text.
function renderInstrument(payload) {
    document.getElementById('instrument-disclaimer').textContent = payload.disclaimer;
    document.getElementById('instrument-name').textContent = payload.name;
    document.getElementById('instrument-citation').textContent = payload.citation;
    document.getElementById('instrument-adaptation').textContent = payload.adaptation_note;

    const container = document.getElementById('questionnaire-items');
    container.textContent = '';

    payload.subscales.forEach(function (subscale) {
        const items = payload.items.filter(item => item.subscale === subscale.id);
        if (!items.length) { return; }

        const group = document.createElement('div');
        group.className = 'question-group';

        const title = document.createElement('h3');
        title.textContent = subscale.label;
        group.appendChild(title);

        const description = document.createElement('p');
        description.className = 'question-group-description';
        description.textContent = subscale.description;
        group.appendChild(description);

        items.forEach(function (item) {
            group.appendChild(renderItem(item, payload.scales[item.scale]));
        });
        container.appendChild(group);
    });

    updateQuestionnaireProgress();
}

// One item is a fieldset of radios: a radio group is the honest control for "pick exactly
// one of five ordered options", and a fieldset/legend gives a screen reader the question
// text with each option instead of five bare labels.
function renderItem(item, scale) {
    const fieldset = document.createElement('fieldset');
    fieldset.className = 'question';
    fieldset.dataset.itemId = item.id;

    const legend = document.createElement('legend');
    legend.textContent = item.text;
    fieldset.appendChild(legend);

    const options = document.createElement('div');
    options.className = 'question-options';

    scale.options.forEach(function (option) {
        const label = document.createElement('label');
        label.className = 'question-option';

        const input = document.createElement('input');
        input.type = 'radio';
        input.name = 'item-' + item.id;
        input.value = String(option.value);
        input.addEventListener('change', updateQuestionnaireProgress);

        const text = document.createElement('span');
        text.textContent = option.label;

        label.appendChild(input);
        label.appendChild(text);
        options.appendChild(label);
    });

    fieldset.appendChild(options);
    return fieldset;
}

// Collect answers as item id → integer. Unanswered items are simply absent, which is what
// makes the count below meaningful.
function collectAnswers() {
    const answers = {};
    document.querySelectorAll('#questionnaire-items fieldset.question').forEach(function (field) {
        const chosen = field.querySelector('input[type="radio"]:checked');
        if (chosen) { answers[field.dataset.itemId] = parseInt(chosen.value, 10); }
    });
    return answers;
}

function updateQuestionnaireProgress() {
    if (!instrument) { return; }

    const answered = Object.keys(collectAnswers()).length;
    const total = instrument.items.length;
    const complete = answered === total;

    document.getElementById('questionnaire-submit').disabled = !complete;
    document.getElementById('questionnaire-progress').textContent = complete
        ? 'All questions answered.'
        : answered + ' of ' + total + ' questions answered.';
}

function submitQuestionnaire() {
    const answers = collectAnswers();
    if (!instrument || Object.keys(answers).length !== instrument.items.length) { return; }

    const body = { responses: answers };
    if (donationId !== null) { body.donation_id = donationId; }

    fetch('/api/questionnaire', {
        method: 'POST',
        headers: consentHeaders(),
        body: JSON.stringify(body)
    })
    .then(response => response.json().then(payload => ({ ok: response.ok, body: payload })))
    .then(result => {
        if (!result.ok) {
            alert(result.body.error || 'Your answers could not be scored.');
            return;
        }
        showQuestionnaireResult(result.body);
    })
    .catch(() => alert('Your answers could not be scored. Please try again.'));
}

function showQuestionnaireResult(result) {
    const scores = document.getElementById('questionnaire-scores');
    scores.textContent = '';

    Object.keys(result.subscale_scores).forEach(function (subscale) {
        const row = document.createElement('div');
        row.className = 'score-row';

        const name = document.createElement('span');
        name.textContent = result.subscale_labels[subscale] || subscale;

        const value = document.createElement('strong');
        // Out of 100 is stated every time: a bare "62" invites being read as a percentage
        // of something, or as a probability of being burned out. It is neither.
        value.textContent = result.subscale_scores[subscale] + ' out of 100';

        row.appendChild(name);
        row.appendChild(value);
        scores.appendChild(row);
    });

    document.getElementById('questionnaire-band').textContent =
        'Overall: ' + result.overall_score + ' out of 100 — ' + result.band + '.';
    document.getElementById('questionnaire-caveat').textContent = result.caveat;
    document.getElementById('questionnaire-storage').textContent = result.storage_note;

    document.getElementById('questionnaire-result').classList.remove('is-hidden');
    announce('Questionnaire scored. ' + result.band + '. ' + result.caveat);
}

// Offered once a session has been analysed, so the questionnaire labels a real session
// rather than arriving out of context.
function showQuestionnaire() {
    showCard('questionnaire-card');
    loadInstrument();
}

loadConsentPolicy();
restoreConsent();

// -------------------------------------------------------------------------------------
// Accessibility
//
// The page carries `#announcer`, a polite live region, and offers Ctrl+Enter as a
// shortcut. Both were markup-only until now: a screen-reader user got no notification
// when a card swapped, and the advertised shortcut did nothing.
//
// PRIVACY: announcements are fixed strings and server-supplied labels. Nothing derived
// from the typed text is ever put here — a live region is read aloud, so it is the last
// place content should be able to reach.
// -------------------------------------------------------------------------------------

function announce(message) {
    const region = document.getElementById('announcer');
    if (!region) { return; }
    // Clearing first makes a repeat of the same message announce again; assistive
    // technology ignores a write that does not change the text.
    region.textContent = '';
    region.textContent = message;
}

// Ctrl+Enter (Cmd+Enter on a Mac) analyses without leaving the keyboard. `event.key` is
// compared and discarded exactly as the Backspace check is — no key identity is stored.
function handleShortcut(event) {
    if (!(event.ctrlKey || event.metaKey) || event.key !== 'Enter') { return; }
    if (analyzeBtn.disabled) { return; }
    event.preventDefault();
    analyzeTyping();
}

typingArea.addEventListener('keydown', handleShortcut);

function recordKeyDown(event) {
    const timestamp = performance.now() / 1000; // Convert to seconds

    if (startTime === null) {
        startTime = timestamp;
        updateInterval = setInterval(updateDuration, 100);
    }

    // Record keystroke (only timing, not content)
    keystrokeData.push({
        timestamp: timestamp,
        is_backspace: event.key === 'Backspace'
    });

    updateStats();
}

function updateStats() {
    const keyCount = keystrokeData.length;
    const backspaceCount = keystrokeData.filter(k => k.is_backspace).length;
    const duration = startTime ? (performance.now() / 1000 - startTime) : 0;
    const speed = duration > 0 ? (keyCount / duration).toFixed(1) : '0.0';

    document.getElementById('key-count').textContent = keyCount;
    document.getElementById('backspace-count').textContent = backspaceCount;
    document.getElementById('speed').textContent = speed;

    // Enable analyze button once the session reaches the server's documented minimum
    // (MIN_KEYSTROKE_EVENTS in keystress/api/predict.py). The client gate and the server
    // contract must not drift apart: anything the server would accept, the page allows.
    analyzeBtn.disabled = keyCount < 5;
}

function updateDuration() {
    if (startTime) {
        const duration = (performance.now() / 1000 - startTime).toFixed(1);
        document.getElementById('duration').textContent = duration + 's';
    }
}

function resetTest() {
    keystrokeData = [];
    startTime = null;
    if (updateInterval) {
        clearInterval(updateInterval);
        updateInterval = null;
    }
    
    typingArea.value = '';
    document.getElementById('key-count').textContent = '0';
    document.getElementById('backspace-count').textContent = '0';
    document.getElementById('duration').textContent = '0.0s';
    document.getElementById('speed').textContent = '0.0';
    analyzeBtn.disabled = true;
}

function analyzeTyping() {
    if (keystrokeData.length < 5) {
        alert('Please type at least 5 keystrokes - shorter sessions carry too little timing signal to analyze.');
        return;
    }

    // Show loader
    hideCard('test-card');
    document.getElementById('loader-card').classList.add('show');
    announce('Analysing your typing session.');

    // Prepare data for API
    const data = {
        keystroke_events: keystrokeData.map(k => ({
            timestamp: k.timestamp,
            is_backspace: k.is_backspace
        }))
    };

    // Send to API. The consent token travels in the header; without a valid one the
    // server refuses to analyse at all (F2).
    fetch('/api/predict', {
        method: 'POST',
        headers: consentHeaders(),
        body: JSON.stringify(data)
    })
    .then(response => response.json().then(body => ({
        ok: response.ok, status: response.status, body: body
    })))
    .then(result => {
        if (result.ok) {
            displayResults(result.body);
            return;
        }
        // A rejected request must never be rendered as a result. 403 means the token is
        // gone or withdrawn, which is a consent problem, not a typing problem.
        returnToTest();
        if (result.status === 403) {
            forgetConsent();
            showConsentGate('Analysis needs your consent. Please agree again to continue.');
        } else {
            alert(result.body.error || 'An error occurred. Please try again.');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred. Please try again.');
        returnToTest();
    });
}

// Undo the loader/hidden-card state set before a request, so a failure leaves the page
// usable rather than stuck on a spinner.
function returnToTest() {
    document.getElementById('loader-card').classList.remove('show');
    showCard('test-card');
}

// Human-readable qualifier for a data_source value. Every number rendered on this
// page passes through here — an unqualified metric should be impossible to display.
function sourceQualifier(dataSource) {
    if (dataSource === 'real') { return 'on real validated data'; }
    if (dataSource === 'synthetic') { return 'on synthetic data'; }
    return 'on data of unknown origin';
}

// Result icons as inline SVGs (stroke-based, currentColor). Replaces the Font
// Awesome glyphs so the page needs no icon font. Each string is self-contained
// and aria-hidden; the surrounding .result-icon circle already conveys meaning.
const RESULT_ICON_SVGS = {
    low: '<svg class="icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" ' +
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" ' +
        'aria-hidden="true" focusable="false"><path d="M20 6 9 17l-5-5"/></svg>',
    medium: '<svg class="icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" ' +
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" ' +
        'aria-hidden="true" focusable="false"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>' +
        '<path d="M12 9v4"/><path d="M12 17h.01"/></svg>',
    high: '<svg class="icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" ' +
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" ' +
        'aria-hidden="true" focusable="false"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>'
};

const RESULT_ICON_UNKNOWN =
    '<svg class="icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" ' +
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" ' +
    'aria-hidden="true" focusable="false"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>' +
    '<path d="M12 17h.01"/></svg>';

function displayResults(result) {
    // Hide loader, show results
    document.getElementById('loader-card').classList.remove('show');
    document.getElementById('results-card').classList.add('show');

    const dataSource = result.data_source || 'unknown';
    const qualifier = sourceQualifier(dataSource);
    const modelVersion = result.model_version || 'unknown';

    document.getElementById('disclaimer-text').textContent = result.disclaimer || '';

    const icon = document.getElementById('result-icon');
    const levelEl = document.getElementById('result-level');
    const probabilitySection = document.getElementById('probability-section');
    const sourceNote = document.getElementById('source-note');

    // Insufficient signal: say so plainly rather than rendering an invented score.
    if (result.insufficient_data) {
        icon.className = 'result-icon medium';
        icon.innerHTML = RESULT_ICON_UNKNOWN;
        levelEl.textContent = result.label;
        levelEl.className = 'result-level medium';
        document.getElementById('result-confidence').textContent = '';
        document.getElementById('result-description').textContent = result.description;
        probabilitySection.classList.add('is-hidden');
        // A session with too little signal is not worth donating, and saying so is
        // clearer than leaving the storage note blank.
        document.getElementById('donation-note').textContent =
            'This session carried too little timing signal to analyse, so nothing was stored.';
        sourceNote.textContent =
            'No indicator was produced, so there is no number to report. '
            + 'Model ' + modelVersion + ' (trained ' + qualifier + ').';
        announce(result.label + '. ' + result.description);
        return;
    }

    probabilitySection.classList.remove('is-hidden');

    icon.className = 'result-icon ' + result.level_class;
    icon.innerHTML = RESULT_ICON_SVGS[result.level_class] || RESULT_ICON_UNKNOWN;

    levelEl.textContent = result.label;
    levelEl.className = 'result-level ' + result.level_class;

    // Confidence never appears without its source. It is also a raw model score,
    // not a calibrated probability — F7 addresses that; until then we say so.
    document.getElementById('result-confidence').textContent =
        'Model confidence: ' + (result.confidence * 100).toFixed(0) + '% '
        + '(uncalibrated, ' + qualifier + ')';

    document.getElementById('result-description').textContent = result.description;

    // Probabilities are ordered by class index (ARCHITECTURE.md 4.3); labels come
    // from the response so the page never hard-codes them.
    const probs = result.probabilities;
    const labels = result.labels || ['Low (indicator)', 'Medium (indicator)', 'High (indicator)'];
    const slots = ['low', 'medium', 'high'];

    document.getElementById('probability-heading').textContent =
        'Indicator breakdown (' + qualifier + ')';

    slots.forEach(function (slot, i) {
        document.getElementById('label-' + slot).textContent = labels[i];
        document.getElementById('prob-' + slot).textContent = (probs[i] * 100).toFixed(0) + '%';
        document.getElementById('bar-' + slot).style.width = (probs[i] * 100) + '%';
    });

    sourceNote.textContent =
        'Every percentage above was produced by model ' + modelVersion
        + ', trained ' + qualifier + '. '
        + (dataSource === 'synthetic'
            ? 'The training categories were defined by hand in the data generator, '
              + 'so these numbers describe how separable those authored categories are '
              + '- not any demonstrated ability to detect real burnout.'
            : '');

    // The announcement carries the same qualifier the page shows, so a screen-reader
    // user is never told a bare number either.
    announce('Result: ' + result.label + '. Model confidence '
        + (result.confidence * 100).toFixed(0) + ' percent, uncalibrated, ' + qualifier + '.');

    donateSession();
    showQuestionnaire();
}

function newTest() {
    document.getElementById('results-card').classList.remove('show');
    showCard('test-card');
    document.getElementById('donation-note').textContent = '';
    hideCard('questionnaire-card');
    donationId = null;
    resetTest();
    announce('Ready for a new typing session.');
}
