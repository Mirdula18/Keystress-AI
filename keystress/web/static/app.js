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
    ['delete-btn', 'click', deleteMyData]
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
    document.getElementById('consent-card').style.display = 'none';
    document.getElementById('test-card').style.display = 'block';
    document.getElementById('data-card').style.display = 'block';
    document.getElementById('donate-toggle').checked = donateConsent;
    document.getElementById('data-status').textContent = donateConsent
        ? 'This session’s timing features will be stored for research when you analyse.'
        : 'Nothing is being stored. Analysis runs and the result is discarded.';
}

// Show the gate again, e.g. after deletion or a token the server no longer knows.
function showConsentGate(message) {
    document.getElementById('consent-card').style.display = 'block';
    document.getElementById('test-card').style.display = 'none';
    document.getElementById('data-card').style.display = 'none';
    document.getElementById('results-card').classList.remove('show');
    document.getElementById('consent-analysis').checked = false;
    document.getElementById('consent-donate').checked = false;
    document.getElementById('data-output').textContent = '';
    updateConsentButton();
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
    .then(response => {
        note.textContent = response.ok
            ? 'You opted in, so this session’s five timing features were stored for '
              + 'research. Use "Your data" below to see or delete them.'
            : 'This session could not be stored, so nothing was saved for it.';
    })
    .catch(() => {
        note.textContent = 'This session could not be stored, so nothing was saved for it.';
    });
}

loadConsentPolicy();
restoreConsent();

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
    document.getElementById('test-card').style.display = 'none';
    document.getElementById('loader-card').classList.add('show');

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
    document.getElementById('test-card').style.display = 'block';
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
        probabilitySection.style.display = 'none';
        // A session with too little signal is not worth donating, and saying so is
        // clearer than leaving the storage note blank.
        document.getElementById('donation-note').textContent =
            'This session carried too little timing signal to analyse, so nothing was stored.';
        sourceNote.textContent =
            'No indicator was produced, so there is no number to report. '
            + 'Model ' + modelVersion + ' (trained ' + qualifier + ').';
        return;
    }

    probabilitySection.style.display = 'block';

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

    donateSession();
}

function newTest() {
    document.getElementById('results-card').classList.remove('show');
    document.getElementById('test-card').style.display = 'block';
    document.getElementById('donation-note').textContent = '';
    resetTest();
}
