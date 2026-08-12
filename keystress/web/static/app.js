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

    // Send to API
    fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        displayResults(result);
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred. Please try again.');
        document.getElementById('loader-card').classList.remove('show');
        document.getElementById('test-card').style.display = 'block';
    });
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
}

function newTest() {
    document.getElementById('results-card').classList.remove('show');
    document.getElementById('test-card').style.display = 'block';
    resetTest();
}
