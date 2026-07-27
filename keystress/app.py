"""
Keystress-AI Flask application factory.

A privacy-preserving research prototype examining whether typing rhythm relates to
academic wellbeing. It produces research *indicators*, not assessments or diagnoses.

The served model is trained on synthetic data whose classes were hand-authored by the
generator, so every number it produces is labelled ``data_source: "synthetic"`` and means
only that. See ``docs/CLAUDE.md`` §1.

Structure
---------
This module is a thin entrypoint: it builds the app, wires the model registry, and
registers the API blueprints. Request handling lives in ``keystress.api``, and the
domain logic in ``keystress.core``. There is no module-level mutable model state — the
registry attached to ``app.extensions`` owns the loaded model (F11).

The page markup below is still an inline string. Extracting it into ``web/`` is F10, the
next feature; it is left untouched here so that change stays behaviour-preserving and
independently reviewable.
"""

from __future__ import annotations

import logging

from flask import Flask

from .config import Settings, load_settings
from .core.model import ModelRegistry, ModelUnavailableError

logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """
    Configure application logging.

    Replaces the inherited ``print`` calls (F11). Emoji are deliberately absent: they
    raise ``UnicodeEncodeError`` on legacy Windows console codepages, which turned a
    cosmetic banner into a startup crash.

    Parameters:
        level: Log level name; an unrecognised value falls back to INFO.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_model(registry: ModelRegistry, settings: Settings) -> None:
    """
    Load the model, training one from synthetic data if none exists.

    Parameters:
        registry: The registry to populate.
        settings: Resolved settings supplying artifact paths.

    Note:
        A failure here is logged and swallowed rather than raised. The app must still
        start so that ``/api/health`` and ``/readyz`` can report the degraded state —
        HARD RULE 6 asks for a clear message, and a process that refuses to boot cannot
        deliver one.
    """
    try:
        registry.load(settings.model_path, settings.scaler_path, settings.metadata_path)
        return
    except ModelUnavailableError as exc:
        logger.warning("%s", exc)

    if not settings.auto_train:
        logger.error("No model available and auto-training is disabled")
        return

    logger.info("Training a model from synthetic data (this happens once)")
    try:
        from .ml.synthetic import generate_synthetic_typing_data, save_synthetic_data
        from .ml.train import train_and_evaluate

        if not settings.data_path.exists():
            save_synthetic_data(
                generate_synthetic_typing_data(n_samples=1500), settings.data_path
            )

        train_and_evaluate(
            data_path=settings.data_path,
            model_path=settings.model_path,
            scaler_path=settings.scaler_path,
            metadata_path=settings.metadata_path,
        )
        registry.load(settings.model_path, settings.scaler_path, settings.metadata_path)
    except (ModelUnavailableError, FileNotFoundError, ValueError, OSError) as exc:
        logger.error("Could not train a model: %s. Predictions will be unavailable.", exc)


def create_app(settings: Settings | None = None,
               registry: ModelRegistry | None = None,
               load_model: bool = True) -> Flask:
    """
    Build the Flask application.

    Parameters:
        settings: Configuration; read from the environment when omitted.
        registry: Model registry to use. Injectable so tests can supply a fixture model
            without touching disk — the thing the inherited module globals made impossible.
        load_model: Whether to load or train a model at startup.

    Returns:
        Flask: The configured application.
    """
    settings = settings if settings is not None else load_settings()
    registry = registry if registry is not None else ModelRegistry()

    app = Flask(__name__)
    app.config["KEYSTRESS_SETTINGS"] = settings
    app.extensions["keystress_registry"] = registry

    from .api.health import bp as health_bp
    from .api.predict import bp as predict_bp

    app.register_blueprint(predict_bp)
    app.register_blueprint(health_bp)

    @app.route("/")
    def index() -> str:
        """Render the single-page application."""
        return HTML_TEMPLATE

    if load_model and not registry.is_loaded:
        ensure_model(registry, settings)

    return app


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Keystress-AI | Academic Burnout Detection</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --secondary: #22d3ee;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --dark: #1e293b;
            --light: #f8fafc;
            --gray: #64748b;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2rem;
        }

        .container {
            max-width: 800px;
            width: 100%;
        }

        /* Header */
        .header {
            text-align: center;
            color: white;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
        }

        .header h1 i {
            color: var(--secondary);
        }

        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        /* Card */
        .card {
            background: white;
            border-radius: 20px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
            padding: 2rem;
            margin-bottom: 1.5rem;
        }

        .card-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid #f1f5f9;
        }

        .card-header i {
            font-size: 1.5rem;
            color: var(--primary);
        }

        .card-header h2 {
            font-size: 1.25rem;
            color: var(--dark);
        }

        /* Typing Area */
        .typing-prompt {
            background: linear-gradient(135deg, #f0f4ff 0%, #e8f4f8 100%);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }

        .typing-prompt p {
            color: var(--dark);
            font-size: 1rem;
            line-height: 1.6;
        }

        .typing-prompt .label {
            display: inline-block;
            background: var(--primary);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
        }

        #typing-area {
            width: 100%;
            min-height: 150px;
            padding: 1.25rem;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            font-size: 1rem;
            font-family: inherit;
            resize: none;
            transition: all 0.3s ease;
        }

        #typing-area:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1);
        }

        #typing-area::placeholder {
            color: #94a3b8;
        }

        /* Stats */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }

        .stat-item {
            background: #f8fafc;
            padding: 1rem;
            border-radius: 12px;
            text-align: center;
        }

        .stat-item .value {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--primary);
        }

        .stat-item .label {
            font-size: 0.75rem;
            color: var(--gray);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Buttons */
        .btn-group {
            display: flex;
            gap: 1rem;
            margin-top: 1.5rem;
        }

        .btn {
            flex: 1;
            padding: 1rem 1.5rem;
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            transition: all 0.3s ease;
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            color: white;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(99, 102, 241, 0.3);
        }

        .btn-primary:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .btn-secondary {
            background: #f1f5f9;
            color: var(--dark);
        }

        .btn-secondary:hover {
            background: #e2e8f0;
        }

        /* Results */
        .results {
            display: none;
        }

        .results.show {
            display: block;
        }

        .result-header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .result-icon {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 1rem;
            font-size: 2rem;
        }

        .result-icon.low {
            background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
            color: var(--success);
        }

        .result-icon.medium {
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            color: var(--warning);
        }

        .result-icon.high {
            background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
            color: var(--danger);
        }

        .result-level {
            font-size: 1.75rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .result-level.low { color: var(--success); }
        .result-level.medium { color: var(--warning); }
        .result-level.high { color: var(--danger); }

        .result-confidence {
            color: var(--gray);
            font-size: 0.9rem;
        }

        .result-description {
            background: #f8fafc;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1.5rem 0;
            text-align: center;
            color: var(--dark);
            line-height: 1.6;
        }

        /* Probability bars */
        .probability-section h3 {
            font-size: 0.9rem;
            color: var(--gray);
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .probability-bar {
            margin-bottom: 0.75rem;
        }

        .probability-bar .label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.25rem;
            font-size: 0.85rem;
        }

        .probability-bar .bar {
            height: 8px;
            background: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
        }

        .probability-bar .fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }

        .probability-bar.low .fill { background: var(--success); }
        .probability-bar.medium .fill { background: var(--warning); }
        .probability-bar.high .fill { background: var(--danger); }

        /* Privacy notice */
        .privacy-notice {
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            border: 1px solid #86efac;
            border-radius: 12px;
            padding: 1rem 1.25rem;
            margin-top: 1.5rem;
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
        }

        .privacy-notice i {
            color: var(--success);
            font-size: 1.25rem;
            margin-top: 0.125rem;
        }

        .privacy-notice p {
            font-size: 0.875rem;
            color: #166534;
            line-height: 1.5;
        }

        /* Research-status banner (F1: the synthetic caveat is not fine print) */
        .research-banner {
            border-left: 5px solid var(--warning);
            padding: 1.25rem 1.5rem;
        }

        .research-banner p {
            font-size: 0.9rem;
            color: var(--dark);
            line-height: 1.6;
        }

        /* Data-source note attached to every displayed metric */
        .source-note {
            margin-top: 1.25rem;
            padding: 0.875rem 1rem;
            background: #f8fafc;
            border-radius: 10px;
            font-size: 0.8rem;
            color: var(--gray);
            line-height: 1.5;
        }

        .disclaimer-notice {
            background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
            border-color: #fcd34d;
        }

        .disclaimer-notice i {
            color: var(--warning);
        }

        .disclaimer-notice p {
            color: #78350f;
        }

        /* Footer */
        .footer {
            text-align: center;
            color: rgba(255, 255, 255, 0.8);
            margin-top: 2rem;
            font-size: 0.875rem;
        }

        .footer a {
            color: white;
            text-decoration: none;
        }

        .footer a:hover {
            text-decoration: underline;
        }

        /* Loader */
        .loader {
            display: none;
            text-align: center;
            padding: 2rem;
        }

        .loader.show {
            display: block;
        }

        .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid #e2e8f0;
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Responsive */
        @media (max-width: 600px) {
            .header h1 {
                font-size: 1.75rem;
            }
            
            .btn-group {
                flex-direction: column;
            }
            
            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1><i class="fas fa-brain"></i> Keystress-AI</h1>
            <p>A research prototype exploring whether typing rhythm relates to academic wellbeing</p>
        </header>

        <div class="card research-banner">
            <p><strong>Research prototype, not a burnout test.</strong> The model behind this page
            was trained on <strong>synthetic data</strong> whose burnout categories were written by
            hand in the data generator. It has never been tested against real burnout, so it cannot
            tell you whether you are burned out, and nothing here is a medical assessment.</p>
        </div>

        <!-- Typing Test Card -->
        <div class="card" id="test-card">
            <div class="card-header">
                <i class="fas fa-keyboard"></i>
                <h2>Typing Session</h2>
            </div>

            <div class="typing-prompt">
                <span class="label">TYPE THIS TEXT</span>
                <p id="prompt-text">The quick brown fox jumps over the lazy dog. This simple sentence contains every letter of the alphabet and helps us understand your typing patterns. Please type naturally and at your own pace.</p>
            </div>

            <textarea id="typing-area" placeholder="Start typing here..." autocomplete="off" spellcheck="false"></textarea>

            <div class="stats-grid">
                <div class="stat-item">
                    <div class="value" id="key-count">0</div>
                    <div class="label">Keys Pressed</div>
                </div>
                <div class="stat-item">
                    <div class="value" id="backspace-count">0</div>
                    <div class="label">Corrections</div>
                </div>
                <div class="stat-item">
                    <div class="value" id="duration">0.0s</div>
                    <div class="label">Duration</div>
                </div>
                <div class="stat-item">
                    <div class="value" id="speed">0.0</div>
                    <div class="label">Keys/Sec</div>
                </div>
            </div>

            <div class="btn-group">
                <button class="btn btn-secondary" onclick="resetTest()">
                    <i class="fas fa-redo"></i> Reset
                </button>
                <button class="btn btn-primary" id="analyze-btn" onclick="analyzeTyping()" disabled>
                    <i class="fas fa-chart-line"></i> Analyze Typing Session
                </button>
            </div>

            <div class="privacy-notice">
                <i class="fas fa-shield-alt"></i>
                <p><strong>Privacy First:</strong> We only analyze timing patterns between keystrokes. Your actual typed content is never stored or transmitted. Your privacy is our priority.</p>
            </div>
        </div>

        <!-- Loading State -->
        <div class="card loader" id="loader-card">
            <div class="spinner"></div>
            <p>Analyzing your typing patterns...</p>
        </div>

        <!-- Results Card -->
        <div class="card results" id="results-card">
            <div class="result-header">
                <div class="result-icon" id="result-icon">
                    <i class="fas fa-check"></i>
                </div>
                <div class="result-level" id="result-level"></div>
                <div class="result-confidence" id="result-confidence"></div>
            </div>

            <div class="result-description" id="result-description"></div>

            <div class="probability-section" id="probability-section">
                <h3 id="probability-heading">Indicator breakdown</h3>
                <div class="probability-bar low">
                    <div class="label">
                        <span id="label-low">Low (indicator)</span>
                        <span id="prob-low">0%</span>
                    </div>
                    <div class="bar">
                        <div class="fill" id="bar-low" style="width: 0%"></div>
                    </div>
                </div>
                <div class="probability-bar medium">
                    <div class="label">
                        <span id="label-medium">Medium (indicator)</span>
                        <span id="prob-medium">0%</span>
                    </div>
                    <div class="bar">
                        <div class="fill" id="bar-medium" style="width: 0%"></div>
                    </div>
                </div>
                <div class="probability-bar high">
                    <div class="label">
                        <span id="label-high">High (indicator)</span>
                        <span id="prob-high">0%</span>
                    </div>
                    <div class="bar">
                        <div class="fill" id="bar-high" style="width: 0%"></div>
                    </div>
                </div>
            </div>

            <div class="source-note" id="source-note"></div>

            <div class="btn-group">
                <button class="btn btn-primary" onclick="newTest()">
                    <i class="fas fa-redo"></i> Run Another Session
                </button>
            </div>

            <div class="privacy-notice disclaimer-notice">
                <i class="fas fa-info-circle"></i>
                <p><strong>Disclaimer:</strong> <span id="disclaimer-text"></span></p>
            </div>
        </div>

        <footer class="footer">
            <p>Built with ❤️ for academic wellness | <a href="https://github.com/Mirdula18/Keystress-AI" target="_blank">GitHub</a></p>
        </footer>
    </div>

    <script>
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

            // Enable analyze button after minimum keystrokes
            analyzeBtn.disabled = keyCount < 20;
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
            if (keystrokeData.length < 20) {
                alert('Please type at least 20 keystrokes - shorter sessions carry too little timing signal to analyze.');
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
                icon.innerHTML = '<i class="fas fa-question"></i>';
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
            const iconMap = {
                'low': 'fa-check',
                'medium': 'fa-exclamation',
                'high': 'fa-times'
            };
            icon.innerHTML = '<i class="fas ' + iconMap[result.level_class] + '"></i>';

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
    </script>
</body>
</html>
'''


def main() -> int:
    """
    Run the development server.

    Returns:
        int: Process exit code.
    """
    settings = load_settings()
    configure_logging(settings.log_level)

    logger.info("Keystress-AI: typing-dynamics research prototype")
    logger.info(
        "Research indicators only - not a diagnostic tool. The shipped model is "
        "trained on synthetic data; no real-world performance has been established."
    )

    app = create_app(settings)

    if not settings.is_loopback:
        # HARD RULE 5: local-first. A wider bind is allowed but never silent.
        logger.warning(
            "Bound to %s, which may be reachable from your network. Raw keystroke "
            "timing is sensitive; prefer 127.0.0.1.", settings.host,
        )

    logger.info("Serving on http://%s:%d", settings.host, settings.port)
    app.run(debug=settings.debug, host=settings.host, port=settings.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
