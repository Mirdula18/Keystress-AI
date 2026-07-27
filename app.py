"""
Keystress-AI: Flask Web Application

A privacy-preserving research prototype that examines whether typing rhythm relates to
academic wellbeing. It produces research *indicators*, not assessments or diagnoses.

This application provides:
- A UI for recording a typing session
- Keystroke metadata collection (timing and correction flags only, never content)
- A burnout risk *indicator* with its data source and uncertainty attached
- Privacy-first design (no content storage)

The served model is trained on synthetic data whose classes were hand-authored by the
generator, so every number it produces is labelled ``data_source: "synthetic"`` and means
only that. See ``docs/CLAUDE.md`` §1.
"""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify
import numpy as np

# Import local modules
from src.feature_engineering import extract_typing_features
from src.predict import predict_burnout, get_prediction_details, BURNOUT_LABELS

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))

# Global model and scaler (loaded on startup)
model = None
scaler = None


def load_models():
    """Load trained model and scaler on startup."""
    global model, scaler
    try:
        from src.predict import load_trained_model
        model, scaler = load_trained_model()
        print("Model loaded successfully")
    except FileNotFoundError:
        print("Model not found. Running training first...")
        from src.train_model import train_and_evaluate
        from src.generate_synthetic_data import generate_synthetic_typing_data, save_synthetic_data
        
        # Generate data if not exists
        if not os.path.exists('data/synthetic_typing_data.csv'):
            print("Generating synthetic data...")
            df = generate_synthetic_typing_data(n_samples=1500)
            save_synthetic_data(df)
        
        # Train model
        results = train_and_evaluate()
        model = results['model']
        scaler = results['scaler']
        print("Model trained and loaded")


# HTML template as a string (for self-contained app)
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


@app.route('/')
def index():
    """Render the main application page."""
    return HTML_TEMPLATE


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for burnout prediction.
    
    Expects JSON with 'keystroke_events' containing list of
    {timestamp, is_backspace} objects.
    
    Returns JSON with prediction results.
    """
    global model, scaler
    
    try:
        data = request.get_json()
        
        if not data or 'keystroke_events' not in data:
            return jsonify({'error': 'No keystroke data provided'}), 400
        
        keystroke_events = data['keystroke_events']
        
        if len(keystroke_events) < 5:
            return jsonify({'error': 'Insufficient keystroke data'}), 400
        
        # Process keystroke data to extract features
        from src.collect_typing_data import process_keystroke_data
        session_data = process_keystroke_data(keystroke_events)
        
        # Extract typing features
        features = extract_typing_features(session_data)
        
        # Get prediction. The response carries data_source, model_version, disclaimer,
        # and insufficient_data by construction (see src/predict.py).
        result = get_prediction_details(features, model=model, scaler=scaler)

        # Determine level class for styling. An absent prediction (insufficient signal)
        # must not fall back to 'low' — that would style a non-result as a reassuring one.
        level_classes = {0: 'low', 1: 'medium', 2: 'high'}
        result['level_class'] = level_classes.get(result['prediction'], 'unknown')

        return jsonify(result)
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'An error occurred while processing your request'}), 500


@app.route('/api/health')
def health_check():
    """
    Health check endpoint.

    Reports which model is loaded and what data it was trained on, so an operator can
    never be unsure whether a running instance is serving synthetic-trained predictions.
    """
    from src.predict import load_model_metadata

    metadata = load_model_metadata()
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_version': metadata.get('model_version', 'unknown'),
        'data_source': metadata.get('data_source', 'unknown'),
        'feature_set': metadata.get('feature_set', 'unknown'),
    })


if __name__ == '__main__':
    print("\n" + "=" * 72)
    print("KEYSTRESS-AI: typing-dynamics research prototype")
    print("Research indicators only - not a diagnostic tool. Model trained on")
    print("synthetic data; no real-world performance has been established.")
    print("=" * 72)

    # Load models
    load_models()

    # Get debug mode from environment (default: False for security)
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'

    # Local-first (docs/CLAUDE.md HARD RULE 5): bind loopback only. Exposing this to a
    # network is an explicit opt-in, never the default. Full config handling is F3/F14.
    host = os.environ.get('KEYSTRESS_HOST', '127.0.0.1')
    port = int(os.environ.get('KEYSTRESS_PORT', '5000'))

    print(f"\nStarting Flask development server on http://{host}:{port}")
    if host not in ('127.0.0.1', 'localhost', '::1'):
        print(f"WARNING: bound to {host}, which may be reachable from your network.")
        print("         Raw keystroke timing is sensitive data - prefer 127.0.0.1.")
    print("Press Ctrl+C to stop the server")
    print("=" * 72 + "\n")

    app.run(debug=debug_mode, host=host, port=port)
