"""
Keystress-AI: Flask Web Application

A privacy-preserving burnout detection system that analyzes
typing behavior patterns to assess academic stress levels.

This application provides:
- A modern, responsive UI for typing tests
- Real-time keystroke metadata collection
- AI-powered burnout risk assessment
- Privacy-first design (no content storage)
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
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print("⚠ Model not found. Running training first...")
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
        print("✓ Model trained and loaded")


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
            <p>Detect academic burnout through typing patterns</p>
        </header>

        <!-- Typing Test Card -->
        <div class="card" id="test-card">
            <div class="card-header">
                <i class="fas fa-keyboard"></i>
                <h2>Typing Test</h2>
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
                    <i class="fas fa-chart-line"></i> Analyze Burnout Risk
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
                <div class="result-level" id="result-level">Low Burnout</div>
                <div class="result-confidence" id="result-confidence">Confidence: 85%</div>
            </div>

            <div class="result-description" id="result-description">
                Your typing patterns indicate healthy, consistent behavior.
            </div>

            <div class="probability-section">
                <h3>Risk Assessment Breakdown</h3>
                <div class="probability-bar low">
                    <div class="label">
                        <span>Low Burnout</span>
                        <span id="prob-low">0%</span>
                    </div>
                    <div class="bar">
                        <div class="fill" id="bar-low" style="width: 0%"></div>
                    </div>
                </div>
                <div class="probability-bar medium">
                    <div class="label">
                        <span>Medium Burnout</span>
                        <span id="prob-medium">0%</span>
                    </div>
                    <div class="bar">
                        <div class="fill" id="bar-medium" style="width: 0%"></div>
                    </div>
                </div>
                <div class="probability-bar high">
                    <div class="label">
                        <span>High Burnout</span>
                        <span id="prob-high">0%</span>
                    </div>
                    <div class="bar">
                        <div class="fill" id="bar-high" style="width: 0%"></div>
                    </div>
                </div>
            </div>

            <div class="btn-group">
                <button class="btn btn-primary" onclick="newTest()">
                    <i class="fas fa-redo"></i> Take Another Test
                </button>
            </div>

            <div class="privacy-notice">
                <i class="fas fa-info-circle"></i>
                <p><strong>Disclaimer:</strong> This tool provides an estimate based on typing patterns and is not a medical diagnosis. If you're experiencing burnout, please consult a healthcare professional or counselor.</p>
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
                alert('Please type more (at least 20 keystrokes) for accurate analysis.');
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

        function displayResults(result) {
            // Hide loader, show results
            document.getElementById('loader-card').classList.remove('show');
            document.getElementById('results-card').classList.add('show');

            // Update result icon
            const icon = document.getElementById('result-icon');
            icon.className = 'result-icon ' + result.level_class;
            
            const iconMap = {
                'low': 'fa-check',
                'medium': 'fa-exclamation',
                'high': 'fa-times'
            };
            icon.innerHTML = '<i class="fas ' + iconMap[result.level_class] + '"></i>';

            // Update level and confidence
            const levelEl = document.getElementById('result-level');
            levelEl.textContent = result.label;
            levelEl.className = 'result-level ' + result.level_class;

            document.getElementById('result-confidence').textContent = 
                'Confidence: ' + (result.confidence * 100).toFixed(0) + '%';

            // Update description
            document.getElementById('result-description').textContent = result.description;

            // Update probability bars
            const probs = result.probabilities;
            document.getElementById('prob-low').textContent = (probs['Low Burnout'] * 100).toFixed(0) + '%';
            document.getElementById('prob-medium').textContent = (probs['Medium Burnout'] * 100).toFixed(0) + '%';
            document.getElementById('prob-high').textContent = (probs['High Burnout'] * 100).toFixed(0) + '%';

            document.getElementById('bar-low').style.width = (probs['Low Burnout'] * 100) + '%';
            document.getElementById('bar-medium').style.width = (probs['Medium Burnout'] * 100) + '%';
            document.getElementById('bar-high').style.width = (probs['High Burnout'] * 100) + '%';
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
        
        # Get prediction
        result = get_prediction_details(features, model=model, scaler=scaler)
        
        # Determine level class for styling
        level_classes = {0: 'low', 1: 'medium', 2: 'high'}
        result['level_class'] = level_classes.get(result['prediction'], 'low')
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'An error occurred while processing your request'}), 500


@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("🧠 KEYSTRESS-AI: Academic Burnout Detection")
    print("=" * 50)
    
    # Load models
    load_models()
    
    print("\n🚀 Starting Flask server...")
    print("📍 Open http://127.0.0.1:5000 in your browser")
    print("⚡ Press Ctrl+C to stop the server")
    print("=" * 50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
