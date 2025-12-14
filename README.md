# 🧠 Keystress-AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

**Keystress-AI is a privacy-preserving machine learning system that detects early academic burnout by analyzing typing behavior patterns such as speed, pauses, and correction frequency — without capturing typed content.**

[Features](#-features) • [How It Works](#-how-it-works) • [Quick Start](#-quick-start) • [Project Structure](#-project-structure) • [Privacy](#-privacy--ethics)

</div>

---

## 📋 Overview

Academic burnout is a growing concern among students, often going undetected until it significantly impacts performance and well-being. Keystress-AI offers a **non-intrusive**, **privacy-first** approach to early detection by analyzing how students type rather than what they type.

### Problem Statement

Students experiencing burnout often exhibit changes in their typing behavior:
- **Slower typing speed** due to fatigue or difficulty concentrating
- **Longer pauses** while thinking or losing focus
- **More corrections** (backspaces) due to errors from distraction
- **Inconsistent patterns** reflecting cognitive load variability

### Our Solution

A machine learning-based system that:
1. Collects only typing **metadata** (timing patterns)
2. Never stores actual typed content
3. Provides real-time burnout risk assessment
4. Offers actionable insights for self-care

---

## ✨ Features

- 🔒 **Privacy-First Design** — Only timing metadata is collected; no content storage
- 🤖 **AI-Powered Analysis** — Random Forest classifier with 85%+ accuracy
- ⚡ **Real-Time Detection** — Instant burnout risk assessment
- 🎨 **Modern UI** — Clean, responsive Flask web interface
- 📊 **Detailed Insights** — Probability breakdown and feature analysis
- 🧪 **Synthetic Training** — Pre-generated realistic dataset included

---

## 🔬 How It Works

### 1. Data Collection
The system captures only timing-related metadata:
- Timestamp of each key press
- Time between consecutive keystrokes
- Number of backspaces (corrections)
- Total typing duration

### 2. Feature Engineering
From the raw metadata, we compute five key features:

| Feature | Description | Burnout Indicator |
|---------|-------------|-------------------|
| `avg_typing_speed` | Keys per second | ↓ Lower = Higher risk |
| `avg_inter_key_delay` | Average pause between keys | ↑ Higher = Higher risk |
| `max_pause_duration` | Longest pause | ↑ Higher = Higher risk |
| `backspace_ratio` | Corrections / Total keys | ↑ Higher = Higher risk |
| `typing_consistency` | Std. deviation of delays | ↑ Higher = Higher risk |

### 3. Machine Learning Model
- **Algorithm**: Random Forest Classifier
- **Classes**: Low (0), Medium (1), High (2) Burnout
- **Training Data**: 1,500 synthetic samples with realistic distributions
- **Metrics**: ~90% accuracy, balanced precision/recall

### 4. Prediction
Real-time classification with confidence scores and actionable recommendations.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

```bash
# Clone the repository
git clone https://github.com/Mirdula18/Keystress-AI.git
cd Keystress-AI

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Generate synthetic data and train the model (automatic on first run)
python app.py
```

Open your browser and navigate to `http://127.0.0.1:5000`

### Manual Training (Optional)

```bash
# Generate synthetic data
python -m src.generate_synthetic_data

# Train the model
python -m src.train_model
```

---

## 📁 Project Structure

```
Keystress-AI/
│
├── data/
│   └── synthetic_typing_data.csv    # Generated training dataset
│
├── models/
│   ├── burnout_model.pkl            # Trained Random Forest model
│   └── scaler.pkl                   # Feature scaler
│
├── src/
│   ├── __init__.py
│   ├── generate_synthetic_data.py   # Synthetic data generation
│   ├── collect_typing_data.py       # Keystroke metadata collector
│   ├── feature_engineering.py       # Feature extraction
│   ├── train_model.py               # Model training & evaluation
│   └── predict.py                   # Prediction module
│
├── app.py                           # Flask web application
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── LICENSE                          # MIT License
└── .gitignore                       # Git ignore rules
```

---

## 🛠 Technologies Used

| Category | Technology |
|----------|------------|
| Language | Python 3.8+ |
| ML Framework | scikit-learn |
| Data Processing | NumPy, Pandas |
| Web Framework | Flask |
| Frontend | HTML5, CSS3, JavaScript |
| UI Design | Custom CSS with modern gradients |

---

## 🔒 Privacy & Ethics

### Data Privacy Principles

1. **No Content Storage**: We never store, log, or transmit the actual characters typed
2. **Timing Only**: Only keystroke timing metadata is collected
3. **Local Processing**: All analysis happens locally on your machine
4. **Transparent Design**: Open-source code for full transparency

### Ethical Considerations

- This tool is **not a medical diagnostic device**
- Results are **estimates** based on typing patterns
- Users should consult healthcare professionals for burnout concerns
- No personal identifiers are collected or stored

### What We DON'T Collect
- ❌ Typed content or characters
- ❌ Personal information
- ❌ IP addresses or device identifiers
- ❌ Browsing history or cookies

### What We DO Collect
- ✅ Timestamp of keystrokes
- ✅ Time intervals between keys
- ✅ Backspace count (not what was deleted)
- ✅ Session duration

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~90% |
| Precision (weighted) | ~90% |
| Recall (weighted) | ~90% |
| F1-Score (weighted) | ~90% |

### Feature Importance
1. `avg_typing_speed` — Most predictive
2. `typing_consistency` — High importance
3. `avg_inter_key_delay` — Moderate importance
4. `max_pause_duration` — Moderate importance
5. `backspace_ratio` — Lower importance

---

## 🎯 Use Cases

- **Student Self-Assessment**: Quick check on academic stress levels
- **Wellness Programs**: Non-intrusive screening tool
- **Research**: Study typing behavior and cognitive load
- **Educational Demos**: ML/AI project showcase

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

This application is for **educational and demonstration purposes only**. It is not intended to diagnose, treat, cure, or prevent any medical condition. If you are experiencing burnout or mental health concerns, please seek help from a qualified healthcare professional.

---

<div align="center">

**Built with ❤️ for academic wellness**

⭐ Star this repo if you find it helpful!

</div>
