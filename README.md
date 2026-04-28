# 🚀 PlagiarismAI — Premium AI Assistant

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-lightgrey.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A polished, enterprise-grade plagiarism detection and rewrite assistant built for academic authenticity and advanced linguistic reconstruction. This project features a sophisticated Flask-driven backend paired with a premium, glassmorphism-inspired frontend.

**✨ Authors:** Manisha, Joita, Debasmita, Suchitra

---

## 📋 Table of Contents

- [🌟 Premium Features](#-premium-features)
- [🆕 Latest Updates](#-latest-updates)
- [📸 Screenshots](#-screenshots)
- [⚙️ Installation](#-installation)
- [🚀 Usage](#-usage)
- [🧠 Model & Training](#-model--training)
- [📂 Dataset and Utility Scripts](#-dataset-and-utility-scripts)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🌟 Premium Features

`PlagiarismAI` provides a state-of-the-art experience with:

- **🔄 Full Auto Workflow**: A 4-step automated pipeline that generates plagiarism, detects it, reconstructs clean text, and verifies meaning preservation.
- **🧠 Deep Neural Analysis**: Multi-step loading sequences showing real-time decomposition, semantic vector scanning, and cross-referencing.
- **✏️ Advanced Rewriting Engine**: Adjustable rewrite strength (Low, Medium, Aggressive) and specialized modes for **Removing Plagiarism** or **Humanizing AI Text**.
- **👤 Human-Like Reconstruction**: Preserves semantic integrity with sentence-level semantic checks and deterministic fallback reconstruction.
- **📝 Short-Sentence Safe Rewrite**: Short/simple inputs are paraphrased using phrase-level replacements instead of unsafe clause reordering.
- **🛡️ Grammar-Safe Token Protection**: Core grammar words (including `a`, `an`, `the`) are protected from aggressive synonym replacement.
- **📊 Activity History**: Local storage integration to track your previous scans and rewrites.
- **🎨 Professional UI**: Glassmorphism design, pulsing preloader, and high-performance top-bar progress indicators.

---

## 🆕 Latest Updates (April 2026)

- 🐛 Fixed `restructure_sentence()` behavior to avoid corrupting short/simple sentences.
- 🔄 Added fallback logic so single-clause sentences use phrase paraphrasing when clause movement is not possible.
- 🛡️ Added stronger protected-word skipping in aggressive synonym replacement to prevent article movement (`a`, `an`, `the`) and grammar breakage.
- 📈 Expanded phrase-level paraphrase coverage for paragraph-definition style academic sentences.
- ⚡ Improved anti-plagiarism fallback chain so rewrites avoid returning near-identical output.

---

## 📸 Screenshots

*Coming soon - Add screenshots of the application interface here.*

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- pip package manager
- Optional (Advanced Features)
pip install sentence-transformers nltk python-dotenv

### Core Dependencies
```bash
# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1

# Install core packages
pip install flask pandas numpy scipy scikit-learn joblib
```

### Advanced Features (Optional)
For the current rewrite and semantic-check pipeline:
```bash
pip install sentence-transformers nltk python-dotenv
```

For OpenAI/Gemini support, create a `.env` file in the `plagiarism_app/` directory:
```text
OPENAI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

---

## 🚀 Usage

1. **Start the Application**:
   ```bash
   cd plagiarism_app
   python app.py
   ```

2. **Access the Dashboard**:
   Open your browser and navigate to: `http://localhost:5000`

3. **Use the Interface**:
   - Upload or paste text for plagiarism detection
   - Choose rewrite strength and mode
   - View results with detailed analysis

> **💡 Note**: The app initializes with a professional neural engine preloader. During analysis, you can follow the progress through the dynamic multi-step status track.

---

## 🧠 Model & Training

The system uses a TF-IDF + Logistic Regression pipeline trained on **1,000,000+** samples for high-accuracy detection.

### Training Artifacts (`plagiarism_app/models/`)
- `plagiarism_model.pkl` — Trained classifier
- `tfidf_vectorizer.pkl` — Feature extractor
- `model_metadata.pkl` — Accuracy and F1 metrics displayed on the dashboard

### Retraining
```bash
python train_model.py --sample 500000
```

---

## 📂 Dataset and Utility Scripts

This repository includes tools for generating training data and testing rewrite behavior.

- `generate_million.py` — Script to build the 1M+ sample dataset.
- `generate_paragraph_dataset.py` — Creates paragraph-level datasets for broader similarity testing.
- `rewrite_para.py` — Runs a local rewrite test outside of the Flask UI.

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with ❤️ by the PlagiarismAI team
- Special thanks to the open-source community for the amazing libraries used in this project

---

## 🧩 Repository Structure

- `plagiarism_app/` — Core application (Flask, Templates, Static)
- `plagiarism_app/paraphrase_engine.py` — The "brain" behind linguistic reconstruction
- `plagiarism_app/models/` — Serialized AI weights and metadata
- `dataset.csv` — Generated training dataset
- `.gitignore` — Configured to protect `.env` and local caches

---

## ⚠️ Known Issues & Work in Progress

This project is **still in active development**. Several bugs and limitations are currently being addressed:

### Current Limitations
- ⚠️ **Environment Setup Sensitivity**: Missing Python dependencies can disable parts of rewriting and semantic scoring.
- ⚠️ **LLM Optionality**: OpenAI/Groq/Gemini rewriting requires valid API keys; local rewrite fallback is used otherwise.
- ⚠️ **Long Document Handling**: Very long inputs can still lead to slower processing and occasional inconsistency.
- ⚠️ **Rewrite Variability**: Rewrites are intentionally non-deterministic in some steps, so exact output wording can differ between runs.

### Planned Fixes
- ✅ Stabilize environment bootstrap with a pinned requirements file
- ✅ Expand phrase and synonym banks for domain-specific academic text
- ✅ Improve long-input chunking and timeout handling
- ✅ Add comprehensive rewrite regression tests

### Contributing
If you encounter bugs or have suggestions, please open an issue on GitHub.

---


This project is licensed under the MIT License. See `LICENSE` for details.
