# PlagiarismAI — Mini Project

A polished, end-to-end plagiarism detection and rewriting system built for academic authenticity evaluation and writing assistance. This repository combines a Flask web application with dataset generation and model training utilities, making it ideal for students, educators, and developers who want a self-contained plagiarism analysis workflow.

**Authors:** Manisha, Joita, Debasmita, Suchitra

---

## 🌟 Project Overview

`PlagiarismAI` is designed to:
- detect text similarity and plagiarism using machine learning
- analyze sentence-level similarity with TF-IDF and custom features
- generate readable rewritten text that reduces plagiarism risk
- provide a clean web interface for easy use and review

The core components include:
- `plagiarism_app/` — Flask application and frontend UI
- data generation scripts for synthetic datasets
- model training pipeline that outputs reusable `.pkl` artifacts

---

## 🚀 Web Application (`plagiarism_app`)

The Flask app serves a modern plagiarism checker with a responsive UI and API endpoints for checking, comparing, and rewriting text.

### Features

- ✅ **Machine learning powered plagiarism detection**
  - loads a trained model from `plagiarism_app/models`
  - uses TF-IDF vectorization and engineered similarity features
- ✅ **Text rewrite assistant**
  - rewrites plagiarized or similar sentences into more original wording
  - supports sentence-level analysis and phrase substitution
- ✅ **Clean browser experience**
  - lightweight UI with history tracking and quick copy support
- ✅ **Health and status endpoints**
  - includes `/api/health` for model readiness checks

### Run the application

```bash
cd plagiarism_app
python app.py
```

Then open your browser at:

```text
http://localhost:5000
```

> Note: Keep the trained files in `plagiarism_app/models` so the app can load the model, vectorizer, and metadata.

---

## 🧠 Model Training

The training script builds the classification pipeline and persists all required artifacts.

### Available files saved during training

- `plagiarism_model.pkl` — trained classifier model
- `tfidf_vectorizer.pkl` — TF-IDF vectorizer
- `label_encoder.pkl` — label encoder for transformation metadata
- `model_metadata.pkl` — metrics and model configuration

### Train the model

```bash
cd plagiarism_app
python train_model.py
```

If you want to use a smaller dataset for faster training:

```bash
python train_model.py --sample 200000
```

---

## 📂 Dataset and Utility Scripts

This repository includes tools for dataset generation and paragraph rewriting.

### `generate_dataset.py`

Creates a dataset file such as `dataset.csv` for model training and experimentation.

```bash
python generate_dataset.py
```

### `rewrite_para.py`

A lightweight script for testing text rewriting locally without starting the Flask app.

```bash
python rewrite_para.py
```

### `generate_paragraph_dataset.py`

Generates larger paragraph-level datasets to support broader sentence similarity scenarios.

---

## 🧩 Repository Structure

- `dataset.csv` — default generated dataset
- `paragraph_dataset*.csv` — paragraph-based training datasets
- `plagiarism_app/` — Flask app, training logic, and web UI
- `plagiarism_app/models/` — model artifacts
- `README.md` — this project documentation

---

## ✅ Usage Notes

- Keep required model files in `plagiarism_app/models/` for the app to run.
- If the model artifacts are removed, regenerate them with `python train_model.py`.
- You can customize datasets using the generation scripts before retraining.

---

## 📄 License

This repository is provided for educational and experimental use. Please respect the authorship and credit the project contributors when reusing or sharing the code.

