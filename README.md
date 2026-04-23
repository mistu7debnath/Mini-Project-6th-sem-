# PlagiarismAI — Mini Project

A polished, end-to-end plagiarism detection and rewrite assistant built for academic authenticity evaluation and writing support. This repository combines a Flask, HTML, CSS ans JS web application with dataset generation utilities and a model training pipeline, making it easy to run, retrain, and extend.

**Authors:** Manisha, Joita, Debasmita, Suchitra

---

## 🌟 Project Overview

`PlagiarismAI` is designed to:
- detect text similarity and plagiarism using a trained classifier
- analyze sentence-level and corpus-level similarity
- generate rewritten text that preserves meaning while reducing plagiarism risk
- provide a responsive browser UI for quick review, comparison, and history
- show expandable sentence analysis with "Read more" toggles for full sentence detail

The core components include:
- `plagiarism_app/` — Flask web application, API endpoints, and frontend UI
- `plagiarism_app/train_model.py` — training script for model artifacts
- `plagiarism_app/models/` — serialized model, vectorizer, encoder, and metadata
- dataset and generation scripts for experimentation and retraining

---

## ⚙️ Prerequisites

Recommended Python version: `3.10+`

Install the main dependencies before running the app:

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install flask pandas numpy scipy scikit-learn joblib
```

If you want optional OpenAI rewriting support, also install:

```bash
pip install openai
```

---

## 🚀 Run the Application:

Start the Flask app from the `plagiarism_app` folder:

```bash
cd plagiarism_app
python app.py
```

Then open your browser at:

```text
http://localhost:5000
```

> Note: The app expects model artifacts in `plagiarism_app/models/`.
>
> Tip: In the sentence analysis panel, long sentences are shown as previews with a `Read more` button so you can expand them fully.

---

## 🧠 Model Training

The training script creates the classifier pipeline and saves required artifacts for the web app.

### Saved artifacts

- `plagiarism_model.pkl` — trained classifier model
- `tfidf_vectorizer.pkl` — TF-IDF vectorizer
- `label_encoder.pkl` — label encoder
- `model_metadata.pkl` — training summary and metadata

### Train with a dataset

```bash
cd plagiarism_app
python train_model.py
```

If you want to train on a smaller sample for faster iteration:

```bash
python train_model.py --sample 200000
```

The script auto-detects available dataset files if the default path is missing.

---

## 📂 Dataset and Utility Scripts

This repository includes tools for generating training data and testing rewrite behavior.

### `generate_dataset.py`

Generates the default `dataset.csv` file used for training and experimentation.

```bash
python generate_dataset.py
```

### `generate_paragraph_dataset.py`

Creates paragraph-level datasets for broader similarity and rewrite testing.

```bash
python generate_paragraph_dataset.py
```

### `rewrite_para.py`

Runs a local rewrite test outside of the Flask UI.

```bash
python rewrite_para.py
```

---

## 🧩 Repository Structure

- `dataset.csv` — generated training dataset
- `paragraph_dataset*.csv` — paragraph-based dataset variants
- `plagiarism_app/` — web app, model training, and UI files
- `plagiarism_app/models/` — trained model artifacts
- `README.md` — project documentation

---

## 📌 Usage Notes

- Keep the necessary model files in `plagiarism_app/models/` for the web app to work.
- If you remove the model artifacts, rerun `python train_model.py`.
- Configure `OPENAI_API_KEY` only if you want optional GPT-based rewrite enhancements.

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

