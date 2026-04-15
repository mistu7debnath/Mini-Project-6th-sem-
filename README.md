# PlagiarismAI — Mini Project

This repository serves as a powerful AI-driven application and dataset generation suite specifically built for detecting plagiarism and algorithmically generating human-like, non-plagiarized rewrites. 

**Copyright claim by:** Manisha, Joita, Debasmita, Suchitra

---

## 🚀 The Web Application (`/plagiarism_app`)

The core of this project is a fully-featured, backend-driven web application built with **Flask**, featuring a professional, zero-distraction SaaS dashboard. 

### Key Application Features:
1. **Intelligent Plagiarism Detection:** 
   - Powered by a local machine learning model (`RandomForestClassifier`) trained on over 1,000,000 sentence pairs.
   - Evaluates TF-IDF vector combinations to instantly supply a semantic similarity and hard plagiarism grade.
2. **Iterative Rewriting Engine (Datamuse API):**
   - Incorporates the Datamuse API to fetch and cache vast amounts of contextual synonyms via asynchronous `ThreadPoolExecutor`.
   - Iteratively loops text through voice changes (Active/Passive), structural reorganization, and human-like phrase mapping.
   - Intelligently stops rewriting the moment the Plagiarism confidence threshold successfully drops below safe limits.
3. **Dual Operating Modes:**
   - **Direct Compare Mode:** Compare a student's text directly against a known reference source.
   - **Source Corpus Mode:** Scan massive walls of text against a large collection of source documents to highlight specifically stolen sentences.
4. **Professional UI & History Tracking:**
   - Designed with an ultra-clean corporate Light Theme layout.
   - Includes a persistent **History Sidebar** tracked locally in your browser. View past activities, see similarity scores, permanently delete records, or immediately copy the rewritten outcomes to your clipboard.

### How to Run the App
```bash
cd plagiarism_app
python app.py
```
*Wait for the `ML Model Loaded` signal, then open your browser to `http://localhost:5000`.*

---

## 🛠 Dataset Generation Suite

Alongside the main application, this project includes synthetic data generation scripts to create localized, independent datasets.

### `generate_dataset.py`
Generates a highly-customizable dataset (`dataset.csv`) detailing Original Sentences, Transformed Sentences, Transformation Types (e.g. passive_to_active), and Similarity metrics.

```bash
python generate_dataset.py
```

### `rewrite_para.py`
A local CLI testing tool that parses sentences/paragraphs natively to show immediate variations of text without launching the full Flask server.

```bash
python rewrite_para.py
```

---

*This application was built utilizing custom machine learning models and parallel API connections to ensure rapid, offline-capable evaluations for academic authenticity.*
