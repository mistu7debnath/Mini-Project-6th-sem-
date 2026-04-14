"""
Plagiarism Checker — Model Training Script
============================================
Trains a machine-learning model on the sentence_similarity_dataset.csv file
located in the parent directory.

The model pipeline:
  1. TF-IDF vectorization of original and modified sentences
  2. Cosine similarity as an engineered feature
  3. Logistic Regression classifier (plagiarized / not plagiarized)
  4. Saves the trained model + vectorizer to disk as .pkl files

Usage:
    python train_model.py
    python train_model.py --sample 200000   # use only 200k rows for faster training
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, csr_matrix

# Fix Windows console encoding for emoji/unicode
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "sentence_similarity_dataset.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "plagiarism_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.pkl")

# Similarity threshold to classify as "plagiarized"
PLAGIARISM_THRESHOLD = 0.60


def load_dataset(dataset_path: str, sample_size: int | None = None) -> pd.DataFrame:
    """Load and prepare the dataset."""
    print(f"📂 Loading dataset from: {dataset_path}")
    start = time.time()

    df = pd.read_csv(dataset_path)
    print(f"   Loaded {len(df):,} rows in {time.time() - start:.1f}s")
    print(f"   Columns: {list(df.columns)}")

    # Handle column name variations
    col_map = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if "original" in col_lower:
            col_map[col] = "original"
        elif "modified" in col_lower or "transformed" in col_lower:
            col_map[col] = "modified"
        elif col_lower == "type" or "transformation" in col_lower:
            col_map[col] = "type"
        elif "similarity" in col_lower or "score" in col_lower:
            col_map[col] = "similarity"

    df = df.rename(columns=col_map)
    print(f"   Mapped columns: {col_map}")

    # Drop rows with missing values
    df = df.dropna(subset=["original", "modified", "similarity"])
    print(f"   After cleaning: {len(df):,} rows")

    # Create binary label: plagiarized if similarity >= threshold
    df["is_plagiarized"] = (df["similarity"].astype(float) >= PLAGIARISM_THRESHOLD).astype(int)

    plagiarized_count = df["is_plagiarized"].sum()
    non_plagiarized_count = len(df) - plagiarized_count
    print(f"   Plagiarized (sim >= {PLAGIARISM_THRESHOLD}): {plagiarized_count:,}")
    print(f"   Not plagiarized: {non_plagiarized_count:,}")

    # Sample if requested
    if sample_size and sample_size < len(df):
        print(f"   Sampling {sample_size:,} rows...")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    return df


def create_features(df: pd.DataFrame, vectorizer: TfidfVectorizer = None, fit: bool = True):
    """
    Create feature matrix from the dataset.
    
    Features:
      1. TF-IDF vectors of original sentences
      2. TF-IDF vectors of modified sentences
      3. Cosine similarity between the TF-IDF vectors
      4. Length ratio (shorter / longer)
      5. Word overlap ratio
      6. Character length difference
    """
    print("🔧 Creating features...")
    start = time.time()

    # Combine text for TF-IDF fitting
    all_text = pd.concat([df["original"], df["modified"]]).fillna("")

    if fit:
        vectorizer = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            min_df=2,
            max_df=0.95,
        )
        vectorizer.fit(all_text)

    # Transform sentences
    tfidf_original = vectorizer.transform(df["original"].fillna(""))
    tfidf_modified = vectorizer.transform(df["modified"].fillna(""))

    # Compute cosine similarities row-by-row
    print("   Computing cosine similarities...")
    dot_products = np.array((tfidf_original.multiply(tfidf_modified)).sum(axis=1)).flatten()
    norm_orig = np.sqrt(np.array(tfidf_original.multiply(tfidf_original).sum(axis=1)).flatten())
    norm_mod = np.sqrt(np.array(tfidf_modified.multiply(tfidf_modified).sum(axis=1)).flatten())
    denom = norm_orig * norm_mod
    denom[denom == 0] = 1e-10
    cosine_sim = dot_products / denom

    # Additional handcrafted features
    print("   Computing handcrafted features...")
    orig_lens = df["original"].fillna("").str.len().values.astype(float)
    mod_lens = df["modified"].fillna("").str.len().values.astype(float)

    max_lens = np.maximum(orig_lens, mod_lens)
    max_lens[max_lens == 0] = 1
    min_lens = np.minimum(orig_lens, mod_lens)
    length_ratio = min_lens / max_lens

    # Word overlap
    def word_overlap(row):
        words_orig = set(str(row["original"]).lower().split())
        words_mod = set(str(row["modified"]).lower().split())
        if not words_orig or not words_mod:
            return 0.0
        intersection = words_orig & words_mod
        union = words_orig | words_mod
        return len(intersection) / len(union) if union else 0.0

    word_overlaps = df.apply(word_overlap, axis=1).values.astype(float)

    # Char length difference (normalized)
    char_diff = np.abs(orig_lens - mod_lens) / max_lens

    # Combine all features
    extra_features = np.column_stack([cosine_sim, length_ratio, word_overlaps, char_diff])
    extra_sparse = csr_matrix(extra_features)

    # Combine TF-IDF features with handcrafted features
    # Use the absolute difference of TF-IDF vectors + handcrafted
    tfidf_diff = abs(tfidf_original - tfidf_modified)
    X = hstack([tfidf_diff, extra_sparse])

    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Feature creation took {time.time() - start:.1f}s")

    return X, vectorizer


def train_model(X, y, test_size: float = 0.2):
    """Train the plagiarism classifier."""
    print(f"\n🏋️ Training model...")
    start = time.time()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print(f"   Train size: {X_train.shape[0]:,}")
    print(f"   Test size:  {X_test.shape[0]:,}")

    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        class_weight="balanced",
        n_jobs=-1,
        verbose=0,
    )

    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"   Training completed in {elapsed:.1f}s")

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print(f"\n📊 Evaluation Results:")
    print(f"   Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"   Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"   Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"   F1 Score:  {f1_score(y_test, y_pred):.4f}")

    print(f"\n   Classification Report:")
    report = classification_report(y_test, y_pred, target_names=["Not Plagiarized", "Plagiarized"])
    for line in report.split("\n"):
        print(f"   {line}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n   Confusion Matrix:")
    print(f"   {cm[0]}")
    print(f"   {cm[1]}")

    return model, {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
        "training_time": elapsed,
    }


def save_artifacts(model, vectorizer, label_encoder, metadata):
    """Save all model artifacts to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"\n💾 Saving model artifacts to: {MODEL_DIR}")
    joblib.dump(model, MODEL_PATH)
    print(f"   ✅ Model saved: {MODEL_PATH}")

    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"   ✅ Vectorizer saved: {VECTORIZER_PATH}")

    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    print(f"   ✅ Label encoder saved: {LABEL_ENCODER_PATH}")

    joblib.dump(metadata, METADATA_PATH)
    print(f"   ✅ Metadata saved: {METADATA_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Train the plagiarism detection model")
    parser.add_argument("--sample", type=int, default=None,
                        help="Number of rows to sample from dataset (default: use all)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to dataset CSV (default: auto-detect)")
    args = parser.parse_args()

    dataset_path = args.dataset or DATASET_PATH
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        # Try looking for dataset.csv
        alt_path = os.path.join(os.path.dirname(__file__), "..", "dataset.csv")
        if os.path.exists(alt_path):
            print(f"   Found alternative: {alt_path}")
            dataset_path = alt_path
        else:
            print("   No dataset file found. Please generate the dataset first.")
            sys.exit(1)

    total_start = time.time()

    # 1. Load data
    df = load_dataset(dataset_path, sample_size=args.sample)

    # 2. Create label encoder for transformation types
    le = LabelEncoder()
    if "type" in df.columns:
        le.fit(df["type"].fillna("unknown"))
        print(f"\n   Transformation types: {list(le.classes_)}")

    # 3. Create features
    X, vectorizer = create_features(df, fit=True)
    y = df["is_plagiarized"].values

    # 4. Train the model
    model, metrics = train_model(X, y)

    # 5. Add extra metadata
    metrics["plagiarism_threshold"] = PLAGIARISM_THRESHOLD
    metrics["total_dataset_rows"] = len(df)
    metrics["feature_count"] = X.shape[1]
    metrics["tfidf_max_features"] = 15000
    if "type" in df.columns:
        metrics["transformation_types"] = list(le.classes_)

    # 6. Save everything
    save_artifacts(model, vectorizer, le, metrics)

    total_time = time.time() - total_start
    print(f"\n🎉 All done in {total_time:.1f}s!")
    print(f"   Model ready for deployment at: {MODEL_DIR}")


if __name__ == "__main__":
    main()
