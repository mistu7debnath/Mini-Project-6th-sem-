"""
Plagiarism Checker & Rewriter — Flask Application
=====================================================
Serves the trained model through a web API and a beautiful frontend.

Features:
  1. Detect plagiarism between two texts
  2. Show plagiarized sentences highlighted
  3. Rewrite plagiarized sentences into human-like original text
  4. Keep the meaning identical in both versions

Endpoints:
    GET  /              → Main page (HTML UI)
    POST /api/check     → Check plagiarism + rewrite plagiarized text
    POST /api/compare   → Corpus scan + rewrite flagged sentences
    GET  /api/health    → Health check + model info

Usage:
    python app.py
    python app.py --port 8080
"""

import os
import sys
import argparse
import re
import random
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify
from scipy.sparse import hstack, csr_matrix
from difflib import SequenceMatcher

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "plagiarism_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.pkl")

app = Flask(__name__)

# ─────────────────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────────────────
model = None
vectorizer = None
metadata = None


def load_model():
    """Load trained model artifacts."""
    global model, vectorizer, metadata

    if not os.path.exists(MODEL_PATH):
        print("⚠️  Model not found! Please run train_model.py first.")
        print(f"   Expected at: {MODEL_PATH}")
        return False

    print("🔄 Loading model artifacts...")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    metadata = joblib.load(METADATA_PATH)

    print(f"   ✅ Model loaded (accuracy: {metadata.get('accuracy', 'N/A'):.4f})")
    print(f"   ✅ Vectorizer loaded ({metadata.get('tfidf_max_features', 'N/A')} features)")
    return True


# ─────────────────────────────────────────────────────────
# SYNONYM & REWRITING ENGINE
# ─────────────────────────────────────────────────────────

SYNONYM_MAP = {
    # Verbs
    "completed": ["finished", "accomplished", "concluded", "wrapped up"],
    "finished": ["completed", "accomplished", "concluded"],
    "submitted": ["handed in", "turned in", "delivered"],
    "discussed": ["talked about", "deliberated", "debated"],
    "achieved": ["accomplished", "attained", "reached"],
    "organized": ["arranged", "coordinated", "planned"],
    "reviewed": ["examined", "analyzed", "evaluated"],
    "implemented": ["executed", "carried out", "put into practice"],
    "prepared": ["arranged", "set up", "got ready"],
    "evaluated": ["assessed", "appraised", "judged"],
    "designed": ["created", "crafted", "developed"],
    "built": ["constructed", "assembled", "developed"],
    "tested": ["examined", "evaluated", "checked"],
    "fixed": ["repaired", "mended", "corrected"],
    "installed": ["set up", "configured", "put in"],
    "created": ["developed", "produced", "generated"],
    "analyzed": ["examined", "studied", "investigated"],
    "optimized": ["improved", "enhanced", "refined"],
    "deployed": ["launched", "released", "rolled out"],
    "automated": ["mechanized", "streamlined", "systematized"],
    "examined": ["inspected", "checked", "investigated"],
    "recovered": ["healed", "got better", "recuperated"],
    "visited": ["went to", "called on", "stopped by"],
    "traveled": ["journeyed", "went", "toured"],
    "climbed": ["ascended", "scaled", "went up"],
    "explored": ["investigated", "surveyed", "ventured through"],
    "photographed": ["captured", "took a photo of", "snapped"],
    "discovered": ["found", "uncovered", "came across"],
    "enjoyed": ["appreciated", "relished", "liked"],
    "celebrated": ["commemorated", "marked", "honored"],
    "followed": ["adhered to", "obeyed", "complied with"],
    "scored": ["earned", "achieved", "netted"],
    "practiced": ["rehearsed", "trained", "drilled"],
    "competed": ["contended", "participated", "vied"],
    "performed": ["executed", "carried out", "delivered"],
    "published": ["released", "issued", "put out"],
    "painted": ["illustrated", "decorated", "colored"],
    "resolved": ["solved", "settled", "sorted out"],
    "answered": ["responded to", "replied to", "addressed"],
    "wrote": ["composed", "authored", "penned"],
    "read": ["perused", "studied", "went through"],
    "watched": ["observed", "viewed", "witnessed"],
    "helped": ["assisted", "aided", "supported"],
    "started": ["began", "commenced", "initiated"],
    "bought": ["purchased", "acquired", "obtained"],
    "gave": ["provided", "presented", "offered"],
    "took": ["grabbed", "picked up", "collected"],
    "made": ["created", "produced", "crafted"],
    "said": ["stated", "mentioned", "declared"],
    "told": ["informed", "notified", "advised"],
    "asked": ["inquired", "questioned", "requested"],
    "showed": ["demonstrated", "displayed", "revealed"],
    "played": ["engaged in", "participated in", "took part in"],
    "walked": ["strolled", "ambled", "wandered"],
    "ran": ["sprinted", "dashed", "jogged"],
    "won": ["triumphed", "prevailed", "succeeded"],
    "documented": ["recorded", "logged", "chronicled"],
    "presented": ["showed", "displayed", "demonstrated"],
    "explained": ["clarified", "described", "elaborated on"],
    "teaches": ["instructs", "educates", "trains"],
    "teaches": ["instructs", "educates", "coaches"],
    "manages": ["handles", "oversees", "supervises"],
    "maintains": ["keeps up", "sustains", "preserves"],
    "monitors": ["tracks", "observes", "watches"],
    "collected": ["gathered", "assembled", "accumulated"],
    "selected": ["chose", "picked", "opted for"],
    "improved": ["enhanced", "upgraded", "bettered"],
    "reduced": ["decreased", "lowered", "cut down"],
    "increased": ["raised", "boosted", "elevated"],
    "established": ["set up", "founded", "created"],
    "conducted": ["carried out", "performed", "executed"],
    "developed": ["created", "built", "produced"],
    "described": ["explained", "outlined", "portrayed"],
    "determined": ["decided", "established", "figured out"],
    "identified": ["recognized", "spotted", "detected"],
    "produced": ["created", "generated", "manufactured"],
    "received": ["got", "obtained", "acquired"],
    "considered": ["thought about", "contemplated", "pondered"],
    "included": ["contained", "encompassed", "comprised"],
    "provided": ["gave", "supplied", "offered"],
    "required": ["needed", "demanded", "necessitated"],
    "suggested": ["proposed", "recommended", "advised"],
    "supported": ["backed", "endorsed", "upheld"],

    # Nouns
    "market": ["marketplace", "bazaar", "store"],
    "house": ["home", "residence", "dwelling"],
    "car": ["vehicle", "automobile", "ride"],
    "road": ["street", "path", "route"],
    "money": ["cash", "funds", "currency"],
    "work": ["job", "task", "assignment"],
    "book": ["publication", "volume", "text"],
    "food": ["meal", "cuisine", "nourishment"],
    "friend": ["companion", "pal", "buddy"],
    "children": ["kids", "youngsters", "little ones"],
    "people": ["individuals", "folks", "persons"],
    "school": ["academy", "institution", "educational facility"],
    "problem": ["issue", "challenge", "difficulty"],
    "idea": ["concept", "notion", "thought"],
    "place": ["location", "spot", "site"],
    "country": ["nation", "state", "land"],
    "student": ["learner", "pupil", "scholar"],
    "teacher": ["educator", "instructor", "professor"],
    "scientist": ["researcher", "scholar", "investigator"],
    "doctor": ["physician", "medical professional", "practitioner"],
    "company": ["firm", "organization", "business"],
    "project": ["undertaking", "initiative", "venture"],
    "report": ["document", "paper", "account"],
    "meeting": ["gathering", "conference", "assembly"],
    "strategy": ["plan", "approach", "method"],
    "team": ["group", "squad", "crew"],
    "findings": ["results", "discoveries", "conclusions"],
    "research": ["study", "investigation", "analysis"],
    "species": ["type", "variety", "kind"],
    "conference": ["meeting", "symposium", "convention"],
    "forest": ["woods", "woodland", "jungle"],
    "rainforest": ["tropical forest", "jungle", "tropical woodland"],
    "topic": ["subject", "theme", "matter"],
    "budget": ["financial plan", "allocation", "funds"],

    # Adjectives
    "beautiful": ["gorgeous", "stunning", "lovely"],
    "important": ["crucial", "vital", "significant"],
    "new": ["fresh", "novel", "recent"],
    "old": ["ancient", "aged", "vintage"],
    "good": ["excellent", "great", "fine"],
    "bad": ["terrible", "awful", "poor"],
    "difficult": ["challenging", "tough", "demanding"],
    "easy": ["simple", "straightforward", "effortless"],
    "interesting": ["fascinating", "intriguing", "captivating"],
    "annual": ["yearly", "once-a-year", "regular"],
    "entire": ["whole", "complete", "full"],
    "successful": ["triumphant", "prosperous", "accomplished"],
    "clearly": ["plainly", "obviously", "evidently"],
    "successfully": ["effectively", "triumphantly", "with great success"],

    # Adverbs & others
    "very": ["extremely", "really", "incredibly"],
    "quickly": ["rapidly", "swiftly", "speedily"],
    "carefully": ["cautiously", "meticulously", "attentively"],
    "immediately": ["instantly", "right away", "promptly"],
    "yesterday": ["the day before", "the previous day", "a day ago"],
    "together": ["collectively", "as a group", "jointly"],
    "also": ["additionally", "moreover", "furthermore"],
    "however": ["nevertheless", "nonetheless", "yet"],
    "therefore": ["consequently", "thus", "as a result"],
    "because": ["since", "as", "due to the fact that"],
}

# Sentence restructuring templates
RESTRUCTURE_TEMPLATES = [
    # Move time/place to front
    ("at the {place}", "At the {place}, "),
    ("in the {place}", "In the {place}, "),
    ("on the {place}", "On the {place}, "),
]

# Transition phrases to make text sound more natural/human
HUMAN_TRANSITIONS = [
    "In other words, ",
    "To put it differently, ",
    "Essentially, ",
    "Put simply, ",
    "What this means is that ",
    "In simple terms, ",
]


def rewrite_sentence_human(sentence: str, source_sentence: str = "") -> dict:
    """
    Rewrite a plagiarized sentence into a human-like original version.
    
    Strategy:
      1. Replace words with synonyms (multiple passes)
      2. Restructure the sentence (reorder clauses)
      3. Change voice (active <-> passive) if possible
      4. Ensure the meaning stays the same
    
    Returns a dict with the original plagiarized text and the rewritten clean version.
    """
    if not sentence or not sentence.strip():
        return {
            "plagiarized": sentence,
            "rewritten": sentence,
            "changes_made": [],
            "similarity_to_original": 1.0
        }

    original = sentence.strip()
    rewritten = original
    changes = []

    # ─── Step 1: Synonym Replacement (aggressive — replace many words) ───
    words = rewritten.split()
    new_words = []
    for word in words:
        # Strip punctuation for lookup
        clean = word.lower().rstrip(".,!?;:'\"")
        trail = word[len(word.rstrip(".,!?;:'\"")):]
        
        if clean in SYNONYM_MAP:
            synonyms = SYNONYM_MAP[clean]
            # Pick a synonym that's different from the source too
            chosen = random.choice(synonyms)
            
            # If we have a source sentence, avoid picking words that are in it
            if source_sentence:
                source_lower = source_sentence.lower()
                safe_synonyms = [s for s in synonyms if s.lower() not in source_lower]
                if safe_synonyms:
                    chosen = random.choice(safe_synonyms)
            
            # Preserve capitalization
            if word[0].isupper():
                chosen = chosen[0].upper() + chosen[1:]
            
            new_words.append(chosen + trail)
            changes.append(f"'{clean}' → '{chosen}'")
        else:
            new_words.append(word)
    
    rewritten = " ".join(new_words)

    # ─── Step 2: Sentence restructuring ───
    restructured = restructure_sentence(rewritten)
    if restructured != rewritten:
        changes.append("sentence restructured")
        rewritten = restructured

    # ─── Step 3: Voice change (active ↔ passive) ───
    voice_changed = try_voice_change(rewritten)
    if voice_changed != rewritten:
        changes.append("voice changed")
        rewritten = voice_changed

    # ─── Step 4: If still too similar to source, apply more changes ───
    if source_sentence:
        sim = compute_text_similarity(rewritten, source_sentence)
        if sim > 0.7:
            # Add transitional phrasing
            rewritten = add_human_phrasing(rewritten)
            changes.append("human phrasing added")

    # Compute final similarity to the plagiarized input (should be high — same meaning)
    meaning_similarity = compute_text_similarity(original, rewritten)
    
    # Compute similarity to source (should be low — shows plagiarism is reduced)
    source_similarity = 0.0
    if source_sentence:
        source_similarity = compute_text_similarity(rewritten, source_sentence)

    return {
        "plagiarized": original,
        "rewritten": rewritten,
        "changes_made": changes,
        "meaning_preserved": meaning_similarity,
        "source_similarity_before": compute_text_similarity(original, source_sentence) if source_sentence else 0,
        "source_similarity_after": source_similarity,
    }


def restructure_sentence(sentence: str) -> str:
    """Restructure a sentence by moving clauses around."""
    words = sentence.split()
    if len(words) < 5:
        return sentence

    # Remove trailing punctuation
    punct = ""
    if sentence and sentence[-1] in ".!?":
        punct = sentence[-1]
        sentence_clean = sentence[:-1]
    else:
        sentence_clean = sentence

    # Strategy 1: If sentence has a prepositional phrase, move it to front
    prep_words = {"in", "at", "on", "to", "for", "from", "with", "by", "about",
                  "after", "before", "during", "through", "across", "near"}
    
    words_clean = sentence_clean.split()
    
    # Find a preposition in middle-to-end of sentence
    for i in range(len(words_clean) // 2, len(words_clean)):
        if words_clean[i].lower() in prep_words and i < len(words_clean) - 1:
            # Move the prepositional phrase to the front
            prep_phrase = words_clean[i:]
            main_clause = words_clean[:i]
            
            if len(prep_phrase) >= 2 and len(main_clause) >= 2:
                # Capitalize prep phrase start, lowercase main clause start
                prep_phrase[0] = prep_phrase[0].capitalize()
                main_clause[0] = main_clause[0][0].lower() + main_clause[0][1:]
                
                result = " ".join(prep_phrase) + ", " + " ".join(main_clause) + punct
                return result
            break

    # Strategy 2: Convert "X did Y" to "Y was done by X" style phrasing
    # But only if we haven't changed already
    if len(words_clean) >= 4:
        # Swap first half and second half with connecting word
        mid = len(words_clean) // 2
        second = words_clean[mid:]
        first = words_clean[:mid]
        
        second[0] = second[0].capitalize()
        first[0] = first[0][0].lower() + first[0][1:]
        
        # Only apply if it produces a significantly different result
        candidate = " ".join(second) + " — " + " ".join(first) + punct
        if compute_text_similarity(sentence, candidate) < 0.85:
            return candidate

    return sentence + ("" if sentence.endswith(tuple(".!?")) else "")


def try_voice_change(sentence: str) -> str:
    """Try to change active voice to passive or vice versa."""
    words = sentence.split()
    if len(words) < 4:
        return sentence
    
    # Remove trailing punctuation
    punct = ""
    if sentence and sentence[-1] in ".!?":
        punct = sentence[-1]
        words[-1] = words[-1].rstrip(".!?")
    
    # Check for passive voice markers ("was/were/is/are ... by")
    lower_sentence = sentence.lower()
    
    if " by " in lower_sentence and any(w in lower_sentence for w in ["was ", "were ", "is ", "are "]):
        # Passive → Active
        try:
            # Find "by" and reconstruct
            by_idx = None
            for i, w in enumerate(words):
                if w.lower() == "by":
                    by_idx = i
                    break
            
            if by_idx and by_idx < len(words) - 1:
                agent = " ".join(words[by_idx + 1:])
                # Find the auxiliary verb
                aux_idx = None
                for i, w in enumerate(words):
                    if w.lower() in ("was", "were", "is", "are"):
                        aux_idx = i
                        break
                
                if aux_idx is not None:
                    subject = " ".join(words[:aux_idx])
                    verb_phrase = " ".join(words[aux_idx + 1:by_idx])
                    result = f"{agent.capitalize()} {verb_phrase} {subject.lower()}{punct}"
                    return result
        except (IndexError, ValueError):
            pass
    
    else:
        # Active → Passive (simple cases)
        # Pattern: Subject Verb Object
        if len(words) >= 3:
            subject = words[0]
            verb = words[1]
            obj_words = words[2:]
            obj = " ".join(obj_words)
            
            # Only convert if verb looks past-tense
            if verb.lower().endswith("ed") or verb.lower() in ("wrote", "gave", "took", "made", "found", "built", "ran"):
                verb_map = {
                    "wrote": "written", "gave": "given", "took": "taken",
                    "made": "made", "found": "found", "built": "built",
                    "ran": "run",
                }
                past_participle = verb_map.get(verb.lower(), verb.lower())
                
                result = f"{obj.capitalize()} was {past_participle} by {subject.lower()}{punct}"
                return result
    
    return sentence


def add_human_phrasing(sentence: str) -> str:
    """Add natural human transitions and phrasing to make text sound more original."""
    # Don't add if sentence is very short
    if len(sentence.split()) < 4:
        return sentence
    
    strategies = [
        "rephrase_start",
        "add_context",
        "combine_clauses",
    ]
    strategy = random.choice(strategies)
    
    if strategy == "rephrase_start":
        # Add a human-like opening
        openers = [
            "It is worth noting that ",
            "Notably, ",
            "In particular, ",
            "Specifically, ",
            "From what can be observed, ",
            "As it turns out, ",
            "Looking at this closely, ",
        ]
        opener = random.choice(openers)
        # Lowercase the first char of the original
        if sentence[0].isupper():
            sentence = sentence[0].lower() + sentence[1:]
        return opener + sentence
    
    elif strategy == "add_context":
        # Add contextual ending
        endings = [
            ", which is quite significant",
            ", demonstrating a clear pattern",
            ", as one might expect",
            ", reflecting the overall trend",
            ", highlighting an important aspect",
        ]
        # Remove trailing period
        if sentence.endswith("."):
            sentence = sentence[:-1]
        ending = random.choice(endings)
        return sentence + ending + "."
    
    else:  # combine_clauses
        # Rephrase using "this means that" or "in other words"
        connector = random.choice([
            "This essentially means that ",
            "To put it another way, ",
            "In simpler terms, ",
        ])
        if sentence[0].isupper():
            sentence = sentence[0].lower() + sentence[1:]
        return connector + sentence
    
    return sentence


# ─────────────────────────────────────────────────────────
# Text Processing Utilities
# ─────────────────────────────────────────────────────────
def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def compute_text_similarity(text1: str, text2: str) -> float:
    """Compute similarity between two texts using SequenceMatcher."""
    if not text1 or not text2:
        return 0.0
    matcher = SequenceMatcher(None, text1.lower(), text2.lower())
    return round(matcher.ratio(), 4)


def compute_word_overlap(text1: str, text2: str) -> float:
    """Compute Jaccard word overlap."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return round(len(intersection) / len(union), 4) if union else 0.0


def create_features_for_pair(text1: str, text2: str):
    """Create feature vector for a single text pair."""
    if vectorizer is None:
        return None

    tfidf1 = vectorizer.transform([text1])
    tfidf2 = vectorizer.transform([text2])

    # Cosine similarity from TF-IDF vectors
    dot = np.array((tfidf1.multiply(tfidf2)).sum(axis=1)).flatten()
    norm1 = np.sqrt(np.array(tfidf1.multiply(tfidf1).sum(axis=1)).flatten())
    norm2 = np.sqrt(np.array(tfidf2.multiply(tfidf2).sum(axis=1)).flatten())
    denom = norm1 * norm2
    denom[denom == 0] = 1e-10
    cosine_sim = dot / denom

    # Length ratio
    len1 = max(len(text1), 1)
    len2 = max(len(text2), 1)
    length_ratio = min(len1, len2) / max(len1, len2)

    # Word overlap
    word_overlap = compute_word_overlap(text1, text2)

    # Character difference
    char_diff = abs(len1 - len2) / max(len1, len2)

    # Combine
    extra = np.array([[cosine_sim[0], length_ratio, word_overlap, char_diff]])
    tfidf_diff = abs(tfidf1 - tfidf2)
    X = hstack([tfidf_diff, csr_matrix(extra)])

    return X


def find_matching_segments(text1: str, text2: str, min_length: int = 4) -> list[dict]:
    """Find matching word sequences between two texts."""
    words1 = text1.lower().split()
    words2 = text2.lower().split()

    if not words1 or not words2:
        return []

    matcher = SequenceMatcher(None, words1, words2)
    matches = []

    for block in matcher.get_matching_blocks():
        if block.size >= min_length:
            matched_words = words1[block.a:block.a + block.size]
            matches.append({
                "text": " ".join(matched_words),
                "position_text1": block.a,
                "position_text2": block.b,
                "length": block.size,
            })

    return matches


def analyze_pair(text1: str, text2: str) -> dict:
    """Full analysis of a text pair."""
    # ML prediction
    prediction = None
    confidence = None

    if model is not None and vectorizer is not None:
        try:
            X = create_features_for_pair(text1, text2)
            if X is not None:
                prediction = int(model.predict(X)[0])
                proba = model.predict_proba(X)[0]
                confidence = float(max(proba))
        except AttributeError:
            # Fallback for sklearn version mismatch
            prediction = 1 if compute_text_similarity(text1, text2) > 0.6 else 0
            confidence = compute_text_similarity(text1, text2)

    # Rule-based metrics
    sequence_sim = compute_text_similarity(text1, text2)
    word_overlap = compute_word_overlap(text1, text2)
    matching_segments = find_matching_segments(text1, text2)

    # Determine plagiarism level
    if prediction is not None:
        is_plagiarized = bool(prediction)
    else:
        is_plagiarized = sequence_sim >= 0.6 or word_overlap >= 0.5

    # Compute overall score (0-100)
    if confidence is not None:
        overall_score = round(confidence * 100 if is_plagiarized else (1 - confidence) * 100, 1)
    else:
        overall_score = round(max(sequence_sim, word_overlap) * 100, 1)

    # Severity
    if overall_score >= 80:
        severity = "high"
        verdict = "High Plagiarism Detected"
    elif overall_score >= 50:
        severity = "medium"
        verdict = "Moderate Similarity Found"
    elif overall_score >= 30:
        severity = "low"
        verdict = "Minor Similarities"
    else:
        severity = "none"
        verdict = "Original Content"

    return {
        "is_plagiarized": is_plagiarized,
        "overall_score": overall_score,
        "severity": severity,
        "verdict": verdict,
        "metrics": {
            "sequence_similarity": sequence_sim,
            "word_overlap": word_overlap,
            "ml_confidence": confidence,
            "ml_prediction": prediction,
        },
        "matching_segments": matching_segments,
    }


# ─────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the main page."""
    model_loaded = model is not None
    return render_template("index.html", model_loaded=model_loaded, metadata=metadata)


@app.route("/api/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "metadata": metadata if metadata else {},
    })


@app.route("/api/check", methods=["POST"])
def check_plagiarism():
    """
    Check plagiarism between two texts.
    
    Flow:
      1. Detect plagiarism (ML + rule-based)
      2. Split suspected text into sentences
      3. For each plagiarized sentence → produce a human-like rewrite
      4. Return: plagiarism score, plagiarized text, rewritten clean text
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    text1 = data.get("text1", "").strip()
    text2 = data.get("text2", "").strip()

    if not text1 or not text2:
        return jsonify({"error": "Both text1 and text2 are required"}), 400

    # Overall analysis
    result = analyze_pair(text1, text2)

    # Sentence-level analysis + rewriting
    sentences_text2 = split_into_sentences(text2)
    sentences_text1 = split_into_sentences(text1)
    
    sentence_analysis = []
    rewritten_sentences = []
    
    for sent2 in sentences_text2:
        # Find the best matching source sentence
        best_source = ""
        best_sim = 0
        for sent1 in sentences_text1:
            sim = compute_text_similarity(sent2, sent1)
            if sim > best_sim:
                best_sim = sim
                best_source = sent1
        
        # Determine if this sentence is plagiarized
        is_plag = best_sim >= 0.5
        
        # Rewrite if plagiarized
        if is_plag:
            rewrite_result = rewrite_sentence_human(sent2, best_source)
        else:
            rewrite_result = {
                "plagiarized": sent2,
                "rewritten": sent2,  # Keep as is — it's clean
                "changes_made": [],
                "meaning_preserved": 1.0,
                "source_similarity_before": best_sim,
                "source_similarity_after": best_sim,
            }
        
        sentence_analysis.append({
            "sentence": sent2,
            "source_match": best_source,
            "similarity": round(best_sim, 3),
            "is_plagiarized": is_plag,
            "rewrite": rewrite_result,
        })
        
        rewritten_sentences.append(rewrite_result["rewritten"])
    
    # Combine all rewritten sentences into a clean paragraph
    rewritten_full_text = " ".join(rewritten_sentences)
    
    # Build the plagiarized text (the input that was flagged)
    plagiarized_full_text = text2
    
    result["sentence_analysis"] = sentence_analysis
    result["plagiarized_text"] = plagiarized_full_text
    result["rewritten_text"] = rewritten_full_text
    result["total_sentences"] = len(sentences_text2)
    result["plagiarized_count"] = sum(1 for s in sentence_analysis if s["is_plagiarized"])
    
    return jsonify(result)


@app.route("/api/compare", methods=["POST"])
def compare_corpus():
    """
    Compare text against a corpus (sentence-by-sentence).
    
    For each flagged sentence → produce a human-like rewrite.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    input_text = data.get("text", "").strip()
    corpus_text = data.get("corpus", "").strip()

    if not input_text or not corpus_text:
        return jsonify({"error": "Both text and corpus are required"}), 400

    # Split into sentences
    input_sentences = split_into_sentences(input_text)
    corpus_sentences = split_into_sentences(corpus_text)

    if not input_sentences:
        return jsonify({"error": "Input text has no sentences"}), 400
    if not corpus_sentences:
        return jsonify({"error": "Corpus text has no sentences"}), 400

    # Compare each input sentence against all corpus sentences
    results = []
    total_plagiarism_score = 0
    rewritten_sentences = []

    for input_sent in input_sentences:
        best_match = None
        best_score = 0
        best_analysis = None

        for corpus_sent in corpus_sentences:
            analysis = analyze_pair(input_sent, corpus_sent)
            score = analysis["overall_score"]

            if score > best_score:
                best_score = score
                best_match = corpus_sent
                best_analysis = analysis

        # Determine if plagiarized
        is_plag = best_score >= 50
        
        # Rewrite if plagiarized
        if is_plag:
            rewrite_result = rewrite_sentence_human(input_sent, best_match or "")
        else:
            rewrite_result = {
                "plagiarized": input_sent,
                "rewritten": input_sent,
                "changes_made": [],
                "meaning_preserved": 1.0,
                "source_similarity_before": best_score / 100,
                "source_similarity_after": best_score / 100,
            }

        results.append({
            "input_sentence": input_sent,
            "best_match": best_match,
            "match_score": best_score,
            "analysis": best_analysis,
            "is_plagiarized": is_plag,
            "rewrite": rewrite_result,
        })
        total_plagiarism_score += best_score
        rewritten_sentences.append(rewrite_result["rewritten"])

    # Average plagiarism score
    avg_score = round(total_plagiarism_score / len(input_sentences), 1) if input_sentences else 0

    # Overall severity
    if avg_score >= 70:
        overall_severity = "high"
        overall_verdict = "Significant Plagiarism Detected"
    elif avg_score >= 40:
        overall_severity = "medium"
        overall_verdict = "Moderate Plagiarism Detected"
    elif avg_score >= 20:
        overall_severity = "low"
        overall_verdict = "Minor Similarities Found"
    else:
        overall_severity = "none"
        overall_verdict = "Content Appears Original"

    # Combine rewritten text
    rewritten_full_text = " ".join(rewritten_sentences)

    return jsonify({
        "overall_score": avg_score,
        "overall_severity": overall_severity,
        "overall_verdict": overall_verdict,
        "total_sentences": len(input_sentences),
        "flagged_sentences": sum(1 for r in results if r["is_plagiarized"]),
        "sentence_results": results,
        "original_text": input_text,
        "rewritten_text": rewritten_full_text,
    })


# ─────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Plagiarism Checker Web Application")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    load_model()

    print(f"\n🌐 Starting Plagiarism Checker on http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
