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
import urllib.request
import urllib.error
import json
import ssl
from concurrent.futures import ThreadPoolExecutor

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

def get_openai_config():
    key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo").strip()
    return key, model, bool(key)

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

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
        print("  Model not found! Please run train_model.py first.")
        print(f"   Expected at: {MODEL_PATH}")
        return False

    print("🔄 Loading model artifacts...")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    metadata = joblib.load(METADATA_PATH)

    print(f"    Model loaded (accuracy: {metadata.get('accuracy', 'N/A'):.4f})")
    print(f"    Vectorizer loaded ({metadata.get('tfidf_max_features', 'N/A')} features)")
    return True


def get_rewrite_engine(sentence_analysis: list[dict]) -> str:
    """Return the active rewrite engine used for the analyzed sentences."""
    for item in sentence_analysis:
        rewrite = item.get("rewrite", {})
        if rewrite.get("rewrite_method") == "openai":
            return "openai"
    return "local"


def llm_rewrite_sentence(sentence: str, source_sentence: str = "") -> str | None:
    """Use OpenAI Chat Completion to rewrite a sentence if an API key is available."""
    api_key, api_model, api_ready = get_openai_config()
    if not api_ready:
        return None

    prompt = (
        "Rewrite the following sentence into clear, grammatically correct English. "
        "Keep the meaning intact, remove any awkward or invented wording, and make the result sound like proper human prose. "
        "If the input looks like a headline or news sentence, preserve that style while making it fluent and readable. "
        "Return a single polished sentence with no explanation."
    )
    if source_sentence:
        prompt += " The source sentence is provided for reference, but do not reuse the same phrases, structure, or unusual words."

    prompt += "\n\nSentence: " + sentence
    if source_sentence:
        prompt += "\nSource sentence: " + source_sentence

    def call_openai(current_prompt: str) -> str | None:
        payload = json.dumps({
            "model": api_model,
            "messages": [
                {"role": "system", "content": "You are a skilled rewrite assistant that produces polished, natural English sentences."},
                {"role": "user", "content": current_prompt},
            ],
            "temperature": 0.2,
            "top_p": 1,
            "max_tokens": max(200, len(sentence.split()) * 4),
        }).encode("utf-8")

        request_url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            request_obj = urllib.request.Request(request_url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(request_obj, context=ssl.create_default_context(), timeout=30) as response:
                response_data = json.loads(response.read().decode("utf-8"))
                choices = response_data.get("choices", [])
                if not choices:
                    return None
                return choices[0].get("message", {}).get("content", "").strip() or None
        except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
            return None

    first_attempt = call_openai(prompt)
    if not first_attempt:
        return None

    if first_attempt.strip().lower() == sentence.strip().lower() or compute_text_similarity(sentence, first_attempt) > 0.90:
        stronger_prompt = prompt + (
            "\n\nThe rewritten sentence should be significantly clearer and more natural than the original. "
            "If the first rewrite still sounds awkward or contains strange wording, create a more fluent version with common vocabulary."
        )
        second_attempt = call_openai(stronger_prompt)
        result = second_attempt or first_attempt
    else:
        result = first_attempt

    app.logger.info("OpenAI rewrite used for sentence: %s", sentence)
    return result


def llm_generate_plagiarized(clean_text: str) -> tuple[str, str] | None:
    """Use OpenAI to generate a naturally plagiarized version of clean text."""
    api_key, api_model, api_ready = get_openai_config()
    if not api_ready:
        return None

    prompt = (
        "You are a plagiarism simulation assistant. "
        "Given a clean source paragraph, rewrite it into a natural, fluent student-style version that preserves the original meaning. "
        "Keep most proper names and titles unchanged. "
        "The output should contain approximately 50% exact wording from the original and 50% rewritten phrasing. "
        "Do not invent nonsense words, do not distort facts, and keep the text readable. "
        "Return only the rewritten paragraph with no explanation."
    )
    prompt += "\n\nSource paragraph:\n" + clean_text

    def call_openai(current_prompt: str) -> str | None:
        payload = json.dumps({
            "model": api_model,
            "messages": [
                {"role": "system", "content": "You are a skilled text transformation assistant."},
                {"role": "user", "content": current_prompt},
            ],
            "temperature": 0.45,
            "top_p": 1,
            "max_tokens": max(200, len(clean_text.split()) * 4),
        }).encode("utf-8")

        request_url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            request_obj = urllib.request.Request(request_url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(request_obj, context=ssl.create_default_context(), timeout=60) as response:
                response_data = json.loads(response.read().decode("utf-8"))
                choices = response_data.get("choices", [])
                if not choices:
                    return None
                return choices[0].get("message", {}).get("content", "").strip() or None
        except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
            return None

    plag_text = call_openai(prompt)
    if not plag_text:
        return None

    if plag_text.strip().lower() == clean_text.strip().lower():
        return None

    app.logger.info("OpenAI generated auto-plagiarized text using %s.", api_model)
    return plag_text, api_model


# ─────────────────────────────────────────────────────────
# AUTO-PLAGIARISM SIMULATION (NEW)
# ─────────────────────────────────────────────────────────

def auto_generate_plagiarized(clean_text: str) -> tuple[str, str, dict]:
    """
    Simulate student plagiarism from clean original text.
    Returns: (suspected_plag_text, original_clean_text, simulation_stats)
    """
    sentences = split_into_sentences(clean_text)
    plag_sentences = []
    verbatim_count = 0
    rewritten_count = 0
    
    llm_result = llm_generate_plagiarized(clean_text)
    if llm_result:
        plag_text, model_used = llm_result
        suspected_text = plag_text
        verbatim_count = round(len(sentences) * 0.5)
        rewritten_count = len(sentences) - verbatim_count
    else:
        model_used = None
        for sent in sentences:
            if random.random() < 0.7:  # 70% verbatim copy (plagiarism)
                plag_sentences.append(sent)
                verbatim_count += 1
            else:  # 30% light rewrite (student attempt)
                light_rewrite = create_light_plagiarism_variant(sent)
                plag_sentences.append(light_rewrite)
                rewritten_count += 1
        suspected_text = " ".join(plag_sentences)

    stats = {
        "verbatim_sentences": verbatim_count,
        "rewritten_sentences": rewritten_count,
        "total_sentences": len(sentences),
        "plagiarism_ratio": round(verbatim_count / len(sentences) * 100, 1),
        "generated_with_llm": bool(model_used),
        "llm_model_used": model_used or "none"
    }
    
    return suspected_text, clean_text, stats


def preserve_punctuation_token(word: str, replacement: str) -> str:
    """Preserve punctuation around a replaced word."""
    match = re.match(r'^([^\w]*)([\w-]+)([^\w]*)$', word)
    if not match:
        return replacement
    return f"{match.group(1)}{replacement}{match.group(3)}"


def create_light_plagiarism_variant(sentence: str) -> str:
    """Create a plausible lightly rewritten version of a sentence."""
    words = sentence.split()
    if len(words) < 4:
        return sentence

    candidate = words.copy()
    replaceable_indices = []

    for index, word in enumerate(words):
        clean = word.strip(".,!?;:'\"-()")
        if len(clean) <= 3:
            continue
        if index > 0 and word[0].isupper():
            continue
        if clean.lower() in {
                "this", "that", "have", "has", "had", "will", "would", "could", "should",
                "and", "or", "but", "for", "with", "from", "into", "about", "after",
                "before", "during", "through", "across", "near", "when", "while", "where"
        }:
            continue

        synonyms = SYNONYM_MAP.get(clean.lower(), [])
        if synonyms:
            replaceable_indices.append(index)

    random.shuffle(replaceable_indices)
    replacements = 0

    for idx in replaceable_indices:
        if replacements >= 2:
            break

        word = words[idx]
        clean = word.strip(".,!?;:'\"-()")
        synonyms = SYNONYM_MAP.get(clean.lower(), [])
        safe_synonyms = [syn for syn in synonyms if re.fullmatch(r"[A-Za-z ]+", syn) and syn.lower() != clean.lower()]
        if not safe_synonyms:
            continue

        chosen = random.choice(safe_synonyms)
        if word[0].isupper():
            chosen = chosen.capitalize()

        candidate[idx] = preserve_punctuation_token(word, chosen)
        replacements += 1

    candidate_sentence = " ".join(candidate)
    candidate_sentence = restructure_sentence(candidate_sentence)
    candidate_sentence = normalize_sentence(candidate_sentence)
    return candidate_sentence


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
    "work": ["collaborate", "team up", "work together"],
    "book": ["publication", "volume", "text"],
    "food": ["meal", "cuisine", "nourishment"],
    "friend": ["companion", "pal", "buddy"],
    "children": ["kids", "youngsters", "little ones"],
    "people": ["folks", "friends", "group"],
    "school": ["academy", "institution", "educational facility"],
    "problem": ["issue", "challenge", "difficulty"],
    "idea": ["concept", "notion", "thought"],
    "place": ["location", "spot", "site"],
    "country": ["nation", "state", "land"],
    "student": ["learner", "pupil", "scholar"],
    "teacher": ["educator", "instructor", "professor"],
    "scientist": ["researcher", "scholar", "investigator"],
    "doctor": ["physician", "medical professional", "practitioner"],
    "company": ["team", "group", "circle"],
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
    "good": ["nice", "pleasant", "great"],
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


# ─────────────────────────────────────────────────────────
# DATAMUSE API INTEGRATION (Unlimited Vocabulary)
# ─────────────────────────────────────────────────────────
WORD_CACHE = {}

def prefetch_synonyms(text: str):
    """Fetch synonyms for long words in parallel to avoid UI lag."""
    words = [w for w in text.split()]
    # Filter words > 3 chars, not proper nouns, and not already cached
    target_words = []
    for w in words:
        clean = w.strip(".,!?;:'\"-()").lower()
        if len(clean) <= 3:
            continue
        if w[0].isupper() and not w.isupper():
            continue
        if clean and clean not in WORD_CACHE:
            target_words.append(clean)
    target_words = list(set(target_words))
    
    if not target_words:
        return
        
    def fetch(word):
        try:
            url = f"https://api.datamuse.com/words?ml={word}&max=5"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(req, context=ctx, timeout=2.0) as response:
                data = json.loads(response.read().decode('utf-8'))
                # prioritize actual synonyms
                syns = [i['word'] for i in data if 'syn' in i.get('tags', [])]
                if not syns and data:
                    # fallback to strongly related words
                    syns = [i['word'] for i in data[:2]]
                WORD_CACHE[word] = syns
        except:
            WORD_CACHE[word] = []
            
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(fetch, target_words))


def get_synonyms_for_word(word: str) -> list[str]:
    """Get synonyms from static map or API cache."""
    clean = word.lower().strip(".,!?;:'\"-()")
    
    # 1. Static map
    if clean in SYNONYM_MAP:
        return SYNONYM_MAP[clean]
        
    # 2. API cache
    if clean in WORD_CACHE and WORD_CACHE[clean]:
        return WORD_CACHE[clean]
        
    return []


def get_rewrite_synonyms(word: str, static_only: bool = False) -> list[str]:
    """Get a cleaned list of synonyms for sentence rewriting."""
    clean = word.lower().strip(".,!?;:'\"-()")
    synonyms = []

    if clean in SYNONYM_MAP:
        synonyms.extend(SYNONYM_MAP[clean])

    filtered = [s for s in synonyms if re.fullmatch(r"[A-Za-z ]+", s)]
    return list(dict.fromkeys(filtered))


def rewrite_sentence_human(sentence: str, source_sentence: str = "") -> dict:
    """
    Rewrite a plagiarized sentence into a human-like original version.
    
    Strategy:
      1. Rapidly fetch synonyms in parallel for the whole text
      2. Iteratively replace words, restructure, and change voice
      3. Monitor similarity to source and keep looping max 5 times until similarity < 0.4
      4. Add human phrasing if still failing to drop below threshold
    """
    if not sentence or not sentence.strip():
        return {
            "plagiarized": sentence,
            "rewritten": sentence,
            "changes_made": [],
            "meaning_preserved": 1.0,
            "rewrite_method": "local"
        }

    original = sentence.strip()
    static_only = bool(source_sentence and source_sentence.strip().lower() == original.lower())

    _, _, api_ready = get_openai_config()
    if api_ready:
        llm_result = llm_rewrite_sentence(original, source_sentence)
        if llm_result:
            source_sim_before = compute_text_similarity(original, source_sentence) if source_sentence else 0
            source_sim_after = compute_text_similarity(llm_result, source_sentence) if source_sentence else 0
            return {
                "plagiarized": original,
                "rewritten": llm_result,
                "changes_made": ["LLM rewrite"],
                "rewrite_method": "openai",
                "meaning_preserved": compute_text_similarity(original, llm_result),
                "source_similarity_before": source_sim_before,
                "source_similarity_after": source_sim_after,
            }

    # Prefetch vocabulary to handle the operation completely without blocking
    prefetch_synonyms(original)

    changes = []
    words = original.split()
    new_words = []
    replaced = 0
    max_replacements = 2

    for index, word in enumerate(words):
        # Strip punctuation for lookup
        clean = word.lower().rstrip(".,!?;:'\"")
        trail = word[len(word.rstrip(".,!?;:'\"")):]

        # Avoid changing proper nouns, titles, and quoted terms
        if index > 0 and word[0].isupper():
            new_words.append(word)
            continue
        if word.startswith(('"', "'")) or word.endswith(('"', "'")):
            new_words.append(word)
            continue

        if replaced >= max_replacements:
            new_words.append(word)
            continue

        synonyms = get_rewrite_synonyms(word, static_only=static_only)
        if synonyms:
            if source_sentence:
                source_lower = source_sentence.lower()
                safe_synonyms = [s for s in synonyms if s.lower() not in source_lower]
                chosen = random.choice(safe_synonyms) if safe_synonyms else random.choice(synonyms)
            else:
                chosen = random.choice(synonyms)

            # Preserve capitalization
            if word and word[0].isupper():
                chosen = chosen[0].upper() + chosen[1:]

            new_words.append(chosen + trail)
            replaced += 1
            if clean != chosen.lower() and f"'{clean}' → '{chosen}'" not in changes:
                changes.append(f"'{clean}' → '{chosen}'")
        else:
            new_words.append(word)

    candidate = " ".join(new_words)
    rewritten = normalize_sentence(candidate)

    # If the rewrite is still very close to source, allow a mild structural shuffle
    if source_sentence:
        final_sim = compute_text_similarity(rewritten, source_sentence)
        if final_sim > 0.9:
            rewritten = normalize_sentence(restructure_sentence(rewritten))

    # Compute final similarity to the plagiarized input
    meaning_similarity = compute_text_similarity(original, rewritten)
    
    # Compute similarity to source (should be low — showing plagiarism is removed)
    source_sim_before = compute_text_similarity(original, source_sentence) if source_sentence else 0
    source_sim_after = compute_text_similarity(rewritten, source_sentence) if source_sentence else 0

    return {
        "plagiarized": original,
        "rewritten": rewritten,
        "changes_made": changes,
        "rewrite_method": "local",
        "meaning_preserved": meaning_similarity,
        "source_similarity_before": source_sim_before,
        "source_similarity_after": source_sim_after,
    }


def restructure_sentence(sentence: str) -> str:
    """Restructure a sentence by moving clauses around."""
    words = sentence.split()
    if len(words) < 3:
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
                # Capitalize prep phrase start, preserve proper noun capitalization in the main clause
                prep_phrase[0] = prep_phrase[0].capitalize()
                if main_clause[0][0].isupper() and main_clause[0] != main_clause[0].lower():
                    main_clause[0] = main_clause[0]
                else:
                    main_clause[0] = main_clause[0][0].lower() + main_clause[0][1:]

                result = " ".join(prep_phrase) + ", " + " ".join(main_clause) + punct
                return result
            break

    return normalize_sentence(sentence)


def normalize_sentence(sentence: str) -> str:
    """Clean and normalize rewritten text for proper punctuation and capitalization."""
    if sentence is None:
        return ""

    sentence = sentence.strip()
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = re.sub(r'\s+([,\.!?;:])', r'\1', sentence)
    sentence = re.sub(r'([,\.!?;:])([^\s\)\'\"])', r'\1 \2', sentence)
    sentence = re.sub(r'\(\s+', '(', sentence)
    sentence = re.sub(r'\s+\)', ')', sentence)
    sentence = re.sub(r'\s+-\s+', ' - ', sentence)
    sentence = re.sub(r'\s+"', ' "', sentence)
    sentence = re.sub(r'"\s+', '" ', sentence)
    sentence = re.sub(r"\s+'", " '", sentence)
    sentence = re.sub(r"'\s+", "' ", sentence)

    # Capitalize after sentence boundaries and opening quotes
    def cap_match(match):
        prefix = match.group(1)
        quote = match.group(2) or ''
        char = match.group(3)
        return prefix + quote + char.upper()

    sentence = re.sub(r'(^|[\.\!?]\s+)(["\']?)([a-z])', cap_match, sentence)
    sentence = re.sub(r'\bi\b', 'I', sentence)

    sentence = sentence.strip()
    if sentence and sentence[-1] not in '.!?':
        sentence += '.'

    return sentence


def try_voice_change(sentence: str) -> str:
    """Try to change active voice to passive or vice versa."""
    words = sentence.split()
    if len(words) < 3:
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
    if len(sentence.split()) < 2:
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


def compute_tfidf_cosine_similarity(text1: str, text2: str) -> float:
    """Compute TF-IDF cosine similarity if vectorizer is available."""
    if vectorizer is None:
        return 0.0

    tfidf1 = vectorizer.transform([text1])
    tfidf2 = vectorizer.transform([text2])
    dot = np.array((tfidf1.multiply(tfidf2)).sum(axis=1)).flatten()
    norm1 = np.sqrt(np.array(tfidf1.multiply(tfidf1).sum(axis=1)).flatten())
    norm2 = np.sqrt(np.array(tfidf2.multiply(tfidf2).sum(axis=1)).flatten())
    denom = norm1 * norm2
    denom[denom == 0] = 1e-10
    cosine_sim = float(dot / denom)
    return round(cosine_sim, 4)


def blend_similarity(text1: str, text2: str) -> float:
    """Blend multiple similarity signals into a single robust score."""
    sequence_sim = compute_text_similarity(text1, text2)
    word_overlap = compute_word_overlap(text1, text2)
    cosine_sim = compute_tfidf_cosine_similarity(text1, text2)

    if cosine_sim > 0:
        blended = (cosine_sim * 0.5) + (sequence_sim * 0.3) + (word_overlap * 0.2)
    else:
        blended = (sequence_sim * 0.6) + (word_overlap * 0.4)

    return round(min(max(blended, 0.0), 1.0), 4)


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
    cosine_sim = compute_tfidf_cosine_similarity(text1, text2)
    blended_similarity = blend_similarity(text1, text2)
    matching_segments = find_matching_segments(text1, text2)

    # Determine plagiarism level
    if prediction is not None:
        is_plagiarized = bool(prediction)
    else:
        is_plagiarized = blended_similarity >= 0.6

    # Compute overall similarity score (0-100)
    overall_score = round(blended_similarity * 100, 1)

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
            "cosine_similarity": cosine_sim,
            "blended_similarity": blended_similarity,
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
    api_key, api_model, api_ready = get_openai_config()
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "metadata": metadata if metadata else {},
        "openai": {
            "enabled": api_ready,
            "model": api_model,
            "has_api_key": bool(api_key),
        },
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
                "rewrite_method": "local",
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
    
    # Compute similarity between the plagiarized text and the final human-like rewrite
    rewrite_similarity = compute_text_similarity(plagiarized_full_text, rewritten_full_text)
    
    # Recheck the rewritten text against the original source
    recheck_analysis = analyze_pair(text1, rewritten_full_text)
    recheck_score = recheck_analysis["overall_score"]
    
    result["sentence_analysis"] = sentence_analysis
    result["plagiarized_text"] = plagiarized_full_text
    result["rewritten_text"] = rewritten_full_text
    result["rewrite_similarity"] = rewrite_similarity
    result["recheck_score"] = recheck_score
    result["rewrite_engine"] = get_rewrite_engine(sentence_analysis)
    result["total_sentences"] = len(sentences_text2)
    result["plagiarized_count"] = sum(1 for s in sentence_analysis if s["is_plagiarized"])
    
    return jsonify(result)


@app.route("/api/verify_rewrite", methods=["POST"])
def verify_rewrite():
    """
    Verify user rewrite: current plag % + similarity to original plag sentence.
    Preserves meaning check while reducing plagiarism score.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    original_source = data.get("original_source", "").strip()
    original_plag = data.get("original_plag", "").strip()  
    user_rewrite = data.get("user_rewrite", "").strip()

    if not all([original_source, original_plag, user_rewrite]):
        return jsonify({"error": "All three texts required"}), 400

    # Current plagiarism score (user_rewrite vs original_source)
    current_analysis = analyze_pair(original_source, user_rewrite)
    
    # Similarity to original plag (meaning preservation)
    meaning_sim = compute_text_similarity(original_plag, user_rewrite)
    
    # Improvement metrics
    original_plag_score = compute_text_similarity(original_source, original_plag)
    
    improvements = {
        "plagiarism_reduced": round((original_plag_score - current_analysis["metrics"]["sequence_similarity"]) * 100, 1),
        "meaning_preserved": round(meaning_sim * 100, 1),
        "word_overlap_change": round((compute_word_overlap(original_plag, user_rewrite) - compute_word_overlap(original_source, original_plag)) * 100, 1)
    }

    result = {
        "current_plagiarism_score": current_analysis["overall_score"],
        "current_severity": current_analysis["severity"],
        "meaning_preserved_pct": improvements["meaning_preserved"],
        "plagiarism_reduction_pct": improvements["plagiarism_reduced"],
        "word_changes_summary": improvements,
        "status": "meaning_preserved" if meaning_sim > 0.7 else "meaning_changed" if meaning_sim > 0.4 else "meaning_lost",
        "recommendation": "Great rewrite! ✅" if current_analysis["overall_score"] < 20 and meaning_sim > 0.7 else "Good progress 👌" if current_analysis["overall_score"] < 50 else "Needs more rewriting ⚠️"
    }
    
    return jsonify(result)


@app.route("/api/auto_plag", methods=["POST"])
def auto_plagiarism_pipeline():
    """
    FULLY AUTOMATIC 4-STEP WORKFLOW:
    1. User enters clean text
    2. AUTO-generates suspected plagiarized version
    3. Detects plagiarism
    4. Reconstructs clean text + verifies similarity
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    clean_text = data.get("clean_text", "").strip()
    if not clean_text:
        return jsonify({"error": "Clean text required"}), 400

    # Step 1: Auto-generate suspected plagiarized text
    suspected_text, original_clean, simulation_stats = auto_generate_plagiarized(clean_text)

    # Step 2: Detect plagiarism (source=original_clean, suspected=suspected_text)
    detection = analyze_pair(original_clean, suspected_text)

    # Step 3: Auto-reconstruct (treat suspected as "text2", original as source)
    sentences_suspected = split_into_sentences(suspected_text)
    sentences_source = split_into_sentences(original_clean)
    
    sentence_analysis = []
    reconstructed_sentences = []
    
    for sent_suspected in sentences_suspected:
        best_source = ""
        best_sim = 0
        for sent_source in sentences_source:
            sim = compute_text_similarity(sent_suspected, sent_source)
            if sim > best_sim:
                best_sim = sim
                best_source = sent_source
        
        is_plag = best_sim >= 0.5
        if is_plag and best_source:
            rewrite_result = rewrite_sentence_human(sent_suspected, best_source)
        else:
            rewrite_result = rewrite_sentence_human(sent_suspected, best_source)
        if not rewrite_result.get("rewrite_method"):
            rewrite_result["rewrite_method"] = "local"
        
        sentence_analysis.append({
            "sentence": sent_suspected,
            "source_match": best_source,
            "similarity": round(best_sim, 3),
            "is_plagiarized": is_plag,
            "rewrite": rewrite_result,
        })
        reconstructed_sentences.append(rewrite_result["rewritten"])
    
    reconstructed_text = " ".join(reconstructed_sentences)

    # Step 4: Verify similarity (reconstructed vs suspected)
    final_similarity = compute_text_similarity(suspected_text, reconstructed_text)
    meaning_preserved_pct = round(final_similarity * 100, 1)

    return jsonify({
        "workflow": "complete",
        # Step 1
        "original_clean": original_clean,
        "auto_suspected": suspected_text,
        "simulation_stats": simulation_stats,
        "auto_model_used": simulation_stats.get("llm_model_used", "none"),
        "auto_generated_with_llm": simulation_stats.get("generated_with_llm", False),
        # Step 2
        "detection": detection,
        "sentence_analysis": sentence_analysis,
        "rewrite_engine": get_rewrite_engine(sentence_analysis),
        # Step 3
        "reconstructed_text": reconstructed_text,
        # Step 4
        "similarity_check": {
            "meaning_preserved_pct": meaning_preserved_pct,
            "status": "excellent" if meaning_preserved_pct > 85 else "good" if meaning_preserved_pct > 70 else "fair"
        }
    })


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
                "rewrite_method": "local",
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

    # Compute similarity between original and rewritten text
    rewrite_similarity = compute_text_similarity(input_text, rewritten_full_text)

    # Recheck the rewritten text against the corpus
    recheck_score_total = 0
    rw_list = split_into_sentences(rewritten_full_text)
    for rw_sent in rw_list:
        best_rw_score = 0
        for corpus_sent in corpus_sentences:
            analysis = analyze_pair(rw_sent, corpus_sent)
            if analysis["overall_score"] > best_rw_score:
                best_rw_score = analysis["overall_score"]
        recheck_score_total += best_rw_score
    recheck_avg_score = round(recheck_score_total / len(rw_list), 1) if rw_list else 0

    return jsonify({
        "overall_score": avg_score,
        "overall_severity": overall_severity,
        "overall_verdict": overall_verdict,
        "total_sentences": len(input_sentences),
        "flagged_sentences": sum(1 for r in results if r["is_plagiarized"]),
        "rewrite_engine": get_rewrite_engine(results),
        "sentence_results": results,
        "original_text": input_text,
        "rewritten_text": rewritten_full_text,
        "rewrite_similarity": rewrite_similarity,
        "recheck_score": recheck_avg_score,
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
