"""
Fixed Deep Paraphrase Engine v3
Professional Plagiarism Remover + AI Humanizer
"""

import re
from difflib import SequenceMatcher

# ----------------------------------
# Load Models
# ----------------------------------

HAS_NEURAL = False
paraphraser = None
similarity_model = None
util = None

try:
    from transformers import pipeline
    from sentence_transformers import SentenceTransformer, util

    paraphraser = pipeline(
        "text2text-generation",
        model="Vamsi/T5_Paraphrase_Paws"
    )

    similarity_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    HAS_NEURAL = True

    print('HAS_NEURAL:', HAS_NEURAL)

except Exception:
    HAS_NEURAL = False
    print("HAS_NEURAL =", HAS_NEURAL)


# ----------------------------------
# Safety Lists
# ----------------------------------

FORBIDDEN_WORDS = {
    "otiose",
    "doom",
    "missive",
    "vocable",
    "wherefore",
    "thereupon"
}

PROTECTED_TERMS = {
    "scientists",
    "algorithm",
    "algorithms",
    "english alphabet",
    "19th century",
    "plagiarism",
    "ai",
    "machine learning"
}

SAFE_SYNONYMS = {
    "quick":"swift",
    "jumps":"leaps",
    "lazy":"sleepy",
    "important":"crucial",
    "shows":"demonstrates",
    "help":"assist",
    "improve":"enhance"
}


# ----------------------------------
# Utilities
# ----------------------------------

def normalize(text):

    text = re.sub(r'\s+',' ',text).strip()

    if len(text)>0 and text[-1] not in ".!?":
        text += "."

    if len(text)>1:
        text=text[0].upper()+text[1:]

    return text


def contains_forbidden(text):

    t=text.lower()

    for word in FORBIDDEN_WORDS:
        if word in t:
            return True

    return False


def semantic_similarity(a,b):

    if not HAS_NEURAL:
        return SequenceMatcher(
            None,
            a.lower(),
            b.lower()
        ).ratio()

    try:
        emb1 = similarity_model.encode(
            a,
            convert_to_tensor=True
        )

        emb2 = similarity_model.encode(
            b,
            convert_to_tensor=True
        )

        return float(
            util.cos_sim(emb1,emb2)[0][0]
        )

    except:
        return SequenceMatcher(
            None,
            a.lower(),
            b.lower()
        ).ratio()


# ----------------------------------
# Humanizer
# ----------------------------------

def humanize(text):

    replacements = {

        "utilize":"use",
        "commence":"start",
        "therefore":"so",
        "demonstrates":"shows",

        "in order to":"to",
        "a number of":"several"

    }

    for old,new in replacements.items():
        text=text.replace(old,new)

    return text


# ----------------------------------
# Safe Backup Synonym Rewrite
# ----------------------------------

def safe_rewrite(text):

    words=text.split()

    out=[]

    for w in words:

        core=w.lower().strip(".,!?")

        if core in SAFE_SYNONYMS:
            replacement=SAFE_SYNONYMS[core]

            if w[0].isupper():
                replacement=replacement.capitalize()

            if w[-1] in ".,!?":
                replacement+=w[-1]

            out.append(replacement)

        else:
            out.append(w)

    return " ".join(out)


# ----------------------------------
# Neural Rewrite
# ----------------------------------

def neural_rewrite(text):

    if not HAS_NEURAL:
        return safe_rewrite(text)

    prompt=f"""
Paraphrase this sentence.

Rules:
1. Keep meaning identical
2. Use different vocabulary
3. Change sentence structure
4. Reorder clauses where possible
5. Sound natural and human-written
6. Avoid repeating original phrases

Sentence:
{text}
"""

    try:

        outputs=paraphraser(
            prompt,
            max_length=250,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            num_return_sequences=3
        )

        best=text
        best_score=-999

        for item in outputs:

            candidate=item["generated_text"]

            candidate=humanize(
                normalize(candidate)
            )

            if contains_forbidden(candidate):
                continue

            sem=semantic_similarity(
                text,
                candidate
            )

            lex=SequenceMatcher(
                None,
                text.lower(),
                candidate.lower()
            ).ratio()

            # Combined score
            score=(sem*0.7)- (lex*0.3)

            if sem >= 0.80 and score > best_score:
                best_score=score
                best=candidate

        return best

    except:
        return safe_rewrite(text)


# ----------------------------------
# Iterative Plagiarism Reduction
# ----------------------------------

def reduce_plagiarism(text):

    current=text

    for _ in range(3):

        rewritten=neural_rewrite(
            current
        )

        if contains_forbidden(rewritten):
            return current

        sim=SequenceMatcher(
            None,
            text.lower(),
            rewritten.lower()
        ).ratio()

        # lower lexical similarity desired
        if sim < 0.75:
            return rewritten

        current=rewritten

    return current


# ----------------------------------
# Public API
# ----------------------------------

def deep_paraphrase(sentence):

    print("INPUT:", sentence)
    if len(sentence.split()) < 3:
        print("MODEL OUTPUT:", sentence)
        return sentence

    result = reduce_plagiarism(sentence)

    print("MODEL OUTPUT:", result)

    # Final meaning protection
    if semantic_similarity(
        sentence,
        result
    ) < 0.80:
        print("MODEL OUTPUT:", sentence)
        return sentence

    # --- FORCE TEST RETURN ---
    # Uncomment the next line to test Flask pipeline:
    # return "A paragraph consists of several sentences focused on one central idea."

    return result


def deep_paraphrase_paragraph(paragraph):

    sentences=re.split(
        r'(?<=[.!?])\s+',
        paragraph.strip()
    )

    outputs=[]

    for s in sentences:

        if s.strip():
            outputs.append(
                deep_paraphrase(s)
            )

    return " ".join(outputs)



# ----------------------------------
# Test
# ----------------------------------

if __name__=="__main__":

    test="""
The quick brown fox jumps over the lazy dog.
This is a classic pangram sentence that contains every letter of the English alphabet.
Scientists use it to test font rendering and text processing algorithms.
"""

    print("ORIGINAL:")
    print(test)

    print("\nREWRITTEN:")
    print(
        deep_paraphrase_paragraph(
            test
        )
    )