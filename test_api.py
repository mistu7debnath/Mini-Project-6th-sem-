import sys
sys.path.insert(0, 'Mini-Project-6th-sem-/plagiarism_app')

from app import app, load_model, analyze_pair, rewrite_sentence_human

# Load the model
load_model()

# Test analyze_pair
try:
    result = analyze_pair("The quick brown fox jumps over the lazy dog.", "The swift brown fox leaps over the sleepy dog.")
    print("analyze_pair OK:", result)
except Exception as e:
    import traceback
    print("ERROR in analyze_pair:")
    traceback.print_exc()

# Test rewrite_sentence_human
try:
    result = rewrite_sentence_human("The quick brown fox jumps over the lazy dog.", "The swift brown fox leaps over the sleepy dog.", 2, "remove_plagiarism")
    print("rewrite_sentence_human OK:", result)
except Exception as e:
    import traceback
    print("ERROR in rewrite_sentence_human:")
    traceback.print_exc()

