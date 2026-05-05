history = []

def save_entry(original, rewritten, similarity):
    history.append({
        "original": original,
        "rewritten": rewritten,
        "similarity": similarity
    })

def get_history():
    return history