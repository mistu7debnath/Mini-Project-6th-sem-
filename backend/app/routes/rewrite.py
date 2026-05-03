from fastapi import APIRouter
from app.services.llm_service import rewrite_text
from app.services.similarity import compute_similarity
from app.services.text_utils import clean_text

router = APIRouter()

THRESHOLD = 0.30
MAX_ITER = 5

@router.post("/rewrite")
def rewrite(input_data: dict):
    original = clean_text(input_data["text"])
    rewritten = original

    for _ in range(MAX_ITER):
        rewritten = rewrite_text(rewritten)
        similarity = compute_similarity(original, rewritten)

        if similarity < THRESHOLD:
            break

    return {
        "original": original,
        "rewritten": rewritten,
        "similarity": similarity,
        "plagiarism_percent": similarity * 100
    }