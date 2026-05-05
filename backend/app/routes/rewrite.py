from fastapi import APIRouter
from app.services.llm_service import rewrite_text
from app.services.similarity import compute_similarity
from app.db.memory_store import save_entry   # ✅ import

router = APIRouter()

@router.post("/rewrite")
def rewrite(data: dict):
    text = data["text"]   # ✅ defined here

    rewritten = rewrite_text(text)
    similarity = compute_similarity(text, rewritten)

    # ✅ SAVE INSIDE FUNCTION
    save_entry(text, rewritten, similarity)

    return {
        "original": text,
        "rewritten": rewritten,
        "similarity": similarity,
        "plagiarism_percent": similarity * 100
    }