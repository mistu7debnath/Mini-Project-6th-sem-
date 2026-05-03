from fastapi import APIRouter
from app.services.similarity import compute_similarity, sentence_level_similarity

router = APIRouter()

@router.post("/analyze")
def analyze(data: dict):
    original = data["original"]
    rewritten = data["rewritten"]

    sim = compute_similarity(original, rewritten)
    sentence_sim = sentence_level_similarity(original, rewritten)

    return {
        "similarity": sim,
        "plagiarism_percent": sim * 100,
        "sentence_matches": sentence_sim
    }