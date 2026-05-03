from fastapi import APIRouter

router = APIRouter()

history = []

@router.post("/history")
def save(entry: dict):
    history.append(entry)
    return {"status": "saved"}

@router.get("/history")
def get_history():
    return history

@router.delete("/history")
def clear():
    history.clear()
    return {"status": "cleared"}