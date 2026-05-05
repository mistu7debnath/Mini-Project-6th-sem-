from pydantic import BaseModel

class TextRequest(BaseModel):
    text: str

class AnalyzeRequest(BaseModel):
    original: str
    rewritten: str