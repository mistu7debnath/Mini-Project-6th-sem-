import requests

def rewrite_text(text):
    prompt = f"""
Rewrite the following text to reduce plagiarism.
Keep meaning but change structure.

Text:
{text}
"""

    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        return res.json()["response"]
    except:
        return "Fallback rewritten text."