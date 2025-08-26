# tester.py
from main import rag_response

payload = {
    "application": "Akuiteo",
    "problem": "Plafonds des notes de frais",
    "summary": "Quels sont les plafonds et comment ajouter une dépense",
    "confidence": True,
    "ask_user": None
}

print("🤖", rag_response(payload, session_id="demo"))
